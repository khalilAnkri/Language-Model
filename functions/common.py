"""
common.py - code shared by 01_baseline.ipynb and 02_attention.ipynb.

Everything that must be IDENTICAL between the baseline and the attention model
lives here: the corpus subset, the train/val/test split, the vocabulary and the
sliding windows. If both notebooks call prepare_data(CFG) with the same CFG,
they are guaranteed to see exactly the same data.

It also holds the model (baseline LSTM and the attention variants), the training
loop with early stopping and the evaluation function, so that the baseline and
the attention model are trained and scored by the SAME code.
"""
import copy
import gzip
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CFG = dict(
    # --- data (fixed for the WHOLE project, never changed between runs) ---
    data_seed=42,            # controls article sampling + split (NOT model init)
    n_articles=1000,         # size of the Wikipedia subset
    min_article_words=200,   # skip stubs / very short pages
    train_frac=0.8,          # split BY ARTICLE: 80 / 10 / 10
    val_frac=0.1,            # (test fraction = 1 - train_frac - val_frac)
    min_freq=5,              # words seen < 5 times in TRAIN become <UNK>
    window=20,               # context length (number of input words)
    cache_path="data/articles.json.gz",  # the sampled subset is saved here
)

UNK = "<UNK>"
UNK_ID = 0


# --------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Call before building/training a model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """'cuda' if a GPU is available, otherwise the CPU fallback."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# 1. Corpus: a small, fixed, saved subset of English Wikipedia
# --------------------------------------------------------------------------
def load_articles(cfg=CFG):
    """
    Return a list of {"id", "title", "text"} dicts.

    First call: streams the Hugging Face dataset (no 20 GB download), shuffles it
    with the data seed, keeps the first n_articles that are long enough, and saves
    them to cfg["cache_path"]. Next calls (and your teammate, if the file is
    shared through git) read the saved file, so the subset is identical.
    """
    n, path = cfg["n_articles"], cfg["cache_path"]

    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= n:
            return cached[:n]          # same seed -> a smaller run is a prefix
        print(f"Cache has {len(cached)} < {n} articles, downloading again.")

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=cfg["data_seed"], buffer_size=10_000)

    articles = []
    for ex in ds:
        if len(ex["text"].split()) < cfg["min_article_words"]:
            continue
        articles.append({"id": ex["id"], "title": ex["title"], "text": ex["text"]})
        if len(articles) == n:
            break

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(articles, f)
    return articles


# --------------------------------------------------------------------------
# 2. Tokenization
# --------------------------------------------------------------------------
def tokenize(text: str):
    """Word-level tokenization with NLTK (same tokenizer as the original notebook).
    Case is kept: the GoogleNews Word2Vec vectors are case-sensitive."""
    from nltk.tokenize import word_tokenize
    return word_tokenize(text, language="english")


def tokenize_articles(articles):
    import nltk
    nltk.download("punkt_tab", quiet=True)
    return [tokenize(a["text"]) for a in articles]


# --------------------------------------------------------------------------
# 3. Split BY ARTICLE (so no article is shared between train / val / test)
# --------------------------------------------------------------------------
def split_indices(n_articles: int, cfg=CFG):
    rng = np.random.RandomState(cfg["data_seed"])
    perm = rng.permutation(n_articles)
    n_train = int(n_articles * cfg["train_frac"])
    n_val = int(n_articles * cfg["val_frac"])
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


# --------------------------------------------------------------------------
# 4. Vocabulary (TRAIN ONLY) and encoding
# --------------------------------------------------------------------------
def build_vocab(train_token_lists, min_freq: int):
    """
    word -> id. Id 0 is <UNK>. Only words seen >= min_freq times in the TRAINING
    articles get their own id; every other word is mapped to <UNK>. This is what
    lets the model actually SEE <UNK> during training.
    Sorted by (-count, word) so the ids are deterministic.
    """
    counts = Counter(w for toks in train_token_lists for w in toks)
    kept = sorted((w for w, c in counts.items() if c >= min_freq),
                  key=lambda w: (-counts[w], w))
    word_to_id = {UNK: UNK_ID}
    for w in kept:
        word_to_id[w] = len(word_to_id)
    return word_to_id, counts


def encode(token_lists, word_to_id):
    """List of token lists -> list of int64 arrays (unknown words -> UNK_ID)."""
    return [np.array([word_to_id.get(w, UNK_ID) for w in toks], dtype=np.int64)
            for toks in token_lists]


# --------------------------------------------------------------------------
# 5. Sliding windows for next-word prediction
# --------------------------------------------------------------------------
def make_windows(id_arrays, window: int):
    """
    For every article with ids = [w0, w1, ..., wn]:
        X[i] = ids[i : i+window]      (the `window` previous words)
        y[i] = ids[i+window]          (the word to predict = X shifted by one)
    Windows never cross an article boundary.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    Xs, ys = [], []
    for ids in id_arrays:
        if len(ids) <= window:
            continue                                   # too short: no window
        w = sliding_window_view(ids, window + 1)       # (n - window, window + 1)
        Xs.append(w[:, :window])
        ys.append(w[:, window])
    X = np.concatenate(Xs).astype(np.int64)
    y = np.concatenate(ys).astype(np.int64)
    return X, y


# --------------------------------------------------------------------------
# 6. Everything together
# --------------------------------------------------------------------------
def prepare_data(cfg=CFG, articles=None, tokenizer=None):
    """
    Build the full dataset. `articles` / `tokenizer` can be injected for testing.
    Returns a dict with X_/y_ arrays for train/val/test, the vocabulary, and the
    bookkeeping needed by sanity_checks().
    """
    if articles is None:
        articles = load_articles(cfg)
    if tokenizer is None:
        tokens = tokenize_articles(articles)
    else:
        tokens = [tokenizer(a["text"]) for a in articles]

    idx = split_indices(len(articles), cfg)
    tok = {s: [tokens[i] for i in idx[s]] for s in idx}

    word_to_id, train_counts = build_vocab(tok["train"], cfg["min_freq"])
    ids = {s: encode(tok[s], word_to_id) for s in tok}

    data = {
        "cfg": dict(cfg),
        "word_to_id": word_to_id,
        "id_to_word": {i: w for w, i in word_to_id.items()},
        "vocab_size": len(word_to_id),
        "train_counts": train_counts,
        "article_ids": {s: [articles[i]["id"] for i in idx[s]] for s in idx},
        "ids": ids,   # per-article id arrays (kept for the sanity checks)
    }
    for s in ("train", "val", "test"):
        data[f"X_{s}"], data[f"y_{s}"] = make_windows(ids[s], cfg["window"])
    return data


# --------------------------------------------------------------------------
# 7. Sanity checks and reference scores
# --------------------------------------------------------------------------
def sanity_checks(data, verbose=True):
    """Assert the properties we rely on and return summary statistics."""
    cfg, V, L = data["cfg"], data["vocab_size"], data["cfg"]["window"]
    splits = ("train", "val", "test")

    # (a) no article appears in two splits -> no leakage between splits
    sets = {s: set(data["article_ids"][s]) for s in splits}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])

    # (b) the vocabulary comes from TRAIN only, with the min_freq rule
    for w, i in data["word_to_id"].items():
        if w != UNK:
            assert data["train_counts"][w] >= cfg["min_freq"], w
    assert data["word_to_id"][UNK] == UNK_ID

    stats = {}
    for s in splits:
        X, y = data[f"X_{s}"], data[f"y_{s}"]
        # (c) shapes and id range
        assert X.ndim == 2 and X.shape[1] == L and y.shape == (len(X),)
        assert X.min() >= 0 and X.max() < V and y.min() >= 0 and y.max() < V

        # (d) number of windows = sum over articles of max(len - L, 0)
        expected = sum(max(len(a) - L, 0) for a in data["ids"][s])
        assert len(X) == expected, (s, len(X), expected)

        # (e) target = input shifted by one (checked inside the first long article)
        first = next(a for a in data["ids"][s] if len(a) > L + 1)
        k = len(first) - L
        assert np.array_equal(X[0], first[:L]) and y[0] == first[L]
        assert np.array_equal(X[1, :-1], X[0, 1:]) and X[1, -1] == y[0]
        assert k >= 2

        n_tok = sum(len(a) for a in data["ids"][s])
        n_unk = sum(int((a == UNK_ID).sum()) for a in data["ids"][s])
        stats[s] = {
            "articles": len(data["article_ids"][s]),
            "tokens": n_tok,
            "windows": len(X),
            "unk_rate_%": round(100 * n_unk / n_tok, 2),
            "unk_target_%": round(100 * float((y == UNK_ID).mean()), 2),
        }

    # (f) <UNK> must actually occur in training, otherwise it is never learned
    assert stats["train"]["unk_rate_%"] > 0, "UNK never seen in training!"

    if verbose:
        print(f"vocab_size = {V}   window = {L}   min_freq = {cfg['min_freq']}")
        print(f"{'split':<6} {'articles':>8} {'tokens':>9} {'windows':>9} "
              f"{'UNK in text %':>14} {'UNK as target %':>16}")
        for s in splits:
            r = stats[s]
            print(f"{s:<6} {r['articles']:>8} {r['tokens']:>9} {r['windows']:>9} "
                  f"{r['unk_rate_%']:>14} {r['unk_target_%']:>16}")
        print("All sanity checks passed.")
    return stats


def reference_perplexities(data):
    """
    Perplexities of two trivial 'models' on the val/test targets, to put the
    neural scores in perspective:
      uniform : every word has probability 1/V          -> perplexity = V
      unigram : p(w) from TRAIN target counts (add-one) -> ignores the context
    A useful LSTM must clearly beat the unigram reference.
    """
    V = data["vocab_size"]
    counts = np.bincount(data["y_train"], minlength=V).astype(np.float64)
    p = (counts + 1) / (counts.sum() + V)
    out = {"uniform": float(V)}
    for s in ("val", "test"):
        nll = -np.log(p[data[f"y_{s}"]]).mean()
        out[f"unigram_{s}"] = float(np.exp(nll))
    return out


# ==========================================================================
# 8. Word2Vec input
# ==========================================================================
def load_word2vec(name="word2vec-google-news-300"):
    """Pre-trained GoogleNews vectors (300-d). Downloaded once by gensim (~1.6 GB)."""
    import gensim.downloader as api
    return api.load(name)


def build_embedding_matrix(data, kv):
    """
    (vocab_size, 300) float32 matrix: row i = Word2Vec vector of word i.
    Words missing from Word2Vec (and <UNK>) get a ZERO vector.
    Returns (matrix, coverage) where coverage says how many words / training
    tokens actually received a pre-trained vector.
    """
    w2i, counts = data["word_to_id"], data["train_counts"]
    M = np.zeros((len(w2i), kv.vector_size), dtype=np.float32)
    found_types = 0
    found_tokens = 0
    total_tokens = 0
    for w, i in w2i.items():
        if w == UNK:
            continue
        total_tokens += counts[w]
        if w in kv:
            M[i] = kv[w]
            found_types += 1
            found_tokens += counts[w]
    coverage = {
        "word_types_%": round(100 * found_types / (len(w2i) - 1), 2),
        "train_tokens_%": round(100 * found_tokens / total_tokens, 2),
    }
    return M, coverage


# ==========================================================================
# 9. Model: LSTM language model on a window, optional attention head
# ==========================================================================
class WindowLM(nn.Module):
    """
    Predicts the word that follows a window of `L` words.

    input_type : "onehot"   -> one-hot vectors (size = vocab_size) go straight into the LSTM
                 "word2vec" -> FROZEN pre-trained vectors (embedding_matrix) go into the LSTM
    attention  : None       -> baseline: predict from the last hidden state h_t only
                 "single"   -> Salton et al. single score   v . tanh(Ws h_i)
                 "combined" -> Salton et al. combined score v . tanh(Ws h_i + Wq h_t)
                 "mean"     -> CONTROL: plain average of the past states (no learned scores)

    Attention (Salton et al., 2017) works over the LSTM's own previous hidden
    states h_1..h_{t-1} (the "memory"), not over the input words, and it is NOT a
    Transformer: the LSTM still reads the window one word at a time.
    """

    def __init__(self, vocab_size, hidden_size=128, num_layers=1, dropout=0.2,
                 input_type="onehot", embedding_matrix=None, attention=None):
        super().__init__()
        assert input_type in ("onehot", "word2vec")
        assert attention in (None, "single", "combined", "mean")
        self.vocab_size, self.input_type, self.attention = vocab_size, input_type, attention
        H = hidden_size

        if input_type == "word2vec":
            self.emb = nn.Embedding.from_pretrained(
                torch.as_tensor(embedding_matrix), freeze=True)   # frozen
            in_size = self.emb.embedding_dim
        else:
            in_size = vocab_size
        self.rnn = nn.LSTM(in_size, H, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)

        if attention in ("single", "combined"):
            self.W_s = nn.Linear(H, H, bias=False)       # scores the stored state h_i
            self.v = nn.Linear(H, 1, bias=False)         # turns a vector into one scalar score
        if attention == "combined":
            self.W_q = nn.Linear(H, H, bias=False)       # "query": the current state h_t
        if attention is not None:
            self.W_c = nn.Linear(2 * H, H)               # merges [h_t ; c_t]  (with bias)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(H, vocab_size)               # softmax is inside the loss

    def embed(self, x):                                  # x: (B, L) word ids
        if self.input_type == "onehot":
            return F.one_hot(x, self.vocab_size).float() # (B, L, V)
        return self.emb(x)                               # (B, L, 300)

    def encode(self, x):
        out, _ = self.rnn(self.embed(x))                 # (B, L, H): h_1 ... h_L
        return out

    def forward(self, x, return_attn=False):
        out = self.encode(x)                             # (B, L, H)
        h_t = out[:, -1]                                 # (B, H)   current state
        a = None
        if self.attention is None:
            feat = h_t
        else:
            mem = out[:, :-1]                            # (B, L-1, H) previous states only
            if self.attention == "mean":
                a = mem.new_full(mem.shape[:2], 1.0 / mem.size(1))   # uniform weights
            else:
                s = self.W_s(mem)                        # (B, L-1, H)
                if self.attention == "combined":
                    s = s + self.W_q(h_t).unsqueeze(1)   # broadcast h_t over positions
                scores = self.v(torch.tanh(s)).squeeze(-1)           # (B, L-1)
                a = torch.softmax(scores, dim=-1)                    # weights sum to 1
            c = torch.bmm(a.unsqueeze(1), mem).squeeze(1)            # (B, H) context vector
            feat = torch.tanh(self.W_c(torch.cat([h_t, c], dim=-1))) # (B, H)
        logits = self.fc(self.drop(feat))                # (B, V)
        return (logits, a) if return_attn else logits


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ==========================================================================
# 10. Sanity checks for the model
# ==========================================================================
def check_no_future_leak(model, X, device, j=10):
    """Changing the word at position j must NOT change the hidden states before j
    (the LSTM reads left to right), but must change the later ones."""
    model.eval().to(device)
    x = torch.as_tensor(X[:8]).to(device)
    x2 = x.clone()
    x2[:, j] = (x2[:, j] + 1) % model.vocab_size
    with torch.no_grad():
        o1, o2 = model.encode(x), model.encode(x2)
    assert torch.allclose(o1[:, :j], o2[:, :j], atol=1e-6), "future token leaked into the past!"
    assert not torch.allclose(o1[:, j:], o2[:, j:]), "later states ignore the changed token?"
    return True


def overfit_one_batch(model, data, device, steps=300, batch=32, lr=3e-3):
    """Classic debugging test: a correct model must be able to memorise 32 examples."""
    model.to(device).train()
    X = torch.as_tensor(data["X_train"][:batch]).to(device)
    y = torch.as_tensor(data["y_train"][:batch]).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    return first, loss.item()


# ==========================================================================
# 11. Evaluation and training
# ==========================================================================
@torch.no_grad()
def evaluate(model, X, y, device, batch_size=1024):
    """Mean cross-entropy (nats/token), perplexity = exp(CE), top-1 / top-5 accuracy."""
    model.eval().to(device)
    n, tot, c1, c5 = len(X), 0.0, 0, 0
    for i in range(0, n, batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(device)
        yb = torch.as_tensor(y[i:i + batch_size]).to(device)
        logits = model(xb)
        tot += F.cross_entropy(logits, yb, reduction="sum").item()
        top5 = logits.topk(5, dim=-1).indices
        c1 += (top5[:, 0] == yb).sum().item()
        c5 += (top5 == yb[:, None]).any(dim=1).sum().item()
    ce = tot / n
    return {"ce": ce, "ppl": math.exp(ce), "acc": c1 / n, "acc5": c5 / n}


TRAIN_CFG = dict(
    hidden_size=128, num_layers=1, dropout=0.2,   # model
    batch_size=128, lr=1e-3, clip=5.0,            # optimisation (Adam)
    max_epochs=30, patience=3,                    # early stopping on validation CE
)


def train_model(model, data, train_cfg, device, seed, verbose=True):
    """Adam + gradient clipping; stops when validation CE has not improved for
    `patience` epochs and restores the best weights."""
    model.to(device)
    X = torch.as_tensor(data["X_train"]).to(device)
    y = torch.as_tensor(data["y_train"]).to(device)
    N, B = len(X), train_cfg["batch_size"]
    params = [p for p in model.parameters() if p.requires_grad]   # frozen W2V excluded
    opt = torch.optim.Adam(params, lr=train_cfg["lr"])
    gen = torch.Generator().manual_seed(seed)                     # batch order

    best_ce, best_epoch, best_state, bad, history = float("inf"), 0, None, 0, []
    t_start = time.time()
    for epoch in range(1, train_cfg["max_epochs"] + 1):
        model.train()
        t_ep = time.time()
        perm = torch.randperm(N, generator=gen)
        run_loss = torch.zeros((), device=device)
        for i in range(0, N, B):
            idx = perm[i:i + B].to(device)
            xb, yb = X[idx], y[idx]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, train_cfg["clip"])
            opt.step()
            run_loss += loss.detach() * len(idx)
        train_ce = run_loss.item() / N
        val = evaluate(model, data["X_val"], data["y_val"], device)
        history.append({"epoch": epoch, "train_ce": train_ce, "val_ce": val["ce"],
                        "val_ppl": val["ppl"], "sec": time.time() - t_ep})
        if verbose:
            print(f"epoch {epoch:>2} | train CE {train_ce:.4f} | val CE {val['ce']:.4f} "
                  f"| val ppl {val['ppl']:.1f} | {time.time() - t_ep:.0f}s")
        if val["ce"] < best_ce - 1e-4:
            best_ce, best_epoch, bad = val["ce"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= train_cfg["patience"]:
                if verbose:
                    print(f"early stop (best epoch = {best_epoch})")
                break
    model.load_state_dict(best_state)                              # restore best weights
    return {"history": history, "best_epoch": best_epoch,
            "epochs_run": len(history), "train_seconds": time.time() - t_start}


@torch.no_grad()
def attention_profile(model, X, device, n=5000, batch_size=500):
    """Mean attention weight per memory position, and mean normalised entropy
    (1.0 = perfectly uniform weights, 0.0 = all weight on one position)."""
    model.eval().to(device)
    sums, ent, seen = None, 0.0, 0
    for i in range(0, min(n, len(X)), batch_size):
        xb = torch.as_tensor(X[i:min(i + batch_size, n)]).to(device)
        _, a = model(xb, return_attn=True)
        s = a.sum(0)
        sums = s if sums is None else sums + s
        e = -(a * torch.log(a.clamp_min(1e-12))).sum(-1) / math.log(a.size(1))
        ent += e.sum().item()
        seen += len(xb)
    return {"mean_weights": (sums / seen).cpu().tolist(), "entropy_norm": ent / seen}


# ==========================================================================
# 12. Running experiments (results are saved as one small JSON per run)
# ==========================================================================
def run_experiment(name, data, input_type, attention, train_cfg=TRAIN_CFG, seed=0,
                   device=None, w2v_matrix=None, results_dir="results/metrics",
                   verbose=True):
    device = device or get_device()
    set_seed(seed)                                   # model init + dropout
    model = WindowLM(data["vocab_size"], train_cfg["hidden_size"], train_cfg["num_layers"],
                     train_cfg["dropout"], input_type, w2v_matrix, attention)
    info = train_model(model, data, train_cfg, device, seed, verbose)
    val = evaluate(model, data["X_val"], data["y_val"], device)
    test = evaluate(model, data["X_test"], data["y_test"], device)   # used ONCE, at the end
    result = {
        "name": name, "input_type": input_type, "attention": attention or "none",
        "seed": seed, "params": count_params(model), "vocab_size": data["vocab_size"],
        "train_cfg": dict(train_cfg), "data_cfg": data["cfg"],
        "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
        "train_seconds": info["train_seconds"],
        "sec_per_epoch": info["train_seconds"] / info["epochs_run"],
        "val": val, "test": test, "history": info["history"],
        "attn_profile": attention_profile(model, data["X_val"], device) if attention else None,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{name}_seed{seed}.json"), "w") as f:
        json.dump(result, f)
    return result


def run_or_load(name, data, input_type, attention, train_cfg=TRAIN_CFG, seed=0,
                results_dir="results/metrics", **kw):
    """Re-use a saved result if it was produced with the SAME configs, else run."""
    path = os.path.join(results_dir, f"{name}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved["train_cfg"] == train_cfg and saved["data_cfg"] == data["cfg"]:
            print(f"[loaded] {name} seed {seed}  (delete {path} to re-run)")
            return saved
        print(f"[config changed] re-running {name} seed {seed}")
    print(f"[running] {name} seed {seed}")
    return run_experiment(name, data, input_type, attention, train_cfg, seed,
                          results_dir=results_dir, **kw)