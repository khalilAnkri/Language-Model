# Part 1 Documentation: Data Pipeline for the LSTM Language Model

INFO2049 Assignment: Language Model. Covers `common.py` and the first three stages of `01_baseline.ipynb` (setup, data pipeline, sanity checks).

## 1. Purpose and scope

The assignment asks for an RNN language model on English or French text, with an LSTM baseline that compares one-hot and Word2Vec inputs, plus one enhancement (we chose attention, following Salton et al., 2017). Following the professor's instruction, the LSTM-vs-GRU comparison is reduced to LSTM only, and only one of the two enhancements (attention or bilingual) is implemented.

This document describes **only the data pipeline**: how raw Wikipedia text becomes the training, validation and test arrays that every model will use. The pipeline is shared by the baseline and the attention notebook, so both models are guaranteed to be trained and evaluated on exactly the same data. That guarantee is what makes the final comparison fair.

The model, the training loop and the evaluation function are not covered here. They are added in later stages.

## 2. The task: next-word prediction

A language model estimates the probability of a word given the words before it. Over a sequence it factorises as

`p(w1, ..., wN) = p(w1) * p(w2 | w1) * ... * p(wN | w1, ..., wN-1)`

We train the network to solve one concrete problem: given a window of the previous 20 words, predict the next word. The input and the target are the same text, shifted by one position:

| Position | Content |
| --- | --- |
| Input `X[i]` | words `i` to `i+19` of an article |
| Target `y[i]` | word `i+20` of the same article |

The next window `X[i+1]` slides one word to the right, so its last word is exactly the previous target. This is the same formulation as the "stock market prediction, but on words" example from the course. The model never sees the target word as input, so there is no access to the future.

## 3. Corpus

| Property | Value |
| --- | --- |
| Source | English Wikipedia, Hugging Face dataset `wikimedia/wikipedia`, snapshot `20231101.en` |
| Full size | about 6.4 million articles (far too large for a student laptop) |
| Used here | 1,000 articles (configurable), each with at least 200 words |
| Licence | CC-BY-SA 3.0 and GFDL according to the dataset card (verify the card yourself before citing it in the final report) |

**Why Wikipedia.** It is one of the corpora allowed by the assignment, it is free to use, it is large enough to scale up later, and it keeps the project consistent with the original baseline notebook.

**Streaming instead of downloading.** The original notebook loaded the whole dataset, which means downloading many gigabytes. The pipeline instead streams the dataset, shuffles it with a fixed seed and a buffer of 10,000 articles, and keeps the first 1,000 that pass the length filter. Note that this shuffle is pseudo-random: it is not a perfectly uniform sample of Wikipedia.

**Why a length filter.** Many Wikipedia pages are very short stubs or list pages. Articles under 200 words give few training windows and add noise, so they are skipped.

## 4. Pipeline step by step

The function `prepare_data(cfg)` runs the following steps in order.

### 4.1 Sampling and caching

The sampled articles are saved to `data/articles.json.gz`. Later runs read this file instead of the network. If the file is committed to the repository, every team member works on identical text. If a smaller run is requested (for example `n_articles=50` for debugging), the first 50 articles of the cache are used, so a debug run is a prefix of the full run.

### 4.2 Tokenization

Text is split into word tokens with NLTK's `word_tokenize` (the same tokenizer as the original notebook). Punctuation marks become separate tokens. Case is **kept**, because the pre-trained GoogleNews Word2Vec vectors are case-sensitive and lower-casing would lose matches. The price of keeping case is a larger vocabulary.

### 4.3 Splitting by article

Articles are shuffled with the data seed and split 80% / 10% / 10% into training, validation and test sets.

The split is **by article, not by window**. If windows were split at random, window `i` could be in the test set while window `i+1` (which shares 19 of its 20 words) is in training, and the model would be tested on text it has practically seen. Splitting whole articles prevents this.

### 4.4 Vocabulary (training articles only)

The vocabulary maps each word to an integer id. Rules:

- Id 0 is reserved for `<UNK>` (unknown word).
- Only words that occur **at least `min_freq = 5` times in the training articles** get their own id.
- Every other word, in any split, is mapped to `<UNK>`.
- Words are sorted by frequency (ties alphabetically), so the ids are deterministic.

Three reasons for this design:

1. **No leakage.** The vocabulary is built from training text only. Building it from the full corpus would let information about the test set influence the model.
2. **`<UNK>` is actually trained.** The original baseline put every training word in the vocabulary, so `<UNK>` never appeared in training even though it appeared in test data. With a frequency cutoff, rare training words become `<UNK>`, and the model learns what to do with it.
3. **Manageable size.** The one-hot input has one dimension per vocabulary word, and the output layer has one logit per word. A smaller vocabulary keeps both affordable on a CPU.

### 4.5 Encoding and windows

Each article becomes an array of word ids. Windows of length 20 are then cut with a sliding window, and **windows never cross article boundaries**. An article with `n` tokens produces `max(n - 20, 0)` windows. Very short articles produce none.

The output arrays are:

| Array | Shape | Meaning |
| --- | --- | --- |
| `X_train`, `X_val`, `X_test` | (number of windows, 20) | input word ids |
| `y_train`, `y_val`, `y_test` | (number of windows,) | id of the word to predict |

## 5. Reproducibility

Two kinds of randomness are kept apart on purpose.

| Seed | Controls | Rule |
| --- | --- | --- |
| `data_seed` (in `CFG`, value 42) | which articles are sampled and how they are split | Never changes. Baseline and attention must see identical data. |
| `SEED` (set in the notebook) | weight initialisation and batch order | Varies between repetitions, so a mean and spread can be reported. |

Together with the cached subset, this means that anyone running the notebook obtains the same arrays, provided the same library versions and the same cache file are used.

## 6. Data leakage: what it is and how it is prevented

Data leakage means that information from the validation or test data influences training, which makes the reported scores look better than the true performance. In this pipeline:

| Possible leak | Prevention |
| --- | --- |
| Neighbouring windows of one article in different splits | Split by article |
| Vocabulary built with test words | Vocabulary from training articles only |
| Target word visible in the input | Target is the next word, never part of the 20 input words |
| Windows mixing two articles | Windows are cut inside each article |
| Choosing hyperparameters on the test set | A separate validation set exists for early stopping and tuning. The test set is used once at the end. |

## 7. Sanity checks

The function `sanity_checks` does not trust the pipeline. It asserts the properties below and stops with an error if any fails.

| Check | What it verifies |
| --- | --- |
| No shared article ids across splits | No leakage between train, validation and test |
| Every vocabulary word has frequency at least `min_freq` in training | Vocabulary comes from training data only |
| Shapes and id range | `X` is (N, 20), `y` is (N,), all ids lie in `[0, vocab_size)` |
| Window count equals the sum of `max(length - 20, 0)` over articles | No windows lost, none crossing article boundaries |
| `X[1]` equals `X[0]` shifted by one and `X[0]` followed by `y[0]` matches the article | The target really is the next word |
| Training `<UNK>` rate is above zero | `<UNK>` is seen during training |

During development these checks were run on synthetic text. A deliberately injected leak (an article placed in two splits) was detected, which confirms the checks can fail. They have **not** yet been run on the real Wikipedia subset: the numbers in Section 9 must be filled in from your own run.

## 8. Reference perplexities

A perplexity value means little on its own, so two trivial reference "models" are computed on the same validation and test targets.

- **Uniform:** every word has probability `1/V`, so the perplexity is simply the vocabulary size `V`.
- **Unigram:** word probabilities come from training target counts with add-one smoothing, `p(w) = (count(w) + 1) / (N + V)`. The perplexity is `exp(-(1/M) * sum(log p(y)))` over the `M` evaluation targets. This model ignores the context completely.

An LSTM that does not clearly beat the unigram reference has learned nothing useful from the context. These references appear next to every results table.

## 9. Results of this stage (to be filled in)

Nothing in this table has been measured on real data yet. Copy the values printed by the notebook.

| Quantity | Value |
| --- | --- |
| Vocabulary size | (fill in) |
| Training / validation / test articles | (fill in) |
| Training / validation / test windows | (fill in) |
| `<UNK>` rate in text, train / val / test | (fill in) |
| Uniform perplexity | (fill in) |
| Unigram perplexity, validation / test | (fill in) |

## 10. Configuration

All values live in `CFG` in `common.py`. The values below are starting points chosen for a CPU-friendly project, not tuned values.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `data_seed` | 42 | Sampling and split |
| `n_articles` | 1000 | Subset size |
| `min_article_words` | 200 | Skip stubs |
| `train_frac`, `val_frac` | 0.8, 0.1 | Split fractions (test is the remaining 0.1) |
| `min_freq` | 5 | Words rarer than this in training become `<UNK>` |
| `window` | 20 | Number of input words |
| `cache_path` | `data/articles.json.gz` | Saved subset |

## 11. Limitations and deviations

- **Small corpus.** 1,000 articles is a small sample. Absolute scores are not comparable with published results, and the goal is a fair comparison between our own models.
- **Pseudo-random sampling.** Streaming shuffle with a finite buffer is not a perfectly uniform sample.
- **Case kept.** This helps Word2Vec coverage but enlarges the vocabulary.
- **Windows cross sentence boundaries** within an article. Salton et al. reset their memory at sentence boundaries, so this is a deviation from the paper that must be stated when the attention model is described.
- **Perplexity depends on the vocabulary.** Because `<UNK>` counts as a normal target and the vocabulary is cut at `min_freq`, our perplexities cannot be compared with those of papers that use other vocabularies.
- **Single sampled subset.** Results are for one subset. A different `data_seed` could give slightly different numbers.

## 12. How to run

1. Install the dependencies used by this part: `numpy`, `nltk` and `datasets` (PyTorch is needed from the next stage). Pin the exact versions from your working environment in `requirements.txt` once everything runs.
2. Open `01_baseline.ipynb` and run the cells in order. The first run downloads and tokenizes the articles, which can take a few minutes.
3. Commit `data/articles.json.gz` so teammates use the identical subset, and add a CC BY-SA 3.0 attribution to the README.
4. If a sanity check fails, stop and fix the cause before going further.

Set `DEBUG = True` in the notebook for a 50-article run while testing code.



## References

- Salton, G., Ross, R. and Kelleher, J. (2017). Attentive Language Models. In *Proceedings of the 8th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 441-450, Taipei, Taiwan. https://aclanthology.org/I17-1045/
- Wikimedia Foundation. Wikipedia dataset (`wikimedia/wikipedia`, snapshot `20231101.en`), Hugging Face Datasets.
- Bird, S., Klein, E. and Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. (NLTK)
