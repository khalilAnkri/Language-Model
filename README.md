# RNN Language Model with Attention (INFO2049, Assignment 1)

An LSTM language model that predicts the next word of English Wikipedia text.
We compare two input representations (one-hot vs Word2Vec) and add one enhancement:
an attention mechanism inspired by Salton, Ross & Kelleher (2017),
[*Attentive Language Models*](https://aclanthology.org/I17-1045/).



## Task

Given a window of 20 words, predict the next word. Models are scored with test
cross-entropy and perplexity (lower is better), plus top-1 / top-5 accuracy,
parameter count and time per epoch.

## Models compared (all LSTM, 1 layer, 128 units)

| Variant | Description |
|---|---|
| baseline | predicts from the last hidden state only |
| mean (control) | adds the average of the 19 previous hidden states, no learned weights |
| single | attention over previous hidden states, score `v.tanh(Ws h_i)` |
| combined | attention with score `v.tanh(Ws h_i + Wq h_t)` |

Each variant is run with one-hot input and with frozen pre-trained Word2Vec
(GoogleNews, 300-d) input. The LSTM-vs-GRU comparison and the bilingual model were
dropped, following the professor's instruction (LSTM only, one enhancement only).

## Repository

| File | Content |
|---|---|
| `common.py` | shared code: data pipeline, model, training with early stopping, evaluation |
| `01_baseline.ipynb` | baseline: setup, data pipeline, sanity checks (model and training stages in progress) |
| `02_attention.ipynb` | attention enhancement, control, experiments, results and plots |
| `code.ipynb` | first version of the baseline, kept for reference |
| `requirements.txt` | dependencies |
| `data/articles.json.gz` | the fixed Wikipedia subset (created on first run, commit it) |
| `results/metrics/` | one JSON file per run (config, seed, scores) |

## Setup

1. Install PyTorch with the selector at <https://pytorch.org/get-started/locally/>
   (CUDA build for an NVIDIA GPU, otherwise CPU).
2. `pip install -r requirements.txt`
3. Open `02_attention.ipynb`, set `QUICK = True`, and run it once to check that
   everything works (results go to `results/debug`). Then set `QUICK = False`.

The first real run downloads and tokenizes 1,000 Wikipedia articles (a few minutes) and
the Word2Vec vectors (about 1.6 GB, stored in your home folder, not in the repo).
Use `SEEDS = [0, 1, 2]` in the notebook to report mean and spread.

## Design choices that keep the comparison fair

- One shared `common.py`: identical data, split, vocabulary and windows for every model.
- Split **by article** (80/10/10) and vocabulary built from training articles only, so
  there is no data leakage. Words seen fewer than 5 times become `<UNK>`.
- Fixed data seed; the training seed is varied separately.
- Early stopping on validation cross-entropy; the test set is evaluated once per run.
- Reference scores (uniform and unigram perplexity) are reported next to every table.

## Results

*To be filled in after the experiments have been run.* All numbers come from the JSON
files in `results/metrics/`; none are typed by hand.

## Data and licence

English Wikipedia, Hugging Face dataset `wikimedia/wikipedia`, snapshot `20231101.en`
(text under CC BY-SA 3.0 / GFDL according to the dataset card; please verify before
submitting). The saved subset in `data/` is derived from it and carries the same
attribution requirement.

## References

- Salton, G., Ross, R., Kelleher, J. (2017). Attentive Language Models. IJCNLP 2017, pp. 441-450.