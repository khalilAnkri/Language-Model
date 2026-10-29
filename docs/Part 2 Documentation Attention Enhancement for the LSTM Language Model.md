# Part 2 Documentation: Attention Enhancement for the LSTM Language Model

INFO2049 Assignment: Language Model. Covers the model, training and evaluation code added to `common.py` and the notebook `02_attention.ipynb`. It builds on the data pipeline described in the Part 1 documentation.

## 1. Purpose and scope

The assignment asks for an LSTM language model baseline (one-hot vs Word2Vec input) and one enhancement for the bonus point: attention or a bilingual model. Following the professor's instruction, the project uses **LSTM only** (no GRU comparison) and implements **attention only** (no bilingual model).

This document explains the attention enhancement: the idea of the paper it is inspired by, how we adapted it to a windowed next-word task, the exact model and tensor shapes, the training and evaluation protocol, the tests that were run, and how to interpret the results honestly.

The baseline and the attention model use the same data (`prepare_data`), the same training loop and the same evaluation code, all in `common.py`. The only difference between them is the prediction head. This is what makes the comparison fair.

## 2. Research question and hypotheses

**Research question.** Does letting an LSTM look back at its own previous hidden states improve next-word prediction, measured by test perplexity, compared with predicting from the last hidden state only?

**H1.** The attention models (single and combined score) reach lower validation and test perplexity than the plain LSTM.

**H2.** Any gain comes from the *learned* attention weights, and not merely from the extra layer or from averaging past states. This is tested with a control model that averages past states with fixed uniform weights.

These are hypotheses to test, not expected outcomes. A result in which attention does not help, or in which the control does as well as attention, is a valid finding.

## 3. The paper this is inspired by

Salton, Ross and Kelleher (2017), *Attentive Language Models*, IJCNLP 2017, pages 441-450. The description below was checked against the paper's PDF.

### 3.1 The idea

A standard recurrent language model predicts the next word from its current hidden state, so older information tends to fade as new words are processed. The paper keeps the recurrent network but stores the hidden state of every earlier step in a memory buffer, and lets the model look back at them:

1. Compute a score for each stored state `h_i` (for `i < t`).
2. Convert the scores to weights with a softmax: `a_i = softmax(score_i)`. The weights sum to 1.
3. Build a context vector: `c_t = sum_i a_i * h_i`.
4. Merge with the current state: `h'_t = tanh(W_c [h_t ; c_t] + b)`.
5. Predict the next word: `softmax(W h'_t + b')`.

### 3.2 The two score functions

| Name | Score of stored state `h_i` | Interpretation in the paper |
| --- | --- | --- |
| single | `v . tanh(W_s h_i)` | absolute relevance of a state, independent of the current state |
| combined | `v . tanh(W_s h_i + W_q h_t)` | relevance of `h_i` relative to the current state `h_t` |

### 3.3 What the paper reports

The authors report that a medium-sized attentive LSTM achieves a better perplexity than larger standard LSTMs and performs comparably to an ensemble of ten similar-sized LSTMs on the Penn Treebank, and reaches results close to the state of the art on WikiText-2. In their own analysis, however, the learned weights did not act like a sharp alignment: attention was spread fairly evenly over the history, behaving more like a smoothing of past states. This observation motivates our mean-pooling control.

A note on numbers: the parameter counts quoted for the model differ between the ACL Anthology abstract (11M) and the PDF (14.5M in the abstract and table, 10M in the results text). We therefore do not quote any of them.

### 3.4 What it is not

This is **not a Transformer**. The LSTM still reads the words one by one. There are no queries, keys and values computed from the words, no multi-head attention and no positional encodings. The attention is a learned weighted average over the LSTM's own past hidden states.

## 4. Our adaptation

### 4.1 The windowed formulation

The task is the one defined in Part 1: given a window of 20 words, predict the next word. For each window the LSTM runs over the 20 words and produces 20 hidden states `h_1 ... h_20`. The current state is `h_t = h_20`. The memory holds the **19 previous states** `h_1 ... h_19`. Each window starts from a zero state, so windows are independent of one another.

### 4.2 The four model variants

All variants are one class, `WindowLM`, selected by the `attention` argument.

| `attention=` | Variant | Role |
| --- | --- | --- |
| `None` | plain LSTM, predicts from `h_t` only | baseline |
| `"mean"` | context = plain average of the 19 past states, then the same merge layer | control |
| `"single"` | Salton single score | enhancement |
| `"combined"` | Salton combined score | enhancement |

The control uses the same merge layer as the attention models but has no learned scores. Comparing it with the baseline shows the effect of the merge layer and of averaging past states. Comparing attention with the control shows the effect of *learning* the weights.

### 4.3 Tensor shapes

`B` is the batch size, `H = 128` the hidden size and `V` the vocabulary size.

| Step | Tensor | Shape |
| --- | --- | --- |
| input word ids | `x` | (B, 20) |
| one-hot or Word2Vec input | `e` | (B, 20, V) or (B, 20, 300) |
| all LSTM outputs | `out` | (B, 20, H) |
| current state | `h_t = out[:, -1]` | (B, H) |
| memory (previous states only) | `mem = out[:, :-1]` | (B, 19, H) |
| scores | `scores` | (B, 19) |
| attention weights | `a`, softmax over the 19 positions | (B, 19), rows sum to 1 |
| context vector | `c = a @ mem` | (B, H) |
| merged feature | `tanh(W_c [h_t ; c])` | (B, H) |
| logits | output layer applied to the merged feature | (B, V) |

The softmax over the vocabulary is inside the cross-entropy loss, so the model outputs raw logits.

### 4.4 No access to the future or to the target

- The model's `forward` function receives only the 20 input words. The target word is never an input.
- The memory contains only hidden states computed from the 20 preceding words. Because the LSTM reads left to right, the state at position `i` depends only on words `1..i`.
- All windows have the same length, so **no padding mask is needed**. A mask would only be required with variable-length sequences.
- This is tested, not just claimed (see Section 9): changing one input word leaves all earlier hidden states exactly unchanged.

### 4.5 Extra parameters

With `H = 128`, the extra trainable parameters compared with the baseline are independent of the vocabulary and the input type:

| Variant | Extra parameters | Made of |
| --- | --- | --- |
| mean | 32,896 | merge layer `W_c` (256 x 128 weights + 128 biases) |
| single | 49,408 | `W_c` + `W_s` (128 x 128) + `v` (128) |
| combined | 65,792 | `W_c` + `W_s` + `v` + `W_q` (128 x 128) |

Because the attention models have more parameters, a lower perplexity alone does not prove that attention is the cause. That is the reason for the mean control. Parameter counts are reported next to every result.

### 4.6 Deviations from the paper

| Paper | This project | Reason |
| --- | --- | --- |
| PTB and WikiText-2 | small Wikipedia subset | assignment corpus, laptop-friendly |
| 2 layers, 650 or 1000 LSTM units | 1 layer, 128 units | laptop-friendly |
| SGD with a learning-rate schedule, 50-65% dropout | Adam, dropout 0.2 | simpler, standard |
| output matrix tied to the input embedding | no weight tying | incompatible with one-hot input and frozen Word2Vec |
| trained on every position of a sentence, memory reset at sentence boundaries | one prediction per 20-word window, memory = the 19 previous states, windows may cross sentence boundaries | the course's "stock market on words" formulation |
| a single reported run | configurable number of seeds | allows measuring spread |

This is therefore an inspired, scaled-down version and **not a replication**. Our perplexities cannot be compared with the paper's.

## 5. Input representations

**One-hot.** Each of the 20 input words is a vector of length `V` with a single 1. It goes directly into the LSTM, whose input weights then act as a learned lookup table for each word.

**Word2Vec.** We use the pre-trained GoogleNews vectors (300 dimensions) loaded through gensim. We do **not** train Word2Vec ourselves. The vectors are used as a **frozen** lookup table: word id to a 300-dimensional vector, never updated during training. Words missing from Word2Vec, and `<UNK>`, receive a zero vector. The function `build_embedding_matrix` reports how many words and training tokens received a vector.

**Frozen vs trainable.** A frozen embedding keeps the pre-trained knowledge fixed and adds no trainable parameters. A trainable embedding would be initialised with the pre-trained vectors but updated by the loss. We use the frozen version. The one-hot model, in contrast, learns its own word representation inside the LSTM input weights.

**Caveats for the comparison.**

- The GoogleNews model is known to lack some very frequent words and most punctuation, so some common tokens get zero vectors. The coverage numbers show how much this matters.
- The two input types differ in trainable parameter count and in whether the word representation is learned. A difference between them mixes several effects. This is why the main attention comparison is made within each input type.

## 6. Training protocol

All runs use the same settings (`TRAIN_CFG` in `common.py`):

| Setting | Value |
| --- | --- |
| LSTM layers and hidden size | 1 layer, 128 units |
| Dropout | 0.2, applied to the merged feature before the output layer (the LSTM has one layer, so no internal dropout) |
| Optimiser | Adam, learning rate 1e-3 |
| Batch size | 128 |
| Gradient clipping | norm 5.0 |
| Maximum epochs | 30 |
| Early stopping | validation cross-entropy, patience 3 epochs, best weights restored |
| Loss | cross-entropy over the full vocabulary |

Two kinds of seed are separated: the data seed (fixed, in `CFG`) fixes the articles and the split for the whole project; the training seed controls weight initialisation and batch order and can be varied to measure spread. The values above are reasonable starting points, not tuned values.

**Test set discipline.** Early stopping and all decisions use the validation set. The test set is evaluated once per run, after training has ended.

## 7. Experiments

| Group | Variants | Input types |
| --- | --- | --- |
| baseline | plain LSTM | Word2Vec, one-hot |
| control | mean of past states | Word2Vec, one-hot |
| enhancement | single score | Word2Vec, one-hot |
| enhancement | combined score | Word2Vec, one-hot |

That is 8 runs per seed. Each run is saved as one small JSON file (configuration, seed, scores, parameter counts, timing, learning curve, attention profile). `run_or_load` skips a run that already exists with an identical configuration, so work can be stopped and resumed, and it re-runs automatically if a setting changed.

One seed gives a first result. With several seeds the notebook reports the mean and standard deviation, which is needed to judge whether a difference is larger than noise.

## 8. Evaluation

| Metric | Meaning | Direction |
| --- | --- | --- |
| Cross-entropy | mean negative log-probability of the true next word, in nats | lower is better |
| Perplexity | `exp(cross-entropy)`, the effective number of equally likely choices | lower is better |
| Top-1 and top-5 accuracy | share of windows where the correct word is the top prediction or among the top five | higher is better |
| Trainable parameters | model size, excluding the frozen Word2Vec table | context |
| Time per epoch | computational cost | context |
| Attention profile | mean weight per memory position, and normalised entropy (1.0 = uniform, near 0 = focused on one position) | diagnostic |

Two reference scores from Part 1 appear next to every table: the uniform perplexity (equal to the vocabulary size) and the unigram perplexity (a context-free model). A useful model must clearly beat the unigram reference.

**Limits of perplexity.** It depends on the vocabulary: `<UNK>` is a normal target and the vocabulary is cut at a minimum frequency, so scores are only comparable between our own models. Perplexity also says nothing about the quality of generated text.

## 9. Verification that was performed

The following checks were run on **synthetic text** (random words), not on real Wikipedia, so they validate the code and not the scientific result.

| Check | What it shows |
| --- | --- |
| Output shapes for all 8 model configurations | tensor shapes match Section 4.3 |
| Attention weights sum to 1 | the softmax is applied over the right axis |
| Changing an input word does not change earlier hidden states | no information flows from the future into the past |
| Mean control equals manual mean pooling of the 19 past states | the control does what it claims |
| Word2Vec embedding has no gradient | the embedding is frozen |
| A model can memorise 32 training examples (loss drops close to zero) | the training loop and gradients work |
| Same seed gives the same validation score | runs are reproducible on the same machine and library versions |
| Changing a setting triggers a re-run, an unchanged setting reuses the saved result | no stale results can be mixed in |
| The whole notebook ran end to end in a small test mode | the notebook has no code errors |

These checks are also built into the notebook (shapes, weight sums, no-future-leak test and the memorise-32-examples test), so they run on your real data too.

## 10. Results (fill in from your own run)

Nothing in these tables has been measured on real data. Copy the values from the notebook output.

| Input | Variant | Test perplexity (mean, std) | Test top-1 accuracy | Trainable params | Best epoch | Seconds per epoch |
| --- | --- | --- | --- | --- | --- | --- |
| Word2Vec | none (baseline) | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| Word2Vec | mean (control) | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| Word2Vec | single | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| Word2Vec | combined | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| One-hot | none (baseline) | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| One-hot | mean (control) | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| One-hot | single | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| One-hot | combined | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |

Reference perplexities on the test set: uniform (fill in), unigram (fill in). Normalised attention entropy: single (fill in), combined (fill in). Number of seeds: (fill in).

## 11. How to interpret the results

Compare within the same input type.

| Comparison | What it tells you |
| --- | --- |
| attention vs baseline | whether the whole enhancement helps (H1) |
| mean vs baseline | the effect of the merge layer and of averaging past states |
| attention vs mean | the effect of learning the weights (H2), the most informative comparison |
| single vs combined | whether conditioning the score on the current state matters |

Rules for honest conclusions:

- With a single seed, or with differences smaller than the seed-to-seed spread, state that the evidence is inconclusive.
- If the mean control gains as much as attention, the improvement does not come from learned attention. This would be consistent with the paper's own observation of near-uniform weights.
- If attention does not help, discuss possible reasons as hypotheses, not facts: a window of 20 words leaves little long-range context to recover, the training set is small, and the LSTM may already carry recent context well.
- A normalised entropy close to 1 means the weights are close to uniform, so the model behaves like the mean control.
- Always report parameter counts and time per epoch next to perplexity.

## 12. Limitations

- One small corpus (1,000 articles by default) and one window length.
- Results depend on the fixed data split and, with few seeds, on initialisation.
- Word2Vec is frozen, was not trained on our corpus, and gives zero vectors to missing words.
- Hyperparameters were not tuned. Attention models could behave differently with other settings.
- The windowed formulation trains one prediction per window and limits the memory to 19 states, unlike the paper.
- Scores cannot be compared with the paper's numbers.

## 13. How to run

1. Install PyTorch with the selector at pytorch.org (CUDA build for an NVIDIA GPU, otherwise CPU), then `pip install -r requirements.txt`.
2. Make sure `common.py` is the updated version and the data subset from Part 1 is in place.
3. Open `02_attention.ipynb` and run it once with `QUICK = True` to check that everything works. Results of this test go to `results/debug`.
4. Set `QUICK = False` and run the notebook. Start with `SEEDS = [0]`, and use `[0, 1, 2]` if time allows. Remove `"onehot"` from `INPUTS` if it is too slow or too memory-heavy.
5. Copy the printed tables into Section 10 of this document.



## References

- Salton, G., Ross, R. and Kelleher, J. (2017). Attentive Language Models. In *Proceedings of the 8th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 441-450, Taipei, Taiwan. https://aclanthology.org/I17-1045/
- Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Bahdanau, D., Cho, K. and Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In *ICLR 2015*.
- Luong, M.-T., Pham, H. and Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. In *Proceedings of EMNLP 2015*, pages 1412-1421.
- Mikolov, T., Chen, K., Corrado, G. and Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv:1301.3781.
- Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. In *ICLR 2015*.
