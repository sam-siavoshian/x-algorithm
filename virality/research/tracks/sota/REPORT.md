# Track "sota": current techniques for a noisy within-account ranking problem

Worker report, 2026-11-19. All numbers are likes. Intervals are 95%, from resampling accounts. Every setting was chosen on validation rows or on the replicate track's development pairs; the benchmark and launch pairs were scored once per final checkpoint and never used to choose anything. No commits, pushes or pull requests; code is in `patch.diff`.

## Headline

**Nothing tested here beats the control. Eight methods from the 2022 to 2026 literature were measured, most with intervals under one point either way.**

- **Training objectives** (control and candidate trained side by side, two seeds each, 3,000 steps, same data, same batch order):
  - Heteroscedastic pair loss: benchmark +0.2 points (-0.6 to +0.9), ordinary test pairs -0.0 (-0.1 to +0.1).
  - Noise-aware soft-label pair loss: benchmark +0.1 (-0.7 to +0.9), ordinary pairs -0.2 (-0.2 to -0.1).
  - Both pair losses lower the confident share on 2x pairs by 1.2 to 1.5 points.
- **Per-account adaptation layer** (closed-form ridge on the account's history): validation loss falls faster early and ends level. Benchmark -1.6 per member (-2.9 to -0.5), launch pairs -2.8 (-5.5 to -0.3). It hurts the comparison of two versions.
- **Weight averaging** of the control's last three checkpoints: benchmark +0.0 (-0.6 to +0.6). Their prediction ensemble is level with the last checkpoint on ordinary pairs (59.8% and 60.0% against 59.8% and 60.1%).
- **History dropout** was stopped at step 1,500 (validation no better). **Masked-history task** ran to the end with validation level (pairs +0.1 point), so it was not scored on the benchmark; on ordinary test pairs it is +0.1 with a like-head NLL 0.008 lower.
- **Confidence without retraining**, on the shipped three-model baseline:
  - Using each version's predicted spread labels more pairs confident (26% to 31% of 2x pairs) at slightly lower accuracy (70.2% to 69.4%); the surest 40% of 2x pairs is unchanged (67.6% to 67.4%).
  - A selector fitted on development pairs: -0.4 (-1.9 to +1.1) on the surest 40% of 2x pairs.
- **A measurement finding.** 944 of the benchmark's 2,030 "2x" pairs include a post with 0 to 2 likes. The baseline's confident calls are 88.0% right on those (11% of them called) and 65.5% right where both posts have at least 3 likes (38% called). The pooled 70.2% mixes the two.
- **Not done:** the listwise loss and the cyclic-rate snapshot run are implemented but were not trained; scoring under shared CPU took about three times the planned time.

The control reproduces the baseline: two seeds give 60.9% and 58.7% on the 2,852 benchmark pairs, 59.7% together (57.3 to 62.1), 27% of 2x pairs confident at 70.7%. My re-scoring of the stored three-model baseline gives exactly the README row (60.0%, 26% at 70.2%).

## Part A. Review: what 2022 to 2026 work offers this problem

The full review is `REVIEW.md` in this folder: twelve topics, about ninety papers, each with link, claimed gain and data, whether the evaluation controlled for the author, and a transfer verdict. About 65 arXiv ids were confirmed against the arXiv API (marked `[V]`), about 25 links were only seen in search results (`[S]`), a few are marked unverified. Claims come from abstracts and search summaries; no full paper was read in this pass, and effect sizes in the ranking below are estimates made before the tests in Part B.

What the review found, in short:

- **Popularity prediction as published does not control for the author.** Every Social Media Prediction challenge entry from 2022 to 2025 ([overview](https://arxiv.org/abs/2405.10497), [MVP 2025](https://arxiv.org/abs/2507.00950), [HyperFusion 2025](https://arxiv.org/abs/2507.00926)) uses post-level splits with the same users in train and test, and wins with user statistics and gradient-boosted ensembles. The same holds for retrieval-augmented models (MMRA, RAGTrans, SKAPP) and for tabular retrieval ([TabR](https://arxiv.org/abs/2307.14338), [ModernNCA](https://arxiv.org/abs/2407.03257)): the neighbours carry user and topic popularity. In a same-account, same-topic pair those cancel. We already pool 16 neighbours and read 48 posts of history.
- **Where the author is controlled, the field is where we are.** On randomised headline pairs (Upworthy; LOLA, Marketing Science 2025) fine-tuned language models are only marginally above chance on all pairs; high figures (82 to 84%) are for strongly filtered subsets, comparable to our 5x band. This is a correction to `research/06_literature.md`.
- **Cascade models** need an observed prefix of the spread and do not apply to a draft.
- **The goal is a selective-prediction problem under outcome noise.** With noise in the label, the best rule for which pairs to call is a threshold on each pair's estimated chance of being right ([Geifman and El-Yaniv 2017](https://arxiv.org/abs/1705.08500); [Fisch et al. 2022](https://arxiv.org/abs/2208.12084)); separate selection heads and ensemble disagreement do not beat the model's own score, which matches the replicate track. So the candidates are: use the predicted spread in the confidence (heteroscedastic), a pair-level selector, and pair likelihoods that treat near-ties as near-coin-flips (Rao-Kupper, soft targets) instead of dropping them.
- **Noise-robust losses** (generalised cross-entropy, co-teaching, small-loss filtering) and **imbalanced-regression** methods (LDS, RankSim, ConR) address wrong labels and rare targets; our labels are random outcomes with a calibrated HL-Gauss head ([Farebrother et al. 2024](https://arxiv.org/abs/2403.03950)) already in place. Not recommended.
- **Per-account adaptation**: the model is already an in-context learner over the account's history, which is what tabular foundation models ([TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), [TabICL](https://arxiv.org/abs/2502.05564)) do. The untested variant is a closed-form adaptation layer ([Bertinetto et al. 2019](https://arxiv.org/abs/1805.08136)).
- **Weight averaging** ([SWA](https://arxiv.org/abs/1803.05407), [snapshot ensembles](https://arxiv.org/abs/1704.00109), [LAWA](https://arxiv.org/abs/2209.14981)) gives a single model part of an ensemble's gain; the training loop already keeps an EMA.

### The ten most promising, ranked by expected gain per effort (before testing), and what Part B found

| # | Method | Source | Expected before testing | Tested here | Result |
|---|---|---|---|---|---|
| 1 | Pair confidence from the two predicted distributions (heteroscedastic, no training) | Thurstone model; [Seitzer et al. 2022](https://arxiv.org/abs/2203.09168) | accuracy 0, coverage +1 to +4 | yes, B.2 | No gain. On the baseline three: confident share on 2x pairs 26% to 31%, but accuracy there 70.2% to 69.4% and the surest 40% of 2x pairs 67.6% to 67.4%: the threshold moved, the ordering did not |
| 2 | Pair-level selector fitted on development pairs | [Fisch et al. 2022](https://arxiv.org/abs/2208.12084) | accuracy 0, coverage +2 to +5 | yes, B.2 | No gain: surest 40% of 2x pairs -0.4 points (-1.9 to +1.1) on the benchmark, -2.4 (-5.6 to +0.4) on development pairs |
| 3 | Count-aware audit of the gap bands | this review | measurement only | yes, B.3 | Finding: 944 of the 2,030 pairs "2x apart" have a post with 0 to 2 likes. Confident calls are 88% right there (11% of those pairs) and 65.5% right where both posts have 3 or more likes (38% of those pairs) |
| 4 | Soft-target (noise-aware) pair loss | Rao-Kupper ties; soft labels from the repost noise | accuracy 0 to +0.5, coverage +1 to +3 | yes, B.1 `soft` | No gain: ordinary pairs -0.2 (-0.2 to -0.1), benchmark +0.1 (-0.7 to +0.9), confident share on 2x pairs -1.5 (-2.3 to -0.7) |
| 5 | Heteroscedastic pair loss in training | Thurstone; [Stirn et al. 2023](https://arxiv.org/abs/2212.09184) | accuracy 0, coverage 0 to +2 | yes, B.1 `thurstone` | No gain: ordinary pairs -0.0 (-0.1 to +0.1), benchmark +0.2 (-0.6 to +0.9), confident share on 2x pairs -1.2 (-1.9 to -0.5) |
| 6 | Weight averaging and snapshot ensembling | [SWA](https://arxiv.org/abs/1803.05407), [snapshots](https://arxiv.org/abs/1704.00109), [LAWA](https://arxiv.org/abs/2209.14981) | +0.3 to +0.7 for one model, 0 for an ensemble | yes, B.1 `snap`, `lawa` | No gain: weight average of the control's last three checkpoints, ordinary pairs -0.1 (-0.1 to -0.0), benchmark +0.0 (-0.6 to +0.6). The cyclic-rate snapshot run was not trained (time) |
| 7 | Closed-form per-account adaptation layer | [R2D2](https://arxiv.org/abs/1805.08136) | 0 to +0.5 | yes, B.1 `ridge` | Worse on pairs of versions: ordinary pairs +0.1 (-0.2 to +0.3), benchmark -1.6 per member (-2.9 to -0.5), launch pairs -2.8 (-5.5 to -0.3) |
| 8 | Masked-history auxiliary task | BERT4Rec-style masking | 0 to +0.5 ordinary pairs | yes, B.1 `mask` (as a joint task on the training table, not pretraining on 4.48M posts) | No movement on validation (pairs +0.1, NLL -0.001); ordinary test pairs +0.1, like-head test NLL -0.008. Not scored on the benchmark |
| 9 | Dropout of history outcomes and of whole histories | conditioning dropout | 0 to +0.3 | yes, B.1 `histdrop` | Stopped at step 1,500: validation NLL worse than control (+0.004), pairs +0.1 |
| 10 | Listwise objective over the account's posts in a batch | ListNet | 0 | implemented (`pair_loss=listnet`), not trained (time) | Not measured |

Not tested, with reasons: a tabular in-context model or a fine-tuned language-model judge as an extra ensemble member (4 to 8 hours each; the encoder and content tracks already found that stronger content models and a direct pair model add nothing after the cutoff); masked pretraining on all 4.48M posts (12 to 20 hours); cross-account retrieval beyond the existing 16 neighbours (cancels within a pair, and the serving path that the benchmark runs through does not supply neighbours at all).

## Part B. Tests

### B.1 Training candidates against a control

**Protocol.** `scripts/sota/train_sota.py`. Control = the 46-column `small` model of `train_one.py` (1.54M parameters, learning rate 1e-3, batch 64, EMA weights), 3,000 steps (the scale track's optimum is 3,000 to 4,000). A candidate differs from it by one switch. Seeds 1 and 2; the control and the candidate of a seed share initialisation seed and batch order. Six validations per run; the step is chosen by validation NLL on held-out accounts, as in `train.py`. At each validation the averaged weights are also scored on 6,000 fitted training rows, 6,000 test rows and the 2,805 development pairs, for the curves only. A rule fixed before the later runs: a candidate is stopped at step 1,500 unless its mean validation NLL is at least 0.005 below the control's or its validation pair accuracy at least 0.3 points above.

| Name | What changes | Source |
|---|---|---|
| `thurstone` | Pair loss P(i beats j) = Phi(k (mu_i - mu_j) / sqrt(var_i + var_j)) with the variances of the two predicted like distributions, so pairs the model calls noisy pull less on the means | heteroscedastic pair objective |
| `soft` | Pair label = Phi(dy / 0.34), the chance that the observed order is the real one given the repost noise; near-ties are kept as near-coin-flips instead of dropped | noise-aware pair objective |
| `ridge` | A ridge regression from a learned 16-dimension projection of each history post to its outcome, solved in closed form on the account's visible history inside the forward pass, its prediction for each draft added to the draft's token | per-account adaptation |
| `histdrop` | Each history outcome hidden with probability 0.15; whole history removed with probability 0.10 | dropout of history |
| `mask` | 15% of history outcomes hidden and predicted at their own positions (weight 0.5), as a joint task | self-supervision on masked history |
| `lawa` | Uniform average of the control's weights at steps 2,000, 2,500 and 3,000 (no training) | weight averaging |
| `listnet`, `lr_cycles=3` | Listwise loss over the account's posts in an episode; cosine restarts with snapshots | implemented, not trained |

#### Results: the chosen checkpoint of every run

Ordinary pairs: same account, within 7 days, likes (train: 6,000 fitted rows; validation: held-out accounts; test: all audience rows after the cutoff). Benchmark: 2,852 same-topic pairs 1.25x apart (2,030 at 2x). Launch: 850 pairs. `both` = the two seeds' scores averaged. Confident = stated 60% or more; the scale is cross-fitted over accounts on the benchmark and reused on launch pairs.

| Run | Step | Pairs train | Pairs val | Pairs test | Dev pairs 1.25x / 2x | Benchmark, all | Benchmark 2x | Benchmark 2x confident | Surest 40% of 2x | Launch, all | Launch 2x confident |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl s1 | 3000 | 62.7 | 61.6 | 59.8 | 55.1 / 56.9 | 60.9 (58.5 to 63.2) | 61.9 (58.7 to 64.9) | 26% at 70.6 | 67.6 | 62.1 (57.5 to 66.7) | 51% at 73.2 |
| ctl s2 | 3000 | 65.5 | 61.6 | 60.1 | 57.5 / 60.2 | 58.7 (56.6 to 60.7) | 59.7 (56.9 to 62.3) | 25% at 70.2 | 67.7 | 62.9 (58.7 to 67.0) | 58% at 71.6 |
| **ctl both** | | | | 60.2 | 56.7 / 58.7 | 59.7 (57.3 to 62.1) | 60.6 (57.6 to 63.6) | 27% at 70.7 | 68.0 | 62.8 (58.5 to 67.0) | 55% at 72.5 |
| thurstone s1 | 3000 | 63.7 | 61.5 | 59.7 | 55.7 / 57.3 | 59.7 (57.2 to 62.1) | 61.1 (57.9 to 64.4) | 25% at 70.3 | 66.7 | 61.2 (56.8 to 65.5) | 54% at 72.4 |
| thurstone s2 | 3000 | 65.7 | 61.5 | 60.1 | 56.8 / 59.0 | 59.1 (56.9 to 61.2) | 59.6 (56.7 to 62.4) | 24% at 69.8 | 67.4 | 64.1 (59.4 to 68.6) | 58% at 71.7 |
| **thurstone both** | | | | 60.2 | 56.4 / 58.7 | 59.9 (57.5 to 62.5) | 61.1 (57.8 to 64.3) | 25% at 71.1 | 68.0 | 63.4 (59.1 to 67.7) | 58% at 71.8 |
| soft s1 | 3000 | 64.3 | 61.4 | 59.6 | 55.0 / 56.5 | 59.8 (57.5 to 61.9) | 61.4 (58.5 to 64.2) | 24% at 70.0 | 67.5 | 60.2 (55.6 to 64.9) | 51% at 71.3 |
| soft s2 | 3000 | 65.0 | 61.5 | 60.1 | 57.3 / 59.0 | 58.5 (56.4 to 60.6) | 59.5 (56.7 to 62.1) | 24% at 70.0 | 67.2 | 63.3 (58.8 to 67.5) | 57% at 70.8 |
| **soft both** | | | | 60.1 | 56.4 / 58.3 | 59.8 (57.5 to 62.1) | 61.0 (58.0 to 64.1) | 25% at 70.7 | 68.6 | 62.6 (58.5 to 66.7) | 55% at 72.5 |
| ridge s1 | 3000 | 66.5 | 61.5 | 59.8 | 55.6 / 57.6 | 58.3 (56.2 to 60.3) | 59.0 (56.3 to 61.6) | 25% at 68.9 | 67.2 | 62.7 (59.2 to 66.2) | 55% at 71.0 |
| ridge s2 | 3000 | 65.7 | 61.2 | 59.9 | 57.0 / 58.8 | 58.1 (55.7 to 60.3) | 59.3 (56.4 to 62.1) | 21% at 69.9 | 67.1 | 59.6 (55.7 to 63.7) | 53% at 66.8 |
| **ridge both** | | | | 60.3 | 56.8 / 59.1 | 58.7 (56.5 to 60.8) | 59.6 (56.8 to 62.2) | 26% at 70.5 | 68.0 | 60.0 (55.9 to 63.6) | 56% at 69.4 |
| mask s1 | 3000 | 63.4 | 61.5 | 59.9 | 55.2 / 56.6 | - | - | - | - | - | - |
| mask s2 | 3000 | 63.4 | 62.0 | 60.2 | 56.4 / 58.3 | - | - | - | - | - | - |
| **lawa** | | | | 60.1 | 56.6 / 58.6 | 59.7 (57.3 to 62.0) | 61.0 (58.0 to 63.9) | 26% at 71.2 | 67.2 | 62.9 (58.6 to 67.1) | 55% at 73.7 |

#### Candidate minus control, paired, 95% intervals from resampling accounts

`mean of seeds`: the per-pair difference averaged over the two seed-matched pairs of runs. `ensembles`: two-seed candidate ensemble against two-seed control ensemble. Points.

| Candidate | Ordinary test pairs | Dev pairs, all | Dev 2x confident calls per 100 | Benchmark, all (mean of seeds) | Benchmark, all (ensembles) | Benchmark 2x (ensembles) | Benchmark 2x confident share | Benchmark 2x confident calls per 100 | Surest 40% of 2x | Launch, all |
|---|---|---|---|---|---|---|---|---|---|---|
| thurstone | -0.0 (-0.1 to +0.1) | -0.2 (-1.1 to +0.6) | +0.1 (-0.5 to +0.6) | -0.4 (-1.0 to +0.2) | +0.2 (-0.6 to +0.9) | +0.5 (-0.5 to +1.3) | -1.2 (-1.9 to -0.5) | -0.7 (-1.3 to -0.2) | +0.0 (-1.1 to +1.0) | +0.6 (-0.9 to +2.1) |
| soft | -0.2 (-0.2 to -0.1) | -0.2 (-1.2 to +0.6) | -1.1 (-1.9 to -0.3) | -0.6 (-1.2 to -0.1) | +0.1 (-0.7 to +0.9) | +0.4 (-0.5 to +1.3) | -1.5 (-2.3 to -0.7) | -1.0 (-1.7 to -0.5) | +0.6 (-0.6 to +1.7) | -0.2 (-1.4 to +0.9) |
| ridge | +0.1 (-0.2 to +0.3) | +0.1 (-1.6 to +1.8) | +0.6 (-0.5 to +1.7) | -1.6 (-2.9 to -0.5) | -1.0 (-2.4 to +0.5) | -1.0 (-2.8 to +0.9) | -0.8 (-2.5 to +0.9) | -0.6 (-2.0 to +0.7) | +0.0 (-2.0 to +2.0) | -2.8 (-5.5 to -0.3) |
| lawa vs ctl_s1+ctl_s2 | -0.1 (-0.1 to -0.0) | -0.0 (-0.7 to +0.6) | +0.1 (-0.4 to +0.7) | - | +0.0 (-0.6 to +0.6) | +0.4 (-0.3 to +1.2) | -0.3 (-0.9 to +0.2) | -0.1 (-0.6 to +0.4) | -0.7 (-1.8 to +0.2) | +0.1 (-0.8 to +1.2) |

Reading the tables:

- **Ordinary test pairs** (683,152 pairs): every candidate is within 0.2 points of the control, with intervals of ±0.1 to ±0.3.
- **Benchmark**: one control seed scores 60.9% and the other 58.7%, so a single pair of runs cannot show less than about 2 points. The paired, two-seed differences are tighter: ±0.8 for the pair losses and the weight average, ±1.4 for ridge.
- **Confident calls**: no candidate raises the share of 2x pairs called at 60% or the accuracy on the surest 40%. The two pair losses lower the share by 1.2 to 1.5 points (intervals exclude zero) with accuracy there unchanged.
- **Launch pairs**: no candidate helps; ridge is 2.8 points worse.
- **Overfitting**: train pair accuracy climbs from 59% to 63 to 66% while validation is flat from step 2,000 (61.4 to 61.7%) and test from step 2,500. No candidate narrows the train-test gap; ridge widens it slightly (66.5% train at the same test).
- The masked-history task lowers the like-head NLL on test rows (2.962 and 2.964 against 2.970 and 2.972) without changing ranking. It is a small calibration gain, not an accuracy gain, and it was not scored on the benchmark.

#### Curves: train / validation / test at every validation step

Same-account pairs on likes, ordinary posts (train: 6,000 fitted rows; validation: held-out accounts, selects the step by NLL; test: 6,000 rows after the cutoff), like-head NLL, and accuracy on the development pairs with a 2x gap. Two seeds, `s1 / s2`.

| Run | Step | Pairs train | Pairs val | Pairs test | NLL train | NLL val (all heads) | NLL test | Dev pairs 2x |
|---|---|---|---|---|---|---|---|---|
| ctl | 500 | 59.6 / 59.2 | 59.9 / 59.7 | 58.6 / 59.2 | 3.068 / 3.064 | 3.090 / 3.081 | 3.086 / 3.080 | 58.8 / 58.4 |
| ctl | 1000 | 60.5 / 59.6 | 61.0 / 61.1 | 59.0 / 60.2 | 2.997 / 2.996 | 3.010 / 3.004 | 3.021 / 3.021 | 60.4 / 58.9 |
| ctl | 1500 | 61.0 / 62.0 | 61.6 / 61.3 | 59.8 / 60.6 | 2.970 / 2.969 | 2.982 / 2.981 | 3.005 / 2.999 | 58.9 / 59.6 |
| ctl | 2000 | 61.7 / 61.9 | 61.7 / 61.4 | 60.2 / 59.6 | 2.949 / 2.953 | 2.967 / 2.964 | 2.991 / 2.986 | 58.6 / 60.8 |
| ctl | 2500 | 62.3 / 63.7 | 61.7 / 61.7 | 60.7 / 60.7 | 2.936 / 2.940 | 2.959 / 2.959 | 2.982 / 2.979 | 57.8 / 60.6 |
| ctl | 3000 (chosen / chosen) | 62.7 / 65.4 | 61.7 / 61.7 | 60.9 / 60.1 | 2.931 / 2.934 | 2.957 / 2.957 | 2.979 / 2.977 | 56.9 / 60.2 |
| thurstone | 500 | 59.4 / 60.2 | 60.1 / 59.8 | 58.4 / 59.6 | 3.066 / 3.060 | 3.087 / 3.081 | 3.088 / 3.079 | 57.9 / 58.5 |
| thurstone | 1000 | 61.9 / 59.5 | 61.2 / 61.3 | 59.1 / 59.7 | 2.995 / 2.994 | 3.009 / 3.003 | 3.022 / 3.022 | 60.5 / 58.8 |
| thurstone | 1500 | 61.3 / 61.2 | 61.5 / 61.1 | 59.3 / 60.5 | 2.968 / 2.969 | 2.982 / 2.982 | 3.003 / 3.000 | 60.0 / 58.9 |
| thurstone | 2000 | 63.0 / 62.4 | 61.6 / 61.4 | 60.1 / 60.2 | 2.949 / 2.954 | 2.969 / 2.966 | 2.991 / 2.989 | 58.8 / 59.6 |
| thurstone | 2500 | 63.2 / 64.6 | 61.7 / 61.6 | 59.8 / 60.7 | 2.935 / 2.941 | 2.960 / 2.962 | 2.980 / 2.981 | 57.4 / 59.7 |
| thurstone | 3000 (chosen / chosen) | 63.7 / 65.4 | 61.6 / 61.5 | 60.3 / 60.2 | 2.930 / 2.935 | 2.958 / 2.960 | 2.978 / 2.979 | 57.3 / 59.0 |
| soft | 500 | 58.1 / 59.2 | 59.6 / 59.4 | 58.1 / 59.0 | 3.069 / 3.067 | 3.091 / 3.084 | 3.085 / 3.082 | 58.0 / 58.6 |
| soft | 1000 | 61.7 / 58.5 | 60.8 / 61.0 | 58.5 / 59.7 | 2.998 / 2.996 | 3.010 / 3.004 | 3.021 / 3.020 | 60.5 / 58.5 |
| soft | 1500 | 62.0 / 61.3 | 61.3 / 61.1 | 58.9 / 60.3 | 2.969 / 2.968 | 2.982 / 2.981 | 3.002 / 2.997 | 58.3 / 58.3 |
| soft | 2000 | 63.0 / 62.9 | 61.6 / 61.4 | 59.8 / 59.7 | 2.950 / 2.953 | 2.969 / 2.966 | 2.990 / 2.986 | 57.7 / 59.1 |
| soft | 2500 | 63.2 / 64.1 | 61.6 / 61.7 | 59.8 / 60.6 | 2.937 / 2.940 | 2.961 / 2.960 | 2.979 / 2.979 | 57.1 / 59.2 |
| soft | 3000 (chosen / chosen) | 64.3 / 64.7 | 61.5 / 61.6 | 59.9 / 59.8 | 2.931 / 2.934 | 2.958 / 2.958 | 2.977 / 2.977 | 56.5 / 59.0 |
| ridge | 500 | 59.9 / 58.5 | 60.0 / 59.6 | 58.6 / 60.3 | 3.064 / 3.068 | 3.082 / 3.086 | 3.082 / 3.085 | 58.6 / 58.3 |
| ridge | 1000 | 60.2 / 60.6 | 60.8 / 61.2 | 58.9 / 59.1 | 2.987 / 2.990 | 3.002 / 3.000 | 3.017 / 3.014 | 59.2 / 58.8 |
| ridge | 1500 | 62.4 / 62.7 | 61.5 / 61.2 | 59.6 / 60.7 | 2.959 / 2.962 | 2.975 / 2.976 | 2.995 / 2.993 | 57.9 / 58.9 |
| ridge | 2000 | 62.7 / 64.4 | 61.6 / 61.2 | 59.9 / 59.4 | 2.941 / 2.943 | 2.964 / 2.962 | 2.984 / 2.980 | 58.6 / 58.5 |
| ridge | 2500 | 65.3 / 66.1 | 61.6 / 61.3 | 59.8 / 59.2 | 2.928 / 2.930 | 2.957 / 2.958 | 2.978 / 2.978 | 58.7 / 58.8 |
| ridge | 3000 (chosen / chosen) | 66.2 / 65.7 | 61.6 / 61.2 | 60.3 / 59.1 | 2.922 / 2.925 | 2.956 / 2.957 | 2.976 / 2.976 | 57.6 / 58.8 |
| mask | 500 | 60.9 / 58.4 | 60.3 / 59.8 | 59.1 / 58.3 | 3.058 / 3.049 | 3.080 / 3.070 | 3.071 / 3.065 | 57.8 / 59.7 |
| mask | 1000 | 60.2 / 59.6 | 61.0 / 61.3 | 59.2 / 60.5 | 2.986 / 2.985 | 3.005 / 2.999 | 3.007 / 3.007 | 58.2 / 59.9 |
| mask | 1500 | 59.8 / 60.9 | 61.6 / 61.5 | 59.9 / 60.6 | 2.961 / 2.958 | 2.977 / 2.975 | 2.987 / 2.989 | 57.9 / 59.4 |
| mask | 2000 | 62.4 / 61.7 | 61.5 / 61.6 | 60.5 / 60.5 | 2.944 / 2.940 | 2.964 / 2.961 | 2.978 / 2.976 | 57.9 / 58.7 |
| mask | 2500 | 63.9 / 63.0 | 61.4 / 62.0 | 60.9 / 60.6 | 2.932 / 2.928 | 2.958 / 2.955 | 2.972 / 2.970 | 57.6 / 58.8 |
| mask | 3000 (chosen / chosen) | 63.4 / 63.3 | 61.6 / 62.0 | 60.7 / 60.4 | 2.925 / 2.922 | 2.957 / 2.954 | 2.970 / 2.969 | 56.6 / 58.3 |

History dropout, stopped at step 1,500 (control rows at the same steps are in the table above):

| Run | Step | Pairs train | Pairs val | Pairs test | NLL val (all heads) | Dev pairs 2x |
|---|---|---|---|---|---|---|
| histdrop | 500 | 59.5 / 57.0 | 60.5 / 59.5 | 58.1 / 59.3 | 3.1018 / 3.0883 | 58.5 / 58.0 |
| histdrop | 1000 | 60.6 / 60.1 | 61.0 / 61.0 | 59.0 / 58.8 | 3.0110 / 3.0091 | 59.0 / 58.6 |
| histdrop | 1500 | 60.8 / 60.8 | 61.5 / 61.6 | 59.6 / 59.5 | 2.9867 / 2.9843 | 58.8 / 57.2 |

### B.2 Confidence without retraining (the review's first two items)

Both were judged on development pairs first and then scored on the benchmark once, on the shipped three-model baseline (`scripts/sota/selector.py`, `sota_lib.py`; `tac_serve_eval.py` now writes each version's predicted spread and level).

| Rule for "confident" | Dev pairs 2x: share at 60%+ | Correct there | Dev: surest 40% of 2x | Benchmark 2x: share at 60%+ | Correct there | Benchmark: surest 40% of 2x |
|---|---|---|---|---|---|---|
| Score gap, one slope (shipped) | 18% | 70.2% | 66.4% | 26% | 70.2% (65.0 to 74.6) | 67.6% |
| Gap divided by the two predicted spreads | 22% | 71.2% | 66.6% | 31% | 69.4% (64.6 to 73.9) | 67.4% |
| Selector: gap, spread, predicted level (logistic, fitted on development pairs) | 22% | 69.7% | 64.0% | 26% | 71.2% (66.1 to 75.8) | 67.2% |

- The predicted spread varies between posts (0.33 to 0.43 log10 for the middle half), but the spread-scaled gap correlates 0.96 with the plain gap. It relabels pairs near the threshold; it does not find better ones.
- Selector against the gap rule on the surest 40% of 2x pairs: -2.4 (-5.6 to +0.4) on development pairs, -0.4 (-1.9 to +1.1) on the benchmark. The predicted level gets a coefficient of zero.
- This agrees with the replicate track's finding that agreement between members adds nothing to the score gap.

### B.3 What the gap bands contain

Baseline three, benchmark pairs, by the lower like count of the two posts:

| Lower like count | Pairs 2x apart | Correct | Confident share | Correct there |
|---|---|---|---|---|
| 0 to 2 | 944 | 62.7% | 11% | 88.0% |
| 3 or more | 1,086 | 60.2% (56.6 to 63.3) | 38% | 65.5% |
| 10 or more | 771 | 59.8% (55.4 to 63.5) | 43% | 64.6% |
| 30 or more | 461 | 59.4% (53.3 to 64.4) | 46% | 62.9% |
| All | 2,030 | 61.4% (58.7 to 64.0) | 26% | 70.2% |

A pair of 1 like against 3 counts as "3x apart". Where both posts have real counts, confident calls are right about 65% of the time, not 70%, and the model calls more of them. The goal of 75 to 80% on confident 2x calls is further away on accounts with real engagement than the pooled figure suggests. I did not test whether the pair fine-tune's 73.6% splits the same way.

## What worked

Nothing raised accuracy or confident coverage. Two things are worth keeping:

- The masked-history task lowers test NLL of the like head by 0.008 at no cost in ranking. It is a candidate for the forecast's calibration, not for pair accuracy, and it needs a benchmark run before use.
- The count-aware split of the benchmark (B.3) changes how the goal should be read.

## What did not work

- Noise-aware and heteroscedastic pair losses: no accuracy change, slightly fewer confident calls.
- Per-account ridge layer: worse on benchmark and launch pairs.
- Weight averaging and a checkpoint ensemble of one run: no change (the loop's EMA already does this).
- Dropout of history: no validation movement.
- Spread-based confidence and a pair-level selector: no better ordering of pairs than the score gap.

Together with the earlier tracks this covers inputs, capacity, data, objective, regularisation, adaptation and the confidence rule. The remaining untested directions in the review are an independent model family as an ensemble member (tabular in-context model, fine-tuned language-model judge) and masked pretraining on all 4.48M posts; the review expects under one point from each.

## Limits

- Two seeds per arm. Paired intervals on the benchmark are about ±0.8 points for most candidates; an effect smaller than that would not be seen.
- Candidates were trained from scratch for 3,000 steps, not as fine-tunes of the shipped models. The pair-loss fine-tune of the pairs and replicate tracks (which does raise confident coverage) was not combined with any candidate.
- `histdrop` was stopped at half length and `mask` was not scored on benchmark or launch pairs, by the validation rule. `listnet` and the cyclic snapshot run were not trained. The stop rule was written after the first two candidates had finished and while the third (`ridge`) was at step 500.
- The masked-history test is a joint task on the training table, not pretraining on unlabeled posts.
- The benchmark runs through the serving path, which supplies no cross-account neighbours although the model was trained with them. I did not test a model trained without neighbours.
- The launch confidence uses the benchmark's scale, which the replicate track showed is too steep for launch pairs; launch confident shares here are for comparison between arms only.
- Review: claims are from abstracts and search summaries; link status is marked per paper in `REVIEW.md`.
- No OpenAI calls were made: 0 tokens, 0 US dollars.
- One existing test, `tests/test_serve.py::test_product_service_and_http`, fails in this environment on a missing web package; the other 15 model and serving tests pass with the patch.

## Files in `tracks/sota/`

- `REPORT.md`, `REVIEW.md` (the full literature review), `patch.diff`.
- `results.json` (every number in B.1, with histories), `tables.md`, `selector.json` (B.2), `stopped_curves.json`.
- `runs/<name>_s{1,2}/metrics.json` and `best.pt` for `ctl`, `thurstone`, `soft`, `ridge`, `mask`; `runs/ctl_s{1,2}/swa.pt` and `swa.json`.
- `bench/<name>/shard_*.parquet` (per-pair benchmark scores with predicted spread; `lvl_*` = the baseline three with spread and level), `dev/<name>.npz`, `launch/<name>.parquet`.

## Reproduce

From `virality/`, after `scripts/worker_job.sh`-style setup, with the dataset's `work/`, `runs/`, `tracks/pairs/`, `tracks/replicate/`, `tracks/launch/` under `data/`, the symlinks of the replicate report (`data/pairs/benchmark_pairs.parquet`, `data/bench3/cache_{0..3}.pkl`), and `scripts/rep_lib.py`, `rep_launch.py`, `rep_score_launch.py` taken from `tracks/replicate/patch.diff`:

```
bash scripts/sota/queue.sh 1 2 & bash scripts/sota/queue.sh 2 2 &     # control and candidates, one seed per queue
bash scripts/sota/eval_all.sh ctl_s1 ctl_s2 thurstone_s1 ...           # benchmark, launch, development and test-row scores
python scripts/sota/swa.py --run data/sota/runs/ctl_s1 --steps 2000,2500,3000
python scripts/sota/report.py --cands thurstone,soft,ridge,mask --extra "lawa=lawa_s1+lawa_s2|ctl_s1+ctl_s2"
python scripts/sota/selector.py --members lvl_ts46_s1,lvl_ts46_s2,lvl_v2full --out data/sota/results/selector.json
```
