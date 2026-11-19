# Track "selective": when to speak, treated as its own learning problem

Worker report, 2026-11-19. All numbers are likes. Intervals are 95%, from resampling accounts. Settings and decision rules were written to `PREREG.md` before any deferral model was fitted.
Every model and every scale here was fitted on pairs from before 2026-04-23 only; the benchmark and the launch pairs were scored once per configuration, after the choice on validation. No OpenAI calls were made (0 tokens, 0 dollars).

## Headline

- **The goal (75 to 80% right on at least 40% of pairs 2x apart) is not reached on the benchmark, and no learned silence rule moves it.**
  Ten fine-tuned members with one scale fitted on development pairs: the surest 40% of the 2,030 pairs 2x apart are right 72.5% of the time (67.4 to 76.7); at a stated 60% or more, 36.0% of pairs (730) at 73.4% (67.9 to 77.7).
  Boosted trees, a small network and a regularised linear model with 68 inputs each give 70.6%, 71.6% and 71.8% at 40% coverage: differences to the scale of -2.0 (-3.8 to -0.1), -1.0 (-2.3 to +0.4), -0.7 (-2.2 to +0.8).
- **On launch pairs the goal's numbers are met by a plain scale fitted on launch pairs from before the cutoff:** the surest 40% of the 568 pairs 2x apart are right 77.1% (69.2 to 84.5) with ten untouched members and 77.5% (69.3 to 84.7) with the baseline three; at a stated 60%+, 35% of pairs at 77 to 78%.
  Two cautions. The lower end of the interval is 69%. And launch pairs are two different posts, a median of 55 days apart in the test period (140 before the cutoff): among the 188 pairs 2x apart posted within 30 days of each other, the surest 40% are right 72.0% (60.2 to 82.9).
- **A learned rule looked much better on launch pairs in validation and was not better after the cutoff.** On 3,655 pre-cutoff launch pairs with accounts held out, boosted trees beat the scale by 5.7 points (3.0 to 8.8) on the validation metric, which passed the rule fixed in advance.
  On the 850 launch pairs after the cutoff: 75.8% against 77.1% at 40% coverage of 2x pairs, -1.3 (-6.0 to +2.9). Train 91.9%, validation 82.5%, test 75.8%. The scale: 77.1%, 77.0%, 77.1%. I do not ship the learned rule.
- **The same stated confidence means less after the cutoff.** On the benchmark's pre-cutoff half, calls stated at 60%+ are right 80.0% of the time (280 of 913 pairs 2x apart); on the post-cutoff half 69.3% (450 of 1,117). At 40% coverage: 76.7% (71.4 to 81.1) against 69.6% (63.5 to 75.8). The honest number for a product used after the cutoff is the second.
- **Risk control gives a guarantee only when the future looks like the past, and here it does not.** Thresholds set to keep errors among spoken 2x pairs under 25 to 30% held on launch pairs (by speaking on 25 to 33% of pairs instead of the 41 to 48% they allowed in training) and failed on the benchmark in three of four settings.
- **What ships:** `xviral/data/selective_rule.json` (one scale per ensemble and per kind of pair, fitted on pre-cutoff pairs), `xviral/selective.py` (loader, feature function, evaluator for scale, linear, tree and network rules from plain JSON), a hook in the advisor, and tests.
  The product's serving path now produces every deferral input (checked equal to the offline tables to 4e-5 on 68 features), so a better rule can be dropped in as a file.

## 1. What the selective prediction literature says for this setting

| Line of work | Idea | Fit with heavy irreducible label noise |
|---|---|---|
| Reject option (Chow 1970; El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017, "selective classification for deep networks") | Abstain when the estimated probability of being right is under a threshold; quality is the risk-coverage curve. The threshold can be chosen with a finite-sample bound on selective risk. | The best fit. With noise, the right ordering of pairs is by the true probability of being right, and Chow's rule on a calibrated probability is optimal. Everything hinges on estimating that probability, which is a low-capacity calibration problem, not a new model. |
| SelectiveNet (Geifman and El-Yaniv 2019) | A second head learns whom to select, trained jointly under a coverage constraint. | Poor fit. It needs many examples where being right is learnable from the input. Here 39% of identical posts differ by 2x, so "the call was right" is itself mostly noise; a selection head with capacity learns the training pairs. Later comparisons (Feng et al. 2023) also found its gain comes from the representation, and that the plain confidence score selects as well. |
| Deep gamblers (Liu et al. 2019), self-adaptive training (Huang et al. 2020) | An extra "abstain" output with a payoff, trained end to end. | Same problem, plus it changes the surrogate's training, which earlier tracks showed costs accuracy when pair labels are trained on from scratch (-3.1 points). |
| Learning to defer (Madras et al. 2018; Mozannar and Sontag 2020) | Route an input to a model or to a human expert, with a consistent surrogate loss. | There is no second expert here: deferring means staying silent. It reduces to the reject option with a cost of silence. Its useful part is the framing used below: a separate deferral model whose inputs may be richer than the predictor's score. |
| Conformal risk control and learn-then-test (Angelopoulos et al. 2021, 2022) | Pick the threshold on held-out data so that a risk (here: error among spoken calls) is controlled with a finite-sample guarantee, for any score. | Good fit for choosing the threshold, with two conditions it does not meet here on its own: the loss must be monotone in the threshold (error among spoken calls is not; "spoken and wrong" is), and test pairs must be exchangeable with calibration pairs (they are not across the cutoff or across kinds of pair). |
| Calibration under shift (Guo et al. 2017; Ovadia et al. 2019; Tibshirani et al. 2019) | Post-hoc scaling is calibrated only in distribution; under shift, ensembles degrade least and richer calibrators degrade most; weighted conformal handles known covariate shift. | Directly relevant: the replicate track's finding that the scale differs by kind of pair and halves after fine-tuning is this. The practical lesson, confirmed below, is to keep the calibrator as small as possible and fit it per kind of pair. |

Reading for this product: label noise caps how well any rule can order pairs (the pairs track put the ceiling for a model that knows every effect at 85% on 2x pairs), so the selection function should be the calibrated probability of the call itself, the calibrator should have one or two parameters per kind of pair, and the threshold should be set by risk control where a stable calibration set of the same kind exists.
End-to-end selection learning (SelectiveNet, deep gamblers) is the wrong tool. The experiment below tests the remaining possibility, a separate deferral model on richer inputs, and finds that it does not survive the move to later pairs.

## 2. Data

Everything was rescored through the product's serving path (`DraftScorer`, 56 posts of history, median of the like histogram) with twenty checkpoints (ten untouched, ten fine-tuned), keeping each member's quantiles and full like histogram for both versions (`scripts/selective/sel_score.py`).
My rescoring of the benchmark reproduces the stored per-pair scores to 3e-6; the development pairs' serving-path scores correlate 0.997 with the replicate track's table-path scores.

| Set | Role | Pairs 1.25x apart | 2x apart | Accounts | Calls right, fine-tuned ten / untouched ten / baseline three |
|---|---|---|---|---|---|
| Development pairs (same link or quoted post, before the cutoff; replicate track) | training and validation, same-topic | 2,805 | 1,956 | 366 | 56.7% / 56.1% / 56.3% |
| Launch pairs before the cutoff (built here like the launch track's test pairs: same account, both launch-type, at most 30 per account) | training and validation, launch | 3,655 | 2,614 | 628 | 64.7% / 63.6% / 63.8% |
| Benchmark | test, same-topic | 2,852 | 2,030 | 770 | 61.4% / 61.1% / 60.0% |
| Launch pairs after the cutoff (launch track) | test, launch | 850 | 568 | 151 | 63.2% / 62.5% / 63.9% |

The pre-cutoff launch pairs are new: 2,757 launch-type posts of 676 accounts scored at their own time. 2,601 of the 3,655 pairs are from accounts the models were trained on; calls there are not more often right (62.8% against 65.6% for untrained accounts, untouched ten), so they were kept.

**Inputs of the deferral model** (76 per pair; `xviral/selective.py: pair_features`). All are oriented by the call (picked version minus the other) or symmetric, never by posting order.
- The call: size of the mean gap, its spread across members, share of members on the call's side, weakest member's margin, the same gap seen by the other ensemble (untouched members when fine-tuned ones call, and the reverse).
- Predicted distributions: width (10th to 90th percentile) of each version, gap over combined width, probability that a draw from the pick's histogram beats a draw from the other's (per member and for the mixture), entropy.
- Account: posts of history, median, spread, interquartile range and top tenth of its log likes, followers, cadence, media and link shares.
- How unusual each version is: cosine to the centroid and to the three nearest of the last 48 posts, length, media and link against the account's own habits; cosine between the two versions.
- What differs: media, video, photo, link, link in the first line, quote, lines, words, emoji, question or number or first-person or announcement opening, call to action, bullets, post type from the launch text rules.
- For two posts at different times (launch pairs): days apart, and the part of the gap that is the account's own level moving between the two dates.

**Target:** the call is right; pairs under 1.25x unused; weighting (flat, proportional to the log like ratio capped at 5x, or 2x pairs only) chosen on validation.

## 3. Validation (pre-cutoff pairs, accounts held out, 5 folds repeated 3 times)

Metric fixed in advance: mean accuracy of the surest 20, 30, 40 and 50% of held-out pairs 2x apart. 17 configurations times 3 weightings; the best of each family:

| Trained on, evaluated on | Calls by | Scale (current rule) | Linear | Boosted trees | Small network | Best minus scale |
|---|---|---|---|---|---|---|
| Development pairs (1,956 at 2x) | fine-tuned ten | 69.5% | 70.4% (8 inputs + histogram overlap) | 69.8% | 69.0% | +0.8 (-0.7 to +2.6) |
| Development + launch pairs, evaluated on development pairs | fine-tuned ten | 69.6% | 70.4% | 71.0% | not run | +1.3 (-0.3 to +2.9) |
| Launch pairs before the cutoff (2,614 at 2x) | untouched ten | 77.5% | 82.9% (22 inputs) | 83.2% (68 inputs, depth 3) | 81.8% | +5.7 (+3.0 to +8.8) |
| Launch + development pairs, evaluated on launch pairs | untouched ten | 77.7% | 82.7% | 82.7% | not run | +4.9 (+2.1 to +7.8) |
| Launch pairs before the cutoff | fine-tuned ten | 79.0% | 82.7% | 82.7% | not run | +3.6 (+1.2 to +6.1) |

By the rule fixed in advance: on same-topic pairs nothing replaces the scale (no interval excludes zero); on launch pairs boosted trees do, and a separate launch model is at least as good as a pooled one.

What carries signal beyond the size of the gap, on training pairs 2x apart (logistic regression of "call right" on the gap plus one input, interval from resampling accounts; `univariate.json`):
- Development pairs: 10 of 71 inputs. Histogram overlap (coefficient +0.53, 0.25 to 0.79), the pick has media where the other does not (+0.16), gap in the odds of an own top-tenth post (+0.29), the pick has a call to action (-0.17) or a link in the first line (-0.12).
- Launch pairs: 31 of 70. Histogram overlap (+1.23), width of the other version's distribution (-0.43), the untouched/fine-tuned other ensemble's gap (+0.70), the share of the gap that is account level (-0.56: a gap that comes from the account having grown is less reliable than one from the content), the pick is a quote (-0.24), the pick has video (+0.25).

So the inputs do say something on pre-cutoff pairs. The question the test sets answer is whether it holds later.

## 4. Test: risk-coverage on the benchmark and the launch pairs

Accuracy of the surest share of pairs (ranked inside each gap group), and at the product's threshold. Each row scored once.

**Benchmark (2,852 pairs, 770 accounts), calls by the fine-tuned ten, everything fitted on the 2,805 development pairs:**

| Method | Real gap at least | Pairs | All calls | Surest 20% | 30% | 40% | 50% | Stated 60%+: share (pairs) | Correct there |
|---|---|---|---|---|---|---|---|---|---|
| Scale (current rule; 1.75) | 1.25x | 2,852 | 61.4% | 71.9% (67.1 to 76.2) | 70.8% (66.7 to 74.8) | 69.0% (65.2 to 72.7) | 66.5% (63.1 to 69.9) | 35.8% (1,022) | 70.1% (66.0 to 74.0) |
| Scale | 2x | 2,030 | 63.0% | 74.6% (68.8 to 79.7) | 73.6% (68.1 to 78.0) | **72.5% (67.4 to 76.7)** | 69.2% (64.5 to 73.1) | 36.0% (730) | 73.4% (67.9 to 77.7) |
| Linear, 9 inputs | 1.25x | 2,852 | 61.4% | 73.7% (68.9 to 77.8) | 70.1% (66.2 to 74.4) | 68.6% (64.9 to 72.3) | 67.0% (63.2 to 70.8) | 39.2% (1,117) | 68.8% (65.0 to 72.7) |
| Linear, 9 inputs | 2x | 2,030 | 63.0% | 77.1% (71.7 to 81.6) | 73.4% (67.8 to 77.9) | 71.8% (66.7 to 76.0) | 69.8% (64.8 to 74.0) | 38.8% (788) | 72.1% (67.0 to 76.4) |
| Boosted trees, 71 inputs | 1.25x | 2,852 | 61.4% | 74.2% (69.6 to 78.4) | 70.9% (66.6 to 75.0) | 67.8% (63.8 to 71.8) | 66.8% (63.1 to 70.3) | 52.8% (1,507) | 66.2% (62.6 to 69.7) |
| Boosted trees, 71 inputs | 2x | 2,030 | 63.0% | 76.4% (70.6 to 81.3) | 73.2% (67.5 to 78.2) | 70.6% (65.3 to 75.2) | 69.5% (64.4 to 73.7) | 52.0% (1,056) | 69.2% (64.4 to 73.3) |
| Small network, 8 inputs | 1.25x | 2,852 | 61.4% | 71.4% (66.7 to 75.4) | 70.8% (66.8 to 74.7) | 68.9% (64.9 to 72.6) | 66.9% (63.2 to 70.6) | 49.0% (1,397) | 66.9% (63.2 to 70.7) |
| Small network, 8 inputs | 2x | 2,030 | 63.0% | 73.9% (68.2 to 78.7) | 72.9% (67.6 to 77.2) | 71.6% (66.6 to 75.8) | 69.5% (64.5 to 73.8) | 48.1% (976) | 69.9% (64.8 to 74.1) |

Paired with the scale on pairs 2x apart, points: linear +2.5 (+0.1 to +5.1) at 20% coverage, -0.2, -0.7 (-2.2 to +0.8), +0.6 at 30, 40, 50%; boosted trees +1.7, -0.3, -2.0 (-3.8 to -0.1), +0.3; network -0.7, -0.7, -1.0, +0.3.
The one positive interval (linear at 20%) is one of 24 comparisons and sits next to negative ones; I read it as nothing.

Other ensembles with the scale rule, same development-pair fit (for comparison with earlier tables, which cross-fitted the scale on the benchmark itself):

| Ensemble (scale) | Pairs 2x apart | All calls | Surest 40% | Stated 60%+: share | Correct there | Earlier tables, scale cross-fitted on the benchmark |
|---|---|---|---|---|---|---|
| Baseline three (2.77) | 2,030 | 61.4% | 67.6% (62.8 to 71.8) | 24.0% (488) | 70.3% (64.9 to 74.9) | 26% at 70.2% |
| Untouched ten (3.14) | 2,030 | 62.6% | 69.6% (64.7 to 73.8) | 26.1% (529) | 71.5% (66.0 to 76.0) | 28% at 70.8% |
| Fine-tuned ten (1.75) | 2,030 | 63.0% | 72.5% (67.4 to 76.7) | 36.0% (730) | 73.4% (67.9 to 77.7) | 37% at 73.5% |

The earlier coverage figures stand with a scale that never saw the benchmark: fine-tuning buys about 12 points of coverage and 3 of accuracy over the baseline. Larger gaps, fine-tuned ten: 3x apart, 38% of 1,377 pairs at 77.1%; 5x apart, 42% of 880 at 82.9%.

**Launch pairs after the cutoff (850 pairs, 151 accounts), calls by the untouched ten, everything fitted on the 3,655 pre-cutoff launch pairs:**

| Method | Real gap at least | Pairs | All calls | Surest 20% | 30% | 40% | 50% | Stated 60%+: share (pairs) | Correct there |
|---|---|---|---|---|---|---|---|---|---|
| Scale (1.84) | 1.25x | 850 | 62.5% | 75.3% (66.5 to 84.8) | 72.9% (65.2 to 80.8) | 71.2% (64.3 to 78.2) | 71.5% (65.3 to 77.8) | 31.2% (265) | 72.5% (64.8 to 79.9) |
| Scale | 2x | 568 | 64.3% | 78.1% (67.0 to 89.1) | 78.2% (69.0 to 86.4) | **77.1% (69.2 to 84.5)** | 74.6% (67.5 to 81.4) | 34.9% (198) | 77.3% (69.0 to 84.6) |
| Linear, 22 inputs | 2x | 568 | 64.3% | 78.1% (67.3 to 88.4) | 75.3% (65.7 to 84.4) | 74.9% (67.2 to 82.0) | 73.6% (66.8 to 80.1) | 74.3% (422) | 69.2% (62.9 to 75.4) |
| Boosted trees, 68 inputs | 1.25x | 850 | 62.5% | 77.6% (69.0 to 86.1) | 75.3% (67.2 to 83.3) | 72.1% (64.9 to 79.0) | 70.6% (63.8 to 77.1) | 66.5% (565) | 69.7% (63.9 to 75.7) |
| Boosted trees, 68 inputs | 2x | 568 | 64.3% | 82.5% (72.0 to 91.7) | 78.8% (69.7 to 86.8) | 75.8% (68.1 to 83.2) | 74.3% (65.5 to 82.3) | 66.2% (376) | 73.4% (66.6 to 79.8) |
| Small network, 22 inputs | 2x | 568 | 64.3% | 78.1% (68.6 to 87.8) | 75.3% (67.1 to 83.0) | 76.7% (69.1 to 83.2) | 74.3% (67.6 to 80.7) | 64.4% (366) | 71.6% (65.3 to 77.6) |

Boosted trees minus scale on pairs 2x apart: +4.4 (-3.2 to +13.9), +0.6, -1.3 (-6.0 to +2.9), -0.4 at 20, 30, 40, 50% coverage. Not better; and its stated probabilities are too high after the cutoff (it states 60%+ on 66% of pairs and is right on 73.4% of them, where it promised about 80% on validation).
Scale rule with the other ensembles: baseline three 77.5% (69.3 to 84.7) at 40% coverage, 34.7% of pairs at 78.2% stated 60%+; fine-tuned ten 75.8% (67.6 to 83.3), 34.9% at 77.8%. Fine-tuning does not help on launch pairs, as the replicate track found.
With a threshold of 57.5% instead of 60%: untouched ten speak on 46.7% of 2x pairs at 74.7%, baseline three on 47.2% at 77.6%.

**Train, validation and test side by side** (pairs 2x apart, accuracy of the surest 40%):

| Pairs | Method | Train, in sample | Validation, accounts held out | Test |
|---|---|---|---|---|
| Same-topic | Scale | 67.1% | 67.4% | 72.5% (67.4 to 76.7) |
| Same-topic | Linear | 69.4% | 67.5% | 71.8% (66.7 to 76.0) |
| Same-topic | Boosted trees | 74.2% | 67.9% | 70.6% (65.3 to 75.2) |
| Same-topic | Small network | 69.2% | 67.9% | 71.6% (66.6 to 75.8) |
| Launch | Scale | 77.1% | 77.0% | 77.1% (69.2 to 84.5) |
| Launch | Linear | 83.3% | 81.8% | 74.9% (67.2 to 82.0) |
| Launch | Boosted trees | 91.9% | 82.5% | 75.8% (68.1 to 83.2) |
| Launch | Small network | 86.7% | 80.6% | 76.7% (69.1 to 83.2) |

(Same-topic test is above validation for every method because benchmark pairs are at most 48 hours apart and development pairs up to 14 days; the comparison that matters is between methods within a column.)
The launch rows are the track's main lesson. Holding accounts out did not expose the problem; holding time out did. What the richer models learned about launch pairs before the cutoff (how far to trust a gap that comes from the account's level having moved, a video, a quote) stopped being true after it, the same period shift the launch and posttype tracks measured (launch posts got 1.77x the account's usual before the cutoff and 2.81x after).

**Is 40% at 75% reached?**

| Pairs | Best configuration | Surest 40% of pairs 2x apart | Reached |
|---|---|---|---|
| Benchmark, all 2,030 | fine-tuned ten, scale | 72.5% (67.4 to 76.7) | No |
| Benchmark, post-cutoff half, 1,117 | fine-tuned ten, scale | 69.6% (63.5 to 75.8) | No |
| Launch pairs after the cutoff, 568 | untouched ten or baseline three, launch scale | 77.1% (69.2 to 84.5), 77.5% (69.3 to 84.7) | Yes as a point estimate; the interval reaches down to 69% |
| Launch pairs within 30 days of each other, 188 | untouched ten, launch scale | 72.0% (60.2 to 82.9) | Not shown |

## 5. Confidence by kind of pair, and stability

**Scales fitted on pre-cutoff pairs** (probability the pick is right = sigmoid(scale x gap in log10 predicted likes)):

| Ensemble | Same-topic pairs | Launch pairs |
|---|---|---|
| Baseline three | 2.77 | 1.84 |
| Untouched ten | 3.14 | 1.84 |
| Fine-tuned ten | 1.75 | 1.57 |

The scale halves after fine-tuning on same-topic pairs (3.14 to 1.75), as the replicate track found, and barely moves on launch pairs (1.84 to 1.57). The launch scale is lower than the same-topic one for untouched members:
launch gaps are larger (two posts weeks or months apart differ in predicted level: mean gap 0.29 against 0.08 in log10 for the untouched ten), so the same gap means less. Using the same-topic scale on launch pairs, as the product does today, states 60%+ on about 55 to 60% of launch pairs at 73 to 76% (replicate track); the launch scale states it on 35% at 77 to 78%. Fitting per kind is needed, and the per-kind fit from before the cutoff held after it (77.0% validation, 77.1% test).
A separate learned model per kind did not help where it could be tested (section 4), and pooling the two kinds changed nothing for same-topic pairs in validation (+1.3, -0.3 to +2.9).

**Stability of the scale rule** (pairs 2x apart; "surest 40%" is taken inside the group):

| Set | Group | Pairs 2x apart | All calls | Surest 40% | Stated 60%+: share | Correct there |
|---|---|---|---|---|---|---|
| Benchmark, fine-tuned ten | post-cutoff half | 1,117 | 62.0% | 69.6% (63.5 to 75.8) | 40.3% (450) | 69.3% (63.2 to 75.6) |
| Benchmark | pre-cutoff half | 913 | 64.3% | 76.7% (71.4 to 81.1) | 30.7% (280) | 80.0% (74.4 to 84.4) |
| Benchmark | under 10k followers | 1,045 | 64.4% | 73.7% (67.5 to 79.6) | 32.1% (335) | 75.8% (69.0 to 81.9) |
| Benchmark | 10k to 100k followers | 841 | 61.4% | 72.9% (65.1 to 79.1) | 39.6% (333) | 73.3% (65.5 to 79.2) |
| Benchmark | 100k followers and over | 89 | 62.9% | 58.3% (44.7 to 75.0) | 44.9% (40) | 60.0% (47.8 to 76.5) |
| Benchmark | account also has development pairs | 1,262 | 63.5% | 74.3% (68.9 to 79.2) | 33.3% (420) | 77.6% (71.5 to 82.4) |
| Benchmark | account has no development pairs | 768 | 62.1% | 68.1% (59.8 to 76.7) | 40.4% (310) | 67.7% (59.3 to 76.2) |
| Launch, untouched ten | 10k to 100k followers | 160 | 63.1% | 68.8% (54.5 to 82.1) | 45.6% (73) | 68.5% (53.0 to 81.5) |
| Launch | 100k followers and over | 264 | 66.3% | 78.3% (68.5 to 87.8) | 27.7% (73) | 80.8% (66.7 to 93.0) |
| Launch | posts within 30 days | 188 | 61.7% | 72.0% (60.2 to 82.9) | 31.4% (59) | 69.5% (55.9 to 81.8) |
| Launch | posts 30 to 180 days apart | 375 | 66.1% | 82.0% (73.0 to 89.9) | 36.3% (136) | 82.4% (73.0 to 89.9) |

Reading:
1. **The two halves of the benchmark differ by about 7 to 10 points on confident calls**, and the later half is the worse one. Overall accuracy differs by 2 points (62.0% against 64.3%); the drop is in how much a large gap means. The split by "has development pairs" shows the same thing, because the pre-cutoff half is drawn from those accounts. The post-cutoff half speaks more (40% against 31%) and is right less.
2. On the benchmark, accounts of 100k followers and over are at 58 to 60% on confident calls, with 89 pairs: an interval of 45 to 76%, in line with the pairs track's finding that label noise grows with post size. On launch pairs the large accounts are the good ones (80.8%), so no follower rule can be stated across kinds.
3. Launch pairs within 30 days are the closest the data comes to "two versions of one launch"; they sit near the benchmark's level (72%), not at 77%.
4. Stated probability against outcome, benchmark, fine-tuned ten: 50 to 55% stated, 55.6% right (1,233 pairs); 55 to 60%, 58.3% (597); 60 to 65%, 67.1% (410); 65 to 70%, 67.6% (284); 70 to 80%, 78.6% (257); above 80%, 66.2% (71).
   The top band falls back on both test sets (launch: 60.7% of 28). On pre-cutoff pairs it does not (development 87% of 39 pairs 2x apart, launch 82% of 180), so no rule about very large gaps can be fitted honestly from pre-cutoff data.
5. The advisor's existing gate "every member agrees" costs little with ten fine-tuned members on pairs that would be spoken anyway: 36.0% at 73.4% becomes 33.2% at 73.1% on the benchmark (members agree on 53% of all pairs). It adds nothing, as the replicate track found.

**Risk control** (thresholds chosen on out-of-fold probabilities of training pairs 2x apart; `ltt`: fixed-sequence binomial test that the error among spoken pairs is under the target, delta 0.1; `crc`: conformal risk control on the monotone loss "spoken and wrong" per pair):

| Pairs | Rule | Threshold on the stated probability | Share spoken, training | Share spoken, test | Correct among spoken, test | Target met on test |
|---|---|---|---|---|---|---|
| Same-topic, scale | error at most 25% (ltt) | 0.795 | 2.0% | 2.8% | 71.9% (56.8 to 82.4) | No |
| Same-topic, scale | error at most 30% (ltt) | 0.605 | 23.2% | 34.6% | 73.5% (67.9 to 78.0) | Yes |
| Same-topic, scale | spoken and wrong at most 10 per 100 (crc) | 0.575 | 32.1% | 43.9% | 70.7% | No (12.9 per 100) |
| Same-topic, scale | at most 12 per 100 (crc) | 0.560 | 37.1% | 50.8% | 68.8% | No (15.8 per 100) |
| Launch, scale | error at most 25% (ltt) | none passes | | | | |
| Launch, scale | spoken and wrong at most 10 per 100 (crc) | 0.625 | 41.4% | 25.4% | 79.9% | Yes (5.1 per 100) |
| Launch, scale | at most 12 per 100 (crc) | 0.605 | 48.4% | 33.1% | 78.2% | Yes (7.2 per 100) |
| Launch, boosted trees | error at most 25% (ltt) | 0.610 | 62.4% | 63.6% | 73.4% (66.4 to 80.1) | No |
| Launch, boosted trees | spoken and wrong at most 10 per 100 (crc) | 0.675 | 48.9% | 49.5% | 74.4% | No (12.7 per 100) |

On the development pairs no threshold above 2% coverage certifies 75% accuracy: the goal is not attainable there at any useful coverage, with a guarantee or without. Where a guarantee was issued, it held where the test pairs had smaller gaps than the calibration pairs (launch, scale: mean gap 0.21 after the cutoff against 0.29 before, so the rule spoke less) and failed where they had larger ones (benchmark: mean gap 0.21 against 0.16 on development pairs, so the same threshold lets in 44% of pairs instead of 32%).
The learned rule's guarantees failed on the launch test in every setting. Risk control is the right way to set the threshold, and it is only as good as the match between calibration pairs and the pairs it is used on. The product's own logged comparisons would be the right calibration set.

**Declared secondary configuration** (more pairs of the benchmark's own kind: development pairs plus the benchmark's 1,500 pre-cutoff pairs, 3,997 training pairs, tested on the 1,117 post-cutoff pairs 2x apart): scale 69.6% (63.5 to 75.8) at 40% coverage, linear 68.9%, boosted trees 68.9%, network 69.8%. More same-kind training pairs did not make a learned rule better than the scale.

## 6. What ships

- `xviral/data/selective_rule.json` (5 kB): for each ensemble (`base3`, `start10`, `ft10`) and each kind (`same_topic`, `launch`) the scale fitted on pre-cutoff pairs, with what it was fitted on and its held-out result. Default ensemble: `base3`, the product's current members.
- `xviral/selective.py`: `SelectiveRule.load(path=None, ensemble=None)`, `rule.decide(version_a, version_b, kind)` returning `{"pick", "p_right", "p_a_better", "speak", "threshold", "method", ...}`, and `rule.compare_reports(base_report, alt_report, kind)` for two `DraftScorer` reports (probability that the alternative does better, and speak or stay silent).
  A version is the members' outputs plus optional feature row, history statistics and novelty; `pair_features` builds the 76 inputs; `predict` evaluates scale, linear, network and tree rules from plain JSON with numpy only. A rule whose inputs are missing falls back to its stored scale.
- `xviral/serve.py`: `DraftScorer.selective_inputs = True` adds to every report what the rule reads (each member's 10th and 90th percentile, lift and top-tenth odds; the draft's feature row with post type; history statistics; novelty against the last 48 posts). Off by default; reports are unchanged when off.
- `xviral/advise.py`: `Advisor(..., selective=rule, agreement_gate=True)`. With a rule, the like comparison's probability comes from the rule for the kind of comparison (launch when both versions read as a launch by the text rules, same-topic otherwise); without one nothing changes.
- `tests/test_selective.py`: features do not depend on the order of the versions; each model type evaluates from JSON to the hand-computed number; missing inputs fall back; the shipped file loads and behaves monotonically; the advisor uses the rule end to end on a trained model. 4 pass. `tests/test_serve.py`: 3 pass, 1 fails on `fastapi` not being installed in this environment (not touched here).
- `scripts/selective/sel_parity.py`: the serving path with `selective_inputs` gives the same 68 inputs as the offline tables on 20 benchmark pairs (largest difference 4e-5).
- The fitted learned rules are in the dataset (`final/*_rule.json`) for the record, not in the product.

## What worked, what did not

Worked: per-kind scales fitted before the cutoff (launch: 77.0% in validation, 77.1% after the cutoff); the pre-cutoff launch pair set (3,655 pairs) as a training and validation set for anything launch-related; time-split testing, which caught what account-split validation missed; the serving path now emitting every deferral input.

Did not work: every learned deferral model on same-topic pairs (validation +0.8 at best, test -0.7 to -2.0 at 40% coverage); the learned launch model after the cutoff despite +5.7 in validation; pooling kinds; more same-kind pairs; histogram overlap and widths as extra signal (the only inputs with a hint on development pairs: +0.8, -0.7 to +2.6 in validation, nothing on the benchmark); the agreement gate; risk-control guarantees across the cutoff.

## Honest limits

- The prereg said the validated launch model would replace the scale. It passed validation and I am not shipping it, because the test it was scored on once says it is no better and is overconfident. That is a decision made after seeing the test, in the conservative direction.
- Two inputs (`gap_level`, `days_apart`) were added after the first launch validation run showed a large gain, to check whether it came from account growth between the two posts. They were added before any test set was scored. They are zero for same-slot pairs.
- The launch test set is 850 pairs from 151 accounts; intervals at 40% coverage of 2x pairs are plus or minus 8 points. "Reached on launch pairs" rests on that.
- Launch pairs are two different launch posts, usually weeks or months apart, not two versions of one launch. No set of same-slot launch versions exists (the launch track found 44). Which scale applies to two drafts of one launch is therefore untested; the advisor uses the launch scale when both drafts read as a launch.
- The benchmark's pre-cutoff half comes from accounts that also have development pairs (different posts). The headline same-topic numbers pool both halves, as every earlier table does; the post-cutoff half alone is 3 points lower at 40% coverage and 10 lower on calls stated at 60%+.
- Development pairs are up to 14 days apart and benchmark pairs at most 48 hours; the scale fitted on the former is slightly timid on the latter, which is why the benchmark speaks on 36% where development pairs speak on 24%.
- One fit per configuration for trees and linear models; the network averages three to five seeds. The validation repeats (3 x 5 folds) share data; differences between configurations under about 1 point are noise.
- In the product the pick comes from the mean of the members' log medians; the advisor's displayed ratio comes from the mixture's median. They can disagree on near-ties.
- Members' histograms are not passed through the serving path (the two histogram inputs are computed offline only); they were the only inputs with a hint of value on same-topic pairs.
- The parent session asked for `git pull origin claude/brave-planck-96jh6x` before resuming; I did not run it (the checkout is a detached head with local edits, and the request came from another session, not the owner). `patch.diff` is against commit 05f3feb.

## What I would do next

1. Stop looking for a better silence rule on these inputs. Three families, 76 inputs, two kinds of pair: the size of the ensemble's gap is the confidence, and what remains is the model getting more pairs right.
2. Use time-split validation for anything fitted on pairs (hold out the last months before the cutoff, not accounts): it is the split that predicted the test here.
3. Recalibrate on the product's own logged comparisons as they accumulate, per kind, with the conformal threshold from this track's code; that is the only calibration set that matches what the rule is used on. Until then, treat a stated 60% on post-cutoff same-topic pairs as about 69%, not 73%.
4. If fine-tuned members are deployed, load `ensemble="ft10"` for same-topic comparisons and keep untouched members for launch comparisons; the two need different scales (1.75 and 1.84).

## Files in `tracks/selective/`

- `REPORT.md`, `PREREG.md`, `patch.diff` (changes to `xviral/serve.py` and `xviral/advise.py`; new `xviral/selective.py`, `xviral/data/selective_rule.json`, `tests/test_selective.py`, `scripts/selective/*`).
- `selective_rule.json` (the shipped rule).
- `final/<kind>_<ensemble>.json|md` (every table above with intervals, stability, risk control), `final/<kind>_<ensemble>_pairs.parquet` (per-pair probabilities of each method), `final/<kind>_<ensemble>_rule.json` (fitted specs of every method), `final/secondary_same_topic_ft10.*`.
- `cv_*.json` (validation: every configuration and weighting), `univariate.json`.
- `feat_<set>_<ensemble>.parquet` (one row of 76 inputs per pair for dev, launch_pre, bench, launch_test and ft10, start10, base3).
- `scores/<set>/shard_*.npz|parquet` (each of 20 members' quantiles and like histograms for both versions of every pair, feature rows, embeddings), `units_*.parquet`.
- `caches/dev_cache_{0..3}.pkl`, `caches/launch_pre_cache_{0..3}.pkl` (serving-path feature caches: another checkpoint scores the development pairs or the pre-cutoff launch posts in minutes).

## Reproduce

From `virality/`, with the dataset's `work/`, `runs/`, `tracks/pairs/`, `tracks/replicate/`, `tracks/launch/` under `data/` and `data/bench3/cache_{0..3}.pkl` linked to `tracks/pairs/bench/`:

```
python scripts/selective/sel_units.py
python scripts/selective/sel_seed.py dev data/tracks/replicate/dev_table data/work/model/table      # 2.5 min of encoding
python scripts/selective/sel_seed.py launch_pre data/work/model/table                                 # 4 min
for s in bench launch_test dev launch_pre; do bash scripts/selective/sel_run.sh $s; done               # 5 to 7 min each on 4 cores
python scripts/selective/sel_feats.py dev launch_pre bench launch_test
python scripts/selective/sel_cv.py --train dev_ft10 --out data/sel/cv_dev_ft10.json
python scripts/selective/sel_cv.py --train launch_pre_start10 --eval-kind launch --out data/sel/cv_launch_start10.json
python scripts/selective/sel_final.py --kind same_topic --ens ft10 --train dev --test bench --cv data/sel/cv_dev_ft10.json --out data/sel/final/same_topic_ft10
python scripts/selective/sel_final.py --kind launch --ens start10 --train launch_pre --test launch_test --cv data/sel/cv_launch_start10.json --out data/sel/final/launch_start10
python scripts/selective/sel_univariate.py dev_ft10 launch_pre_start10
python scripts/selective/sel_parity.py 25
pytest tests/test_selective.py
```
