# Content track: what the first line says, concreteness, and novelty against the account's own history

Worker session of 2026-11-16. Code changes are in `patch.diff`; every number below is in the JSON files next to this report.

## Headline

- **The 67% to 57% drop of the earlier pair model is posting order, not data source.** 99% of pairs come from one source in both periods. "Pick the later version" scores 66.9% before the cutoff and 55.1% after, and the pair model was reading order through five columns computed against the other version. With order balanced the drop is about 3 points.
- **No content signal improves the surrogate.** First-line signals are at chance (50 to 52% on 1,660 test pairs). Three groups hold weakly on their own (novelty against the account's history 53.3%, concreteness 53.5%, the account's response to the format 52.9%; 54 to 60% on the pairs where the versions differ) and add nothing in a blend (gains within half a point, intervals include zero).
- **As model inputs the three kept groups leave the benchmark unchanged.** Three retrained seeds average 59.3% on the 2,852 pairs against 59.4% for the two baseline seeds; the ensemble with the new inputs is 59.7% against the baseline's 60.0%, 2x pairs 60.6% against 61.4%, confident share 24% at 68.8% against 26% at 70.2%. The track goal (75 to 80% on 40% of 2x pairs) is not reached and this track does not move toward it.
- **One lead:** blending the surrogate with an order-balanced direct pair model keeps accuracy and raises the confident share on 2x test pairs from 27% to 36% (68.1% and 69.5% correct). Intervals overlap; it needs a second look on more pairs.

## 1. Why the earlier pair model scored 67% before the cutoff and 57% after

It is not the data source. 99% of the pairs come from `community_archive` on both sides of the cutoff (pre-cutoff 7,060 of 7,158 pairs, test 2,025 of 2,030; `nitter_2026` contributes 46 same-source pairs before the cutoff and 4 after), so a within-source transfer test for the second source is not possible: there are not enough pairs to train or to test on. Within `community_archive`, trained before and tested after the cutoff, is what every number in this report is.

The cause is posting order, which the pair model could read and which changed between the periods.

| Pairs with a real gap of 1.25x or more | Pairs | Accounts | Under 1 h apart | Later version wins | Structure model prefers the later version | Structure model correct |
|---|---|---|---|---|---|---|
| Before cutoff, all | 5,791 | 963 | 58% | 66.9% | 76.9% | 67.5% |
| Before cutoff, 2023 | 783 | 159 | 58% | 69.1% | 75.6% | 68.3% |
| Before cutoff, 2024 | 1,861 | 236 | 63% | 69.7% | 80.9% | 69.5% |
| Before cutoff, 2025 | 2,172 | 506 | 61% | 68.0% | 77.0% | 68.2% |
| Before cutoff, 2026 (January to 22 April) | 975 | 455 | 44% | 57.3% | 70.4% | 61.7% |
| Test period | 1,660 | 568 | 44% | 55.1% | 70.9% | 56.5% |
| Before cutoff, under 1 h apart | 3,371 | 487 | 100% | 80.8% | 83.6% | 76.3% |
| Before cutoff, 1 h or more apart | 2,420 | 719 | 0% | 47.6% | 67.7% | 55.2% |
| Test, under 1 h apart | 732 | 276 | 100% | 65.4% | 79.2% | 62.4% |
| Test, 1 h or more apart | 928 | 423 | 0% | 47.0% | 64.3% | 51.8% |
| Before cutoff, the 7 accounts with over 100 pairs each | 1,471 | 7 | 76% | 76.8% | 83.1% | 75.1% |
| Before cutoff, later version won | 3,876 | 654 | 70% | 100% | 83.1% | 83.3% |
| Before cutoff, earlier version won | 1,915 | 596 | 34% | 0% | 64.4% | 35.5% |
| Test, later version won | 915 | 377 | 52% | 100% | 74.9% | 74.9% |
| Test, earlier version won | 745 | 346 | 34% | 0% | 66.0% | 33.9% |

("Structure model": logistic regression on the 46 structure columns and four flags of `scripts/evidence/pair_model.py`, out of fold over accounts before the cutoff, trained on all pre-cutoff pairs for the test period. It reproduces the earlier result: 67.5% and 56.5%.)

What happened:

1. **The pair model is mostly an order detector.** It prefers the later version in 77% of pre-cutoff pairs and is right in 83% of the pairs the later version won and 35% of the pairs the earlier version won, below a coin. Training on both orders with the sign flipped removes order as an explicit input but not as something the inputs reveal. Five of its columns (`ts_gap_vs_own`, `ts_len_vs_own`, `ts_media_vs_own`, `ts_link_vs_own`, `ts_lines_vs_own`) are computed against the account's previous post, and in `pair_model.py` the frame handed to `text_structure` holds only the pair posts, so for the later version "the previous post" is the earlier version: the gap column says directly which one came second. The remaining columns also carry order (a follow-up posted minutes later is typically the longer, fuller version).
2. **"Always pick the later version" scores 66.9% before the cutoff and 55.1% after.** The 12-point drop of that rule is the 11-point drop of the pair model. Before the cutoff 58% of pairs are less than an hour apart and the later one wins 81% of those; in the test period 44% are that close and the later one wins 65%. The change starts in January 2026, before the cutoff (2026 pre-cutoff pairs look like test pairs), so it is a change in who is in the archive and how they post, not something at the cutoff date: the 2023 to 2025 pairs are dominated by few heavy accounts (7 accounts hold 25% of the pairs) that post a quick first take with few or no likes and then a second one (the earlier side has zero likes in 26% of pre-cutoff pairs, 13% of test pairs).
3. **With order taken out, the gap is about 3 points, not 11.** Order-balanced accuracy (mean of the accuracy on later-won and earlier-won pairs) of the same model is 59.4% before the cutoff and 54.4% after. Dropping the five leaking columns and weighting the training pairs so that each version wins half the time within each time-apart stratum, with each account capped at the weight of 20 pairs, gives 60.8% out of fold and 57.1% on the test period (58.0% and 56.4% order balanced). The remainder is ordinary drift.
4. **No pre-cutoff stratum transfers better.** Trained only on pairs an hour or more apart, only on 2026 pairs, only on pairs from July 2025 on, or without the heavy accounts, the structure model stays between 54% and 58% on the test period (table `gap_diagnosis.train_stratum_to_test` in `pair_signal_groups.json`).

The surrogate is scored the way the product is used (both versions as drafts for the same moment), so it cannot see order: it prefers the later version in 56.7% of test pairs, and its order-balanced accuracy equals its plain accuracy (60.0% and 60.6%). Any pair model compared with it has to be held to the same standard; a pair model that reads order is measuring a property of the benchmark, not of the draft. The calibration pairs of the benchmark (pre-cutoff, later version wins 68%) have the same problem when they are used to fit a blend: an unweighted blend puts most of its weight on the order-reading pair model and loses 1 to 3 points on the test period.

Consequences applied to everything below: (a) both versions get the same history, the account's posts strictly before the earlier version; (b) no column is computed against the other version; (c) training and blend fitting use order-balanced, account-capped weights; (d) every result is reported with its order-balanced accuracy in the JSON.

## 2. Signal groups, alone and added to the surrogate

Pairs: `scripts/evidence/tac_serve_eval.find_pairs` (same account, same quoted post or link, different wording, at most 48 hours apart). 5,791 pre-cutoff pairs with a real gap of 1.25x or more from 963 accounts train the models; 1,660 test-period pairs from 568 accounts judge them. Models are antisymmetric (inputs are differences, both orders, no intercept): a logistic regression, and for groups under 200 inputs also gradient-boosted trees; the table shows whichever did better on the time-forward check inside the pre-cutoff period (trained before July 2025, scored on July 2025 to 22 April 2026), which is also where the logistic penalty was chosen. Nothing was chosen on the test period. A pair where a group gives both versions the same value counts as half right; "decided share" is how often the group has a preference at all. The surrogate is the mean score difference of `ts46_s1`, `ts46_s2` and `compare_v2_full` from `bench2`; the blend is a two-input logistic regression fitted on the benchmark's 1,192 calibration pairs (pair-model scores out of fold) and judged on its 1,660 test pairs, where the surrogate alone is 60.6% (2x: 61.5% of 1,117; 3x: 64.7% of 733; 5x: 69.5% of 440). Intervals resample accounts.

| Signal group | Model | Before cutoff, out of fold | Late pre-cutoff, time-forward | Test | Test 95% interval | Test, order balanced | Decided share | Test 2x | 3x | 5x | With surrogate: test | Gain over surrogate (95% interval) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 46 structure columns + 4 flags (the earlier pair model's inputs; 5 columns leak posting order) | logit | 60.0% | 66.0% | 58.4% | 55.3 to 62.0 | 57.5% | 100% | 62.3% | 65.8% | 70.9% | 59.9% | -0.7 (-3.1 to +1.8) |
| the same without the 5 order-leaking columns | logit | 60.8% | 62.1% | 57.1% | 53.9 to 60.8 | 56.4% | 99% | 59.4% | 61.0% | 64.2% | 59.4% | -1.2 (-3.5 to +1.4) |
| text embedding difference (bge-small, 384) | logit | 52.2% | 52.3% | 53.6% | 51.2 to 55.6 | 53.3% | 100% | 53.2% | 53.9% | 58.4% | 61.0% | +0.4 (+0.0 to +0.7) |
| first line: concreteness, class, second person, result (21) | gbdt | 52.0% | 54.7% | 50.0% | 47.6 to 52.8 | 49.6% | 97% | 51.7% | 54.5% | 55.0% | 59.9% | -0.7 (-2.6 to +0.8) |
| first line vs rest: similarity (3) | logit | 53.4% | 55.6% | 51.0% | 48.6 to 53.2 | 50.8% | 100% | 51.6% | 51.6% | 51.8% | 60.8% | +0.2 (-1.1 to +1.5) |
| first-line embedding difference (384) | logit | 50.1% | 51.7% | 51.7% | 49.4 to 54.3 | 51.9% | 96% | 52.1% | 52.6% | 51.5% | 58.2% | -2.4 (-3.8 to -1.2) |
| concreteness of the whole post (8) | logit | 53.8% | 55.1% | 53.5% | 51.5 to 55.9 | 53.1% | 67% | 54.5% | 54.3% | 59.1% | 60.7% | +0.1 (-0.4 to +0.5) |
| wording, after Tan, Lee, Pang (15) | gbdt | 53.4% | 54.5% | 52.7% | 50.1 to 54.8 | 52.6% | 98% | 54.1% | 56.0% | 56.4% | 60.7% | +0.1 (-0.6 to +0.6) |
| own-language model: fit to the account's past words (4) | gbdt | 53.2% | 53.5% | 50.8% | 48.4 to 53.1 | 50.7% | 99% | 51.7% | 53.2% | 54.1% | 61.0% | +0.4 (-0.5 to +1.2) |
| hashed words and word pairs (1,243) | logit | 52.4% | 52.9% | 52.2% | 49.8 to 54.7 | 52.2% | 98% | 53.0% | 55.0% | 56.5% | 60.5% | -0.1 (-0.5 to +0.3) |
| media and link specifics (17) | gbdt | 54.6% | 53.5% | 52.3% | 50.9 to 53.6 | 52.1% | 22% | 53.0% | 52.8% | 54.2% | 60.7% | +0.1 (-0.9 to +1.1) |
| novelty vs own history: similarity to last 50, to best/worst 10%, outcome of nearest posts (8) | logit | 53.2% | 55.0% | 53.3% | 51.3 to 55.4 | 53.2% | 100% | 53.3% | 55.2% | 55.9% | 61.1% | +0.5 (-1.1 to +1.7) |
| the account's own response to the draft's format (7) | logit | 55.0% | 55.5% | 52.9% | 51.3 to 54.7 | 52.6% | 45% | 53.7% | 53.7% | 55.7% | 59.8% | -0.8 (-1.6 to -0.1) |
| novelty + format response + own language (19) | logit | 57.3% | 57.8% | 54.3% | 52.1 to 56.9 | 54.1% | 100% | 55.2% | 56.5% | 59.1% | 60.7% | +0.1 (-0.3 to +0.4) |
| all new scalar groups together (83) | logit | 58.8% | 59.3% | 55.8% | 53.1 to 58.5 | 55.4% | 100% | 56.8% | 59.3% | 61.4% | 59.9% | -0.7 (-1.9 to +0.5) |
| structure41 + all new scalars (128) | logit | 59.9% | 60.8% | 57.9% | 54.7 to 61.4 | 57.6% | 100% | 59.4% | 61.8% | 62.7% | 60.3% | -0.3 (-2.4 to +1.7) |
| structure41 + new scalars + embedding (512) | logit | 57.4% | 57.4% | 55.8% | 52.8 to 58.4 | 55.5% | 100% | 56.3% | 57.4% | 60.5% | 60.2% | -0.4 (-1.7 to +0.8) |

Reading it:

- **First-line signals do not separate the two versions.** Concreteness of the first line, claim versus question versus announcement versus imperative, second-person address, a stated result, the first line's embedding, and its similarity to the rest of the post: 50 to 52% on the test period, intervals covering 50%, and nothing gained on top of the surrogate (the first-line embedding costs 2.4 points). The hypothesis that the first line carries the difference is not supported on these pairs. One caveat: about three quarters of posts in a 20,000-post sample of the corpus are a single line, so "first line" is often "the post".
- **Concreteness of the whole post holds, weakly.** The version with more numbers, versions, time spans, multiples and named things wins 55.5% of the 969 test pairs where the two versions differ in it (order balanced, 52.3 to 58.7); the group as a model scores 53.5% (51.5 to 55.9).
- **Novelty against the account's own history holds, weakly, and is the most stable group**: 53.2% out of fold, 55.0% time-forward, 53.3% on the test period (51.3 to 55.4). What carries it is not novelty as such but resemblance to what worked: the version whose nearest posts in the account's last 50 did better wins 54.4% (52.1 to 56.5, 1,653 pairs), the version closer to the account's best 10% wins 53.7% (51.3 to 56.5). Plain similarity to recent posts, to the nearest post, or to the worst 10% is at 50 to 52%.
- **The account's own response to a format holds where the versions differ in format**: 55.9% (52.4 to 59.0) over 751 test pairs, 60.0% (53.9 to 66.4) for media over 208 pairs. Media specifics repeat what was known (more photos: 62.4%, 233 pairs); video versus photo versus gif, link domain class and quote versus original have too few differing pairs to say anything (the two versions of a same-topic pair nearly always share them).
- **Wording features after Tan, Lee and Pang, the own-language model and bag of words: 51 to 53%**, not distinguishable from 50% on the test period. Their 65.6% came from about 11,000 pairs of one design; here the same kinds of feature do not hold up under a time-forward split.
- **Nothing adds to the surrogate in a score blend.** Every gain is within half a point and every interval contains zero or sits below it. Stacked together (83 new scalars, with or without the structure columns) the new signals reach 56 to 58% alone, still under the surrogate's 60.6%, and the blend does not move. The surrogate already has the embedding, the history and its outcomes; these signals are weaker views of the same information.

Per-signal rates (order balanced, both periods, with pair counts and intervals) are in `univariate.json`.

**Kept** (hold on the test period, same direction in all three splits): novelty against own history, response to the format, concreteness. **Dropped**: every first-line group, wording, own-language model, bag of words, and the media and link specifics (already model inputs through `has_photo`, `has_video`, `has_gif`, `media_count`, `domain_class`, `kind`).

### A side result: a direct pair model does not raise accuracy but widens the confident calls

Blending the surrogate with an order-balanced pair model (trees on the 41 leak-free structure columns plus the 83 new scalars) leaves accuracy where it was but makes more calls confident at the same hit rate. Test-period pairs only, blend weights fitted on calibration pairs only, confidence cross-fitted over accounts as in `compare_runs.py`:

| Scores | Real gap at least | Pairs | Correct (95% interval) | 50-55% band | 55-60% | 60-70% | 70%+ | Confident (60%+) share | Confident pairs | Correct there |
|---|---|---|---|---|---|---|---|---|---|---|
| surrogate (baseline) | 1.25x | 1,660 | 60.6% (58.6 to 62.9) | 58.4% of 795 | 58.8% of 410 | 65.4% of 353 | 68.6% of 102 | 27% | 455 | 66.2% |
| surrogate (baseline) | 2.0x | 1,117 | 61.5% (58.6 to 64.2) | 59.9% of 548 | 57.3% of 262 | 67.5% of 243 | 70.3% of 64 | 27% | 307 | 68.1% |
| surrogate (baseline) | 3.0x | 733 | 64.7% (60.7 to 68.4) | 61.6% of 354 | 62.5% of 160 | 71.2% of 170 | 71.4% of 49 | 30% | 219 | 71.2% |
| surrogate (baseline) | 5.0x | 440 | 69.5% (64.0 to 74.6) | 64.0% of 203 | 69.8% of 96 | 77.5% of 111 | 76.7% of 30 | 32% | 141 | 77.3% |
| surrogate + pair model (41 structure columns + 83 new scalars, trees) | 1.25x | 1,660 | 60.7% (57.8 to 64.0) | 53.9% of 636 | 62.3% of 448 | 65.8% of 436 | 70.0% of 140 | 35% | 576 | 66.8% |
| surrogate + pair model (41 structure columns + 83 new scalars, trees) | 2.0x | 1,117 | 62.8% (59.0 to 67.0) | 55.2% of 420 | 64.6% of 291 | 68.2% of 311 | 73.7% of 95 | 36% | 406 | 69.5% |
| surrogate + pair model (41 structure columns + 83 new scalars, trees) | 3.0x | 733 | 67.1% (61.4 to 72.7) | 55.7% of 262 | 71.6% of 183 | 74.0% of 215 | 76.7% of 73 | 39% | 288 | 74.7% |
| surrogate + pair model (41 structure columns + 83 new scalars, trees) | 5.0x | 440 | 71.8% (64.2 to 78.0) | 57.4% of 155 | 78.4% of 97 | 79.1% of 139 | 83.7% of 49 | 43% | 188 | 80.3% |
| surrogate + pair model (41 structure columns, trees) | 1.25x | 1,660 | 59.2% (56.3 to 62.7) | 53.4% of 590 | 58.6% of 473 | 63.8% of 492 | 72.4% of 105 | 36% | 597 | 65.3% |
| surrogate + pair model (41 structure columns, trees) | 2.0x | 1,117 | 61.8% (58.0 to 66.0) | 56.5% of 386 | 58.4% of 315 | 67.3% of 346 | 78.6% of 70 | 37% | 416 | 69.2% |
| surrogate + pair model (41 structure columns, trees) | 3.0x | 733 | 65.3% (60.0 to 70.4) | 56.2% of 242 | 61.5% of 195 | 74.0% of 242 | 81.5% of 54 | 40% | 296 | 75.3% |
| surrogate + pair model (41 structure columns, trees) | 5.0x | 440 | 70.2% (64.1 to 75.3) | 58.7% of 138 | 63.1% of 103 | 81.3% of 155 | 84.1% of 44 | 45% | 199 | 81.9% |
| surrogate + novelty group | 1.25x | 1,660 | 61.1% (58.7 to 63.6) | 58.5% of 763 | 60.3% of 416 | 65.8% of 360 | 66.1% of 121 | 29% | 481 | 65.9% |
| surrogate + novelty group | 2.0x | 1,117 | 61.1% (58.0 to 63.9) | 57.1% of 531 | 61.9% of 260 | 66.9% of 242 | 67.9% of 84 | 29% | 326 | 67.2% |
| surrogate + novelty group | 3.0x | 733 | 65.1% (61.4 to 68.9) | 59.5% of 341 | 67.7% of 167 | 72.2% of 162 | 69.8% of 63 | 31% | 225 | 71.6% |
| surrogate + novelty group | 5.0x | 440 | 69.3% (64.4 to 74.1) | 62.6% of 206 | 69.6% of 92 | 82.2% of 101 | 70.7% of 41 | 32% | 142 | 78.9% |
| surrogate + media and link group | 1.25x | 1,660 | 60.8% (58.5 to 63.4) | 57.7% of 764 | 58.7% of 414 | 67.5% of 357 | 67.2% of 125 | 29% | 482 | 67.4% |
| surrogate + media and link group | 2.0x | 1,117 | 62.0% (59.1 to 64.9) | 59.6% of 530 | 55.8% of 260 | 71.3% of 244 | 69.9% of 83 | 29% | 327 | 70.9% |
| surrogate + media and link group | 3.0x | 733 | 64.7% (60.8 to 68.4) | 61.1% of 342 | 58.9% of 158 | 73.7% of 175 | 74.1% of 58 | 32% | 233 | 73.8% |
| surrogate + media and link group | 5.0x | 440 | 69.1% (63.6 to 74.1) | 62.3% of 199 | 64.8% of 88 | 81.2% of 112 | 78.0% of 41 | 35% | 153 | 80.4% |

The intervals on accuracy overlap, so this is a lead, not a result: on 2x pairs the confident share goes from 27% to 36% at 68 to 70% correct. It is cheap (no retraining) and works at score level.

## 3. The kept signals as model inputs

Added to `NUM_SPEC` (23 columns, `xviral/features/content_signals.MODEL_COLUMNS`): 8 concreteness columns, 8 novelty columns (similarity to the account's last 50 posts, to its best and worst 10% by likes, and the outcome of its nearest posts) and 7 format-response columns. `scripts/add_content_signals.py` adds them to the built table from the table's own embeddings (465,675 rows; history columns are missing for 16% of rows with fewer than 10 earlier posts and get a missing indicator); `xviral/serve.py` computes the same columns for drafts, after embedding, with alternative drafts not seeing each other. Three models were trained with `scripts/train_one.py --features ts46cs --preset small --steps 4000` (seeds 1, 2, 3; 166 numeric inputs against 128) and scored on the same 3,530 benchmark pairs as `bench2`. The changed serving path reproduces the stored `ts46_s1` scores (largest difference 1e-5).

**On the pair benchmark the new inputs change nothing.** Single models, real gap 1.25x or more, all 2,852 pairs: `ts46` seeds average 59.4% and `ts46cs` seeds average 59.3%; on the 1,660 test-period pairs 59.8% and 60.3%. On 2x pairs: 60.4% and 60.5% (all), 60.2% and 60.9% (test). Seeds of one configuration differ by 1 to 2 points, so these are the same. No checkpoint beats the baseline; the three are uploaded for completeness.

All 3,530 benchmark pairs (calibration and test parts, as in the track baseline), confidence cross-fitted over accounts:

| Scores | Real gap at least | Pairs | Correct (95% interval) | 50-55% band | 55-60% | 60-70% | 70%+ | Confident (60%+) share | Confident pairs | Correct there |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 1.25x | 2,852 | 60.0% (57.9 to 62.0) | 55.1% of 1412 | 61.6% of 662 | 66.3% of 564 | 70.1% of 214 | 27% | 778 | 67.4% |
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 2.0x | 2,030 | 61.4% (58.3 to 64.2) | 56.8% of 1065 | 62.0% of 442 | 69.9% of 375 | 70.9% of 148 | 26% | 523 | 70.2% |
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 3.0x | 1,377 | 63.3% (60.2 to 66.6) | 56.7% of 723 | 65.7% of 283 | 74.4% of 262 | 74.3% of 109 | 27% | 371 | 74.4% |
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 5.0x | 880 | 66.5% (62.3 to 71.0) | 57.6% of 462 | 69.4% of 173 | 82.6% of 167 | 78.2% of 78 | 28% | 245 | 81.2% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 1.25x | 2,852 | 59.7% (57.7 to 61.8) | 54.7% of 1455 | 64.0% of 650 | 64.3% of 549 | 69.2% of 198 | 26% | 747 | 65.6% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 2.0x | 2,030 | 60.6% (57.7 to 63.4) | 55.4% of 1096 | 64.5% of 437 | 67.9% of 352 | 71.0% of 145 | 24% | 497 | 68.8% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 3.0x | 1,377 | 63.0% (59.8 to 66.3) | 56.4% of 738 | 68.6% of 293 | 71.4% of 245 | 74.3% of 101 | 25% | 346 | 72.3% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 5.0x | 880 | 65.7% (61.6 to 70.2) | 57.4% of 477 | 70.2% of 181 | 79.5% of 151 | 80.3% of 71 | 25% | 222 | 79.7% |
| ts46_s1 + ts46_s2 | 1.25x | 2,852 | 59.4% (57.2 to 61.6) | 54.5% of 1445 | 60.8% of 643 | 66.2% of 562 | 71.3% of 202 | 27% | 764 | 67.5% |
| ts46_s1 + ts46_s2 | 2.0x | 2,030 | 60.4% (57.2 to 63.2) | 55.7% of 1090 | 61.4% of 422 | 68.8% of 375 | 72.0% of 143 | 26% | 518 | 69.7% |
| ts46_s1 + ts46_s2 | 3.0x | 1,377 | 62.7% (59.5 to 65.9) | 56.4% of 750 | 65.6% of 259 | 73.1% of 264 | 74.0% of 104 | 27% | 368 | 73.4% |
| ts46_s1 + ts46_s2 | 5.0x | 880 | 65.9% (62.0 to 70.2) | 57.4% of 477 | 68.8% of 160 | 80.8% of 172 | 80.3% of 71 | 28% | 243 | 80.7% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 1.25x | 2,852 | 59.7% (57.5 to 62.0) | 53.8% of 1427 | 65.5% of 684 | 64.1% of 543 | 70.2% of 198 | 26% | 741 | 65.7% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 2.0x | 2,030 | 60.3% (57.2 to 63.3) | 54.0% of 1073 | 65.7% of 463 | 68.5% of 352 | 71.1% of 142 | 24% | 494 | 69.2% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 3.0x | 1,377 | 62.7% (59.6 to 66.1) | 54.9% of 723 | 69.4% of 304 | 71.5% of 249 | 77.2% of 101 | 25% | 350 | 73.1% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 5.0x | 880 | 65.9% (61.8 to 70.3) | 56.5% of 471 | 73.5% of 181 | 78.2% of 156 | 81.9% of 72 | 26% | 228 | 79.4% |

Test-period pairs only:

| Scores | Real gap at least | Pairs | Correct (95% interval) | 50-55% band | 55-60% | 60-70% | 70%+ | Confident (60%+) share | Confident pairs | Correct there |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 1.25x | 1,660 | 60.6% (58.6 to 62.9) | 58.4% of 795 | 58.8% of 410 | 65.4% of 353 | 68.6% of 102 | 27% | 455 | 66.2% |
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 2.0x | 1,117 | 61.5% (58.6 to 64.2) | 59.9% of 548 | 57.3% of 262 | 67.5% of 243 | 70.3% of 64 | 27% | 307 | 68.1% |
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 3.0x | 733 | 64.7% (60.7 to 68.4) | 61.6% of 354 | 62.5% of 160 | 71.2% of 170 | 71.4% of 49 | 30% | 219 | 71.2% |
| baseline: ts46_s1 + ts46_s2 + compare_v2_full | 5.0x | 440 | 69.5% (64.0 to 74.6) | 64.0% of 203 | 69.8% of 96 | 77.5% of 111 | 76.7% of 30 | 32% | 141 | 77.3% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 1.25x | 1,660 | 60.8% (58.7 to 63.2) | 57.3% of 812 | 64.3% of 398 | 63.1% of 344 | 67.0% of 106 | 27% | 450 | 64.0% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 2.0x | 1,117 | 61.4% (58.6 to 63.9) | 58.9% of 565 | 60.9% of 253 | 66.1% of 230 | 68.1% of 69 | 27% | 299 | 66.6% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 3.0x | 733 | 64.9% (61.1 to 68.3) | 61.8% of 359 | 65.1% of 169 | 71.0% of 155 | 68.0% of 50 | 28% | 205 | 70.2% |
| new inputs: ts46cs_s1 + ts46cs_s2 + compare_v2_full | 5.0x | 440 | 68.4% (63.2 to 73.7) | 63.4% of 205 | 66.7% of 105 | 78.8% of 99 | 74.2% of 31 | 30% | 130 | 77.7% |
| ts46_s1 + ts46_s2 | 1.25x | 1,660 | 60.1% (57.7 to 62.6) | 56.3% of 824 | 61.2% of 394 | 65.2% of 336 | 68.9% of 106 | 27% | 442 | 66.1% |
| ts46_s1 + ts46_s2 | 2.0x | 1,117 | 60.7% (57.5 to 63.7) | 57.5% of 563 | 58.8% of 257 | 68.4% of 228 | 68.1% of 69 | 27% | 297 | 68.4% |
| ts46_s1 + ts46_s2 | 3.0x | 733 | 64.3% (60.1 to 68.5) | 59.9% of 367 | 64.2% of 151 | 71.6% of 162 | 71.7% of 53 | 29% | 215 | 71.6% |
| ts46_s1 + ts46_s2 | 5.0x | 440 | 68.9% (63.1 to 74.4) | 62.4% of 213 | 71.3% of 87 | 76.9% of 108 | 78.1% of 32 | 32% | 140 | 77.1% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 1.25x | 1,660 | 60.8% (58.4 to 63.3) | 56.1% of 776 | 64.9% of 424 | 63.5% of 351 | 69.7% of 109 | 28% | 460 | 65.0% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 2.0x | 1,117 | 60.7% (58.0 to 63.9) | 56.0% of 539 | 62.2% of 270 | 66.8% of 232 | 69.7% of 76 | 28% | 308 | 67.5% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 3.0x | 733 | 64.1% (60.2 to 67.9) | 59.6% of 349 | 65.1% of 172 | 70.7% of 157 | 70.9% of 55 | 29% | 212 | 70.8% |
| ts46cs_s1 + ts46cs_s2 + ts46cs_s3 | 5.0x | 440 | 67.7% (62.7 to 72.8) | 61.1% of 208 | 67.4% of 95 | 78.4% of 102 | 77.1% of 35 | 31% | 137 | 78.1% |

Single models, all pairs:

| Scores | Real gap at least | Pairs | Correct (95% interval) | 50-55% band | 55-60% | 60-70% | 70%+ | Confident (60%+) share | Confident pairs | Correct there |
|---|---|---|---|---|---|---|---|---|---|---|
| ts46_s1 | 1.25x | 2,852 | 59.9% (57.4 to 62.1) | 55.8% of 1454 | 60.7% of 642 | 64.3% of 554 | 74.8% of 202 | 27% | 756 | 67.1% |
| ts46_s1 | 2.0x | 2,030 | 60.8% (57.7 to 63.8) | 55.9% of 1097 | 61.9% of 425 | 68.6% of 363 | 75.9% of 145 | 25% | 508 | 70.7% |
| ts46_s1 | 3.0x | 1,377 | 62.5% (58.9 to 66.1) | 55.0% of 760 | 68.4% of 256 | 72.1% of 251 | 78.2% of 110 | 26% | 361 | 74.0% |
| ts46_s1 | 5.0x | 880 | 65.2% (60.8 to 70.2) | 55.7% of 485 | 71.3% of 157 | 78.8% of 160 | 84.6% of 78 | 27% | 238 | 80.7% |
| ts46_s2 | 1.25x | 2,852 | 58.9% (57.0 to 61.2) | 53.3% of 1466 | 63.2% of 666 | 65.6% of 552 | 69.6% of 168 | 25% | 720 | 66.5% |
| ts46_s2 | 2.0x | 2,030 | 59.9% (57.1 to 62.4) | 54.8% of 1099 | 64.1% of 443 | 66.2% of 370 | 72.0% of 118 | 24% | 488 | 67.6% |
| ts46_s2 | 3.0x | 1,377 | 61.9% (58.5 to 65.1) | 55.4% of 745 | 66.3% of 291 | 71.1% of 263 | 75.6% of 78 | 25% | 341 | 72.1% |
| ts46_s2 | 5.0x | 880 | 64.8% (61.1 to 68.7) | 55.8% of 477 | 71.8% of 174 | 77.6% of 174 | 80.0% of 55 | 26% | 229 | 78.2% |
| ts46cs_s1 | 1.25x | 2,852 | 59.3% (57.3 to 61.4) | 55.1% of 1531 | 62.9% of 671 | 64.1% of 476 | 70.1% of 174 | 23% | 650 | 65.7% |
| ts46cs_s1 | 2.0x | 2,030 | 60.2% (57.4 to 62.9) | 55.8% of 1158 | 62.4% of 444 | 69.0% of 306 | 72.1% of 122 | 21% | 428 | 69.9% |
| ts46cs_s1 | 3.0x | 1,377 | 63.5% (59.9 to 66.7) | 57.8% of 780 | 67.8% of 298 | 73.3% of 210 | 75.3% of 89 | 22% | 299 | 73.9% |
| ts46cs_s1 | 5.0x | 880 | 65.6% (61.2 to 69.5) | 58.5% of 504 | 69.4% of 186 | 79.4% of 126 | 82.8% of 64 | 22% | 190 | 80.5% |
| ts46cs_s2 | 1.25x | 2,852 | 58.8% (56.5 to 60.9) | 54.0% of 1556 | 63.5% of 649 | 65.0% of 505 | 67.6% of 142 | 23% | 647 | 65.5% |
| ts46cs_s2 | 2.0x | 2,030 | 59.8% (56.7 to 62.2) | 55.0% of 1171 | 64.8% of 420 | 66.2% of 337 | 72.5% of 102 | 22% | 439 | 67.7% |
| ts46cs_s2 | 3.0x | 1,377 | 61.7% (58.7 to 64.9) | 55.8% of 801 | 68.1% of 279 | 70.3% of 222 | 76.0% of 75 | 22% | 297 | 71.7% |
| ts46cs_s2 | 5.0x | 880 | 64.0% (60.3 to 67.6) | 55.4% of 513 | 74.0% of 177 | 77.0% of 139 | 80.4% of 51 | 22% | 190 | 77.9% |
| ts46cs_s3 | 1.25x | 2,852 | 59.9% (57.6 to 62.2) | 54.9% of 1394 | 61.6% of 656 | 66.2% of 606 | 69.9% of 196 | 28% | 802 | 67.1% |
| ts46cs_s3 | 2.0x | 2,030 | 61.4% (58.4 to 64.2) | 56.9% of 1042 | 61.9% of 441 | 69.5% of 406 | 70.2% of 141 | 27% | 547 | 69.7% |
| ts46cs_s3 | 3.0x | 1,377 | 62.6% (59.3 to 66.0) | 57.1% of 706 | 62.6% of 294 | 73.0% of 274 | 72.8% of 103 | 27% | 377 | 72.9% |
| ts46cs_s3 | 5.0x | 880 | 65.1% (61.5 to 68.7) | 57.4% of 460 | 66.1% of 180 | 81.0% of 168 | 75.0% of 72 | 27% | 240 | 79.2% |

On the models' own held-out split (audience accounts, posts after the cutoff, all post pairs of one account rather than same-topic pairs) the new inputs give a small, consistent gain on likes and a mixed one on views:

| Model | Numeric inputs | Likes: same-author pairwise | Likes: within-author rank correlation | Views: same-author pairwise | Views: within-author rank correlation |
|---|---|---|---|---|---|
| ts46_s1 | 128 | 59.8% | 0.296 | 62.3% | 0.274 |
| ts46_s2 | 128 | 60.1% | 0.299 | 61.8% | 0.247 |
| ts46cs_s1 | 166 | 60.3% | 0.310 | 63.7% | 0.282 |
| ts46cs_s2 | 166 | 60.2% | 0.308 | 61.8% | 0.268 |
| ts46cs_s3 | 166 | 60.6% | 0.311 | 63.2% | 0.275 |

Within-author rank correlation on likes rises from 0.296 and 0.299 to 0.308 to 0.311 in all three seeds. So the signals carry something about which of an account's posts does better in general (mostly topic: a post near the account's best posts), and nothing the model can use to separate two versions of one message, which is what the benchmark asks.

## 4. What worked, what did not

Worked:
- The gap diagnosis. It explains the earlier negative result completely and gives a rule for any future pair training: balance order, cap accounts, give both versions the same history, and never compute a column against the other version.
- `gap_table.py` reproduces the track baseline (60.0 / 61.4 / 63.3 / 66.5% on 2,852 / 2,030 / 1,377 / 880 pairs; confident share 27%, 67.4% correct) and reports any score set in the same form.
- Three weak but stable signals (resemblance to the account's best posts, response to the format, concreteness).

Did not work:
- First-line signals of every kind tested.
- Literature wording features, an own-language model and bag of words under a time-forward split.
- Adding any content signal group to the surrogate at score level.

## 5. Honest limits

- One test period, 1,660 pairs with a clear gap from 568 accounts; a 2-point difference between two models is inside the noise (the two baseline seeds `ts46_s1` and `ts46_s2` differ by 1.8 points on the test pairs). Groups were kept because they held on the test period, as the track asked, so their test numbers are selected; the time-forward check inside the pre-cutoff period gives the same ranking and is not affected by that.
- The pair experiments use bge-small embeddings of the raw text and the last 50 posts including recent ones; the model inputs use the table's embeddings of the rendered text (author line, text, quoted text) and use a history post's likes only once it is a day old, to match training. On same-quote pairs the rendered text shares the quoted post, which dilutes the difference between the versions.
- History windows come from the 4.48M-post corpus, so an account's "last 50 posts" are the last 50 that survived the corpus filters.
- Same-topic pairs are 90% quote posts of the same post; signals about links and media types are untestable there.
- Likes only. Views are missing on these pairs.
- The second data source could not be tested for transfer (50 pairs in total).
- Lexicon-based features (sentiment, hedges, named things by capitalisation) are crude; a tagger or an LLM labeller was out of scope (no paid API).

## 6. Files

- `REPORT.md`: this file.
- `pair_signal_groups.json`, `pair_signal_groups.log`: every group, model and weighting; out-of-fold, time-forward and test accuracy by gap; order-balanced accuracy; the blend with the surrogate; the gap diagnosis.
- `univariate.json`: per-signal "version with more of it wins" rates.
- `pair_scores.npz`: per-pair scores of every group model and blend on the benchmark pairs.
- `table_baseline.json`, `table_blends_test.json`, `table_retrain_*.json`: the gap-by-confidence tables.
- `bench3/<model>/shard_*.parquet`: per-pair benchmark scores of the retrained models (same 3,530 pairs as `bench2`).
- `runs/<model>/best.pt`, `metrics.json`: retrained checkpoints (`ts46cs_s1`, `ts46cs_s2`, `ts46cs_s3`); none beats the baseline.
- `patch.diff`: `xviral/features/content_signals.py` (signals), `scripts/evidence/content_signals_eval.py` (pair experiments), `scripts/evidence/gap_table.py` (the table), `scripts/add_content_signals.py` (adds the inputs to a built table), and the wiring in `xviral/model/features.py`, `xviral/serve.py`, `xviral/pipeline.py`, `scripts/train_one.py`; `tac_serve_eval.py` drops the unused part of the corpus after choosing its pairs (4.5 GB less per process).
