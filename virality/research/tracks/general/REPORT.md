# Track "general": a test bed for generalisation, broader training data, and an honest forecast level

2026-11-19 (started 2026-11-17; the session stopped on a usage limit and resumed). Everything is in the dataset under `tracks/general/`.
Code is in `patch.diff` (nothing committed). No OpenAI calls were made: 0 tokens, 0 dollars.

## Headline

- **Test bed built first, and frozen.** 924 accounts that no baseline model trained on, drawn with their own random seed before any
  training data was chosen: 454 full-timeline accounts from twitter100m (about 90 in each of five follower tiers), 434 accounts of the
  2026 view corpus (likes and views), 36 brand accounts (11 that no model at all has seen), 249 accounts under 5,000 followers across
  the three. 50,517 scored posts, 368,970 ordinary like pairs and 124,669 view pairs.
- **Baseline on it.** The current three-model ensemble ranks 59.7% of test-bed like pairs correctly (60.7% on full timelines, 58.3% on
  2026 accounts, 53.7% on brands) and 61.7% of view pairs. Its forecast level is wrong in a structured way, not by one factor:
  1.41x too high on brand timelines, 1.19x too high under 1,000 followers, 0.71x (too low) at 100,000 followers and over, and only
  78% of full-timeline accounts land inside 1.5x. The two baseline seeds disagree with each other on level (0.87x and 1.13x on twitter100m).
- **Broader data fixes that, and nothing overfits.** Training on 824,056 rows of 12,307 accounts (the existing rows, the expand track's
  rows, and 4,945 new accounts sampled across sources and follower sizes at 64 posts each), with the data source and the sampling
  regime as inputs:

| | Baseline (3 models) | Broad (2 seeds) | Difference (95% interval, accounts resampled) |
|---|---|---|---|
| Test bed, like pairs (368,970) | 59.7% | 63.3% | +3.6 (+3.1 to +4.1) |
| Full timelines, like pairs (259,245) | 60.4% | 65.0% | +4.6 (+4.0 to +5.3) |
| 2026 view corpus, like pairs (109,725) | 58.3% | 59.4% | +1.1 (+0.7 to +1.6) |
| 2026 view corpus, view pairs (112,742) | 62.4% | 62.8% | +0.4 (0.0 to +0.9) |
| Brand accounts, like pairs (12,035; 36 accounts) | 53.7% | 58.2% | +4.5 (+1.8 to +7.9) |
| Small accounts under 5k, like pairs (48,091) | 59.3% | 62.1% | +2.8 (+1.6 to +4.0) |
| Tech audience, ordinary test like pairs (683,152) | 60.2% | 60.1% | -0.2 (-0.5 to +0.1) |
| Tech audience, ordinary test view pairs (24,481) | 62.8% | 62.0% | -0.8 (-2.1 to +1.3) |
| 3,530-pair benchmark, pairs 1.25x apart (2,851), against the two-seed control | 59.5% | 59.2% | -0.2 (-2.0 to +1.5) |
| Benchmark, confident calls on 2x pairs: share, correct | 25%, 69.6% | 24%, 70.3% | |
| 850 launch pairs, against the two-seed control | 63.4% | 63.8% | +0.4 (-3.8 to +4.0) |
| Level on full timelines, scored as a customer (source unknown): accounts inside 1.5x | 78% | 91% | |
| Level on brand timelines: median account | 1.41x | 1.10x | |
| Level at 100k followers and over: median account | 0.71x | 0.99x | |

- **So: the gain is generalisation, not tech-audience accuracy.** On accounts unlike the training set the broad model is 3 to 5 points
  better and its level is honest. On the tech audience (ordinary test pairs, the benchmark, the launch pairs) it is level with the
  baseline inside every interval: not worse, not better. That matches every earlier track.
- **Fine-tuning on the tech audience afterwards adds nothing** (tech test like pairs 60.3% against 60.1%; benchmark 58.8% against 59.2%;
  test bed 63.2% against 63.3%) and makes the full-timeline level slightly worse (1.08x against 1.06x). Not recommended.
- **Overfitting.** The broad runs were still improving on every held-out curve at their last step (train like NLL 2.645, new-source
  validation 2.717, test bed 2.646 at step 6,000; no gap opening). The control trained on the existing rows alone does overfit to its
  own distribution: its loss on new-source validation accounts turns up after step 2,500 (2.803 to 2.817) while its own validation is flat.
- **Deliverable for new customers:** the two broad checkpoints (`runs/broad_s1`, `runs/broad_s2`), served with the regime
  "full timeline" and the source "unknown". For the models already in production, `eval/bed_level_correction.json` holds a
  three-number correction fitted on validation accounts; it repairs the follower-size error of the baseline (88% of twitter100m accounts
  inside 1.5x, from 79%) but **not** the brand error (still 1.44x), so the correction is a stopgap and the model is the fix.

## 1. The test bed

Code: `scripts/general/account_stats.py`, `select_accounts.py`, `build_rows.py`, `build_table.py`. Files: `rows/accounts.parquet`
(every sampled account with its role), `rows/new_rows.parquet`, `table/row_info.parquet` (`g_role`, `bed_eval`, `g_tier`, `seen_by_expand`).

**Who is eligible.** An account with at least 30 English originals or quotes in a source's window that is not in
`work/dataset/rows.parquet`, not in the expand track's rows, and not an account of a benchmark calibration pair or a development pair
(495 handles blocked). 13,710 twitter100m accounts, 14,222 accounts of the 2026 view corpus and 8 brand accounts qualify.

**How it was drawn.** With its own random generator (seed 20261117), before validation and training accounts, so the bed does not depend
on how much data is added later:

| Group | Accounts scored | Scored posts | Outcome | Notes |
|---|---|---|---|---|
| twitter100m, full timelines | 454 (92, 96, 92, 88, 86 by tier: under 1k, 1k to 5k, 5k to 10k, 10k to 100k, 100k and over) | 30,308 | likes | 100 drawn per tier, 46 lost to language, volume and hygiene filters; 2020 to April 2023 |
| 2026 view corpus | 434 | 17,083 | likes and views | every eligible account with a known follower count under 10k (71; the count is twitter100m's 2023 reading) plus 420 at random; no follower counts otherwise |
| Brand accounts, full timelines | 36 | 3,126 | likes and views | 11 no model has seen (4 never built before, 7 validation accounts of the expand track) and 25 that the expand models trained on and every model here holds out |
| Small accounts under 5k followers | 249 (188 twitter100m, 54 view corpus, 7 brands) | 14,003 | likes; views for 61 | a slice across the three groups |

Rows are built by the pipeline's own stage functions (the expand track's code, imported): validity, kind, maturity, recency, language,
volume, author stage, hygiene (237 accounts dropped), near-duplicates, features, labels. A bed account keeps its most recent 120 posts;
the posts that are scored (`bed_eval`) are its last 100 minus the first 48 the table holds, so every scored post has the account's
history in front of it, as a customer's draft would. Each post is scored alone with its own history (`model.evaluate.predict_rows`).

**Measures.**
- *Ordinary same-account pairs*: two posts of one account within 7 days whose outcomes differ by 0.1 in log10; share ranked correctly,
  likes and views. Intervals and differences resample accounts (2,000 draws), paired for differences.
- *Level*: per account, log10 of the median predicted count minus log10 of the median actual count over its scored posts (accounts
  with 10 or more). Reported as the median account's factor (1.00x is honest), the median absolute factor, and the share of accounts
  inside 1.5x.

**The 25 brand accounts the expand models have seen** are in the "brand accounts" row; for the baseline and for every model trained
here they are unseen. The row "brands no model has seen" (11 accounts) is clean for all models.

### Baseline table: the current models on the test bed

Baseline is the README's three-model ensemble (`ts46_s1`, `ts46_s2`, `compare_v2/full`); Expanded and Expanded + sampling input are the
expand track's two-seed pairs. Cells: accuracy (interval), or accuracy (difference to the baseline, interval).

#### Ordinary same-account pairs, likes

| Slice | Accounts | Pairs | Baseline (3 models) | Expanded (2) | Expanded + sampling input (2) |
|---|---|---|---|---|---|
| twitter100m full timelines (likes, to April 2023) | 451 | 247,210 | 60.7% (59.6% to 61.7%) | 64.6% (+3.9, +3.3 to +4.5) | 64.9% (+4.2, +3.6 to +4.8) |
|   under 1k followers | 89 | 14,497 | 57.2% (53.5% to 61.1%) | 59.4% (+2.2, +0.1 to +4.4) | 59.9% (+2.7, +0.7 to +4.8) |
|   1k to 5k | 96 | 24,490 | 61.4% (58.8% to 64.3%) | 64.3% (+2.9, +1.5 to +4.3) | 64.6% (+3.2, +1.7 to +4.7) |
|   5k to 10k | 92 | 45,957 | 59.9% (57.0% to 62.8%) | 63.2% (+3.3, +2.3 to +4.3) | 63.5% (+3.6, +2.4 to +4.6) |
|   10k to 100k | 88 | 73,157 | 60.0% (58.0% to 61.9%) | 65.2% (+5.2, +4.0 to +6.5) | 65.4% (+5.4, +4.3 to +6.6) |
|   100k and over | 86 | 89,109 | 62.0% (60.3% to 63.7%) | 65.7% (+3.7, +2.8 to +4.6) | 66.0% (+4.0, +3.0 to +4.9) |
| 2026 view corpus (likes and views) | 432 | 109,725 | 58.3% (57.1% to 59.5%) | 58.5% (+0.3, -0.2 to +0.7) | 58.7% (+0.4, -0.1 to +0.9) |
| brand accounts, full timelines | 36 | 12,035 | 53.7% (51.5% to 56.1%) | 59.0% (+5.3, +2.2 to +9.0) | 58.8% (+5.1, +2.1 to +8.5) |
|   brands no model has seen | 11 | 4,867 | 51.9% (48.4% to 55.2%) | 58.8% (+6.9, +0.6 to +14.4) | 58.1% (+6.2, +0.5 to +12.4) |
| small accounts, under 5k followers (all sources) | 246 | 48,091 | 59.3% (57.3% to 61.2%) | 61.5% (+2.2, +1.1 to +3.2) | 61.9% (+2.6, +1.6 to +3.7) |
|   small, 2026 view corpus | 54 | 7,932 | 56.1% (53.2% to 59.1%) | 56.7% (+0.7, -1.0 to +2.7) | 57.6% (+1.5, -0.4 to +4.1) |
| all full-timeline accounts (twitter100m and brands) | 487 | 259,245 | 60.4% (59.3% to 61.5%) | 64.3% (+4.0, +3.4 to +4.5) | 64.6% (+4.2, +3.7 to +4.8) |
| whole test bed | 918 | 368,970 | 59.7% (58.9% to 60.5%) | 62.6% (+2.9, +2.4 to +3.3) | 62.8% (+3.1, +2.6 to +3.6) |
| existing audience test rows | 2,846 | 683,152 | 60.2% (59.0% to 61.3%) | 60.0% (-0.3, -0.6 to +0.0) | 60.2% (-0.1, -0.4 to +0.2) |

#### Ordinary same-account pairs, views

| Slice | Accounts | Pairs | Baseline (3 models) | Expanded (2) | Expanded + sampling input (2) |
|---|---|---|---|---|---|
| 2026 view corpus (likes and views) | 433 | 112,742 | 62.4% (61.3% to 63.6%) | 62.1% (-0.3, -0.8 to +0.2) | 62.2% (-0.2, -0.7 to +0.3) |
| brand accounts, full timelines | 36 | 11,927 | 55.1% (52.2% to 57.8%) | 61.5% (+6.4, +2.8 to +10.5) | 61.4% (+6.3, +2.7 to +10.1) |
|   brands no model has seen | 11 | 4,669 | 53.6% (48.5% to 58.3%) | 61.7% (+8.1, -0.4 to +16.6) | 61.0% (+7.4, -0.4 to +15.7) |
| small accounts, under 5k followers (all sources) | 61 | 10,985 | 62.4% (60.6% to 64.3%) | 62.5% (+0.1, -1.3 to +1.3) | 62.5% (+0.1, -1.0 to +1.3) |
|   small, 2026 view corpus | 54 | 9,235 | 63.3% (61.6% to 65.3%) | 63.1% (-0.2, -1.5 to +1.0) | 63.0% (-0.2, -1.3 to +0.8) |
| all full-timeline accounts (twitter100m and brands) | 36 | 11,927 | 55.1% (52.3% to 58.0%) | 61.5% (+6.4, +2.7 to +10.7) | 61.4% (+6.3, +2.9 to +10.1) |
| whole test bed | 468 | 124,669 | 61.7% (60.6% to 62.8%) | 62.0% (+0.3, -0.3 to +1.0) | 62.1% (+0.4, -0.2 to +1.1) |
| existing audience test rows | 851 | 24,481 | 62.8% (60.1% to 64.7%) | 62.2% (-0.6, -1.4 to +0.7) | 62.5% (-0.3, -1.4 to +1.5) |


Level of the current models is in section 4, next to the new ones.

What the baseline table says: the expand track's finding holds on a far larger sample (twitter100m +3.9 to +4.2 points on 451 accounts,
where it had 134), the baseline is at 53.7% on brand accounts, and the 2026 view corpus was not helped by the expand data (+0.3).

## 2. Broader training data

**The sample.** Drawn after the bed, with a second generator. 600 twitter100m accounts per follower tier and 2,500 accounts of the 2026
view corpus for training; 60 per tier and 300 for validation; 10 brand accounts moved from the expand track's training split to
validation. Posts are subsampled, not accounts: the most recent 64 posts of a training account, 100 of a validation account. Training
and validation accounts keep no post from 2026-04-23 on.

| Training rows by origin | Rows | Accounts |
|---|---|---|
| Existing rows (`work/dataset/rows.parquet`, train split) | 302,696 | 6,334 |
| Expand track, twitter100m company and founder accounts | 219,969 | 954 |
| Expand track, brand accounts (35 held out here) | 26,224 | 91 |
| Expand track, 2026 view corpus | 4,548 | 477 |
| General track, twitter100m across five follower tiers | 168,526 | 2,721 |
| General track, 2026 view corpus | 102,093 | 2,224 |
| **Total** | **824,056** | **12,307 distinct** |

Validation accounts of the new sources: 275 twitter100m, 253 view corpus, 10 brands (45,106 rows). The table has 1,154,680 rows of
14,786 accounts; 349,929 rows carry views.

**Cost on this machine.** Rows 4 minutes (5.2 GB peak; only the most recent 250 posts of an account are read at full width).
Embedding 395,969 posts with bge-small under bf16 autocast: 58 minutes on 4 threads (942 posts a second for the shortest posts, 143
for the longest; the long tail of the 2026 corpus is slower than the expand track's 189 a second). Table with neighbour search over
1.15M rows: 5 minutes. One broad run of 6,000 steps with eight evaluations: 36 minutes.

**Inputs for the level bias.** Two categorical inputs: `src_regime` (the expand track's: stream sample, search sample, full timeline)
and a new `src_name` (community_archive, nitter_2026, twitter100m, zenodo_brands). For a random 15% of accounts of every source
`src_name` is "unknown" in every row, so a source never seen in training, which is what a customer's export is, has a trained answer.
At serving `src_name` is derived from `source` when it is a trained source and is "unknown" otherwise (`xviral/model/features.py`).

**Models.** All `small`, 46 text-structure columns, batch 64, learning rate 1e-3, `scripts/general/train_general.py` (the loop of
`xviral.model.train` with more evaluation sets).

| Name | Trained on | Steps | Checkpoint chosen on |
|---|---|---|---|
| Baseline: `ts46_s1`, `ts46_s2`, `compare_v2/full` | existing rows | 4,000 | stored models, not retrained |
| `ctlg_s1`: control under this trainer | existing rows, no source inputs | 4,000 | tech-audience validation |
| `broad_s1`, `broad_s2`: current plus broad data | all 824,056 rows, both source inputs | 6,000 | mean of tech-audience and new-source validation loss |
| `broadft_s1`, `broadft_s2`: broad, then fine-tuned on the tech data | start from `broad_s*`, then existing rows only, learning rate 2e-4 | 1,200 | tech-audience validation |

The selection rule was fixed before the runs. Test-bed, benchmark and launch numbers were computed once, after training.
`ctlg_s1` reproduces the stored controls (tech test like pairs 60.0% against 59.8% and 60.1%; test bed 59.1% against 58.9% and 59.4%),
so the trainer is a fair control.

### Results on the test bed and on the tech-audience test rows

"As customer" scores the same model with `src_name` unknown in every row (the regime stays). Cells: accuracy (difference to the
baseline, 95% interval).

#### Ordinary same-account pairs, likes

| Slice | Accounts | Pairs | Baseline (3 models) | Expanded + sampling input (2) | Broad (2) | Broad then fine-tuned (2) | Broad (2) as customer | Broad then fine-tuned (2) as customer |
|---|---|---|---|---|---|---|---|---|
| twitter100m full timelines (likes, to April 2023) | 451 | 247,210 | 60.7% (59.6% to 61.7%) | 64.9% (+4.2, +3.6 to +4.8) | 65.3% (+4.6, +4.0 to +5.3) | 65.2% (+4.5, +3.8 to +5.1) | 65.3% (+4.6, +3.9 to +5.2) | 65.1% (+4.4, +3.8 to +5.1) |
|   under 1k followers | 89 | 14,497 | 57.2% (53.5% to 61.1%) | 59.9% (+2.7, +0.6 to +5.0) | 60.2% (+3.0, +0.6 to +5.4) | 60.2% (+3.0, +0.4 to +5.5) | 60.1% (+2.9, +0.5 to +5.4) | 60.2% (+3.0, +0.6 to +5.7) |
|   1k to 5k | 96 | 24,490 | 61.4% (58.8% to 64.3%) | 64.6% (+3.2, +1.7 to +4.7) | 64.9% (+3.5, +2.0 to +5.1) | 64.8% (+3.4, +2.0 to +4.9) | 64.9% (+3.5, +2.0 to +5.0) | 64.7% (+3.3, +1.8 to +4.7) |
|   5k to 10k | 92 | 45,957 | 59.9% (57.0% to 62.8%) | 63.5% (+3.6, +2.4 to +4.5) | 64.3% (+4.4, +3.1 to +5.6) | 64.2% (+4.3, +3.2 to +5.5) | 64.3% (+4.4, +3.2 to +5.6) | 64.2% (+4.3, +3.2 to +5.4) |
|   10k to 100k | 88 | 73,157 | 60.0% (58.0% to 61.9%) | 65.4% (+5.4, +4.2 to +6.7) | 66.3% (+6.3, +5.0 to +7.7) | 66.0% (+6.0, +4.7 to +7.5) | 66.3% (+6.3, +4.8 to +7.7) | 65.9% (+5.9, +4.5 to +7.3) |
|   100k and over | 86 | 89,109 | 62.0% (60.3% to 63.7%) | 66.0% (+4.0, +3.1 to +4.9) | 66.0% (+3.9, +2.8 to +5.0) | 65.9% (+3.9, +2.9 to +4.8) | 65.9% (+3.9, +2.8 to +5.0) | 65.8% (+3.8, +2.8 to +4.8) |
| 2026 view corpus (likes and views) | 432 | 109,725 | 58.3% (57.1% to 59.5%) | 58.7% (+0.4, -0.1 to +0.9) | 59.4% (+1.1, +0.7 to +1.6) | 59.3% (+1.0, +0.5 to +1.5) | 59.4% (+1.1, +0.6 to +1.6) | 59.3% (+1.0, +0.5 to +1.5) |
| brand accounts, full timelines | 36 | 12,035 | 53.7% (51.5% to 56.1%) | 58.8% (+5.1, +2.0 to +8.6) | 58.2% (+4.5, +1.8 to +7.9) | 57.4% (+3.7, +1.4 to +6.5) | 57.9% (+4.2, +1.6 to +7.3) | 57.3% (+3.6, +1.3 to +6.4) |
|   brands no model has seen | 11 | 4,867 | 51.9% (48.4% to 55.2%) | 58.1% (+6.2, +0.4 to +12.9) | 58.1% (+6.1, +1.3 to +12.9) | 56.6% (+4.6, +0.2 to +10.6) | 57.8% (+5.9, +1.2 to +12.5) | 56.7% (+4.8, +0.3 to +10.4) |
| small accounts, under 5k followers (all sources) | 246 | 48,091 | 59.3% (57.3% to 61.2%) | 61.9% (+2.6, +1.5 to +3.8) | 62.1% (+2.8, +1.6 to +4.0) | 62.0% (+2.7, +1.6 to +3.8) | 62.1% (+2.8, +1.8 to +4.0) | 61.9% (+2.7, +1.6 to +3.8) |
|   small, 2026 view corpus | 54 | 7,932 | 56.1% (53.2% to 59.1%) | 57.6% (+1.5, -0.5 to +4.2) | 57.3% (+1.2, -0.3 to +2.9) | 56.4% (+0.3, -0.9 to +1.8) | 57.4% (+1.3, -0.2 to +3.0) | 56.4% (+0.3, -0.9 to +1.9) |
| all full-timeline accounts (twitter100m and brands) | 487 | 259,245 | 60.4% (59.3% to 61.5%) | 64.6% (+4.2, +3.7 to +4.8) | 65.0% (+4.6, +4.0 to +5.3) | 64.8% (+4.4, +3.8 to +5.1) | 64.9% (+4.6, +3.9 to +5.2) | 64.7% (+4.4, +3.8 to +5.0) |
| whole test bed | 918 | 368,970 | 59.7% (58.9% to 60.5%) | 62.8% (+3.1, +2.6 to +3.6) | 63.3% (+3.6, +3.1 to +4.1) | 63.2% (+3.4, +2.9 to +3.9) | 63.3% (+3.5, +3.0 to +4.0) | 63.1% (+3.4, +2.9 to +3.8) |
| existing audience test rows | 2,846 | 683,152 | 60.2% (59.0% to 61.3%) | 60.2% (-0.1, -0.4 to +0.2) | 60.1% (-0.2, -0.5 to +0.1) | 60.3% (+0.0, -0.3 to +0.3) |  |  |

#### Ordinary same-account pairs, views

| Slice | Accounts | Pairs | Baseline (3 models) | Expanded + sampling input (2) | Broad (2) | Broad then fine-tuned (2) | Broad (2) as customer | Broad then fine-tuned (2) as customer |
|---|---|---|---|---|---|---|---|---|
| 2026 view corpus (likes and views) | 433 | 112,742 | 62.4% (61.3% to 63.6%) | 62.2% (-0.2, -0.8 to +0.3) | 62.8% (+0.4, -0.0 to +0.9) | 62.8% (+0.4, -0.0 to +0.9) | 62.8% (+0.4, -0.0 to +0.9) | 62.8% (+0.4, -0.1 to +0.9) |
| brand accounts, full timelines | 36 | 11,927 | 55.1% (52.2% to 57.8%) | 61.4% (+6.3, +2.7 to +10.0) | 58.9% (+3.8, +0.5 to +7.7) | 58.5% (+3.4, -0.0 to +7.2) | 58.6% (+3.5, +0.2 to +7.4) | 58.2% (+3.1, -0.3 to +6.9) |
|   brands no model has seen | 11 | 4,669 | 53.6% (48.5% to 58.3%) | 61.0% (+7.4, -0.7 to +15.7) | 60.1% (+6.6, -1.7 to +14.8) | 59.9% (+6.3, -1.7 to +14.6) | 59.6% (+6.1, -1.8 to +13.5) | 59.7% (+6.1, -2.0 to +14.1) |
| small accounts, under 5k followers (all sources) | 61 | 10,985 | 62.4% (60.6% to 64.3%) | 62.5% (+0.1, -1.1 to +1.2) | 62.4% (+0.0, -1.0 to +1.3) | 62.7% (+0.4, -0.7 to +1.4) | 62.4% (+0.0, -1.0 to +1.2) | 62.7% (+0.3, -0.7 to +1.3) |
|   small, 2026 view corpus | 54 | 9,235 | 63.3% (61.6% to 65.3%) | 63.0% (-0.2, -1.3 to +0.7) | 63.1% (-0.2, -1.1 to +0.9) | 63.4% (+0.2, -0.9 to +1.2) | 63.1% (-0.2, -1.1 to +1.1) | 63.5% (+0.2, -0.9 to +1.2) |
| all full-timeline accounts (twitter100m and brands) | 36 | 11,927 | 55.1% (52.3% to 58.0%) | 61.4% (+6.3, +2.8 to +9.7) | 58.9% (+3.8, +0.5 to +7.7) | 58.5% (+3.4, +0.1 to +7.4) | 58.6% (+3.5, +0.3 to +7.2) | 58.2% (+3.1, -0.1 to +6.8) |
| whole test bed | 468 | 124,669 | 61.7% (60.6% to 62.8%) | 62.1% (+0.4, -0.3 to +1.1) | 62.5% (+0.8, +0.2 to +1.3) | 62.4% (+0.7, +0.2 to +1.3) | 62.4% (+0.7, +0.2 to +1.3) | 62.4% (+0.7, +0.1 to +1.2) |
| existing audience test rows | 851 | 24,481 | 62.8% (60.1% to 64.7%) | 62.5% (-0.3, -1.4 to +1.4) | 62.0% (-0.8, -2.1 to +1.3) | 62.3% (-0.5, -1.8 to +1.4) |  |  |


Per model (two seeds of each finalist; no ensembling):

| Slice | Pairs | ctl_s1 | ctl_s2 | ctlg_s1 (control, this trainer) | broad_s1 | broad_s2 | broadft_s1 | broadft_s2 |
|---|---|---|---|---|---|---|---|---|
| twitter100m full timelines (likes, to April 2023): likes pairs | 247,210 | 59.5% | 60.5% | 59.9% | 65.0% | 65.2% | 64.8% | 65.1% |
| 2026 view corpus (likes and views): likes pairs | 109,725 | 58.1% | 57.8% | 58.0% | 59.3% | 59.1% | 59.3% | 58.9% |
| brand accounts, full timelines: likes pairs | 12,035 | 53.6% | 53.0% | 53.1% | 58.3% | 56.9% | 57.7% | 56.3% |
| small accounts, under 5k followers (all sources): likes pairs | 48,091 | 58.4% | 59.5% | 58.1% | 61.9% | 62.2% | 61.8% | 62.1% |
| whole test bed: likes pairs | 368,970 | 58.9% | 59.4% | 59.1% | 63.1% | 63.1% | 63.0% | 63.0% |
| existing audience test rows: likes pairs | 683,152 | 59.8% | 60.1% | 60.0% | 60.0% | 59.7% | 60.3% | 59.7% |
| 2026 view corpus (likes and views): views pairs | 112,742 | 61.7% | 61.9% | 62.0% | 62.6% | 62.5% | 62.8% | 62.2% |
| brand accounts, full timelines: views pairs | 11,927 | 55.2% | 54.6% | 54.8% | 59.1% | 57.8% | 58.7% | 57.2% |
| small accounts, under 5k followers (all sources): views pairs | 10,985 | 61.8% | 62.0% | 61.8% | 62.6% | 62.0% | 62.5% | 61.9% |
| whole test bed: views pairs | 124,669 | 61.1% | 61.2% | 61.3% | 62.2% | 62.0% | 62.4% | 61.7% |
| existing audience test rows: views pairs | 24,481 | 62.3% | 61.8% | 61.7% | 62.0% | 61.2% | 63.1% | 61.4% |
| twitter100m full timelines (likes, to April 2023): like level (median factor; median absolute factor) | | 0.87x; 1.25x | 1.13x; 1.27x | 0.90x; 1.22x | 1.04x; 1.17x | 1.04x; 1.18x | 1.06x; 1.18x | 1.06x; 1.18x |
| 2026 view corpus (likes and views): like level (median factor; median absolute factor) | | 1.05x; 1.21x | 1.10x; 1.23x | 1.07x; 1.23x | 1.09x; 1.21x | 1.08x; 1.20x | 1.08x; 1.21x | 1.08x; 1.20x |
| brand accounts, full timelines: like level (median factor; median absolute factor) | | 1.53x; 1.60x | 1.36x; 1.39x | 1.58x; 1.62x | 1.09x; 1.17x | 1.09x; 1.14x | 1.11x; 1.20x | 1.10x; 1.17x |
| small accounts, under 5k followers (all sources): like level (median factor; median absolute factor) | | 1.03x; 1.21x | 1.21x; 1.25x | 1.05x; 1.21x | 1.08x; 1.18x | 1.07x; 1.19x | 1.10x; 1.19x | 1.10x; 1.19x |
| whole test bed: like level (median factor; median absolute factor) | | 0.99x; 1.23x | 1.13x; 1.26x | 0.99x; 1.23x | 1.07x; 1.18x | 1.06x; 1.19x | 1.08x; 1.19x | 1.07x; 1.19x |

Reading it:
1. **Test bed, likes: +3.6 points** (+3.1 to +4.1) for Broad over the baseline, +0.5 over the expand track's best model. Both seeds
   give 63.1%; both control seeds 58.9% and 59.4%.
2. **Every follower tier of full timelines gains**: +3.0 under 1k, +3.5 at 1k to 5k, +4.4 at 5k to 10k, +6.3 at 10k to 100k, +3.9 at 100k and over.
3. **The 2026 view corpus gains for the first time**: +1.1 on likes (+0.7 to +1.6) and +0.4 on views (0.0 to +0.9), where the expand
   data gave +0.4 and -0.2. This is the 2,224 added 2026 accounts; it is small.
4. **Brands**: +4.5 on likes, +3.8 on views (36 accounts, wide intervals). The expand models are a little ahead on views (61.4% against
   58.9%), but they trained on 25 of these 36 accounts.
5. **Hiding the source costs nothing**: "as customer" is within 0.1 points everywhere.
6. **Fine-tuning** moves nothing beyond noise and loses 0.8 on brand like pairs.

## 3. Overfitting

Each run is evaluated at fixed steps on fixed whole accounts: about 6,000 training rows, 12,000 rows of tech-audience validation
accounts, 12,000 rows of new-source validation accounts and, for reporting only, 6,000 tech-audience test rows and 6,000 test-bed rows.
"Like NLL" is the likelihood loss of the likes head; pairs are ordinary same-account pairs on that sample (small samples: the view
column on tech test rows is a few hundred pairs and jumps).

#### ctlg_s1: 302,696 training rows, roles orig, learning rate 0.001, chosen step 3500 (on val_aud)

| Step | Train: like NLL, like pairs | Validation, tech audience: like NLL, like pairs | Validation, new sources: like NLL, like pairs | Test, tech audience: like NLL, like pairs | Test bed: like NLL, like pairs | Views pairs: validation audience, new sources, test audience, bed |
|---|---|---|---|---|---|---|
| 500 | 3.021, 55.7% | 3.118, 59.6% | 2.884, 54.1% | 2.977, 56.1% | 2.830, 56.0% | 55.7%, 56.0%, 56.0%, 55.7% |
| 1000 | 2.946, 57.8% | 3.057, 61.0% | 2.848, 56.0% | 2.925, 56.9% | 2.783, 59.6% | 58.3%, 57.8%, 62.2%, 60.3% |
| 1500 | 2.919, 58.6% | 3.039, 61.0% | 2.824, 55.9% | 2.897, 57.9% | 2.768, 58.9% | 59.0%, 58.0%, 57.0%, 61.3% |
| 2000 | 2.898, 59.6% | 3.029, 61.2% | 2.808, 56.1% | 2.881, 58.4% | 2.763, 58.3% | 59.4%, 58.0%, 49.6%, 60.3% |
| 2500 | 2.884, 60.0% | 3.019, 61.3% | 2.803, 56.8% | 2.883, 58.7% | 2.753, 58.6% | 59.0%, 57.8%, 49.5%, 60.3% |
| 3000 | 2.872, 60.4% | 3.016, 61.6% | 2.808, 56.8% | 2.880, 58.4% | 2.760, 58.2% | 59.1%, 58.0%, 49.3%, 60.4% |
| 3500 | 2.864, 60.6% | 3.016, 61.5% | 2.813, 56.8% | 2.874, 58.4% | 2.760, 58.4% | 59.0%, 58.0%, 52.3%, 60.5% |
| 4000 | 2.859, 61.1% | 3.016, 61.7% | 2.817, 56.9% | 2.875, 58.8% | 2.770, 58.5% | 58.9%, 58.2%, 52.1%, 60.4% |

#### broad_s1: 824,056 training rows, roles orig,expand,train, learning rate 0.001, chosen step 6000 (on val_aud,val_broad)

| Step | Train: like NLL, like pairs | Validation, tech audience: like NLL, like pairs | Validation, new sources: like NLL, like pairs | Test, tech audience: like NLL, like pairs | Test bed: like NLL, like pairs | Views pairs: validation audience, new sources, test audience, bed |
|---|---|---|---|---|---|---|
| 750 | 2.758, 60.0% | 3.103, 60.1% | 2.805, 59.3% | 2.959, 56.0% | 2.733, 61.5% | 55.6%, 56.9%, 48.3%, 57.0% |
| 1500 | 2.709, 62.0% | 3.066, 60.6% | 2.765, 60.0% | 2.913, 57.8% | 2.683, 62.1% | 56.2%, 57.3%, 48.9%, 59.1% |
| 2250 | 2.685, 62.5% | 3.044, 61.0% | 2.743, 60.9% | 2.895, 58.6% | 2.662, 62.6% | 56.9%, 58.1%, 56.7%, 61.1% |
| 3000 | 2.670, 63.3% | 3.034, 61.2% | 2.733, 61.1% | 2.882, 59.2% | 2.657, 62.7% | 57.9%, 58.0%, 56.3%, 61.3% |
| 3750 | 2.660, 63.8% | 3.024, 61.4% | 2.727, 61.5% | 2.875, 59.3% | 2.658, 62.7% | 58.9%, 58.2%, 60.1%, 61.7% |
| 4500 | 2.651, 63.7% | 3.020, 61.5% | 2.722, 61.7% | 2.865, 59.6% | 2.653, 62.9% | 58.8%, 58.4%, 59.1%, 62.2% |
| 5250 | 2.648, 63.9% | 3.017, 61.4% | 2.719, 62.0% | 2.861, 59.5% | 2.649, 63.0% | 59.2%, 58.7%, 57.5%, 62.0% |
| 6000 | 2.645, 64.1% | 3.015, 61.5% | 2.717, 62.1% | 2.858, 59.6% | 2.646, 63.0% | 59.4%, 59.1%, 57.2%, 62.0% |

#### broad_s2: 824,056 training rows, roles orig,expand,train, learning rate 0.001, chosen step 6000 (on val_aud,val_broad)

| Step | Train: like NLL, like pairs | Validation, tech audience: like NLL, like pairs | Validation, new sources: like NLL, like pairs | Test, tech audience: like NLL, like pairs | Test bed: like NLL, like pairs | Views pairs: validation audience, new sources, test audience, bed |
|---|---|---|---|---|---|---|
| 750 | 2.763, 59.5% | 3.109, 59.8% | 2.813, 58.3% | 2.973, 56.2% | 2.726, 60.8% | 57.1%, 56.7%, 56.3%, 57.8% |
| 1500 | 2.709, 61.4% | 3.065, 59.9% | 2.761, 60.3% | 2.916, 56.6% | 2.696, 61.9% | 57.7%, 57.1%, 54.4%, 59.0% |
| 2250 | 2.689, 62.3% | 3.048, 60.5% | 2.746, 60.6% | 2.899, 57.3% | 2.679, 62.3% | 57.6%, 58.1%, 53.8%, 60.0% |
| 3000 | 2.673, 62.8% | 3.036, 60.6% | 2.737, 61.1% | 2.883, 58.1% | 2.667, 62.6% | 58.4%, 58.7%, 57.3%, 60.6% |
| 3750 | 2.661, 63.0% | 3.027, 61.1% | 2.731, 61.7% | 2.877, 58.3% | 2.663, 62.7% | 58.9%, 59.0%, 55.7%, 60.9% |
| 4500 | 2.656, 63.4% | 3.021, 61.0% | 2.726, 61.9% | 2.877, 57.9% | 2.659, 62.7% | 59.5%, 59.2%, 53.6%, 61.0% |
| 5250 | 2.650, 63.5% | 3.017, 61.1% | 2.724, 62.0% | 2.868, 57.9% | 2.655, 62.7% | 60.2%, 59.3%, 55.8%, 60.9% |
| 6000 | 2.646, 63.7% | 3.016, 61.1% | 2.722, 62.0% | 2.864, 58.0% | 2.655, 62.8% | 60.3%, 59.6%, 56.9%, 61.0% |

#### broadft_s1: 302,696 training rows, roles orig, learning rate 0.0002, chosen step 1200 (on val_aud)

| Step | Train: like NLL, like pairs | Validation, tech audience: like NLL, like pairs | Validation, new sources: like NLL, like pairs | Test, tech audience: like NLL, like pairs | Test bed: like NLL, like pairs | Views pairs: validation audience, new sources, test audience, bed |
|---|---|---|---|---|---|---|
| 0 | 2.888, 59.6% | 3.015, 61.5% | 2.717, 62.1% | 2.858, 59.6% | 2.646, 63.0% | 59.4%, 59.1%, 57.2%, 62.0% |
| 200 | 2.888, 59.5% | 3.020, 61.4% | 2.721, 61.8% | 2.858, 59.7% | 2.650, 62.8% | 59.9%, 58.9%, 59.2%, 62.3% |
| 400 | 2.885, 59.8% | 3.018, 61.7% | 2.727, 61.8% | 2.863, 60.2% | 2.648, 62.7% | 60.0%, 58.9%, 56.6%, 62.4% |
| 600 | 2.881, 59.9% | 3.016, 61.7% | 2.728, 61.6% | 2.857, 60.3% | 2.647, 62.7% | 60.0%, 59.1%, 58.8%, 62.3% |
| 800 | 2.880, 59.9% | 3.015, 61.7% | 2.727, 61.8% | 2.855, 60.2% | 2.649, 62.9% | 60.0%, 59.1%, 59.1%, 62.4% |
| 1000 | 2.878, 60.0% | 3.014, 61.8% | 2.725, 61.9% | 2.855, 60.1% | 2.650, 63.0% | 60.2%, 59.1%, 59.7%, 62.4% |
| 1200 | 2.879, 60.0% | 3.013, 61.8% | 2.725, 61.9% | 2.855, 60.1% | 2.650, 63.0% | 60.0%, 59.2%, 60.4%, 62.5% |

#### broadft_s2: 302,696 training rows, roles orig, learning rate 0.0002, chosen step 1200 (on val_aud)

| Step | Train: like NLL, like pairs | Validation, tech audience: like NLL, like pairs | Validation, new sources: like NLL, like pairs | Test, tech audience: like NLL, like pairs | Test bed: like NLL, like pairs | Views pairs: validation audience, new sources, test audience, bed |
|---|---|---|---|---|---|---|
| 0 | 2.888, 59.0% | 3.016, 61.1% | 2.722, 62.0% | 2.864, 58.0% | 2.655, 62.8% | 60.3%, 59.6%, 56.9%, 61.0% |
| 200 | 2.890, 59.1% | 3.020, 61.1% | 2.731, 61.9% | 2.869, 57.8% | 2.665, 62.6% | 60.8%, 59.6%, 60.0%, 61.5% |
| 400 | 2.887, 59.1% | 3.017, 61.0% | 2.730, 61.9% | 2.868, 58.1% | 2.664, 62.4% | 60.3%, 59.4%, 58.3%, 60.6% |
| 600 | 2.882, 59.2% | 3.016, 61.0% | 2.731, 61.9% | 2.865, 58.0% | 2.665, 62.5% | 60.4%, 59.5%, 57.6%, 60.6% |
| 800 | 2.880, 59.5% | 3.013, 61.2% | 2.731, 62.0% | 2.863, 58.3% | 2.665, 62.3% | 60.5%, 59.4%, 57.8%, 60.2% |
| 1000 | 2.879, 59.5% | 3.012, 61.2% | 2.729, 62.0% | 2.863, 58.4% | 2.665, 62.5% | 60.5%, 59.3%, 57.9%, 60.4% |
| 1200 | 2.878, 59.5% | 3.011, 61.2% | 2.729, 62.1% | 2.863, 58.3% | 2.665, 62.4% | 60.6%, 59.4%, 58.2%, 60.4% |


- **Broad runs**: training and all four held-out curves fall together to the last step. Train and test-bed loss are equal at step
  6,000 (2.645 and 2.646). The runs had not converged; more steps would likely still help a little (not run, for time).
- **Control**: its training loss keeps falling (2.946 to 2.859) while tech validation is flat from step 3,000 and the loss on
  new-source validation accounts rises from 2.803 to 2.817. Its test-bed pairs peak at step 1,000 (59.6%) and decline to 58.5%. That is
  overfitting to the archive distribution, invisible on its own validation set.
- **Fine-tune**: tech validation improves by 0.002 in loss; new-source validation and the test bed get slightly worse. Nothing gained.

**Tech-audience checks (must not degrade).**

| Check | Pairs | Baseline (3) | Control (2 seeds) | Expanded + sampling input (2) | Broad (2) | Broad then fine-tuned (2) |
|---|---|---|---|---|---|---|
| Ordinary test like pairs | 683,152 | 60.2% | | 60.2% | 60.1% (-0.2, -0.5 to +0.1) | 60.3% (0.0, -0.3 to +0.3) |
| Ordinary test view pairs | 24,481 | 62.8% | | 62.5% | 62.0% (-0.8, -2.1 to +1.3) | 62.3% (-0.5, -1.8 to +1.4) |
| Benchmark, 1.25x apart | 2,851 | 59.9% | 59.5% | 59.1% | 59.2% (-0.2 against control, -2.0 to +1.5) | 58.8% (-0.7, -2.6 to +1.1) |
| Benchmark, 2x apart | 2,071 | 61.3% | 60.5% | 60.6% | 60.5% | 59.5% |
| Benchmark, confident on 2x pairs: share, correct | | 25%, 70.1% | 25%, 69.6% | 25%, 70.3% | 24%, 70.3% | 25%, 69.5% |
| Benchmark, 5x apart; confident there | 880 | 66.5%; 28%, 81.2% | 65.9%; 28%, 80.7% | 64.4%; 26%, 78.7% | 64.3%; 26%, 78.9% | 62.7%; 26%, 79.3% |
| Launch pairs | 850 | 63.9% | 63.4% | 64.7% | 63.8% (+0.4 against control, -3.8 to +4.0) | 63.3% (-0.1, -4.3 to +3.4) |
| Launch pairs 2x apart | 568 | 66.6% | 66.2% | 66.0% | 65.3% | 65.3% |
| Launch, confident calls: share, correct | | 26%, 73.9% | 26%, 72.4% | 27%, 73.9% | 32%, 74.4% | 31%, 73.4% |
| Predicted launch level over the account's usual (actual 2.19x) | 576 posts | 1.03x | 0.98x | 1.14x | 1.13x | 1.15x |

Per seed on the benchmark: `broad_s1` 58.8%, `broad_s2` 58.3%, against `ctl_s1` 59.9%, `ctl_s2` 58.9% (-1.1 and -0.6, intervals of about
2 points either side); `broadft_s1` 58.2%, `broadft_s2` 58.8%. Nothing degrades beyond the resolution of these sets, and the point
estimates on the benchmark lean slightly negative for every model that trains on added data (expand and broad alike), most at the
5x gap (64.3% against 65.9%). That lean is inside the intervals; it is the reason not to claim "no cost", only "no measurable cost".

## 4. Forecast level for new customers

A customer gives a full timeline. The honest test is the full-timeline accounts of the bed (twitter100m and brands), with the model
told only what it would be told for a customer: regime "full timeline", source unknown.

#### Level, likes: median account's predicted median over actual median; median absolute factor; share of accounts inside 1.5x

| Slice | Accounts | Baseline (3 models) | Baseline (3 models) + correction | Expanded + sampling input (2) | Broad (2) | Broad (2) as customer | Broad (2) as customer + correction | Broad then fine-tuned (2) as customer |
|---|---|---|---|---|---|---|---|---|
| twitter100m full timelines (likes, to April 2023) | 454 | 1.00x (0.96 to 1.05); 1.23x; 79% | 1.00x (0.97 to 1.02); 1.15x; 88% | 1.05x (1.02 to 1.08); 1.18x; 89% | 1.03x (1.01 to 1.06); 1.18x; 92% | 1.06x (1.03 to 1.08); 1.17x; 91% | 1.02x (0.99 to 1.06); 1.16x; 91% | 1.08x (1.05 to 1.11); 1.18x; 89% |
| &nbsp;&nbsp;under 1k followers | 92 | 1.19x (1.13 to 1.22); 1.23x; 89% | 1.01x (0.99 to 1.03); 1.10x; 91% | 1.17x (1.12 to 1.20); 1.20x; 92% | 1.17x (1.11 to 1.19); 1.19x; 92% | 1.17x (1.13 to 1.19); 1.18x; 92% | 1.15x (1.10 to 1.16); 1.16x; 92% | 1.19x (1.15 to 1.20); 1.20x; 91% |
| &nbsp;&nbsp;1k to 5k | 96 | 1.07x (1.02 to 1.16); 1.21x; 85% | 1.05x (0.98 to 1.09); 1.14x; 91% | 1.05x (1.02 to 1.11); 1.19x; 92% | 1.03x (0.99 to 1.08); 1.15x; 92% | 1.05x (1.01 to 1.09); 1.14x; 91% | 1.03x (0.98 to 1.06); 1.11x; 93% | 1.09x (1.04 to 1.14); 1.15x; 91% |
| &nbsp;&nbsp;5k to 10k | 92 | 1.05x (0.99 to 1.09); 1.18x; 84% | 1.01x (0.95 to 1.04); 1.14x; 88% | 1.08x (1.01 to 1.10); 1.16x; 90% | 1.02x (0.97 to 1.08); 1.17x; 92% | 1.05x (0.98 to 1.10); 1.16x; 91% | 1.00x (0.95 to 1.06); 1.15x; 92% | 1.12x (1.03 to 1.14); 1.17x; 90% |
| &nbsp;&nbsp;10k to 100k | 88 | 0.94x (0.84 to 1.03); 1.24x; 77% | 0.99x (0.95 to 1.06); 1.17x; 88% | 1.01x (0.95 to 1.06); 1.16x; 85% | 1.02x (0.93 to 1.05); 1.18x; 91% | 1.04x (0.96 to 1.07); 1.15x; 89% | 0.99x (0.92 to 1.05); 1.17x; 90% | 1.05x (1.00 to 1.12); 1.17x; 88% |
| &nbsp;&nbsp;100k and over | 86 | 0.71x (0.67 to 0.76); 1.42x; 59% | 0.89x (0.84 to 0.94); 1.23x; 80% | 0.94x (0.87 to 1.03); 1.20x; 83% | 0.98x (0.93 to 1.04); 1.17x; 92% | 0.99x (0.92 to 1.06); 1.17x; 91% | 0.97x (0.92 to 1.04); 1.18x; 88% | 0.97x (0.90 to 1.03); 1.18x; 84% |
| 2026 view corpus (likes and views) | 433 | 1.06x (1.03 to 1.09); 1.21x; 80% | 1.03x (1.00 to 1.05); 1.20x; 81% | 1.07x (1.04 to 1.11); 1.21x; 79% | 1.09x (1.04 to 1.13); 1.20x; 83% | 1.08x (1.04 to 1.13); 1.20x; 83% | 1.03x (0.99 to 1.08); 1.18x; 86% | 1.06x (1.02 to 1.10); 1.20x; 84% |
| brand accounts, full timelines | 36 | 1.41x (1.33 to 1.74); 1.43x; 56% | 1.44x (1.16 to 1.64); 1.46x; 53% | 1.06x (0.93 to 1.21); 1.19x; 92% | 1.09x (1.00 to 1.20); 1.16x; 89% | 1.10x (1.00 to 1.20); 1.15x; 89% | 1.06x (0.96 to 1.09); 1.14x; 89% | 1.14x (1.01 to 1.20); 1.16x; 86% |
| &nbsp;&nbsp;brands no model has seen | 11 | 1.50x (1.26 to 1.76); 1.50x; 55% | 1.55x (1.24 to 2.06); 1.55x; 36% | 0.97x (0.84 to 1.21); 1.19x; 91% | 1.03x (0.92 to 1.10); 1.09x; 100% | 1.06x (0.96 to 1.15); 1.10x; 100% | 1.04x (0.89 to 1.10); 1.10x; 100% | 1.03x (0.97 to 1.16); 1.11x; 100% |
| small accounts, under 5k followers (all sources) | 249 | 1.12x (1.08 to 1.17); 1.21x; 86% | 1.03x (1.00 to 1.05); 1.14x; 88% | 1.10x (1.05 to 1.16); 1.19x; 90% | 1.08x (1.04 to 1.13); 1.18x; 91% | 1.09x (1.05 to 1.13); 1.18x; 91% | 1.05x (1.03 to 1.09); 1.16x; 92% | 1.10x (1.06 to 1.16); 1.18x; 90% |
| &nbsp;&nbsp;small, 2026 view corpus | 54 | 1.04x (0.94 to 1.11); 1.18x; 81% | 1.01x (0.90 to 1.07); 1.17x; 81% | 1.06x (0.96 to 1.14); 1.19x; 83% | 1.08x (0.97 to 1.16); 1.18x; 87% | 1.06x (0.96 to 1.13); 1.17x; 87% | 1.02x (0.91 to 1.08); 1.17x; 87% | 1.02x (0.92 to 1.12); 1.18x; 87% |
| all full-timeline accounts (twitter100m and brands) | 490 | 1.03x (0.99 to 1.06); 1.24x; 78% | 1.01x (0.99 to 1.03); 1.16x; 85% | 1.05x (1.02 to 1.08); 1.18x; 89% | 1.04x (1.01 to 1.06); 1.18x; 92% | 1.06x (1.03 to 1.09); 1.17x; 91% | 1.03x (1.00 to 1.06); 1.16x; 91% | 1.08x (1.05 to 1.11); 1.18x; 89% |
| whole test bed | 922 | 1.05x (1.03 to 1.08); 1.23x; 79% | 1.02x (1.00 to 1.03); 1.18x; 83% | 1.06x (1.03 to 1.08); 1.19x; 84% | 1.05x (1.03 to 1.08); 1.19x; 88% | 1.06x (1.04 to 1.09); 1.18x; 87% | 1.03x (1.00 to 1.05); 1.17x; 89% | 1.07x (1.05 to 1.10); 1.19x; 86% |

#### Level, views: median account's predicted median over actual median; median absolute factor; share of accounts inside 1.5x

| Slice | Accounts | Baseline (3 models) | Baseline (3 models) + correction | Expanded + sampling input (2) | Broad (2) | Broad (2) as customer | Broad (2) as customer + correction | Broad then fine-tuned (2) as customer |
|---|---|---|---|---|---|---|---|---|
| 2026 view corpus (likes and views) | 433 | 1.10x (1.06 to 1.13); 1.22x; 81% | 1.06x (1.02 to 1.10); 1.21x; 81% | 1.08x (1.06 to 1.13); 1.21x; 81% | 1.10x (1.07 to 1.13); 1.19x; 83% | 1.11x (1.07 to 1.13); 1.19x; 83% | 1.06x (1.02 to 1.08); 1.17x; 84% | 1.09x (1.06 to 1.12); 1.19x; 82% |
| brand accounts, full timelines | 36 | 1.49x (1.35 to 1.83); 1.69x; 44% | 1.44x (1.16 to 1.78); 1.64x; 44% | 1.13x (1.07 to 1.20); 1.15x; 86% | 1.17x (1.09 to 1.26); 1.20x; 72% | 1.15x (1.09 to 1.25); 1.21x; 72% | 1.10x (1.01 to 1.23); 1.24x; 75% | 1.17x (1.07 to 1.28); 1.23x; 72% |
| &nbsp;&nbsp;brands no model has seen | 11 | 1.50x (1.24 to 1.84); 1.63x; 45% | 1.63x (1.17 to 1.88); 1.65x; 27% | 1.17x (1.07 to 1.50); 1.21x; 73% | 1.16x (1.08 to 1.38); 1.19x; 73% | 1.15x (1.10 to 1.45); 1.16x; 73% | 1.11x (1.06 to 1.41); 1.14x; 82% | 1.16x (1.13 to 1.38); 1.18x; 73% |
| small accounts, under 5k followers (all sources) | 61 | 1.14x (1.03 to 1.23); 1.25x; 80% | 1.10x (1.00 to 1.19); 1.25x; 79% | 1.07x (0.97 to 1.13); 1.21x; 84% | 1.14x (1.05 to 1.19); 1.21x; 84% | 1.14x (1.04 to 1.19); 1.22x; 84% | 1.10x (1.00 to 1.14); 1.21x; 84% | 1.09x (1.02 to 1.16); 1.19x; 82% |
| &nbsp;&nbsp;small, 2026 view corpus | 54 | 1.09x (1.02 to 1.17); 1.22x; 83% | 1.05x (0.99 to 1.13); 1.19x; 81% | 1.04x (0.93 to 1.13); 1.21x; 85% | 1.09x (1.01 to 1.18); 1.20x; 87% | 1.10x (1.00 to 1.18); 1.20x; 87% | 1.05x (0.96 to 1.13); 1.16x; 85% | 1.06x (1.00 to 1.13); 1.18x; 85% |
| all full-timeline accounts (twitter100m and brands) | 36 | 1.49x (1.35 to 1.83); 1.69x; 44% | 1.44x (1.17 to 1.79); 1.64x; 44% | 1.13x (1.07 to 1.20); 1.15x; 86% | 1.17x (1.09 to 1.26); 1.20x; 72% | 1.15x (1.09 to 1.25); 1.21x; 72% | 1.10x (1.02 to 1.25); 1.24x; 75% | 1.17x (1.07 to 1.29); 1.23x; 72% |
| whole test bed | 468 | 1.12x (1.07 to 1.16); 1.24x; 78% | 1.08x (1.03 to 1.11); 1.22x; 78% | 1.09x (1.07 to 1.13); 1.20x; 82% | 1.10x (1.08 to 1.13); 1.19x; 82% | 1.11x (1.08 to 1.14); 1.19x; 82% | 1.06x (1.03 to 1.09); 1.17x; 84% | 1.10x (1.07 to 1.13); 1.19x; 82% |


#### Level correction fitted on validation accounts

Full timelines: log10(error) = a + b*log10(followers+1) + c*log10(median likes of the visible history+1), subtracted from the forecast. Search sample: one offset.

| Model | a | b | c | Validation accounts | Median absolute error before | after | Search-sample offset |
|---|---|---|---|---|---|---|---|
| Baseline (3 models) | +0.005 | +0.035 | -0.151 | 275 | 1.23x | 1.19x | 1.03x |
| Expanded + sampling input (2) | -0.041 | +0.025 | -0.048 | 275 | 1.18x | 1.17x | 1.05x |
| Broad (2) | -0.033 | +0.018 | -0.032 | 275 | 1.17x | 1.16x | 1.05x |
| Broad then fine-tuned (2) | -0.025 | +0.023 | -0.052 | 275 | 1.18x | 1.15x | 1.05x |
| Broad (2) as customer | -0.030 | +0.021 | -0.039 | 275 | 1.18x | 1.15x | 1.05x |
| Broad then fine-tuned (2) as customer | -0.020 | +0.025 | -0.058 | 275 | 1.19x | 1.14x | 1.05x |

What it says:
1. **The baseline is not uniformly 1.5x high.** On 454 twitter100m timelines its median account is 1.00x, but that hides
   1.19x under 1k followers and 0.71x at 100k and over, and one account in five is outside 1.5x. On brand timelines it is 1.41x high
   (1.50x on the 11 unseen brands), which is the expand track's "about 1.5x" and is specific to accounts with large passive followings.
   The baseline's two seeds differ (0.87x and 1.13x on twitter100m): its level on full timelines is not pinned down by its training data.
2. **The broad model as a customer would use it**: 1.06x on twitter100m (91% inside 1.5x), 1.10x on brands (89%), 0.99x at 100k and
   over, 1.06x on all full timelines. Both seeds agree (1.04x and 1.04x with the source given). With the three-number correction on
   top: 1.02x, 1.06x, 1.03x overall.
3. **What stays off**: accounts under 1,000 followers are forecast 1.17x high by every model, new or old (1.15x after correction).
   Their median post has one or two likes, where a count cannot follow a smooth forecast; the absolute error is a fraction of a like.
4. **Views**: on brand timelines the baseline is 1.49x high, the broad model 1.15x (36 accounts). On the 2026 corpus every model is
   about 1.10x high on views; that corpus is a search sample and its level was not the question.
5. **The correction alone is not enough for the current models**: it lifts the baseline from 79% to 88% of twitter100m accounts inside
   1.5x and fixes the follower-size slope, but brand timelines stay at 1.44x, because the validation accounts it is fitted on are 265
   twitter100m accounts and 10 brands. A correction cannot know an account is of a kind the model never saw; the broad model does.

**Delivered.** `runs/broad_s1/best.pt`, `runs/broad_s2/best.pt` (use as a two-member ensemble). `eval/bed_level_correction.json`:
for each model family, `log10_error = a + b*log10(followers+1) + c*log10(median likes of the visible history+1)` for full timelines
(subtract it from the log10 forecast) and one offset for search samples. For the baseline: a=+0.005, b=+0.035, c=-0.151. For the
broad model as a customer: a=-0.030, b=+0.021, c=-0.039.

## 5. Recommendation

- **Training set**: the 824,056-row table in `table/` (existing rows, the expand rows, the general sample). Adding 4,945 accounts at
  64 posts each bought more than the expand track's 1,566 accounts at up to 400: breadth of accounts, not depth per account.
- **Recipe**: `small`, 46 text-structure columns, inputs `src_regime` and `src_name` with 15% of accounts hidden, 6,000 steps (or
  more; the curves had not flattened), checkpoint on the mean of tech and new-source validation loss, two or three seeds averaged.
  No fine-tune on the tech audience.
- **Serving a new customer**: regime full timeline, source unknown; optionally the correction above.
- **Whether to replace the production models**: for a customer whose account is not a Community Archive regular, yes: better ranking
  on their kind of account (+3 to +5 points) and an honest level. For the tech audience it is a wash, and the benchmark's point
  estimate is 0.2 to 0.7 lower with a 2-point interval. A safe path is to add the broad pair as members next to the existing three.
  That mixed ensemble was not scored here.

## 6. What did not work or was not done

- Tech-audience accuracy did not move, again. Twice the accounts (6,334 to 12,307) gave -0.2 on ordinary test like pairs, where the
  scale track's curve suggested about +0.4 per doubling for accounts of the same kind. Accounts of other kinds do not count as the same kind.
- Fine-tuning after broad pretraining: no gain anywhere.
- Small accounts with views remain thin: 54 accounts of the 2026 corpus with a known count under 5k (+1.2 on likes, -0.2 on views, both inside noise).
- Not run, for time: the ablation without the two source inputs on the broad table (the expand track ran that comparison on its
  table: the input moved level, not ranking); more than 6,000 steps; a larger sample (about 9,000 eligible twitter100m accounts and
  11,000 view-corpus accounts remain unused); the pair-loss fine-tune on the broad models; a mixed ensemble.
- `ctlg_s1` is one seed; the stored `ts46_s1` and `ts46_s2` are the two-seed control everywhere else.

## 7. Honest limits

- The full-timeline bed is twitter100m: likes only, 2020 to April 2023, before the current ranker, with one follower reading from
  April 2023. There is no public full-timeline data with views from 2026 other than 36 brand accounts. A customer's 2026 timeline is
  not in this bed.
- The broad models trained on other accounts of the same sources as the bed. The bed measures generalisation to unseen accounts of
  seen sources, not to an unseen source. Hiding the source name changes nothing, which is some evidence; the regime input is still given.
- Brands: 36 accounts, 11 unseen by every model; large consumer brands, not startups.
- "Small accounts" are 75% twitter100m. Follower counts of the 2026 accounts are 2023 readings.
- Only the most recent 250 posts of an account were read, so `history_n` and the trailing statistics of the earliest kept rows see a
  shorter past than the pipeline would give them; a bed account's scored posts have at least 48 posts before them.
- New accounts are flagged general (`in_audience` false); the tech-audience slices are the existing rows only.
- Evaluation samples inside the curves are small (6,000 to 12,000 rows); the final tables use all rows.
- The level measure compares medians per account; it says nothing about the tails (launch posts are still forecast at 1.13x of usual
  against 2.19x actual; the post-type correction is still needed).
- Tests: 138 pass, 1 skipped; 2 in `tests/test_app.py` fail on a module that is not installed on this machine, with and without these changes.

## Exact commands

From `virality/`, after the setup of `scripts/worker_job.sh` and the downloads listed in the expand report plus `tracks/expand/table/*`,
`tracks/expand/rows/*`, `tracks/replicate/dev_pairs.parquet`:

```bash
python scripts/general/account_stats.py                       # data/general/account_stats.parquet
python scripts/general/select_accounts.py                     # data/general/accounts.parquet (bed first, own seed)
python scripts/general/build_rows.py                          # data/general/new_rows.parquet, 395,969 rows
OMP_NUM_THREADS=4 python scripts/expand/embed_rows.py --rows data/general/new_rows.parquet --out data/general/emb
python scripts/general/build_table.py                         # data/general/table
R=data/general/runs
python scripts/general/train_general.py --out $R/broad_s1 --seed 1 --roles orig,expand,train --steps 6000      # and seed 2
python scripts/general/train_general.py --out $R/broadft_s1 --seed 1 --roles orig --steps 1200 --lr 2e-4 --warmup 100 --evals 6 --init $R/broad_s1/best.pt --select val_aud
python scripts/general/train_general.py --out $R/ctlg_s1 --seed 1 --roles orig --steps 4000 --drop-cat src_name,src_regime --select val_aud
python scripts/general/predict_rows.py --customer --run broad_s1=$R/broad_s1/best.pt --run ...                 # data/general/pred
python scripts/general/report.py --model "Baseline (3 models)=ctl_s1,ctl_s2,v2full" --model "Broad (2)=broad_s1,broad_s2" ... --customer "Broad (2)" --base "Baseline (3 models)" --out data/general/eval/bed
python scripts/general/curves.py $R/ctlg_s1 $R/broad_s1 ...   # eval/curves.md
bash scripts/expand/after_train.sh g_broad_s1 $R/broad_s1/best.pt                                             # benchmark and launch scores
python scripts/evidence/gap_table.py ... --out data/general/eval/bench.json; python scripts/expand/launch_check.py ... --out data/general/eval/launch.json
```
`scripts/general/queue.sh` and `queue2.sh` are the queues as they ran.

## Code changes (`patch.diff`)

- `xviral/model/features.py`: the categorical input `src_name` (in `CAT_SPEC`; derived from `source` at serving, "unknown" otherwise).
- New, under `scripts/general/`: `account_stats.py`, `select_accounts.py`, `build_rows.py`, `build_table.py`, `train_general.py`,
  `predict_rows.py`, `report.py`, `compact.py`, `curves.py`, `queue.sh`, `queue2.sh`.

## Files (`tracks/general/`)

| Path | Content |
|---|---|
| `REPORT.md`, `patch.diff` | this report; the code |
| `rows/accounts.parquet`, `account_stats.parquet`, `new_rows.parquet`, `build_counts.json`, `filter_log.csv` | the sampled accounts with roles; per-account counts by source; the 395,969 built rows (`g_role`, `g_tier`); counts by stage |
| `table/` | the model table, 1,154,680 rows; `row_info.parquet` has roles, the bed flag and tiers |
| `runs/<name>/best.pt`, `curves.json` | `broad_s1`, `broad_s2` (recommended), `broadft_s1`, `broadft_s2`, `ctlg_s1` |
| `eval/baseline.md`, `.json` | the current models on the bed |
| `eval/bed.md`, `.json`, `bed_level_correction.json`, `level.md` | every model on the bed; the correction; the level table |
| `eval/bed_seeds.json`, `seeds.md`, `curves.md` | per-seed results; curves |
| `eval/bench.json`, `launch.json` | the 3,530-pair benchmark and the 850 launch pairs |
| `bench/<name>/shard_*.parquet`, `launch/<name>.parquet` | per-pair and per-post scores of the four new models |
