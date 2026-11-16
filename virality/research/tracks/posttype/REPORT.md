# Track "posttype": telling the model that a launch is a launch

2026-11-16. Files in this folder of the dataset (`tracks/posttype/`): this report, `patch.diff`, `type_correction.json`,
`launch_eval.json`, `bench_table.json`, `bench_table_corr.json`, `feature_checks.json`, `table_calibration_test.json`,
`table_calibration_train.json`, `scores/` (per-post draft-path predictions of every model), `bench2/` (per-pair
benchmark scores of the new models), `post_type.parquet` (the type and the 34 new columns of every table row),
`checkpoints/` (the three retrained models, see "What to ship").

## Headline

- **Retraining with the new inputs fixes about half of the level error and does not improve ranking.** Three `small`
  models with 34 added inputs (post type with the own-announcement test, launch traits, trailing record by type; two
  seeds plus one with own announcements weighted 10x) predict launch posts at 1.50x the account's usual likes
  (baseline ensemble 1.10x, actual 2.19x), funding at 3.88x (1.54x, actual 6.07x) and feature posts at 0.97x (0.91x,
  actual 1.13x). On the 850 launch pairs they are 63.1% correct against 63.9% for the baseline ensemble (difference
  -0.8 points, -4.2 to +2.3), and on the general benchmark 59.1% against 60.0% (-0.8, -2.2 to +0.3) with fewer
  confident calls (24% against 27% of pairs). **No retrained checkpoint beats the baseline on pairs.**
- **A seven-number correction on the unchanged baseline does the same for the level at no cost to ranking.** One
  shift in log likes per post type, fitted on the baseline's errors on own announcements before the cutoff, applied
  in `DraftScorer`: launch 1.10x to 1.50x, funding 1.54x to 3.36x, feature 0.91x to 0.94x; launch pairs 63.8%
  (-0.1, -0.6 to +0.4); general benchmark unchanged (18 of 3,530 pairs contain an own announcement). This is what I
  recommend shipping.
- **The rest of the level gap cannot be learned from data before the cutoff.** In the training table launch posts got
  1.77x the account's median before 2026-04-23 and 2.81x after it; funding 3.63x and 6.30x. A model that fitted the
  training period perfectly would still be about a third short on the test period. Ordinary posts are also
  under-predicted in the test period (control sample: 0.81x predicted, 1.00x actual), which is a general drift and
  not a post-type matter.
- **The trait weights did not get right.** The retrained ensemble prefers the version that names the product in 68.9%
  of pairs (baseline 61.4%, it won 73.9%), which is better, and the video version in 74.1% (baseline 67.7%, it won
  50.5%), which is worse. Before the cutoff video won 73.6% of launch pairs (launch track), so the model learned the
  old effect faithfully; the effect itself moved.

## What was built

`xviral/features/post_type.py` (new), wired into `xviral/model/features.py` (NUM_SPEC) and `xviral/serve.py`
(`feature_frame`, so drafts get the same values). 34 columns:

| Group | Columns | Definition |
|---|---|---|
| Type (7) | `pt_funding`, `pt_hiring`, `pt_launch`, `pt_open_source`, `pt_feature`, `pt_milestone`, `pt_demo` | `launch_labels.add_types` then `launch_labels.own_expr`, called as they are; all zero = ordinary post |
| Rule parts (5) | `pt_raw_typed`, `pt_fp_head`, `pt_opener`, `pt_techy`, `pt_company_voice` | the pieces of the own-announcement test |
| Traits (8) | `pt_names_product_first_line`, `pt_lines3`, `pt_video`, `pt_image`, `pt_open_problem`, `pt_open_number`, `pt_ext_link`, `pt_is_quote` | `post_type.traits()`, called per post |
| Traits on launch posts (8) | `pt_launch_<trait>` | the trait, only when the post is the account's own launch or open-source release |
| Trailing record (6) | `pt_prev_launch_n/_lift`, `pt_prev_ann_n/_lift`, `pt_prev_same_n/_lift` | among the account's previous 50 posts: how many were own launch-type posts / any own announcement / the same type as this post, and their mean log likes minus the median log likes of those 50 posts |

One definition had to change. `own_expr` needs the account kind (a company account's "Introducing X" counts without
"we/our"). The launch track labelled accounts from their whole timeline, which serving cannot do and which lets later
posts into earlier rows. Here the kind is read from the previous 50 posts only: under 10% use I/my and either 12%+ use
we/our or 15%+ are typed tech posts, with at least 8 previous posts. On the table this flag is true for 81% of the rows
of the launch track's company accounts and 6% of other rows; 96.8% of the track's launch rows are typed launch here,
and 98.9% of rows typed here have the same type in the track (`feature_checks.json`).

**Leakage.** Every trailing value comes from posts of the same account created strictly earlier (asof joins that
refuse equal timestamps), and drafts never count as history. Tested on 300 accounts (38,081 posts): multiplying the
likes of a post and every later post changes none of the 20,347 rows at or before it; deleting every later post
changes none either.

**Training and serving agree.** For 400 table rows (200 typed), the post was rebuilt as a draft against its previous
56 posts through `serve.feature_frame`: all 34 columns identical in all 400.

**Table.** `scripts/posttype/build_table.py` reuses the stored table (embeddings, semantic IDs, neighbours, labels)
and rewrites only `num.npy` and `meta.json`: 128 to 165 numeric inputs. Own announcements are rare: 3,658 of 302,696
training rows (launch 2,056, feature 778, demo 317, hiring 255, funding 125, open source 91, milestone 36).

## Results

Models. *baseline*: ts46_s1, ts46_s2, compare_v2_full (the three 46-column `small` models). *retrained*: ptb_s1,
ptb_s2 (new inputs, seeds 1 and 2) and ptbw10_s1 (new inputs, own announcements weighted 10x), same preset, 4,000
steps, `scripts/train_one.py --features ts46`. *baseline + correction*: the baseline with `type_correction.json`.
*six*: all six averaged. Ensembles here average the members' log likes per post or per pair (score level). All numbers
are from the test period (2026-04-23 on); nothing was tuned on it.

### (a) General pair benchmark (same flags as `worker_bench.sh`; 3,530 pairs scored, 2,852 with a 1.25x gap)

Confidence cross-fitted over accounts as in `compare_runs.py`; "confident" = stated 60% or more.

| Real gap at least | Pairs | Baseline correct | confident share, correct | Retrained correct | confident share, correct | Baseline + correction | Six correct | confident share, correct |
|---|---|---|---|---|---|---|---|---|
| 1.25x | 2,852 | 60.0% | 27% (778), 67.3% | 59.1% | 24% (676), 66.4% | 60.0%, 27%, 67.4% | 60.3% | 26% (729), 67.3% |
| 2x | 2,030 | 61.4% | 26% (523), 70.2% | 60.3% | 23% (463), 69.3% | 61.4%, 26%, 70.2% | 61.5% | 24% (496), 70.6% |
| 3x | 1,377 | 63.3% | 27% (371), 74.4% | 63.2% | 23% (322), 74.2% | 63.3%, 27%, 74.4% | 64.3% | 25% (350), 74.6% |
| 5x | 880 | 66.5% | 28% (245), 81.2% | 66.9% | 24% (212), 81.1% | 66.5%, 28%, 81.2% | 67.6% | 27% (234), 81.6% |

By confidence band (pairs, correct):

| Stated | Baseline | Retrained | Six |
|---|---|---|---|
| 50 to 55% | 1,412, 55.1% | 1,520, 55.3% | 1,425, 56.6% |
| 55 to 60% | 662, 61.6% | 656, 60.4% | 698, 60.6% |
| 60 to 70% | 564, 66.3% | 526, 64.8% | 528, 65.9% |
| 70 to 80% | 172, 72.1% | 126, 76.2% | 168, 74.4% |
| 80%+ | 42, 61.9% | 24, 50.0% | 33, 54.5% |

Paired differences against the baseline ensemble (accounts resampled): retrained -0.8 points at 1.25x (-2.2 to +0.3),
-1.1 at 2x (-2.6 to +0.2); the two seeds alone -0.9 (-2.1 to +0.2); six +0.4 (-0.4 to +1.1); baseline + correction 0.0.
Single retrained models are 1.1 to 1.7 points below the baseline ensemble, and two of the three intervals exclude
zero (ptb_s2 -1.6, -2.8 to -0.3; ptbw10_s1 -1.7, -3.1 to -0.5), where the single baseline model ts46_s1 is -0.1.
So the added inputs cost about a point on ordinary pairs.

### (b) The 850 launch pairs (same account, both launch-type, test period; built as `model_check.py` builds them)

My pair construction reproduces the launch track's file exactly (850 pairs, 63.65%, 73.3% confident). Confidence
uses the baseline's scale 1.5156 for every model, as the launch track did.

| Real gap at least | Pairs | Baseline correct | confident share (pairs), correct | Retrained correct | confident share (pairs), correct | Baseline + correction | Six |
|---|---|---|---|---|---|---|---|
| 1.25x | 850 | 63.9% (59.7-68.1) | 26% (218), 73.9% | 63.1% (58.5-68.0) | 28% (237), 73.8% | 63.8%, 25% (213), 73.7% | 63.2%, 26% (217), 75.6% |
| 2x | 568 | 66.5% | 28% (159), 78.6% | 64.1% | 31% (177), 74.6% | 66.5%, 27% (155), 78.1% | 64.4%, 28% (159), 78.0% |
| 3x | 401 | 68.3% | 31% (123), 78.9% | 64.8% | 32% (129), 76.0% | 68.3%, 29% (118), 78.0% | 65.8%, 30% (121), 78.5% |
| 5x | 240 | 70.0% | 32% (77), 74.0% | 63.7% | 34% (81), 72.8% | 69.6%, 31% (75), 73.3% | 67.9%, 32% (78), 74.4% |

By confidence band (pairs, correct):

| Stated | Baseline | Retrained | Baseline + correction |
|---|---|---|---|
| 50 to 55% | 393, 55.0% | 372, 55.1% | 399, 54.6% |
| 55 to 60% | 239, 69.5% | 241, 64.7% | 238, 70.2% |
| 60 to 70% | 163, 76.1% | 181, 72.4% | 158, 76.0% |
| 70%+ | 55, 67.3% | 56, 78.6% | 55, 67.3% |

Paired differences against the baseline ensemble: retrained -0.8 (-4.2 to +2.3) overall, -2.5 (-6.7 to +1.7) at 2x;
confident calls 0.0 (-6.8 to +4.9). Single models: ptb_s1 62.1%, ptb_s2 61.5%, ptbw10_s1 63.9% (ts46_s1 63.8%).
None of these differences is distinguishable from zero on 850 pairs from 151 accounts.

Pairs differing in one trait (share of pairs in which the model prefers the version with it; what really won):

| Trait | Pairs | It won | Baseline | Retrained | ptb_s1 | ptb_s2 | ptbw10_s1 |
|---|---|---|---|---|---|---|---|
| First line names the product | 264 | 73.9% | 61.4% | 68.9% | 66.3% | 70.1% | 65.5% |
| Video | 220 | 50.5% | 67.7% | 74.1% | 66.8% | 75.5% | 73.6% |

### (c) Predicted against actual level by post type (draft path, test period)

Each post scored at its own time against the account's previous 56 posts. "x" = median of likes over the median
likes of the account's previous 30 posts (the launch track's definition). Types are the launch track's labels.
"Other" is a control sample of ordinary posts by the same accounts.

| Type | Posts | Actual | Baseline | Baseline + correction | Retrained | ptb_s1 | ptb_s2 | ptbw10_s1 | Share beating own median: actual / baseline / retrained |
|---|---|---|---|---|---|---|---|---|---|
| Launch | 1,080 | 2.19x | 1.10x | 1.50x | 1.50x | 1.52x | 1.67x | 1.36x | 75% / 58% / 74% |
| Funding | 44 | 6.07x | 1.54x | 3.36x | 3.88x | 3.44x | 4.04x | 4.61x | 93% / 77% / 95% |
| Feature | 435 | 1.13x | 0.91x | 0.94x | 0.97x | 1.04x | 1.00x | 0.91x | 57% / 42% / 47% |
| Hiring | 103 | 1.77x | 1.11x | 1.16x | 1.19x | 1.20x | 1.15x | 1.12x | 63% / 60% / 59% |
| Open source | 32 | 1.47x | 1.15x | 2.01x | 2.26x | 2.51x | 1.94x | 1.97x | 62% / 59% / 91% |
| Other (control) | 600 | 1.00x | 0.81x | 0.81x | 0.83x | 0.85x | 0.85x | 0.81x | 47% / 32% / 36% |

Mean absolute error in log likes on launch posts: baseline 1.14, baseline + correction 1.02, retrained 1.01; on
funding 1.42, 0.82, 0.73. Rank correlation of predicted and actual lift among launch posts: 0.48, 0.48, 0.50.
Open-source releases are over-corrected by both fixes (32 posts; 91 in training). Relative to the control sample
(which removes the general drift), launch is predicted at 1.81x by the retrained models and 1.85x by the corrected
baseline against 2.19x actual.

The same comparison inside the feature table (`table_calibration_*.json`, lift against the median of the visible
history), which shows the period shift:

| | Train period: actual | ts46_s1 | ptb_s1 | ptbw10_s1 | Test period: actual | ts46_s1 | ptb_s1 | ptbw10_s1 |
|---|---|---|---|---|---|---|---|---|
| Launch (1,757 / 753 posts) | 1.77x | 1.36x | 1.52x | 1.65x | 2.81x | 1.53x | 1.65x | 1.76x |
| Funding (94 / 31) | 3.63x | 1.61x | 2.92x | 3.46x | 6.30x | 1.98x | 3.93x | 4.07x |
| Feature (691 / 270) | 1.13x | 1.17x | 1.18x | 1.18x | 1.18x | 1.27x | 1.24x | 1.24x |
| Launch pairs in the table | 6,991 | 65.3% | 67.4% | 71.6% | 985 | 65.8% | 63.6% | 65.0% |

Weighting own announcements 10x fits the training period (launch 1.65x of 1.77x, funding 3.46x of 3.63x, launch
pairs 71.6%) and does not carry to the test period (launch pairs 65.0%, below the baseline's 65.8%): the extra fit is
memorised, not learned.

## What was tried, in order

1. **Inputs alone** (ptb_s1, ptb_s2): level halfway, pairs not better. Above.
2. **Upweighting own announcements 10x** (ptbw10_s1): best single model on funding level (4.61x) and equal to the
   baseline on launch pairs (63.9%), worse on launch level (1.36x) and the worst on the general benchmark (-1.7).
3. **Type-conditioned correction head** (`scripts/posttype/fit_correction.py`): ridge regression of the baseline
   ensemble's error in log10 likes on the seven type columns, 3,990 own announcements of 1,499 accounts before the
   cutoff. Weights (log10): funding +0.339, open source +0.241, launch +0.133, milestone +0.052, demo +0.044,
   hiring +0.017, feature +0.015. Fitted on half the accounts and measured on the other half, it takes the mean
   error from +0.105 to +0.003. A version with the eight launch-trait terms was fitted once, before the type bug below was
   fixed, and not refitted: the trait weights were all under 0.05 except video (+0.087), it moved the video
   preference further the wrong way and gave no pair gain, so the shipped correction has types only.
4. **Six-model ensemble** (baseline + retrained): +0.4 on the general benchmark and -0.7 on launch pairs, both inside
   their intervals; level in between (launch 1.28x). Not worth six forward passes.

A first pair of models (pt_s1, ptw10_s1) was trained with a bug in my feature code: the company-voice flag was never
true, so 30% of company-account launches (the ones without "we/our", which got 3.06x) were typed as ordinary posts.
Found by comparing types with the launch track on the test posts, fixed, table rebuilt, checks rerun, models
retrained. Those two models are not in the results and not uploaded.

## What to ship

1. **The correction, on the current models.** `DraftScorer(checkpoints, embed_fn, sid_codebook=..., type_correction=
   "research/tracks/posttype/type_correction.json")`, or `xviral.app.service.load(..., type_correction=...)`. It shifts
   the likes forecast (q10, q50, q90 and "x times your usual") of a draft the rules type as an own announcement and
   returns the shift as `type_correction_log10`; every other draft is untouched, so the pair benchmark and its
   confidence scale stay valid. Verified through the serving path on 103 launch posts: served value equals the
   ensemble score plus the correction to within 0.0006 in log10. It is fitted to these three checkpoints; refit with
   `fit_correction.py` for any other set. The views forecast is not corrected (view counts exist for 205 launch
   posts).
2. **Not the retrained checkpoints.** They are uploaded under `checkpoints/` for reproduction (ptb_s1, ptb_s2,
   ptbw10_s1; they need the patch, since their inputs include the 34 columns), but they are about a point worse on
   ordinary pairs, make fewer confident calls, and give no more than the correction on level.
3. `patch.diff` is the exact patch: `xviral/features/post_type.py` (new), `xviral/model/features.py` (3 lines),
   `xviral/serve.py` (post-type columns in `feature_frame`, correction, reuse of stored embeddings),
   `xviral/app/service.py` (the `type_correction` argument), `scripts/evidence/tac_serve_eval.py` (`--emb-table`,
   frees the corpus after choosing pairs), `scripts/posttype/*`. Old checkpoints are unaffected: a model reads only
   the columns named in its own metadata. Existing tests: 113 pass; 3 in `tests/test_serve.py` fail in this checkout
   with or without the patch (missing `xviral/data/edit_evidence.json` and a missing module).

## Honest limits

- The level is still a third short for launch and funding after either fix, because launch posts did better after
  the cutoff than before it. The correction should be refitted whenever the models are retrained on newer data.
- 44 funding posts and 32 open-source posts in the test period: their rows are rough. Open source is over-corrected.
- Launch pairs: 850 pairs from 151 accounts; an interval of about +/- 4.5 points. A real difference of 2 points in
  either direction would not show.
- The correction is a constant per type. It does not know which launch will do well; ranking inside launch posts is
  exactly the baseline's.
- The account kind read from 50 previous posts disagrees with the launch track's whole-timeline label on 19% of
  company-account rows; a product that knows the account is a company could pass that in and skip the estimate
  (not implemented).
- The control sample shows the model under-predicting ordinary posts by about 20% in the test period against the
  30-post median. That is outside this track and affects every "x times your usual" number.
- One seed of the weighted variant; 4,000 steps as the baseline; no search over the weight (10x only).

## Why ranking did not move (for the next track)

- The signal is thin: 2,056 launch rows in training, and the launch track's own rule score fitted before the cutoff
  ranked the test pairs at 55.7%.
- The strongest pre-cutoff trait, video, stopped working after the cutoff; a model given explicit trait inputs
  learns it harder (74% preference against 68%).
- 37 extra inputs for 1.2% of rows cost about a point on the other 98.8%. A narrower set (the seven type columns
  and the trailing record only, no traits) would be the next thing to try if retraining is wanted; the correction
  already covers what the type columns can do for the level.

## Reproduce

From `virality/` with the data in `data/` (see `scripts/worker_job.sh`), all under `scripts/posttype/`:
`build_table.py`, `check_features.py`, `weight_table.py --weight 10`, then `scripts/train_one.py --table
data/work/model/table_pt --features ts46 --seed 1` (and seed 2, and `table_pt_w10`), `score_posts.py --checkpoint
name=path ... --shard i/4` (four single-threaded processes; the feature cache is shared by all models),
`fit_correction.py`, `apply_correction.py`, `eval_launch.py`, `bench_shard.sh <slice> name=checkpoint ...` for slices
0 to 5 (set `POSTS` to a file holding only the accounts with pairs to stay within 15 GB), `bench_correction.py`,
`bench_table.py`, `table_calibration.py`.
