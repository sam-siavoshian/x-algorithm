# Track "expand": the company and founder accounts of the new sources, added to training

2026-11-16. Everything is in the dataset under `tracks/expand/`. Code changes are in `patch.diff` (nothing committed).

## Headline

- **Built.** 293,036 posts of 1,566 company and founder accounts (1,202 of them new to training) from the three converted sources, taken
  through the pipeline's own stages and appended to the 465,675 existing rows: 758,711 rows, 8,423 accounts. By source: twitter100m 248,472
  (likes only, 2020 to April 2023), the brand set 38,368 (all with views, 2023 to 2025), root posts of the 2026 view corpus 6,196 (all with
  views). 44,589 added rows have views; 2,119 are launch-type own posts (520 with views); 129,586 come from accounts under 10,000
  followers. The existing rows, their split and the benchmark posts are untouched.
- **Levels.** Like counts agree between sources (the same post read by two sources: median ratio 1.08 to 1.09). Account levels do not:
  the Community Archive holds posts that were already travelling, so the same account shows about four times the likes there that it
  shows in its full twitter100m timeline (67 accounts, median ratio 0.24), and at equal follower counts archive posts carry 3 to 10 times
  the likes. One model family therefore gets an input that says how the source sampled the account (`src_regime`).
- **Result: adding the data neither helps nor hurts.** Two seeds each, the control retrained under the same conditions (it reproduces the
  stored baseline bit for bit):

| | Control | Expanded | Expanded + sampling input |
|---|---|---|---|
| Ordinary same-account pairs, likes, existing test rows (683,152 pairs) | 59.97% | 59.77%, -0.19 (-0.5 to +0.1) | 59.89%, -0.08 (-0.4 to +0.2) |
| The same, views (24,481 pairs) | 62.05% | 61.98%, -0.07 (-1.0 to +1.2) | 61.99%, -0.06 (-1.1 to +1.5) |
| The same, company and founder accounts, likes (34,831 pairs, 470 accounts) | 64.19% | 64.49%, +0.30 (-0.3 to +0.8) | 64.91%, +0.73 (+0.2 to +1.3) |
| The same, accounts under 10,000 followers, likes (335,638 pairs) | 57.87% | 57.85%, -0.02 (-0.5 to +0.4) | 57.87%, -0.01 (-0.5 to +0.4) |
| Benchmark, 2,851 same-topic pairs 1.25x apart, two-seed ensemble | 59.5% | 59.8%, +0.3 (-1.4 to +1.7) | 59.1%, -0.3 (-1.9 to +1.2) |
| Benchmark, confident calls on pairs 2x apart (2,071 pairs): share, correct | 25%, 69.6% | 23%, 69.8% | 25%, 70.3% |
| Launch pairs (850 pairs, 151 accounts), two-seed ensemble | 63.4% | 63.8%, +0.4 (-2.7 to +3.1) | 64.7%, +1.3 (-1.8 to +4.2) |
| Launch pairs, company accounts (601) / founder accounts (119) | 63.6% / 64.7% | 63.6% / 65.5% | 64.7% / 66.4% |
| Predicted level of a launch post over the account's usual (actual 2.19x, 576 posts) | 0.98x | 1.00x | 1.14x |
| Accounts held out from twitter100m, likes (427,248 pairs, 134 accounts) | 57.39% | 60.66%, +3.27 (+2.2 to +4.4) | 61.04%, +3.65 (+2.2 to +4.9) |

  Mean of the two seeds for the table rows, differences against the control in points with a 95% interval from resampling accounts.
- **With the pair-loss fine-tune on top** (the pairs track's recipe, applied because the expanded models are level with the control):
  benchmark 60.4% against 60.2% for the fine-tuned control (+0.2, -1.2 to +1.6), 59.7% with the sampling input (-0.6, -2.1 to +0.8);
  launch pairs 65.6% and 64.8% against 63.8% (+1.9, -1.5 to +5.2; +1.1, -1.7 to +3.7); confident calls on benchmark pairs 2x apart
  31% and 29% of pairs at 70.5% and 71.1%, against 32% at 71.3%. Still level.
- **Nearly twice the steps does not change it.** One expanded model trained 7,450 steps (as many passes over the data as the control gets):
  benchmark 58.8% against 59.9% for its control seed (-1.1, -2.7 to +0.7), launch pairs 63.5% against 63.8%, existing test pairs -0.06
  (-0.4 to +0.2). So the expanded models are not held back by seeing each existing row less often.
- **Where the added data does show.** The expanded models rank posts of held-out accounts of the new kinds 3.3 points better, and they
  forecast a full timeline at the right level where the existing models are about 1.5x too high (held-out brand accounts: predicted
  1.45x to 1.58x the account's own median against an outcome of 0.95x; expanded 0.92x to 1.02x; seven accounts). A customer's own account is a
  full timeline. With the sampling input there is also a small gain on existing company and founder accounts (+0.7 points, 0.2 to 1.3;
  one slice of about twenty).
- **Not fixed by volume.** Small accounts of 2026 (no change), the launch level (0.98x to 1.14x against 2.19x), same-topic pairs.
  85% of what public data adds is likes-only, before 2023, from a loosely defined list of organisations; only 6,196 added rows are from 2026.

## 1. The expanded training set

Code: `scripts/expand/build_rows.py` (rows), `scripts/expand/added_report.py` (the tables below), `scripts/expand/source_levels.py` (section 2).
Files: `rows/new_rows.parquet` (the 293,036 added rows, same columns as `work/dataset/rows.parquet` plus `expand_src`, `acct_kind`,
`kind_from`, `in_brand_set`, `acct_in_original`), `rows/build_counts.json`, `rows/filter_log.csv`, `rows/added.md` and `.json` (every cut,
for the new and the existing rows), `rows/typed_posts.parquet` (post types). The expanded set is the existing rows, unchanged, plus these.

**Stages.** The same functions as `xviral/pipeline.py`, called in the same order on the new posts only:

| Stage | What ran | Rows after (twitter100m / zenodo_brands / nitter_2026_replies) |
|---|---|---|
| Account selection, before any text is read | handle in `startup_accounts.parquet` (19,320), minus 373 handles of the benchmark's calibration accounts; narrow scan first | 740,571 / 523,246 / 44,831 rows of 1,502 / 182 / 3,018 accounts |
| Full-width read | originals and quotes from 2020 on (the kind and recency filters, pushed into the scan) | 455,028 / 218,699 / 30,632 |
| Unify | `unify.merge_posts`: one row per post id (6,038 rows shared an id between twitter100m and the brand set); 4,416 posts already training rows dropped | 708,775 unique |
| Validity | `filters.validity` | 455,028 / 218,699 / 30,632 |
| Maturity | 72 hours after posting; the 2026 view corpus only before 2026-05-01 (its conversion marks every row mature, the reader's rule was re-applied) | 453,462 / 218,690 / 12,695 |
| Recency | 2020 for twitter100m, 2023 for the rest (config) | 453,462 / 94,002 / 12,695 |
| Language | the pipeline's detector, English kept | 420,127 / 80,448 / 11,200 |
| Volume | 20 posts, or already a training account | 417,824 / 80,329 / 8,770 |
| Authors | followers at post time, trailing statistics, self-repeats, author table: over the account's new posts together with its 31,325 posts in `work/filtered/posts.parquet` | |
| Hygiene | the pipeline's automated-account rules: 118 accounts dropped, nearly all for more than 20 self-repeated texts (3 for 200 posts in a day, 3 for one link domain), 83,730 posts | 342,367 / 74,548 / 6,278 |
| Cap | the most recent 400 new posts per account (the audience cap of the existing run) | 248,447 / 38,393 / 6,196 |
| Near-duplicates | `dedup.near_duplicate_clusters` on the selected new posts: 4,297 of 274,407 in a cluster | |
| Features, labels | the feature stage's joins and functions, `labels.build` (recency weights 0.25 to 2023, 0.5 for 2024, 1 after) | |
| Split | test from 2026-04-23; an account already in training stays on its side; a tenth of the new accounts is validation | train 261,152, validation 31,297, test 587 |

**Check that this is the pipeline.** The same code was run on 150 existing accounts from their posts in `work/filtered/posts.parquet` and compared
with their 10,276 stored training rows: every feature column agrees in at least 99.5% of rows (`rows/replication_check.json`). Not reproduced:
`reply_share` for existing accounts (the activity table is not in the shared data; new accounts get it from the new sources' own reply rows,
existing accounts keep their stored value) and the near-duplicate clusters, which the existing run computed over its own rows.

**Memory.** The first build was killed three times. Two causes: reading 1.3M rows at full width before filtering (8 GB; now the narrow scan
selects accounts and the full-width read applies the kind and recency floor in the scan, 5 GB peak with the feature stage run in chunks of
40 accounts), and a real defect in my first version: two handles of one account gave the author table two rows for it, the join duplicated
its posts, and the feature stage's self-joins multiplied them. An assertion now guards the author table, and the output is checked to have
one row per post.

**Kept out of training.** No new row is a training row already, none belongs to an account with a benchmark calibration pair (asserted in
the script), and every new row from 2026-04-23 on is in the test split. The existing rows, their splits and their arrays are untouched
(the table builder asserts the old arrays are bit-identical inside the new table).

### What was added

| | Posts | Accounts | With views | Train | Validation | Test | Launch-type own posts | of them with views |
|---|---|---|---|---|---|---|---|---|
| Existing rows | 465,675 | 7,221 | 154,660 | 302,696 | 32,131 | 130,848 | 3,186 | 297 |
| Added | 293,036 | 1,566 | 44,589 | 261,152 | 31,297 | 587 | 2,119 | 520 |
| twitter100m (2020 to April 2023) | 248,472 | 1,088 | 25 | 219,969 | 28,503 | 0 | 1,599 | 0 |
| zenodo_brands (2023 to 2025) | 38,368 | 133 | 38,368 | 36,633 | 1,735 | 0 | 401 | 401 |
| nitter_2026_replies, root posts (2026) | 6,196 | 561 | 6,196 | 4,550 | 1,059 | 587 | 119 | 119 |

1,202 of the 1,566 accounts are new to the training set; 364 were in it already and gain 22,195 older or newer posts (67 of them 20,184 twitter100m posts).

By account kind (the `kind` of `startup_accounts.parquet`):

| Added | Posts | Accounts | With views | Launch-type own posts | of them with views |
|---|---|---|---|---|---|
| company | 186,774 | 888 | 41,848 | 1,730 | 480 |
| founder | 106,262 | 678 | 2,741 | 389 | 40 |
| company, twitter100m | 144,951 | 646 | 25 | 1,250 | 0 |
| founder, twitter100m | 103,521 | 442 | 0 | 349 | 0 |
| company, zenodo_brands | 38,368 | 133 | 38,368 | 401 | 401 |
| company, nitter_2026_replies | 3,455 | 218 | 3,455 | 79 | 79 |
| founder, nitter_2026_replies | 2,741 | 343 | 2,741 | 40 | 40 |

For scale: the existing rows hold 13,750 posts of 243 company accounts and 33,251 of 516 founder accounts (launch track labels, then the startup list).

By follower tier at post time:

| Tier | Added posts | Accounts | With views | Launch-type own posts | Existing posts | Existing launch-type |
|---|---|---|---|---|---|---|
| under 100 | 14,739 | 142 | 51 | 79 | 2,959 | 4 |
| 100 to 1k | 39,703 | 262 | 536 | 262 | 31,966 | 49 |
| 1k to 10k | 75,144 | 352 | 6,875 | 462 | 103,857 | 296 |
| 10k to 100k | 93,021 | 413 | 22,589 | 653 | 134,659 | 909 |
| 100k to 1M | 48,908 | 208 | 7,657 | 411 | 57,243 | 591 |
| 1M and over | 19,223 | 64 | 4,583 | 170 | 10,879 | 278 |
| unknown | 2,298 | 137 | 2,298 | 82 | 124,112 | 1,059 |

Posts from accounts under 10,000 followers go from 138,782 to 268,368, and launch-type posts there from 349 to 1,152. All but 80 of the added
ones are twitter100m: likes only, 2020 to April 2023.

By post type (`xviral/text/launch_labels.py`, own announcements only):

| Type | Added | Accounts | With views | Existing |
|---|---|---|---|---|
| launch | 2,105 | 624 | 516 | 3,069 |
| open-source release | 14 | 10 | 4 | 117 |
| feature release | 638 | 293 | 80 | 1,133 |
| hiring | 210 | 121 | 10 | 336 |
| funding | 44 | 35 | 1 | 163 |
| demo | 42 | 27 | 42 | 385 |
| milestone | 16 | 15 | 1 | 51 |
| not an own announcement | 289,967 | 1,564 | 43,935 | 460,421 |

**Who these accounts are.** The startup list is loose where its only evidence is the bio, and that is where most of the volume comes from:

| Evidence for the account | Posts | Accounts |
|---|---|---|
| Bio only | 139,357 | 702 |
| Two or more own announcements, no list or label | 70,700 | 250 |
| YC or Product Hunt list, launch-track label, or brand set | 82,979 | 614 |

The bio-only "companies" in twitter100m include NASA, CNN Breaking News, The New York Times, Netflix, a US senator, game studios and local
businesses next to OpenAI, Coinbase, Infisical and Obsidian. The hygiene stage removed the wire-style ones (ABC, Forbes, Sky News, Yahoo News:
dozens of repeated texts). What is left is "organisations and people with a product", not "startups". The data track measured this too
(company bios: about 7 of 14 are product companies). The brand set is consumer hardware and app brands.

## 2. Are outcome levels comparable between sources?

`scripts/expand/source_levels.py`, `rows/source_levels.json`. Two different questions, with opposite answers.

**The like counts themselves are comparable.** The same post read by two sources:

| Same post in | Posts | Median ratio of the two like counts | Within 1.25x |
|---|---|---|---|
| twitter100m (read April 2023) and the Community Archive | 3,046 | 1.09 | 87% |
| twitter100m (read April 2023) and the brand set (read December 2025) | 3,680 | 1.08 | 99.5% |
| the brand set and the existing corpus | 30 | 1.01 | 77% |

twitter100m reads 8 to 9% higher than a later reading of the same post, at every age of the post at its snapshot (3 to 14 days: 1.08; 14 to
60 days: 1.10; over 60 days: 1.08), so its counts were mature; the likely cause is likes lost later to deleted accounts. That is 0.035 in
log10 and was left alone.

**The level of an account is not comparable, because the sources sample an account differently.** The Community Archive rows of most
accounts are posts that showed up in archive users' timelines (DATA_PIPELINE.md 15.2): posts that were already travelling. twitter100m and
the brand set are whole timelines.

| Comparison | Accounts or posts | Result |
|---|---|---|
| One account on both sides: median likes of its last 30 twitter100m posts over its first 30 existing rows | 67 accounts (median gap between the two blocks 47 days) | 0.24 (a tenth of accounts under 0.07, a tenth over 0.67); 0.21 for the 41 accounts whose blocks are within 180 days |
| Likes per follower, accounts of 1k to 10k followers: twitter100m against the Community Archive | 68,269 and 78,107 posts | 0.28x (log10 medians -2.88 and -2.33) |
| the same, 10k to 100k followers | 70,457 and 97,335 posts | 0.10x (-3.34 and -2.32) |
| the same, 100k to 1M | 41,251 and 39,455 posts | 0.11x (-3.54 and -2.57) |
| Likes per follower, 10k to 100k: brand set against the 2026 view corpus | 21,365 and 37,324 posts | 0.23x (-3.57 and -2.93) |
| Views per follower, 10k to 100k: brand set against the 2026 view corpus | same | 0.39x (-1.51 and -1.11) |
| Likes per view, 10k to 100k: brand set against the 2026 view corpus | same | 0.58x (-2.05 and -1.81) |

So the same account looks about four times bigger in the archive than in its full timeline, and at a given follower count the
archive's posts carry 3 to 10 times the likes of full timelines. The brand accounts sit lower again on every ratio (views per follower,
likes per view): that is who they are (large passive followings), not how they were read. The follower count of twitter100m is one reading
from April 2023 applied to posts back to 2020, which lowers likes per follower for older posts of accounts that grew.

**What was done about it.**

1. Period: the pipeline's recency weights apply as they do to existing rows (0.25 up to 2023, 0.5 for 2024, 1 from 2025), so the
   twitter100m rows count a quarter each in the loss.
2. Sampling: a model cannot tell a twitter100m row from an archive row (both have likes, no views, a follower count). One model family
   therefore gets a new categorical input, `src_regime` (stream sample: Community Archive; search sample: the 2026 view corpus; full
   timeline: twitter100m, the brand set, and what a customer's own scrape or analytics export would be). A draft takes the regime of its
   account's history (`model.features.with_src_regime`), so the benchmark's drafts are scored as stream-sampled posts, which they are.
   The other family is trained without the input, to show what the input does.
3. Not corrected: the 67 accounts whose history crosses from twitter100m into the archive carry the fourfold step inside one history
   (20,184 of the 293,036 added rows). The existing pipeline already joins archive and view-corpus posts of one account the same way.
4. Inputs that mark the source without carrying information in the existing rows were blanked in the new rows of the table: the verified
   flag (null in every existing row) and video durations (no existing source has them; a video is "duration unknown" there). Left in,
   they would enter on an unfitted scale and tell the model which source a row came from. `new_rows.parquet` keeps the real values.

## 3. The table

`scripts/expand/embed_rows.py`, `scripts/expand/build_table.py`; the table itself is in `table/` (1.3 GB).

- **Embeddings.** BAAI/bge-small-en-v1.5 on the same rendered text (`features.render_texts`), bf16 autocast. Checked first on 2,000 rows
  of the existing table: cosine to the stored vectors 0.99999 on average, 0.99994 at worst. 293,036 posts in 26 minutes on 3 threads (189 posts a second).
- **Transforms.** The new rows go through `features._arrays` with the existing table's stored means, deviations, missing-indicator
  columns and categorical layout. The two tables share one input space: a model trained on either can be scored on either, and on the
  benchmark's cached features, without refitting anything.
- **Existing rows.** Copied array by array from the old table; the builder asserts they are bit-identical in the new one.
- **Semantic ids.** Assigned with the existing codebook; on 3,000 existing rows the recomputed ids equal the stored ones in all cases.
- **Neighbours.** For the new rows, `neighbors.global_neighbors` over all 758,711 rows (whole days strictly before the post; 293,002 of
  the new rows have at least one); existing rows keep their stored neighbours, so they never point at a new row.
- **Result.** 758,711 rows, 8,423 accounts; train 563,848, validation 63,428, test 131,435; 199,249 rows with views; 564,884 audience rows
  (the brand accounts are flagged as general accounts, as the data track advised; every other added account is audience, as seed handles
  are in `filters.account_selection`). Regimes: 311,015 stream sample, 160,856 search sample, 286,840 full timeline.
- **Validation rows differ between the runs.** Model selection inside a run uses the audience validation rows of its table: 20,684 for
  the control, 50,229 for the expanded runs (the validation accounts of the added sources come on top). Test metrics are always on the same rows.

## 4. Models

All models: `scripts/train_one.py`, small preset, the 46 text-structure columns (`--features ts46`), 4,000 steps, batch 64, learning rate 1e-3,
CPU, seeds 1 and 2. Same machine, same code, one job at a time.

| Name | Trained on | Rows it trains on | Note |
|---|---|---|---|
| `ctl_s1`, `ctl_s2` (Control) | the existing table | 302,696 | `ctl_s1` was retrained here and is bit-identical to the stored `runs/ts46_s1` (largest weight difference 0.0, same metrics), so training is deterministic on these machines and the stored `runs/ts46_s2` is used as `ctl_s2` without retraining |
| `exp_s1`, `exp_s2` (Expanded) | the expanded table without `src_regime` | 563,848 | the plain answer to "does adding the rows help" |
| `expsrc_s1`, `expsrc_s2` (Expanded + sampling input) | the expanded table with `src_regime` | 563,848 | the answer to section 2 |
| `explong_s1` | as Expanded, 7,450 steps | 563,848 | the same number of passes over the data as the control gets in 4,000 steps; one seed |
| `ctlft_*`, `expft_*`, `expsrcft_*` | each of the above fine-tuned with the pair loss | | section 5 |

A two-seed "ensemble" below is the mean of the two members' scores (log10 predicted likes), as in `compare_runs.py`. The README baseline is a
three-model ensemble (the two control seeds plus `compare_v2/full`); `v2full` is listed alone on the benchmark for reference.

**How each evaluation is run.**

- *Ordinary same-account pairs*: `model.evaluate` on table rows, each row scored alone with its own history; pairs are two posts of one
  account within 7 days whose outcomes differ by 0.1 in log10. Every model is scored on the same rows of the expanded table
  (`scripts/expand/eval_table.py`); for the control this reproduces its own test metrics exactly (59.81% on 683,152 pairs for `ctl_s1`).
  Differences carry a 95% interval from resampling accounts, pooled over the two seeds.
- *The benchmark*: the fixed 3,530 pairs, `scripts/bench_pairs.sh` with the pairs track's feature caches, table and intervals from
  `scripts/evidence/gap_table.py` (confidence scale cross-fitted over accounts).
- *Launch pairs and level*: the launch track's 582 test-period launch posts scored through the draft path with 56 posts of history
  (`scripts/expand/score_launch.py`, the launch track's scorer plus a feature cache; my run of the baseline ensemble reproduces its stored
  scores: 850 pairs, 63.65%, predicted level 1.03x against 2.19x). "Confident" uses the launch track's scale (1.5156), so the column means
  what it means in its report.

Slices of the table evaluation: `orig_test_audience` is the test set every existing model reports (audience accounts, existing rows, from
2026-04-23). `_company`, `_founder`, `_startup` restrict it to accounts labelled by the launch track or, failing that, the startup list;
`_under_10k` to accounts with a known follower count under 10,000. `new_val_<source>` are validation accounts of an added source:
accounts no model trained on, all before the cutoff. `new_test` are the 587 added rows from the test period. `*_launch_posts` are the
launch-type own posts inside a slice.

#### Ordinary same-account test pairs, each model's own run (existing audience test rows)

| Model | Rows | Likes pairs | Likes correct | Views pairs | Views correct | Within-account rho likes | Like NLL | Best step |
|---|---|---|---|---|---|---|---|---|
| ctl_s1 | 95,357 | 683,152 | 59.81% | 24,481 | 62.26% | 0.2963 | 2.9662 | 3330 |
| ctl_s2 | 95,357 | 683,152 | 60.12% | 24,481 | 61.84% | 0.2993 | 2.9634 | 3996 |
| exp_s1 | 95,357 | 683,152 | 59.63% | 24,481 | 61.94% | 0.2985 | 2.9666 | 4000 |
| exp_s2 | 95,357 | 683,152 | 59.92% | 24,481 | 62.03% | 0.3014 | 2.9678 | 4000 |
| expsrc_s1 | 95,357 | 683,152 | 59.68% | 24,481 | 61.65% | 0.2956 | 2.9679 | 4000 |
| expsrc_s2 | 95,357 | 683,152 | 60.09% | 24,481 | 62.33% | 0.3039 | 2.9741 | 4000 |
| explong_s1 | 95,357 | 683,152 | 59.75% | 24,481 | 62.21% | 0.2992 | 2.9538 | 7450 |

#### Same-account pairs by slice, likes (all models on the same rows of the expanded table)

| Slice | Rows | Accounts with pairs | Pairs | Control (mean of seeds) | Expanded (mean of seeds) | Expanded + sampling input (mean of seeds) | Expanded 7,450 steps (mean of seeds) | Expanded minus Control, points | Expanded + sampling input minus Control, points | Expanded 7,450 steps minus Control, points |
|---|---|---|---|---|---|---|---|---|---|---|
| orig_test_audience | 95,357 | 2,846 | 683,152 | 59.97% [59.8%, 60.1%] | 59.77% [59.6%, 59.9%] | 59.89% [59.7%, 60.1%] | 59.75% [59.7%] | -0.19 (-0.5 to +0.1) | -0.08 (-0.4 to +0.2) | -0.06 (-0.4 to +0.2) |
| orig_test_company | 3,160 | 130 | 11,184 | 64.92% [65.2%, 64.7%] | 65.61% [65.9%, 65.4%] | 65.71% [65.7%, 65.7%] | 66.01% [66.0%] | +0.69 (-0.5 to +1.7) | +0.79 (+0.3 to +1.7) | +0.85 (-0.7 to +1.6) |
| orig_test_founder | 7,717 | 340 | 23,647 | 63.84% [63.3%, 64.4%] | 63.96% [63.1%, 64.8%] | 64.54% [64.2%, 64.9%] | 64.42% [64.4%] | +0.12 (-0.6 to +0.8) | +0.70 (-0.1 to +1.5) | +1.15 (-0.1 to +2.1) |
| orig_test_startup | 10,877 | 470 | 34,831 | 64.19% [63.9%, 64.5%] | 64.49% [64.0%, 65.0%] | 64.91% [64.7%, 65.2%] | 64.93% [64.9%] | +0.30 (-0.3 to +0.8) | +0.73 (+0.2 to +1.3) | +1.05 (+0.1 to +1.8) |
| orig_test_under_10k | 38,289 | 854 | 335,638 | 57.87% [57.7%, 58.1%] | 57.85% [57.8%, 58.0%] | 57.87% [57.7%, 58.0%] | 57.77% [57.8%] | -0.02 (-0.5 to +0.4) | -0.01 (-0.5 to +0.4) | +0.09 (-0.3 to +0.5) |
| orig_test_startup_under_10k | 1,153 | 65 | 2,252 | 63.12% [63.3%, 63.0%] | 63.81% [63.2%, 64.4%] | 64.25% [64.3%, 64.2%] | 63.45% [63.5%] | +0.69 (-1.2 to +1.5) | +1.13 (-1.1 to +2.1) | +0.18 (-1.8 to +1.8) |
| new_test | 587 | 85 | 2,155 | 62.09% [61.5%, 62.7%] | 62.65% [62.4%, 62.9%] | 61.58% [61.0%, 62.1%] | 61.72% [61.7%] | +0.56 (-2.1 to +4.6) | -0.51 (-4.1 to +4.5) | +0.23 (-3.8 to +5.7) |
| orig_test_launch_posts | 736 | 44 | 92 | 65.22% [68.5%, 62.0%] | 63.59% [66.3%, 60.9%] | 65.76% [67.4%, 64.1%] | 61.96% [62.0%] | -1.63 (-7.6 to +4.4) | +0.54 (-6.1 to +6.6) | -6.52 (-14.7 to +2.0) |
| orig_test_launch_posts_startup | 545 | 39 | 85 | 64.71% [68.2%, 61.2%] | 62.94% [65.9%, 60.0%] | 65.88% [68.2%, 63.5%] | 62.35% [62.4%] | -1.76 (-8.2 to +4.5) | +1.18 (-5.8 to +7.4) | -5.88 (-14.3 to +2.6) |
| new_val_launch_posts | 244 | 16 | 32 | 59.38% [62.5%, 56.2%] | 62.50% [62.5%, 62.5%] | 64.06% [62.5%, 65.6%] | 65.62% [65.6%] | +3.12 (-2.8 to +8.8) | +4.69 (-6.2 to +14.9) | +3.12 (-16.0 to +19.2) |
| new_val_launch_posts_company | 215 | 14 | 30 | 56.67% [60.0%, 53.3%] | 60.00% [60.0%, 60.0%] | 61.67% [60.0%, 63.3%] | 63.33% [63.3%] | +3.33 (-3.2 to +9.1) | +5.00 (-6.9 to +15.2) | +3.33 (-18.2 to +20.8) |
| new_val_twitter100m | 28,503 | 134 | 427,248 | 57.39% [57.1%, 57.6%] | 60.66% [61.0%, 60.3%] | 61.04% [61.2%, 60.9%] | 60.84% [60.8%] | +3.27 (+2.2 to +4.4) | +3.65 (+2.2 to +4.9) | +3.71 (+2.2 to +5.4) |
| new_val_twitter100m_under_10k | 14,828 | 86 | 103,550 | 60.40% [60.1%, 60.7%] | 62.65% [63.2%, 62.1%] | 62.74% [63.3%, 62.1%] | 63.33% [63.3%] | +2.26 (+0.8 to +3.8) | +2.34 (+0.8 to +4.1) | +3.20 (+1.0 to +5.0) |
| new_val_zenodo_brands | 1,735 | 7 | 14,240 | 51.67% [51.6%, 51.7%] | 55.05% [55.2%, 54.9%] | 55.46% [55.0%, 56.0%] | 57.49% [57.5%] | +3.37 (-0.7 to +5.7) | +3.79 (-0.4 to +6.5) | +5.88 (+0.7 to +7.1) |
| new_val_zenodo_brands_under_10k | 646 | 3 | 2,148 | 53.12% [53.1%, 53.2%] | 52.14% [52.4%, 51.9%] | 52.93% [51.8%, 54.1%] | 53.35% [53.4%] | -0.98 (-1.8 to +0.7) | -0.19 (-1.0 to +1.0) | +0.28 (-3.2 to +2.6) |
| new_val_nitter_2026_replies | 1,059 | 31 | 9,208 | 60.08% [59.9%, 60.3%] | 60.70% [60.2%, 61.2%] | 61.09% [60.6%, 61.5%] | 60.25% [60.3%] | +0.61 (-0.6 to +6.7) | +1.00 (-0.4 to +8.8) | +0.35 (-1.5 to +11.4) |

#### Same-account pairs by slice, views (all models on the same rows of the expanded table)

| Slice | Rows | Accounts with pairs | Pairs | Control (mean of seeds) | Expanded (mean of seeds) | Expanded + sampling input (mean of seeds) | Expanded 7,450 steps (mean of seeds) | Expanded minus Control, points | Expanded + sampling input minus Control, points | Expanded 7,450 steps minus Control, points |
|---|---|---|---|---|---|---|---|---|---|---|
| orig_test_audience | 95,357 | 851 | 24,481 | 62.05% [62.3%, 61.8%] | 61.98% [61.9%, 62.0%] | 61.99% [61.7%, 62.3%] | 62.21% [62.2%] | -0.07 (-1.0 to +1.2) | -0.06 (-1.1 to +1.5) | -0.05 (-1.2 to +1.7) |
| orig_test_company | 3,160 | 21 | 304 | 64.47% [62.5%, 66.4%] | 64.64% [64.8%, 64.5%] | 63.32% [64.1%, 62.5%] | 63.16% [63.2%] | +0.16 (-5.6 to +3.1) | -1.15 (-4.7 to +2.3) | +0.66 (-11.7 to +6.7) |
| orig_test_founder | 7,717 | 34 | 550 | 70.27% [69.5%, 71.1%] | 70.73% [70.5%, 70.9%] | 69.82% [69.6%, 70.0%] | 71.64% [71.6%] | +0.45 (-2.7 to +2.6) | -0.45 (-4.2 to +2.3) | +2.18 (-0.9 to +4.8) |
| orig_test_startup | 10,877 | 55 | 854 | 68.21% [67.0%, 69.4%] | 68.56% [68.5%, 68.6%] | 67.51% [67.7%, 67.3%] | 68.62% [68.6%] | +0.35 (-2.3 to +2.1) | -0.70 (-3.1 to +1.4) | +1.64 (-2.7 to +4.6) |
| orig_test_under_10k | 38,289 | 178 | 2,080 | 63.75% [64.6%, 62.9%] | 63.56% [63.9%, 63.2%] | 64.28% [64.7%, 63.8%] | 64.28% [64.3%] | -0.19 (-1.6 to +1.2) | +0.53 (-1.0 to +2.1) | -0.29 (-2.4 to +1.9) |
| orig_test_startup_under_10k | 1,153 | 6 | 44 | 76.14% [79.5%, 72.7%] | 73.86% [72.7%, 75.0%] | 69.32% [70.5%, 68.2%] | 75.00% [75.0%] | -2.27 (-5.9 to +2.3) | -6.82 (-12.0 to +0.0) | -4.55 (-16.1 to +4.1) |
| new_test | 587 | 85 | 1,860 | 61.53% [61.5%, 61.6%] | 63.52% [62.7%, 64.4%] | 60.86% [60.7%, 61.0%] | 61.40% [61.4%] | +1.99 (-0.4 to +4.9) | -0.67 (-2.5 to +2.2) | -0.11 (-2.2 to +2.5) |
| new_val_zenodo_brands | 1,735 | 7 | 14,025 | 54.30% [54.1%, 54.5%] | 56.88% [57.4%, 56.3%] | 57.25% [56.7%, 57.8%] | 56.88% [56.9%] | +2.58 (+0.2 to +5.1) | +2.96 (+0.9 to +5.1) | +2.80 (-1.7 to +5.7) |
| new_val_zenodo_brands_under_10k | 646 | 3 | 1,802 | 59.18% [60.1%, 58.3%] | 59.16% [58.6%, 59.7%] | 60.74% [61.0%, 60.5%] | 59.10% [59.1%] | -0.03 (-1.9 to +3.1) | +1.55 (-4.1 to +6.4) | -1.00 (-7.2 to +1.5) |
| new_val_nitter_2026_replies | 1,059 | 35 | 7,372 | 58.82% [58.4%, 59.2%] | 58.91% [58.7%, 59.1%] | 59.24% [58.9%, 59.6%] | 57.76% [57.8%] | +0.09 (-0.4 to +3.3) | +0.43 (-1.1 to +4.3) | -0.68 (-1.5 to +3.4) |

#### Level of the forecast on table slices, likes (median over rows with 5+ visible history posts; ratio to the account's own median)

| Slice | Rows | Accounts | Actual / own median | Control: predicted / own median | Expanded: predicted / own median | Expanded + sampling input: predicted / own median | Expanded 7,450 steps: predicted / own median | Control: actual above q90 | Expanded: actual above q90 | Expanded + sampling input: actual above q90 | Expanded 7,450 steps: actual above q90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orig_test_audience | 94,545 | 3,666 | 1.17x | 1.11x / 1.13x | 1.12x / 1.16x | 1.10x / 1.14x | 1.13x | 11% / 11% | 11% / 10% | 11% / 11% | 11% |
| orig_test_company | 3,153 | 188 | 1.42x | 1.02x / 1.05x | 1.06x / 1.08x | 1.08x / 1.00x | 1.08x | 15% / 16% | 13% / 14% | 13% / 15% | 15% |
| orig_test_founder | 7,694 | 452 | 1.40x | 1.04x / 1.03x | 1.09x / 1.11x | 1.07x / 1.02x | 1.11x | 13% / 14% | 12% / 12% | 12% / 12% | 11% |
| orig_test_startup | 10,847 | 640 | 1.41x | 1.04x / 1.04x | 1.08x / 1.10x | 1.07x / 1.01x | 1.10x | 14% / 14% | 12% / 12% | 12% / 13% | 12% |
| orig_test_under_10k | 37,754 | 1,035 | 1.08x | 1.09x / 1.13x | 1.11x / 1.13x | 1.09x / 1.14x | 1.14x | 12% / 12% | 12% / 11% | 13% / 12% | 12% |
| orig_test_startup_under_10k | 1,146 | 82 | 1.40x | 1.06x / 1.09x | 1.10x / 1.08x | 1.07x / 1.07x | 1.11x | 13% / 14% | 13% / 14% | 14% / 14% | 13% |
| new_test | 583 | 201 | 0.85x | 0.87x / 0.88x | 0.88x / 0.92x | 0.90x / 0.93x | 0.92x | 9% / 9% | 9% / 9% | 10% / 9% | 9% |
| orig_test_launch_posts | 734 | 398 | 3.09x | 1.49x / 1.45x | 1.55x / 1.55x | 1.53x / 1.45x | 1.58x | 22% / 24% | 20% / 19% | 20% / 21% | 22% |
| orig_test_launch_posts_startup | 544 | 237 | 3.34x | 1.51x / 1.43x | 1.59x / 1.56x | 1.56x / 1.44x | 1.63x | 23% / 25% | 20% / 19% | 20% / 21% | 22% |
| orig_test_launch_posts_under_10k | 89 | 75 | 3.30x | 1.78x / 1.86x | 1.72x / 1.88x | 1.88x / 1.78x | 1.92x | 16% / 18% | 17% / 15% | 17% / 16% | 17% |
| new_val_launch_posts | 226 | 72 | 1.24x | 1.15x / 1.38x | 1.24x / 1.23x | 1.26x / 1.24x | 1.23x | 19% / 16% | 14% / 14% | 14% / 14% | 15% |
| new_val_launch_posts_under_10k | 121 | 40 | 1.00x | 1.21x / 1.43x | 1.18x / 1.19x | 1.20x / 1.19x | 1.18x | 12% / 10% | 12% / 11% | 12% / 12% | 12% |
| new_val_launch_posts_company | 200 | 55 | 1.15x | 1.16x / 1.38x | 1.24x / 1.22x | 1.25x / 1.23x | 1.23x | 18% / 14% | 12% / 12% | 12% / 12% | 13% |
| new_val_launch_posts_founder | 26 | 17 | 1.59x | 1.09x / 1.38x | 1.23x / 1.30x | 1.32x / 1.30x | 1.24x | 35% / 31% | 27% / 27% | 27% / 23% | 27% |
| new_val_twitter100m | 27,229 | 133 | 1.00x | 0.92x / 1.18x | 1.07x / 1.07x | 1.07x / 1.08x | 1.07x | 13% / 9% | 9% / 9% | 9% / 9% | 10% |
| new_val_twitter100m_under_10k | 14,265 | 86 | 1.00x | 1.07x / 1.25x | 1.16x / 1.16x | 1.17x / 1.16x | 1.16x | 8% / 6% | 9% / 8% | 8% / 8% | 9% |
| new_val_zenodo_brands | 1,693 | 7 | 0.95x | 1.58x / 1.45x | 0.93x / 1.02x | 0.92x / 1.01x | 0.89x | 3% / 4% | 7% / 6% | 9% / 7% | 8% |
| new_val_zenodo_brands_under_10k | 630 | 3 | 0.87x | 1.26x / 1.30x | 0.88x / 0.95x | 0.94x / 1.02x | 0.85x | 5% / 5% | 8% / 7% | 8% / 7% | 10% |
| new_val_nitter_2026_replies | 1,021 | 59 | 1.00x | 0.89x / 0.85x | 0.93x / 0.96x | 1.01x / 1.01x | 1.03x | 10% / 12% | 10% / 10% | 10% / 9% | 11% |
| new_val_nitter_2026_replies_under_10k | 23 | 8 | 0.82x | 0.83x / 0.98x | 0.95x / 0.97x | 0.83x / 0.84x | 0.96x | 4% / 4% | 0% / 4% | 4% / 4% | 4% |

#### The 3,530-pair benchmark (same account, same topic; likes), by size of the real gap

| Model | Real gap at least | Pairs | Correct (95% interval) | Confident (60%+) share (pairs) | Correct there (95% interval) |
|---|---|---|---|---|---|
| ctl_s1 | 1.25x | 2,851 | 59.9% (57.4 to 62.2) | 26% (755) | 67.0% (62.9 to 70.6) |
| ctl_s1 | 2.0x | 2,071 | 60.9% (57.9 to 63.8) | 25% (516) | 70.5% (65.1 to 75.0) |
| ctl_s1 | 3.0x | 1,390 | 62.2% (58.4 to 65.7) | 26% (363) | 73.8% (67.0 to 79.7) |
| ctl_s1 | 5.0x | 880 | 65.2% (60.8 to 70.2) | 27% (238) | 80.7% (74.6 to 85.8) |
| ctl_s2 | 1.25x | 2,851 | 58.9% (57.0 to 61.1) | 25% (719) | 66.5% (61.5 to 70.9) |
| ctl_s2 | 2.0x | 2,071 | 59.7% (57.3 to 62.2) | 24% (495) | 67.5% (61.8 to 72.0) |
| ctl_s2 | 3.0x | 1,390 | 61.9% (58.5 to 65.2) | 25% (342) | 71.9% (65.2 to 77.1) |
| ctl_s2 | 5.0x | 880 | 64.8% (61.1 to 68.7) | 26% (229) | 78.2% (70.9 to 83.2) |
| exp_s1 | 1.25x | 2,851 | 58.6% (56.0 to 60.9) | 23% (653) | 67.5% (62.3 to 71.8) |
| exp_s1 | 2.0x | 2,071 | 59.6% (56.7 to 62.2) | 22% (456) | 71.3% (65.8 to 76.1) |
| exp_s1 | 3.0x | 1,390 | 61.1% (57.3 to 64.1) | 23% (317) | 72.9% (66.1 to 79.4) |
| exp_s1 | 5.0x | 880 | 63.0% (59.2 to 67.0) | 23% (202) | 79.7% (71.7 to 86.1) |
| exp_s2 | 1.25x | 2,851 | 59.9% (58.1 to 61.8) | 24% (671) | 66.0% (61.2 to 70.2) |
| exp_s2 | 2.0x | 2,071 | 61.3% (59.1 to 63.7) | 23% (467) | 68.7% (63.7 to 73.7) |
| exp_s2 | 3.0x | 1,390 | 63.2% (59.5 to 66.5) | 23% (323) | 71.8% (64.3 to 78.3) |
| exp_s2 | 5.0x | 880 | 65.7% (62.1 to 69.4) | 24% (213) | 78.4% (69.8 to 85.0) |
| expsrc_s1 | 1.25x | 2,851 | 59.1% (56.7 to 61.5) | 27% (760) | 68.3% (64.8 to 72.0) |
| expsrc_s1 | 2.0x | 2,071 | 60.4% (57.6 to 63.0) | 25% (525) | 71.2% (66.2 to 76.2) |
| expsrc_s1 | 3.0x | 1,390 | 61.9% (58.3 to 64.9) | 26% (357) | 73.7% (67.3 to 79.8) |
| expsrc_s1 | 5.0x | 880 | 64.3% (60.6 to 68.0) | 26% (231) | 79.7% (72.6 to 85.6) |
| expsrc_s2 | 1.25x | 2,851 | 59.1% (57.1 to 61.2) | 24% (698) | 66.9% (62.7 to 70.6) |
| expsrc_s2 | 2.0x | 2,071 | 60.7% (58.4 to 62.9) | 23% (480) | 70.6% (65.9 to 75.5) |
| expsrc_s2 | 3.0x | 1,390 | 62.7% (58.9 to 66.1) | 23% (321) | 74.8% (68.2 to 80.2) |
| expsrc_s2 | 5.0x | 880 | 65.6% (61.7 to 69.8) | 24% (211) | 79.1% (71.6 to 86.4) |
| explong_s1 | 1.25x | 2,851 | 58.8% (56.8 to 60.8) | 23% (649) | 67.8% (63.8 to 71.5) |
| explong_s1 | 2.0x | 2,071 | 59.9% (57.3 to 62.3) | 22% (450) | 70.7% (65.8 to 75.4) |
| explong_s1 | 3.0x | 1,390 | 61.7% (58.6 to 64.6) | 22% (306) | 73.2% (66.8 to 79.5) |
| explong_s1 | 5.0x | 880 | 64.2% (61.0 to 67.6) | 22% (192) | 82.8% (77.1 to 87.6) |
| v2full | 1.25x | 2,851 | 60.2% (58.1 to 62.5) | 26% (739) | 66.2% (61.7 to 69.7) |
| v2full | 2.0x | 2,071 | 60.8% (58.2 to 63.7) | 24% (505) | 69.7% (64.1 to 74.5) |
| v2full | 3.0x | 1,390 | 63.0% (59.8 to 66.6) | 25% (350) | 72.6% (66.3 to 77.5) |
| v2full | 5.0x | 880 | 66.4% (62.1 to 71.1) | 25% (223) | 79.8% (73.0 to 85.4) |
| Control | 1.25x | 2,851 | 59.5% (57.3 to 61.6) | 27% (763) | 67.5% (63.0 to 70.8) |
| Control | 2.0x | 2,071 | 60.5% (57.5 to 63.3) | 25% (523) | 69.6% (63.9 to 74.5) |
| Control | 3.0x | 1,390 | 62.5% (59.3 to 65.7) | 27% (370) | 73.2% (66.4 to 78.5) |
| Control | 5.0x | 880 | 65.9% (62.0 to 70.2) | 28% (243) | 80.7% (74.4 to 86.1) |
| Expanded | 1.25x | 2,851 | 59.8% (57.6 to 61.9) | 25% (702) | 66.8% (62.8 to 70.8) |
| Expanded | 2.0x | 2,071 | 60.7% (58.3 to 63.0) | 23% (484) | 69.8% (64.6 to 74.3) |
| Expanded | 3.0x | 1,390 | 62.3% (58.8 to 65.5) | 24% (330) | 72.7% (66.5 to 78.8) |
| Expanded | 5.0x | 880 | 63.9% (60.3 to 68.1) | 24% (212) | 78.8% (71.6 to 85.1) |
| Expanded + sampling input | 1.25x | 2,851 | 59.1% (56.9 to 61.5) | 27% (760) | 66.8% (62.3 to 70.8) |
| Expanded + sampling input | 2.0x | 2,071 | 60.6% (57.9 to 63.1) | 25% (522) | 70.3% (64.5 to 74.9) |
| Expanded + sampling input | 3.0x | 1,390 | 62.2% (58.6 to 65.3) | 25% (353) | 73.4% (66.5 to 79.4) |
| Expanded + sampling input | 5.0x | 880 | 64.4% (60.4 to 68.6) | 26% (230) | 78.7% (70.8 to 84.8) |

Paired differences on the benchmark (points, accounts resampled):

- exp minus ctl: 2,852 pairs, all pairs +0.3 (-1.4 to +1.7), among confident calls -0.7 (-2.8 to +1.4)
- exp_s1 minus ctl_s1: 2,852 pairs, all pairs -1.3 (-3.1 to +0.4), among confident calls +0.5 (-2.5 to +3.5)
- exp_s2 minus ctl_s2: 2,852 pairs, all pairs +1.0 (-0.8 to +2.9), among confident calls -0.5 (-3.1 to +2.4)
- expsrc minus ctl: 2,852 pairs, all pairs -0.3 (-1.9 to +1.2), among confident calls -0.7 (-3.1 to +1.6)
- expsrc_s1 minus ctl_s1: 2,852 pairs, all pairs -0.8 (-2.7 to +1.3), among confident calls +1.3 (-1.2 to +3.9)
- expsrc_s2 minus ctl_s2: 2,852 pairs, all pairs +0.2 (-1.5 to +1.8), among confident calls +0.4 (-2.6 to +3.8)
- explong_s1 minus ctl_s1: 2,852 pairs, all pairs -1.1 (-2.7 to +0.7), among confident calls +0.7 (-2.1 to +3.7)

#### The 850 launch pairs (same account, both launch posts, test period)

| Model | All | Gap 2x+ | Gap 3x+ | Company | Founder | Under 10k followers | Confident 60%+ | Confident and 2x+ |
|---|---|---|---|---|---|---|---|---|
| ctl_s1 | 63.8% (59.4 to 68.2) n=850 | 66.5% (61.7 to 71.4) n=568 | 68.1% (62.4 to 73.8) n=401 | 63.1% (58.3 to 68.3) n=601 | 69.8% (61.3 to 78.6) n=119 | 70.5% (51.0 to 83.3) n=78 | 72.6% (63.8 to 81.2) n=208, 24% of pairs | 75.6% (66.4 to 84.7) n=156, 27% of 2x pairs |
| ctl_s2 | 62.4% (57.9 to 67.1) n=850 | 63.9% (58.7 to 69.2) n=568 | 66.3% (60.2 to 72.7) n=401 | 62.4% (56.9 to 68.1) n=601 | 61.3% (53.6 to 71.4) n=119 | 71.8% (54.8 to 82.3) n=78 | 72.9% (64.8 to 80.5) n=247, 29% of pairs | 77.2% (68.6 to 85.6) n=180, 32% of 2x pairs |
| exp_s1 | 62.5% (58.2 to 66.8) n=850 | 65.8% (60.6 to 71.2) n=568 | 68.1% (61.9 to 74.2) n=401 | 62.6% (57.4 to 68.0) n=601 | 62.2% (53.5 to 73.6) n=119 | 66.7% (47.3 to 79.1) n=78 | 75.5% (66.7 to 83.3) n=233, 27% of pairs | 77.3% (67.8 to 85.8) n=176, 31% of 2x pairs |
| exp_s2 | 63.5% (59.5 to 67.6) n=850 | 64.1% (58.8 to 69.4) n=568 | 66.3% (59.9 to 73.1) n=401 | 62.6% (57.6 to 67.6) n=601 | 66.4% (57.9 to 76.2) n=119 | 70.5% (52.8 to 82.3) n=78 | 75.2% (67.9 to 82.7) n=262, 31% of pairs | 79.0% (70.6 to 86.5) n=195, 34% of 2x pairs |
| expsrc_s1 | 62.8% (58.7 to 67.1) n=850 | 63.6% (58.3 to 68.9) n=568 | 65.6% (59.4 to 72.1) n=401 | 63.6% (58.6 to 68.9) n=601 | 63.9% (54.7 to 75.3) n=119 | 66.7% (47.2 to 79.0) n=78 | 75.5% (68.0 to 83.5) n=245, 29% of pairs | 79.6% (71.5 to 87.8) n=181, 32% of 2x pairs |
| expsrc_s2 | 65.6% (61.5 to 69.4) n=850 | 67.8% (62.6 to 72.5) n=568 | 71.3% (65.4 to 77.1) n=401 | 65.7% (60.8 to 70.4) n=601 | 66.4% (57.8 to 76.7) n=119 | 68.0% (48.6 to 79.5) n=78 | 73.2% (64.5 to 81.7) n=239, 28% of pairs | 76.8% (67.8 to 85.6) n=177, 31% of 2x pairs |
| explong_s1 | 63.5% (59.5 to 67.6) n=850 | 65.8% (61.0 to 70.9) n=568 | 68.3% (63.1 to 74.1) n=401 | 63.7% (59.2 to 68.2) n=601 | 64.7% (55.5 to 75.8) n=119 | 74.4% (52.7 to 89.4) n=78 | 76.1% (68.2 to 84.0) n=255, 30% of pairs | 78.6% (70.2 to 87.0) n=192, 34% of 2x pairs |
| Control | 63.4% (59.4 to 67.7) n=850 | 66.2% (61.5 to 71.0) n=568 | 68.8% (62.9 to 74.7) n=401 | 63.6% (59.2 to 68.4) n=601 | 64.7% (56.4 to 73.1) n=119 | 69.2% (50.0 to 82.1) n=78 | 72.4% (64.1 to 80.3) n=217, 26% of pairs | 77.8% (69.1 to 86.1) n=162, 29% of 2x pairs |
| Expanded | 63.8% (59.7 to 67.8) n=850 | 65.7% (60.2 to 71.0) n=568 | 67.1% (60.9 to 73.5) n=401 | 63.6% (58.5 to 68.9) n=601 | 65.5% (57.2 to 76.1) n=119 | 69.2% (51.0 to 81.6) n=78 | 76.1% (68.3 to 83.8) n=239, 28% of pairs | 78.2% (69.9 to 86.7) n=179, 32% of 2x pairs |
| Expanded + sampling input | 64.7% (60.5 to 69.0) n=850 | 66.0% (61.0 to 71.5) n=568 | 67.8% (61.9 to 74.2) n=401 | 64.7% (59.6 to 70.1) n=601 | 66.4% (56.8 to 76.9) n=119 | 66.7% (45.5 to 79.8) n=78 | 73.9% (65.6 to 81.9) n=230, 27% of pairs | 76.7% (68.0 to 85.4) n=176, 31% of 2x pairs |

Paired differences on the launch pairs (points, accounts resampled):

- exp minus ctl: 850 pairs, 151 accounts, +0.4 (-2.7 to +3.1)
- exp_s1 minus ctl_s1: 850 pairs, 151 accounts, -1.3 (-4.7 to +1.9)
- exp_s2 minus ctl_s2: 850 pairs, 151 accounts, +1.2 (-2.6 to +4.6)
- expsrc minus ctl: 850 pairs, 151 accounts, +1.3 (-1.8 to +4.2)
- expsrc_s1 minus ctl_s1: 850 pairs, 151 accounts, -0.9 (-4.3 to +2.3)
- expsrc_s2 minus ctl_s2: 850 pairs, 151 accounts, +3.3 (-0.4 to +6.5)
- explong_s1 minus ctl_s1: 850 pairs, 151 accounts, -0.2 (-3.5 to +2.8)

#### Level of the forecast for launch posts (draft path; likes over the account's trailing median)

| Model | Slice | Posts | Accounts | Actual | Predicted (95% interval) | Actual beats own median | Predicted to beat it | Rank correlation of predicted and actual lift | Median predicted / actual likes |
|---|---|---|---|---|---|---|---|---|---|
| ctl_s1 | all | 576 | 167 | 2.19x | 0.90x (0.80 to 1.02) | 76% | 45% | 0.48 | 0.41 |
| ctl_s1 | company | 326 | 73 | 2.45x | 0.86x (0.75 to 1.03) | 82% | 43% | 0.55 | 0.35 |
| ctl_s1 | founder | 118 | 45 | 2.09x | 0.91x (0.72 to 1.13) | 70% | 47% | 0.45 | 0.42 |
| ctl_s1 | under 10k followers | 69 | 25 | 1.58x | 1.06x (0.90 to 1.20) | 67% | 54% | 0.65 | 0.71 |
| ctl_s2 | all | 576 | 167 | 2.19x | 1.08x (0.95 to 1.20) | 76% | 54% | 0.46 | 0.48 |
| ctl_s2 | company | 326 | 73 | 2.45x | 1.03x (0.89 to 1.21) | 82% | 52% | 0.52 | 0.40 |
| ctl_s2 | founder | 118 | 45 | 2.09x | 1.05x (0.79 to 1.38) | 70% | 55% | 0.45 | 0.48 |
| ctl_s2 | under 10k followers | 69 | 25 | 1.58x | 1.20x (1.02 to 1.44) | 67% | 65% | 0.68 | 0.81 |
| exp_s1 | all | 576 | 167 | 2.19x | 0.98x (0.90 to 1.07) | 76% | 48% | 0.49 | 0.45 |
| exp_s1 | company | 326 | 73 | 2.45x | 0.92x (0.82 to 1.09) | 82% | 46% | 0.55 | 0.38 |
| exp_s1 | founder | 118 | 45 | 2.09x | 1.04x (0.83 to 1.17) | 70% | 51% | 0.50 | 0.47 |
| exp_s1 | under 10k followers | 69 | 25 | 1.58x | 1.11x (0.93 to 1.25) | 67% | 58% | 0.66 | 0.71 |
| exp_s2 | all | 576 | 167 | 2.19x | 1.04x (0.93 to 1.17) | 76% | 53% | 0.47 | 0.49 |
| exp_s2 | company | 326 | 73 | 2.45x | 0.95x (0.86 to 1.13) | 82% | 48% | 0.53 | 0.41 |
| exp_s2 | founder | 118 | 45 | 2.09x | 1.08x (0.91 to 1.30) | 70% | 57% | 0.49 | 0.52 |
| exp_s2 | under 10k followers | 69 | 25 | 1.58x | 1.22x (1.00 to 1.33) | 67% | 65% | 0.58 | 0.85 |
| expsrc_s1 | all | 576 | 167 | 2.19x | 1.05x (0.95 to 1.15) | 76% | 53% | 0.48 | 0.48 |
| expsrc_s1 | company | 326 | 73 | 2.45x | 0.99x (0.89 to 1.17) | 82% | 49% | 0.54 | 0.41 |
| expsrc_s1 | founder | 118 | 45 | 2.09x | 1.08x (0.90 to 1.26) | 70% | 55% | 0.47 | 0.50 |
| expsrc_s1 | under 10k followers | 69 | 25 | 1.58x | 1.19x (1.02 to 1.30) | 67% | 65% | 0.67 | 0.75 |
| expsrc_s2 | all | 576 | 167 | 2.19x | 1.22x (1.10 to 1.36) | 76% | 64% | 0.48 | 0.58 |
| expsrc_s2 | company | 326 | 73 | 2.45x | 1.15x (1.02 to 1.37) | 82% | 59% | 0.53 | 0.50 |
| expsrc_s2 | founder | 118 | 45 | 2.09x | 1.31x (1.08 to 1.60) | 70% | 67% | 0.50 | 0.61 |
| expsrc_s2 | under 10k followers | 69 | 25 | 1.58x | 1.36x (1.21 to 1.50) | 67% | 74% | 0.64 | 0.90 |
| explong_s1 | all | 576 | 167 | 2.19x | 0.93x (0.85 to 1.01) | 76% | 45% | 0.48 | 0.43 |
| explong_s1 | company | 326 | 73 | 2.45x | 0.85x (0.77 to 0.96) | 82% | 40% | 0.55 | 0.37 |
| explong_s1 | founder | 118 | 45 | 2.09x | 0.99x (0.79 to 1.20) | 70% | 48% | 0.51 | 0.43 |
| explong_s1 | under 10k followers | 69 | 25 | 1.58x | 1.19x (1.01 to 1.44) | 67% | 67% | 0.70 | 0.75 |
| Control | all | 576 | 167 | 2.19x | 0.98x (0.88 to 1.09) | 76% | 49% | 0.48 | 0.44 |
| Control | company | 326 | 73 | 2.45x | 0.95x (0.81 to 1.08) | 82% | 47% | 0.54 | 0.38 |
| Control | founder | 118 | 45 | 2.09x | 0.97x (0.78 to 1.28) | 70% | 47% | 0.45 | 0.47 |
| Control | under 10k followers | 69 | 25 | 1.58x | 1.13x (0.97 to 1.28) | 67% | 58% | 0.68 | 0.75 |
| Expanded | all | 576 | 167 | 2.19x | 1.00x (0.92 to 1.13) | 76% | 51% | 0.49 | 0.47 |
| Expanded | company | 326 | 73 | 2.45x | 0.93x (0.84 to 1.12) | 82% | 46% | 0.55 | 0.38 |
| Expanded | founder | 118 | 45 | 2.09x | 1.06x (0.87 to 1.25) | 70% | 54% | 0.50 | 0.50 |
| Expanded | under 10k followers | 69 | 25 | 1.58x | 1.21x (0.95 to 1.28) | 67% | 61% | 0.66 | 0.74 |
| Expanded + sampling input | all | 576 | 167 | 2.19x | 1.14x (1.02 to 1.25) | 76% | 58% | 0.49 | 0.53 |
| Expanded + sampling input | company | 326 | 73 | 2.45x | 1.06x (0.94 to 1.24) | 82% | 54% | 0.55 | 0.46 |
| Expanded + sampling input | founder | 118 | 45 | 2.09x | 1.22x (0.96 to 1.39) | 70% | 62% | 0.48 | 0.55 |
| Expanded + sampling input | under 10k followers | 69 | 25 | 1.58x | 1.25x (1.12 to 1.41) | 67% | 67% | 0.66 | 0.82 |

### What the tables say

1. **Existing test pairs: no change.** Likes: Expanded -0.19 points (-0.5 to +0.1), with the sampling input -0.08 (-0.4 to +0.2), on 683,152
   pairs of 2,846 accounts. Views: -0.07 (-1.0 to +1.2) and -0.06 (-1.1 to +1.5) on 24,481 pairs of 851 accounts.
2. **Company and founder accounts in the existing test rows: a small gain, clear only with the sampling input.** On 34,831 like pairs of 470
   accounts: Expanded +0.30 (-0.3 to +0.8), with the sampling input +0.73 (+0.2 to +1.3). Company accounts alone (11,184 pairs, 130
   accounts): +0.69 (-0.5 to +1.7) and +0.79 (+0.3 to +1.7). Founder accounts (23,647 pairs, 340 accounts): +0.12 (-0.6 to +0.8) and +0.70
   (-0.1 to +1.5). This is the slice the expansion was aimed at, but it is one of about twenty slices read, so treat it as a lead. Views in
   that slice have 854 pairs of 55 accounts and say nothing (+0.35, -2.3 to +2.1; -0.70, -3.1 to +1.4).
3. **Accounts under 10,000 followers in the existing test rows: nothing.** -0.02 (-0.5 to +0.4) and -0.01 (-0.5 to +0.4) on 335,638 like
   pairs of 854 accounts; views -0.19 and +0.53 on 2,080 pairs, intervals of 1.5 points either side. The 129,586 added small-account posts
   (almost all from 2020 to 2023) did not help the small accounts of 2026.
4. **The benchmark: equal.** Two-seed ensembles on 2,851 pairs: Control 59.5%, Expanded 59.8% (+0.3, -1.4 to +1.7), with the sampling
   input 59.1% (-0.3, -1.9 to +1.2). The two seeds point in opposite directions (-1.3 and +1.0 for Expanded), which is the size of seed
   noise here. Confident calls on pairs 2x apart: 25% of pairs at 69.6%, 23% at 69.8%, 25% at 70.3%.
5. **Launch pairs: equal.** 850 pairs of 151 accounts: Control 63.4%, Expanded 63.8% (+0.4, -2.7 to +3.1), with the sampling input 64.7%
   (+1.3, -1.8 to +4.2). Company accounts (601 pairs): 63.6%, 63.6%, 64.7%. Founder accounts (119 pairs): 64.7%, 65.5%, 66.4%. Accounts
   under 10,000 followers: 78 pairs, intervals of 15 points either side. Confident calls 2x apart: 29% of such pairs at 77.8%, 32% at
   78.2%, 31% at 76.7%.
6. **The level of a launch forecast moves a little, in the right direction, and stays far off.** Launch posts get 2.19x the account's
   trailing median; the predicted median is 0.98x (0.88 to 1.09) for Control, 1.00x (0.92 to 1.13) for Expanded and 1.14x (1.02 to 1.25)
   with the sampling input. Company accounts (326 posts, actual 2.45x): 0.95x, 0.93x, 1.06x. Founder accounts (118 posts, actual 2.09x):
   0.97x, 1.06x, 1.22x. Accounts under 10,000 followers (69 posts of 25 accounts, actual 1.58x): 1.13x, 1.21x, 1.25x. The 1,866 launch-type
   posts added to training did not teach the model what a launch is worth; the post-type correction of the posttype track is still needed.
7. **The models did learn the new sources.** On accounts held out from the added sources (no model trained on them): twitter100m +3.27
   (+2.2 to +4.4) on 427,248 like pairs of 134 accounts, and +2.26 (+0.8 to +3.8) for its accounts under 10,000 followers (103,550 pairs, 86
   accounts); the brand set +3.37 on likes (-0.7 to +5.7; 14,240 pairs, 7 accounts) and +2.58 on views (+0.2 to +5.1; 14,025 pairs); root
   posts of the 2026 view corpus +0.61 (-0.6 to +6.7; 9,208 pairs, 31 accounts). So the rows carry signal for their own kind of account.
   It does not carry over to archive accounts in 2026.
8. **A full timeline is forecast too high by the existing models.** On the held-out brand accounts (whole timelines, 1,693 posts) the
   control predicts 1.58x and 1.45x the account's own median where the outcome is 0.95x; the expanded models predict 0.92x to 1.02x. A
   customer's own account is a whole timeline too, so this is the regime the product will be used in. Seven accounts: a lead, not a
   measurement.

9. **Dilution is not the reason.** At 4,000 steps an expanded model sees each existing row about half as often as the control does.
   `explong_s1` trains 7,450 steps (the same number of passes as the control's 4,000; one seed, compared with `ctl_s1`): existing test pairs
   59.75% likes and 62.21% views (-0.06, -0.4 to +0.2; -0.05, -1.2 to +1.7); benchmark 58.8% against 59.9% (-1.1, -2.7 to +0.7); launch
   pairs 63.5% against 63.8% (-0.2, -3.5 to +2.8); launch level 0.93x. Its like NLL on the existing test rows is the best of all runs
   (2.954 against 2.966), and it repeats the two local gains: company and founder accounts +1.05 (+0.1 to +1.8) on 34,831 pairs,
   held-out twitter100m accounts +3.71 (+2.2 to +5.4). So the gain on company and founder accounts appears in all three expanded
   variants (+0.30, +0.73, +1.05), and the same-topic and launch pairs stay where they were.

## 5. The pair-loss fine-tune

The condition was "if the expanded models are at least as good". They are level with the control on every benchmark, so the recipe of the
pairs track was applied as it stands: 600 steps, learning rate 2e-4, warm-up 50, the pair loss at weight 10 on the 11,507 serving-path pairs
of `tracks/pairs/pair_table` (no benchmark pair, no benchmark calibration account), checkpoint chosen on the 2,011 held-out pairs and the
validation loss, never on the benchmark. The ordinary loss runs on each model's own training table. The fine-tuned controls are the pairs
track's own checkpoints (`ft_w10` from `ts46_s1`, `ft_w10_s2` from `ts46_s2`), rescored here. For the models with the sampling input the
pair table got the `src_regime` column (`scripts/expand/pair_table_regime.py`).

#### Ordinary same-account test pairs, each model's own run (existing audience test rows)

| Model | Rows | Likes pairs | Likes correct | Views pairs | Views correct | Within-account rho likes | Like NLL | Best step |
|---|---|---|---|---|---|---|---|---|
| ctlft_s1 | 95,357 | 683,152 | 59.91% | 24,481 | 62.14% | 0.2982 | 2.9890 | 150 |
| ctlft_s2 | 95,357 | 683,152 | 60.21% | 24,481 | 61.67% | 0.2994 | 2.9887 | 150 |
| expft_s1 | 95,357 | 683,152 | 59.77% | 24,481 | 62.05% | 0.2928 | 2.9797 | 150 |
| expft_s2 | 95,357 | 683,152 | 59.79% | 24,481 | 62.02% | 0.2947 | 2.9898 | 450 |
| expsrcft_s1 | 95,357 | 683,152 | 59.68% | 24,481 | 61.84% | 0.2906 | 2.9840 | 150 |
| expsrcft_s2 | 95,357 | 683,152 | 60.15% | 24,481 | 62.62% | 0.2993 | 2.9896 | 150 |

#### The 3,530-pair benchmark (same account, same topic; likes), by size of the real gap

| Model | Real gap at least | Pairs | Correct (95% interval) | Confident (60%+) share (pairs) | Correct there (95% interval) |
|---|---|---|---|---|---|
| ctlft_s1 | 1.25x | 2,851 | 60.3% (58.1 to 62.4) | 31% (890) | 68.3% (64.9 to 71.7) |
| ctlft_s1 | 2.0x | 2,071 | 61.7% (58.8 to 64.4) | 30% (625) | 71.8% (66.5 to 76.0) |
| ctlft_s1 | 3.0x | 1,390 | 64.2% (60.7 to 67.8) | 31% (436) | 76.1% (70.0 to 81.9) |
| ctlft_s1 | 5.0x | 880 | 67.0% (62.5 to 72.1) | 34% (296) | 83.1% (76.2 to 88.0) |
| ctlft_s2 | 1.25x | 2,851 | 59.9% (57.5 to 62.5) | 30% (860) | 68.0% (63.8 to 72.0) |
| ctlft_s2 | 2.0x | 2,071 | 60.7% (57.9 to 63.9) | 29% (610) | 70.3% (65.8 to 74.5) |
| ctlft_s2 | 3.0x | 1,390 | 64.0% (60.0 to 68.1) | 30% (420) | 75.0% (69.5 to 80.2) |
| ctlft_s2 | 5.0x | 880 | 66.7% (61.8 to 71.6) | 34% (298) | 79.2% (72.9 to 84.3) |
| expft_s1 | 1.25x | 2,851 | 59.5% (57.0 to 62.0) | 28% (805) | 67.3% (63.2 to 71.1) |
| expft_s1 | 2.0x | 2,071 | 61.4% (58.3 to 64.6) | 28% (572) | 70.8% (65.5 to 75.7) |
| expft_s1 | 3.0x | 1,390 | 62.7% (58.5 to 66.4) | 28% (395) | 74.2% (68.2 to 79.6) |
| expft_s1 | 5.0x | 880 | 65.2% (60.4 to 70.3) | 30% (268) | 81.0% (74.6 to 86.1) |
| expft_s2 | 1.25x | 2,851 | 60.8% (58.7 to 63.0) | 30% (869) | 67.4% (63.7 to 71.4) |
| expft_s2 | 2.0x | 2,071 | 63.0% (60.7 to 65.5) | 30% (616) | 70.3% (65.7 to 75.1) |
| expft_s2 | 3.0x | 1,390 | 65.7% (61.8 to 69.4) | 31% (425) | 74.8% (67.5 to 80.5) |
| expft_s2 | 5.0x | 880 | 68.6% (64.5 to 73.4) | 33% (293) | 80.9% (73.5 to 87.1) |
| expsrcft_s1 | 1.25x | 2,851 | 60.1% (57.8 to 62.3) | 30% (858) | 68.2% (63.9 to 71.7) |
| expsrcft_s1 | 2.0x | 2,071 | 61.6% (58.9 to 64.2) | 29% (602) | 71.6% (66.9 to 76.1) |
| expsrcft_s1 | 3.0x | 1,390 | 63.5% (59.6 to 67.0) | 30% (416) | 74.8% (69.0 to 80.2) |
| expsrcft_s1 | 5.0x | 880 | 66.6% (62.2 to 71.0) | 32% (278) | 80.6% (73.9 to 85.9) |
| expsrcft_s2 | 1.25x | 2,851 | 59.8% (57.9 to 61.8) | 29% (835) | 66.5% (62.4 to 70.5) |
| expsrcft_s2 | 2.0x | 2,071 | 62.2% (59.8 to 64.5) | 28% (585) | 69.6% (65.2 to 74.3) |
| expsrcft_s2 | 3.0x | 1,390 | 64.5% (60.7 to 67.9) | 29% (402) | 73.9% (68.1 to 79.6) |
| expsrcft_s2 | 5.0x | 880 | 67.3% (62.7 to 72.0) | 30% (268) | 78.7% (70.7 to 85.2) |
| Control, fine-tuned | 1.25x | 2,851 | 60.2% (57.9 to 62.7) | 32% (912) | 69.1% (65.2 to 72.5) |
| Control, fine-tuned | 2.0x | 2,071 | 61.3% (58.3 to 64.4) | 32% (654) | 71.3% (66.6 to 75.5) |
| Control, fine-tuned | 3.0x | 1,390 | 63.9% (59.8 to 67.8) | 33% (457) | 75.3% (69.6 to 80.2) |
| Control, fine-tuned | 5.0x | 880 | 67.3% (62.5 to 72.2) | 35% (311) | 82.6% (76.5 to 87.8) |
| Expanded, fine-tuned | 1.25x | 2,851 | 60.4% (58.1 to 62.9) | 31% (895) | 67.6% (63.5 to 71.5) |
| Expanded, fine-tuned | 2.0x | 2,071 | 62.4% (59.9 to 65.3) | 31% (634) | 70.5% (66.0 to 74.5) |
| Expanded, fine-tuned | 3.0x | 1,390 | 64.2% (59.9 to 67.9) | 32% (444) | 74.8% (68.6 to 79.8) |
| Expanded, fine-tuned | 5.0x | 880 | 66.4% (62.2 to 71.5) | 34% (302) | 81.5% (74.5 to 86.4) |
| Expanded + sampling input, fine-tuned | 1.25x | 2,851 | 59.7% (57.3 to 61.8) | 30% (862) | 67.6% (63.4 to 72.0) |
| Expanded + sampling input, fine-tuned | 2.0x | 2,071 | 61.9% (59.2 to 64.4) | 29% (610) | 71.1% (66.2 to 75.7) |
| Expanded + sampling input, fine-tuned | 3.0x | 1,390 | 64.0% (59.8 to 67.3) | 31% (426) | 74.2% (68.9 to 79.6) |
| Expanded + sampling input, fine-tuned | 5.0x | 880 | 67.3% (62.8 to 71.9) | 32% (280) | 80.4% (74.2 to 86.2) |

Paired differences on the benchmark (points, accounts resampled):

- expft minus ctlft: 2,852 pairs, all pairs +0.2 (-1.2 to +1.6), among confident calls -1.5 (-3.6 to +0.7)
- expft_s1 minus ctlft_s1: 2,852 pairs, all pairs -0.7 (-2.3 to +0.6), among confident calls -1.0 (-3.7 to +1.9)
- expft_s2 minus ctlft_s2: 2,852 pairs, all pairs +0.9 (-0.5 to +2.4), among confident calls -0.6 (-3.3 to +1.9)
- expsrcft minus ctlft: 2,852 pairs, all pairs -0.6 (-2.1 to +0.8), among confident calls -1.4 (-3.8 to +0.9)
- expsrcft_s1 minus ctlft_s1: 2,852 pairs, all pairs -0.2 (-1.7 to +1.2), among confident calls -0.1 (-2.5 to +2.2)
- expsrcft_s2 minus ctlft_s2: 2,852 pairs, all pairs -0.0 (-1.6 to +1.3), among confident calls -1.6 (-4.3 to +1.0)

#### The 850 launch pairs (same account, both launch posts, test period)

| Model | All | Gap 2x+ | Gap 3x+ | Company | Founder | Under 10k followers | Confident 60%+ | Confident and 2x+ |
|---|---|---|---|---|---|---|---|---|
| ctlft_s1 | 64.8% (60.8 to 68.7) n=850 | 68.3% (63.5 to 73.2) n=568 | 70.1% (64.6 to 76.0) n=401 | 64.6% (60.2 to 69.2) n=601 | 69.8% (60.5 to 77.8) n=119 | 70.5% (50.9 to 83.9) n=78 | 72.9% (63.9 to 80.9) n=273, 32% of pairs | 74.1% (65.3 to 83.4) n=201, 35% of 2x pairs |
| ctlft_s2 | 63.1% (58.7 to 67.4) n=850 | 65.1% (59.8 to 70.1) n=568 | 66.8% (60.9 to 73.0) n=401 | 63.4% (58.1 to 68.6) n=601 | 61.3% (53.7 to 72.4) n=119 | 69.2% (51.6 to 80.3) n=78 | 72.9% (65.6 to 80.6) n=273, 32% of pairs | 75.0% (66.5 to 82.8) n=196, 35% of 2x pairs |
| expft_s1 | 63.5% (58.9 to 68.0) n=850 | 67.1% (61.3 to 72.3) n=568 | 68.8% (62.4 to 75.3) n=401 | 63.9% (58.4 to 69.4) n=601 | 62.2% (53.2 to 72.4) n=119 | 69.2% (44.4 to 85.8) n=78 | 75.8% (67.8 to 83.8) n=244, 29% of pairs | 78.3% (69.3 to 87.0) n=184, 32% of 2x pairs |
| expft_s2 | 65.3% (61.1 to 69.5) n=850 | 64.8% (59.4 to 70.0) n=568 | 66.3% (60.2 to 72.2) n=401 | 65.6% (60.2 to 71.4) n=601 | 63.0% (53.3 to 71.3) n=119 | 71.8% (55.4 to 82.8) n=78 | 71.5% (65.2 to 77.4) n=372, 44% of pairs | 73.0% (65.4 to 79.7) n=263, 46% of 2x pairs |
| expsrcft_s1 | 64.5% (60.1 to 68.9) n=850 | 65.7% (60.1 to 71.1) n=568 | 66.6% (60.2 to 73.0) n=401 | 65.4% (60.5 to 70.6) n=601 | 63.9% (55.2 to 74.4) n=119 | 70.5% (46.5 to 87.0) n=78 | 74.2% (67.2 to 81.8) n=279, 33% of pairs | 79.1% (71.4 to 87.1) n=206, 36% of 2x pairs |
| expsrcft_s2 | 65.8% (61.6 to 69.6) n=850 | 67.6% (62.4 to 72.4) n=568 | 71.1% (65.1 to 77.1) n=401 | 65.6% (60.7 to 70.2) n=601 | 67.2% (58.5 to 76.8) n=119 | 69.2% (50.0 to 82.6) n=78 | 74.2% (66.8 to 81.1) n=330, 39% of pairs | 76.8% (69.1 to 83.6) n=241, 42% of 2x pairs |
| Control, fine-tuned | 63.8% (59.7 to 67.8) n=850 | 66.7% (62.0 to 71.3) n=568 | 68.6% (62.9 to 74.4) n=401 | 64.1% (59.5 to 69.0) n=601 | 63.9% (55.0 to 71.9) n=119 | 66.7% (47.5 to 78.8) n=78 | 73.8% (65.8 to 82.1) n=263, 31% of pairs | 76.4% (67.4 to 84.7) n=191, 34% of 2x pairs |
| Expanded, fine-tuned | 65.6% (61.0 to 70.0) n=850 | 66.7% (61.3 to 71.7) n=568 | 68.1% (62.0 to 74.1) n=401 | 65.4% (60.0 to 71.1) n=601 | 64.7% (56.5 to 72.9) n=119 | 74.4% (52.3 to 89.8) n=78 | 73.4% (66.5 to 80.4) n=305, 36% of pairs | 77.4% (69.0 to 84.7) n=221, 39% of 2x pairs |
| Expanded + sampling input, fine-tuned | 64.8% (60.6 to 69.2) n=850 | 66.2% (60.7 to 71.5) n=568 | 68.3% (62.2 to 74.5) n=401 | 65.1% (60.3 to 70.3) n=601 | 65.5% (56.2 to 76.6) n=119 | 70.5% (46.8 to 87.2) n=78 | 74.1% (66.8 to 81.4) n=290, 34% of pairs | 77.0% (68.8 to 84.6) n=213, 38% of 2x pairs |

Paired differences on the launch pairs (points, accounts resampled):

- expft minus ctlft: 850 pairs, 151 accounts, +1.9 (-1.5 to +5.2)
- expft_s1 minus ctlft_s1: 850 pairs, 151 accounts, -1.3 (-4.8 to +1.9)
- expft_s2 minus ctlft_s2: 850 pairs, 151 accounts, +2.2 (-1.2 to +5.5)
- expsrcft minus ctlft: 850 pairs, 151 accounts, +1.1 (-1.7 to +3.7)
- expsrcft_s1 minus ctlft_s1: 850 pairs, 151 accounts, -0.4 (-3.8 to +2.9)
- expsrcft_s2 minus ctlft_s2: 850 pairs, 151 accounts, +2.7 (-0.4 to +5.7)

#### Level of the forecast for launch posts (draft path; likes over the account's trailing median)

| Model | Slice | Posts | Accounts | Actual | Predicted (95% interval) | Actual beats own median | Predicted to beat it | Rank correlation of predicted and actual lift | Median predicted / actual likes |
|---|---|---|---|---|---|---|---|---|---|
| ctlft_s1 | all | 576 | 167 | 2.19x | 1.28x (1.11 to 1.42) | 76% | 60% | 0.49 | 0.57 |
| ctlft_s1 | company | 326 | 73 | 2.45x | 1.30x (1.05 to 1.46) | 82% | 59% | 0.54 | 0.49 |
| ctlft_s1 | founder | 118 | 45 | 2.09x | 1.29x (0.88 to 1.61) | 70% | 58% | 0.49 | 0.64 |
| ctlft_s1 | under 10k followers | 69 | 25 | 1.58x | 1.30x (1.17 to 1.91) | 67% | 70% | 0.61 | 1.02 |
| ctlft_s2 | all | 576 | 167 | 2.19x | 1.48x (1.30 to 1.73) | 76% | 70% | 0.43 | 0.67 |
| ctlft_s2 | company | 326 | 73 | 2.45x | 1.44x (1.24 to 1.72) | 82% | 70% | 0.48 | 0.58 |
| ctlft_s2 | founder | 118 | 45 | 2.09x | 1.48x (1.10 to 2.38) | 70% | 65% | 0.42 | 0.65 |
| ctlft_s2 | under 10k followers | 69 | 25 | 1.58x | 1.44x (1.15 to 2.07) | 67% | 74% | 0.61 | 1.03 |
| expft_s1 | all | 576 | 167 | 2.19x | 1.06x (0.97 to 1.20) | 76% | 54% | 0.46 | 0.49 |
| expft_s1 | company | 326 | 73 | 2.45x | 0.99x (0.88 to 1.18) | 82% | 49% | 0.54 | 0.41 |
| expft_s1 | founder | 118 | 45 | 2.09x | 1.07x (0.89 to 1.38) | 70% | 57% | 0.48 | 0.54 |
| expft_s1 | under 10k followers | 69 | 25 | 1.58x | 1.25x (1.15 to 1.86) | 67% | 72% | 0.56 | 0.92 |
| expft_s2 | all | 576 | 167 | 2.19x | 1.75x (1.52 to 1.94) | 76% | 73% | 0.47 | 0.81 |
| expft_s2 | company | 326 | 73 | 2.45x | 1.60x (1.45 to 1.94) | 82% | 74% | 0.53 | 0.64 |
| expft_s2 | founder | 118 | 45 | 2.09x | 1.84x (1.37 to 2.54) | 70% | 72% | 0.49 | 0.97 |
| expft_s2 | under 10k followers | 69 | 25 | 1.58x | 1.84x (1.22 to 2.56) | 67% | 72% | 0.50 | 1.20 |
| expsrcft_s1 | all | 576 | 167 | 2.19x | 1.20x (1.08 to 1.34) | 76% | 62% | 0.47 | 0.54 |
| expsrcft_s1 | company | 326 | 73 | 2.45x | 1.17x (1.01 to 1.39) | 82% | 59% | 0.54 | 0.45 |
| expsrcft_s1 | founder | 118 | 45 | 2.09x | 1.23x (0.98 to 1.53) | 70% | 63% | 0.46 | 0.54 |
| expsrcft_s1 | under 10k followers | 69 | 25 | 1.58x | 1.51x (1.21 to 1.99) | 67% | 75% | 0.58 | 0.99 |
| expsrcft_s2 | all | 576 | 167 | 2.19x | 1.64x (1.46 to 1.80) | 76% | 75% | 0.47 | 0.74 |
| expsrcft_s2 | company | 326 | 73 | 2.45x | 1.62x (1.41 to 1.86) | 82% | 75% | 0.51 | 0.61 |
| expsrcft_s2 | founder | 118 | 45 | 2.09x | 1.69x (1.26 to 2.10) | 70% | 75% | 0.52 | 0.79 |
| expsrcft_s2 | under 10k followers | 69 | 25 | 1.58x | 1.69x (1.33 to 2.02) | 67% | 80% | 0.54 | 1.16 |
| Control, fine-tuned | all | 576 | 167 | 2.19x | 1.36x (1.22 to 1.57) | 76% | 66% | 0.47 | 0.62 |
| Control, fine-tuned | company | 326 | 73 | 2.45x | 1.35x (1.20 to 1.57) | 82% | 65% | 0.53 | 0.53 |
| Control, fine-tuned | founder | 118 | 45 | 2.09x | 1.36x (0.96 to 1.91) | 70% | 64% | 0.46 | 0.69 |
| Control, fine-tuned | under 10k followers | 69 | 25 | 1.58x | 1.49x (1.12 to 1.88) | 67% | 72% | 0.62 | 1.08 |
| Expanded, fine-tuned | all | 576 | 167 | 2.19x | 1.39x (1.24 to 1.55) | 76% | 67% | 0.48 | 0.64 |
| Expanded, fine-tuned | company | 326 | 73 | 2.45x | 1.30x (1.14 to 1.51) | 82% | 64% | 0.55 | 0.49 |
| Expanded, fine-tuned | founder | 118 | 45 | 2.09x | 1.46x (1.09 to 1.78) | 70% | 68% | 0.49 | 0.69 |
| Expanded, fine-tuned | under 10k followers | 69 | 25 | 1.58x | 1.71x (1.21 to 2.02) | 67% | 70% | 0.53 | 1.03 |
| Expanded + sampling input, fine-tuned | all | 576 | 167 | 2.19x | 1.42x (1.22 to 1.57) | 76% | 70% | 0.48 | 0.62 |
| Expanded + sampling input, fine-tuned | company | 326 | 73 | 2.45x | 1.37x (1.15 to 1.67) | 82% | 68% | 0.54 | 0.52 |
| Expanded + sampling input, fine-tuned | founder | 118 | 45 | 2.09x | 1.54x (1.12 to 1.76) | 70% | 69% | 0.51 | 0.68 |
| Expanded + sampling input, fine-tuned | under 10k followers | 69 | 25 | 1.58x | 1.61x (1.23 to 2.13) | 67% | 78% | 0.57 | 1.07 |

### What the fine-tune tables say

1. **Fine-tuned, the expanded models are still level with the fine-tuned control.** Benchmark, two-seed ensembles on 2,851 pairs: 60.2%
   for the control, 60.4% Expanded (+0.2, -1.2 to +1.6), 59.7% with the sampling input (-0.6, -2.1 to +0.8). Launch pairs (850): 63.8%,
   65.6% (+1.9, -1.5 to +5.2), 64.8% (+1.1, -1.7 to +3.7). Per seed the signs differ again (benchmark -0.7 and +0.9 for Expanded).
2. **The fine-tune does for the expanded models what it does for the control.** Against their own starting ensembles on the benchmark:
   control 59.5% to 60.2%, Expanded 59.8% to 60.4%, with the sampling input 59.1% to 59.7%. Confident calls on pairs 2x apart grow from
   25%, 23% and 25% of pairs to 32%, 31% and 29%, at 71.3%, 70.5% and 71.1% correct (69.6%, 69.8%, 70.3% before). The pairs track's
   result (more confident calls, about the same accuracy on them, under a point overall with two members) is reproduced on models trained
   on different data; it does not grow with the added data.
3. **Ordinary pairs hold.** Existing test rows, likes: control 59.91% and 60.21%, Expanded 59.77% and 59.79%, with the sampling input
   59.68% and 60.15% (683,152 pairs each). Views: 62.14% and 61.67%; 62.05% and 62.02%; 61.84% and 62.62% (24,481 pairs).
4. **A side effect worth knowing: the fine-tune raises the predicted launch level.** From 0.98x, 1.00x and 1.14x to 1.36x (1.22 to 1.57),
   1.39x (1.24 to 1.55) and 1.42x (1.22 to 1.57), against 2.19x; for accounts under 10,000 followers (69 posts, actual 1.58x) to 1.49x,
   1.71x and 1.61x. The pair loss was not built for level, the like NLL rises by about 0.02 as the pairs track reported, and nobody
   should rely on this; it is noted because it is larger than anything the added data did to the level.

## 6. What worked, what did not

Worked:
- The build: 293,036 rows through the pipeline's own functions, checked against stored rows, in about three minutes once memory was under control; a
  table that shares the existing table's input space, so every existing evaluation runs on the new models unchanged.
- The level check found a real incompatibility (archive rows are a success-biased sample, about 4x) that no one had measured, and the
  sampling input is a cheap way to carry it.
- Training is deterministic here: the retrained control is bit-identical to the stored model, which makes paired comparisons clean.
- The expanded models are better on accounts of the new kinds (+3.3 points on held-out twitter100m accounts) and forecast full timelines
  at the right level.

Did not work:
- More data of this kind does not raise accuracy on the benchmark, on launch pairs, or on ordinary pairs of the existing test accounts.
- Small accounts: 129,586 added posts from accounts under 10,000 followers, no change on the existing small accounts.
- Launch level: 1,866 more launch-type training posts, predicted level 0.98x to 1.00x (1.14x with the sampling input) against 2.19x.
- The 2026 view corpus added almost nothing: of 3,018 startup accounts found there, 561 remain with 6,196 rows. It is a sample of posts,
  not timelines, so most accounts have too few mature posts to form a history (30,632 originals and quotes, 12,695 before May 2026, 8,770 from
  accounts with 20 or more).

Why, as far as the numbers go (interpretation, not measured): 85% of the added rows are likes-only timelines from 2020 to April 2023 of
organisations and founders of every kind, weighted a quarter each; the test questions are about archive accounts in mid-2026. Only 44,589
added rows have views and only 6,196 are from 2026. And the benchmark's resolution with two seeds is about 1.5 points, with a label-noise
ceiling the pairs track put near 80%.

## 7. Honest limits

- Two seeds per family. On the benchmark and the launch pairs nothing under about 1.5 and 3 points can be seen; the per-seed differences
  have opposite signs. The +0.7 for company and founder accounts is one slice of about twenty.
- `ctl_s2` is the stored `runs/ts46_s2`, not a retrain; the claim that it would come out identical rests on `ctl_s1` doing so.
- The runs choose their checkpoint on different validation rows (20,684 for the control, 50,229 for the expanded runs). The expanded runs
  were still improving at step 4,000; the control peaked at 3,330 and 3,996.
- The startup account list is loose: 139,357 of the added rows come from accounts whose only evidence is a bio that reads as a company or
  founder. A stricter list would add about 154,000 rows. Not trained separately (`scripts/train_one.py --train-exclude
  data/expand/bio_only_ids.npy` does it without rebuilding the table).
- Only posts from the three new sources were added. Company and founder accounts that sit in `work/filtered/posts.parquet` but were not
  selected into the existing rows were not pulled in.
- New rows differ from existing rows in ways that are documented, not removed: near-duplicate clusters computed among new rows only;
  existing rows' neighbours never point at new rows; `reply_share` of new accounts comes from the new sources' reply rows; the verified
  flag and video durations are blanked in the table.
- 67 accounts have a history that steps from a full timeline into archive rows (section 2).
- twitter100m follower counts are one reading from April 2023, applied to posts back to 2020.
- The launch level uses the launch track's trailing median and its 582 posts; "under 10,000 followers" there is 69 posts of 25 accounts.
- The pair fine-tune was run with the weight the pairs track chose with its benchmark in view; that caveat carries over.
- No step count other than 4,000 was tried for the control, so `explong_s1` is compared with a control that had fewer steps.
- Tests: 122 pass, 1 skipped. `test_serve.py::test_product_service_and_http` fails on a module that is not installed here;
  `test_end_to_end.py::test_phase0_end_to_end` failed once in the full run and passes alone, with and without these changes; the two
  failures the pairs track listed (missing `edit_evidence.json`) were deselected.

## 8. What I would do next

1. Do not merge twitter100m for accuracy; it is safe to merge (nothing got worse) if small-account or full-timeline coverage is wanted, and
   then with the sampling input.
2. Test the sampling input where it should matter: full timelines of 2026 startup accounts with views (the owner's scrape of the 14,732
   listed accounts without posts). The brand accounts say the existing models forecast such accounts about 1.5x too high.
3. Tighten the startup list before the scrape (bio-only evidence is half organisations of other kinds).
4. Keep the launch level with the post-type correction; data volume did not move it.

## Exact commands

From `virality/`, after the setup lines of `scripts/worker_job.sh` (venv, CPU torch, `pip install -e . sentence-transformers peft jax huggingface_hub`)
and with `HF_TOKEN=hf_proxyinjected`. Run one heavy job at a time: two together on these 4-core machines ran at a fifth of the speed.

```bash
# data: the existing work files, the three converted sources, the benchmark caches, the launch tables
python - <<'E'
from huggingface_hub import snapshot_download
snapshot_download("sam-siavoshian/xviral-work", repo_type="dataset", local_dir="data", allow_patterns=[
  "config.yaml", "work/model/table/*", "work/emb/*", "work/filtered/posts.parquet", "work/dataset/rows.parquet",
  "tracks/data/*.parquet", "tracks/data/sources/*", "tracks/data/sources/zenodo_brands/*", "tracks/data/sources/twitter100m/*",
  "tracks/data/sources/nitter_2026_replies/*.parquet", "tracks/pairs/benchmark_pairs.parquet", "tracks/pairs/bench/cache_*.pkl",
  "tracks/pairs/pair_table/*", "tracks/pairs/train_pairs.parquet", "tracks/pairs/runs/ft_w10/*", "tracks/pairs/runs/ft_w10_s2/*",
  "tracks/launch/*", "runs/ts46_s1/*", "runs/ts46_s2/*", "work/model/compare_v2/full/*"])
E
mkdir -p data/launch data/pairs data/bench3 data/expand/launch data/expand/eval
cp data/tracks/launch/launch_posts.parquet data/launch/table.parquet; cp data/tracks/pairs/benchmark_pairs.parquet data/pairs/
for i in 0 1 2 3; do ln -sf $PWD/data/tracks/pairs/bench/cache_$i.pkl data/bench3/cache_$i.pkl; done

# 1. rows
python scripts/expand/build_rows.py --check 150 --out data/expand          # the same code on 150 existing accounts, compared with their stored rows
python scripts/expand/build_rows.py --cap 400 --out data/expand            # data/expand/new_rows.parquet, build_counts.json, filter_log.csv
python scripts/expand/added_report.py                                      # added.md, added.json, typed_posts.parquet
python scripts/expand/source_levels.py                                     # source_levels.json

# 2. embeddings and table
OMP_NUM_THREADS=3 python scripts/expand/embed_rows.py --rows data/expand/new_rows.parquet --out data/expand/emb --check-table data/work/model/table
python scripts/expand/build_table.py --out data/expand/table
python -c "import numpy as np; d='data/work/model/table/'; np.save('data/expand/orig_test_ids.npy', np.load(d+'post_id.npy')[(np.load(d+'split.npy')==2)&np.load(d+'audience.npy')])"

# 3. models (S = 1, 2).  T="--features ts46 --preset small --steps 4000"
python scripts/train_one.py --table data/work/model/table --out data/expand/runs/ctl_s$S --seed $S $T                          # control; identical to runs/ts46_s$S
python scripts/train_one.py --table data/expand/table --out data/expand/runs/exp_s$S --seed $S $T --drop-cat src_regime --eval-rows data/expand/orig_test_ids.npy
python scripts/train_one.py --table data/expand/table --out data/expand/runs/expsrc_s$S --seed $S $T --eval-rows data/expand/orig_test_ids.npy
python scripts/train_one.py --table data/expand/table --out data/expand/runs/explong_s1 --seed 1 --features ts46 --preset small --steps 7450 --drop-cat src_regime --eval-rows data/expand/orig_test_ids.npy

# 4. evaluation
bash scripts/expand/after_train.sh <name> data/expand/runs/<name>/best.pt   # benchmark (scripts/bench_pairs.sh) and launch posts (scripts/expand/score_launch.py)
python scripts/expand/eval_table.py --table data/expand/table --out data/expand/eval/table_slices.json --run ctl_s1=... --run exp_s1=... (every model)
python scripts/expand/summarize.py --families ctl,exp,expsrc,explong --out data/expand/eval/results

# 5. pair-loss fine-tune (the pairs track's recipe).  FT="--pair-info data/tracks/pairs/train_pairs.parquet --features ts46 --steps 600 --lr 2e-4 --w-ctl 10 --eval-rows data/expand/orig_test_ids.npy"
python scripts/train_pairs.py --table data/expand/table --pair-table data/tracks/pairs/pair_table --out data/expand/runs/expft_s$S --init data/expand/runs/exp_s$S/best.pt --drop-cat src_regime $FT
python scripts/expand/pair_table_regime.py                                  # the pair table with the src_regime column, data/expand/pair_table_src
python scripts/train_pairs.py --table data/expand/table --pair-table data/expand/pair_table_src --out data/expand/runs/expsrcft_s$S --init data/expand/runs/expsrc_s$S/best.pt $FT
python scripts/expand/summarize.py --families ctlft,expft,expsrcft --out data/expand/eval/results_ft
```

The queue that ran them in order is `scripts/expand/queue.sh` (job files in `queue/`).

## Code changes (`patch.diff`)

- `xviral/model/features.py`: the optional categorical input `src_regime` (`SRC_REGIME`, `with_src_regime`; only in tables built with the
  column, derived from `source` at serving time), and `FeatureTable.without_cat`.
- `scripts/train_one.py`: `--drop-cat`, `--eval-rows`, `--train-exclude`. `scripts/train_pairs.py`: `--drop-cat`, `--eval-rows`.
- New, under `scripts/expand/`: `build_rows.py`, `added_report.py`, `source_levels.py`, `embed_rows.py`, `build_table.py`, `score_launch.py`
  (the launch track's scorer with a feature cache), `launch_check.py` (its pair and level check for any model), `eval_table.py`,
  `pair_table_regime.py`, `summarize.py`, `after_train.sh`, `queue.sh`.
- Nothing committed.

## Files (`tracks/expand/`)

| Path | Content |
|---|---|
| `REPORT.md`, `patch.diff` | this report; code changes and new files |
| `rows/new_rows.parquet` | the 293,036 added rows (columns of `work/dataset/rows.parquet` plus `expand_src`, `acct_kind`, `kind_from`, `in_brand_set`, `acct_in_original`) |
| `rows/rows_expanded.parquet` | the whole expanded set, 758,711 rows, with `is_new` |
| `rows/build_counts.json`, `filter_log.csv`, `added.md`, `added.json`, `typed_posts.parquet`, `source_levels.json`, `replication_check.json` | counts by stage; what was added by source, kind, tier, type, year and evidence; post types; the level comparison; the check against stored rows |
| `table/` | the expanded model table (`meta.json`, arrays, `texts.parquet`, `row_info.parquet`): ready for `scripts/train_one.py --table` |
| `embedding_check.txt` | bf16 vectors against the stored ones |
| `orig_test_ids.npy` | post ids of the existing audience test rows (`--eval-rows`) |
| `pair_table_src/` | `cat.npy` and `meta.json` of the pair table with `src_regime` (the other arrays are those of `tracks/pairs/pair_table`) |
| `runs/<name>/best.pt`, `metrics.json` | every model: `ctl_s1`, `ctl_s2`, `exp_s1`, `exp_s2`, `expsrc_s1`, `expsrc_s2`, `explong_s1`, and the fine-tuned `ctlft_*`, `expft_*`, `expsrcft_*` |
| `bench/<name>/shard_*.parquet` | per-pair scores on the 3,530 benchmark pairs |
| `launch/<name>.parquet` | forecasts for the 582 launch posts; `baseline_ens.parquet` is the README baseline ensemble rescored |
| `eval/results.md`, `results.json`, `results_bench.json`, `results_launch.json` | every table of section 4 and the numbers behind it; `results_ft*` for section 5 |
| `eval/table_slices.json`, `.npz` | ordinary pairs by slice for every model, with per-account tallies |
| `queue/` | the job files as they ran, with start and end times |
