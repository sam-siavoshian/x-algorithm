# Track "embed_api": large API embeddings and a supervised adapter

2026-11-17 and 2026-11-19 (the session stopped on a usage limit in between), one 4-core CPU machine, about 6 hours of work.

## Answer

**Replacing bge-small with OpenAI `text-embedding-3-large` (shortened to 1,024 dimensions) is the first change to the model's inputs that raises pair accuracy on every test set.** Two seeds against the two existing bge-small seeds (`ts46_s1`, `ts46_s2`), same table, same script, same 4,000 steps, the same embedding at serving time:

| Test set | Pairs | bge-small control | 3-large, 1024 | Difference (accounts resampled) |
|---|---|---|---|---|
| Ordinary same-account pairs, likes, after the cutoff | 683,152 | 60.0% (59.8, 60.1) | **62.3%** (62.4, 62.2) | +2.3, both seeds above both control seeds |
| Same, within-account rank correlation | 95,357 posts | 0.298 | **0.361** | +0.063 |
| Ordinary same-account pairs, views | 24,481 | 62.1% | **64.0%** | +1.9 |
| Same-topic benchmark, gap 1.25x+ | 2,852 | 59.5% | **61.7%** | +2.2 (+0.4 to +3.9) |
| Same-topic benchmark, gap 2x+ | 2,030 | 60.5% | **63.5%** | +3.0 (+1.2 to +5.1) |
| Benchmark, confident calls (60%+) on 2x pairs | 2,030 | 25% of pairs at 69.6% | **32% at 72.1%** | coverage +6.7 (+4.3 to +9.0), accuracy +2.5 (-1.2 to +6.7) |
| Benchmark, test-period pairs only, 1.25x+ | 1,660 | 60.1% | 61.6% | +1.4 (-0.8 to +3.5) |
| Benchmark, test-period pairs only, 2x+ | 1,117 | 60.8% | 63.0% | +2.2 (-0.3 to +4.8) |
| Launch pairs, gap 1.25x+ | 850 | 63.4% | **67.2%** | +3.8 (+0.1 to +7.6) |
| Launch pairs, gap 2x+ | 568 | 66.2% | **72.0%** | +5.8 (+1.4 to +10.0) |
| Launch pairs, confident calls on 2x pairs | 568 | 42% at 77.1% | **54% at 81.8%** | |

The supervised adapter (step 3) is a null result through the full model: it improves a stand-alone ranking of an account's posts by 1 to 2 points over the raw embedding, on held-out accounts, on a later period and on both, but the full model gains nothing from it on bge-small or on the OpenAI embedding (bge-small + adapter: ordinary pairs 59.7% against 60.0%, benchmark +0.1; 3-large + adapter against 3-large alone: ordinary pairs 61.9% against 62.3%, benchmark -0.1, and fewer confident calls).

`text-embedding-3-small` (1,024 dimensions), at 15% of the price, gets a bit over half of the gain: ordinary pairs 61.4%, benchmark +1.8 (-0.2 to +3.3) and +2.3 (+0.5 to +4.1) at 2x with the same rise in confident coverage (25% to 33% of 2x pairs at 72.3%), launch pairs +1.6 (-2.0 to +5.3).

The goal (75 to 80% on confident calls covering 40% of 2x pairs) is not reached on the same-topic benchmark: the surest 40% of 2x pairs are 70.8% correct (control 66.5%). On launch pairs the surest 40% of 2x pairs are 82.4% correct (control 78.0%), and confident calls cover 54% of them at 81.8%.

Spend: $11.54 of the $40 allowed, 153.8M embedding tokens (3-large 76.9M tokens, $10.00; 3-small 76.9M tokens, $1.54; latency test $0.01). No chat model was used.

## Step 1. Probe

`embed_pairs.py`, `probe.py` (the encoder track's, unchanged), `openai_embed.py`; results `probe_api.json`, `probe_api.txt`, per-pair scores `probe_api_scores.parquet`.

Exactly the encoder track's setup: the 9,188 same-topic pairs (7,158 before the cutoff, 2,030 in the test period = the benchmark's test pairs), rendered text, antisymmetric logistic regression on the standardised embedding difference, trained on 5,791 pre-cutoff pairs with a 25%+ gap, regularisation chosen by 5-fold cross-validation over pre-cutoff accounts, scored on the 1,660 test-period pairs with a 25%+ gap. Shorter sizes are the leading coordinates renormalised, which is what the API's `dimensions` parameter returns. Cost: 15,546 distinct texts, 1.87M tokens per model, $0.24 (large) + $0.04 (small).

Embedding difference alone:

| Encoder | Dim | Pre-cutoff, held-out accounts | Test 1.25x (1,660) | 2x (1,117) | 3x (733) | 5x (440) | Confident 60%+: pairs, correct | vs bge-small |
|---|---|---|---|---|---|---|---|---|
| bge-small (control) | 384 | 53.7% | 54.9% | 55.1% | 56.8% | 60.2% | 195 at 57.9% | |
| 3-large | 3072 | 60.1% | 58.0% | 59.8% | 62.9% | 65.0% | 509 at 60.5% | +3.1 [-0.1, +6.2] |
| 3-large | 1024 | 59.0% | 57.7% | 59.5% | 63.4% | 65.2% | 432 at 61.6% | +2.8 [-0.9, +5.8] |
| 3-large | 256 | 56.8% | 57.2% | 58.2% | 61.8% | 63.9% | 279 at 59.1% | +2.3 [-0.7, +5.2] |
| 3-small | 1536 | 58.0% | 58.4% | 58.6% | 61.5% | 63.6% | 391 at 63.9% | +3.6 [+0.5, +6.6] |
| 3-small | 512 | 58.1% | 57.7% | 58.1% | 61.8% | 64.8% | 345 at 60.3% | +2.8 [-0.3, +5.6] |
| bge-small + 3-large 3072 | 3456 | 59.5% | 59.2% | 61.1% | 64.1% | 65.7% | 470 at 61.3% | +4.3 [+1.4, +7.3] |
| bge-small + 3-large 1024 | 1408 | 58.4% | 56.8% | 57.7% | 61.9% | 65.2% | 396 at 62.1% | +1.9 [-1.0, +4.4] |
| bge-small + 3-large 256 | 640 | 56.3% | 56.5% | 57.1% | 59.5% | 62.5% | 312 at 59.3% | +1.6 [-0.4, +3.9] |
| bge-small + 3-small 1536 | 1920 | 57.9% | 58.1% | 58.4% | 61.8% | 63.6% | 401 at 61.3% | +3.3 [+0.7, +5.6] |

Embedding difference plus the 50 structure, media and link differences:

| Encoder | Dim | Pre-cutoff, held-out accounts | Test 1.25x (1,660) | 2x (1,117) | 3x (733) | 5x (440) | Confident 60%+: pairs, correct | vs bge-small |
|---|---|---|---|---|---|---|---|---|
| structure columns only | 50 | 68.2% | 56.6% | 60.1% | 64.0% | 69.5% | 1,035 at 59.4% | |
| bge-small (control) | 434 | 67.4% | 58.3% | 61.2% | 66.4% | 71.8% | 1,019 at 62.2% | |
| 3-large | 3122 | 67.6% | 60.2% | 62.8% | 67.3% | 70.2% | 1,025 at 63.7% | +1.9 [-0.4, +4.3] |
| 3-large | 1074 | 68.2% | 59.7% | 63.8% | 69.8% | 71.6% | 1,117 at 61.5% | +1.4 [-0.7, +3.8] |
| 3-large | 306 | 68.1% | 58.6% | 61.9% | 67.3% | 71.4% | 1,120 at 61.6% | +0.4 [-1.2, +2.7] |
| 3-small | 1586 | 67.2% | 60.2% | 62.6% | 67.7% | 72.5% | 1,014 at 64.1% | +1.9 [-0.2, +4.3] |
| 3-small | 562 | 67.7% | 59.2% | 61.4% | 66.3% | 71.6% | 1,032 at 62.3% | +0.9 [-1.0, +3.0] |
| bge-small + 3-large 3072 | 3506 | 67.3% | 60.8% | 64.5% | 70.7% | 73.6% | 956 at 62.9% | +2.6 [+0.5, +5.1] |
| bge-small + 3-large 1024 | 1458 | 67.8% | 60.1% | 63.0% | 68.8% | 71.6% | 1,053 at 63.3% | +1.9 [-0.4, +4.1] |
| bge-small + 3-large 256 | 690 | 67.4% | 59.5% | 62.9% | 68.1% | 71.1% | 1,051 at 63.1% | +1.2 [-0.7, +3.2] |
| bge-small + 3-small 1536 | 1970 | 67.1% | 59.8% | 63.2% | 67.1% | 71.6% | 954 at 64.4% | +1.6 [-0.6, +4.0] |

Reading: every OpenAI variant is ahead of bge-small, alone by 2 to 4 points and beside the structure columns by 1 to 2.6. Unlike the Twitter-domain encoders of the encoder track, whose pre-cutoff advantage (58 to 59%) mostly vanished after the cutoff, these hold in the test period (57 to 59% against 55%). Size matters little above 512 dimensions; 256 loses about a point. The small model is level with the large one in this probe. No single row is decisive (most intervals touch zero), but all 18 point the same way, so step 2 was run.

## Step 2. The model retrained on the API embedding

`openai_embed.py` (batched, cached on disk, resumable, with a token and dollar ledger), `recipe.py` (how the embedding is assembled, shared by training and serving), `api_table.py`, `bench_api.py`, `score.sh`, `eval_splits.py`, `score_launch_api.py`, `results.py`; results `results.json`, `results.txt`, `results.md`.

**Cost, estimated first.** A 2,000-text sample of the table's rendered texts gave 138.4 tokens per text, projecting 64.5M tokens and $8.38 for 3-large. Actual: 458,237 distinct texts (of 465,675 rows), 66.29M tokens, $8.62 for 3-large at the native 3,072 dimensions (every shorter size derives from it), and $1.33 for 3-small. The benchmark's history posts and drafts and the launch posts added 123,055 texts not in the table: 8.76M tokens, $1.14 (large) and $0.18 (small). Texts are cut at 8,000 characters (the API limit is 8,192 tokens; bge-small reads 512). Both stores are in the dataset under `tracks/embed_api/store/` keyed by a hash of the text, so the same text is never paid for twice; `openai_embed.ApiEmbedder` reads them directly.

**What was changed.** Only the table's post embedding and its semantic IDs (codebook retrained on training-split rows of the new embedding). 3-large is shortened to 1,024 dimensions and used as is: its random-pair cosine is 0.21 +/- 0.09 (bge-small 0.56 +/- 0.07), so no centring was needed for the history-similarity kernel, and the trunk takes 1,024 inputs directly, so no PCA was applied (`api_table.py --pca` exists and is untested beyond a run-through). Neighbour lists stay as built with bge-small, as in the encoder track; serving passes none. 3-small is treated the same way (first 1,024 of its 1,536 dimensions; random-pair cosine 0.21 +/- 0.09). Training: `scripts/train_one.py --features ts46 --preset small --steps 4000`, seeds 1 and 2, unchanged, on the full 465,675-row table. Five arms were trained, ten runs: 3-large, 3-small, bge-small + adapter, 3-large + adapter (step 3), each against the bge-small control. The control is the two existing full-data models `ts46_s1` and `ts46_s2`, which came from the same script with the same arguments; both were re-scored here through the same code path as the new models (their re-scored benchmark results match the stored `bench2` scores: 1 sign flip in 3,530 pairs).

**Serving.** `bench_api.py` rewrites the benchmark's and the launch posts' feature caches so that every history post and draft carries the new embedding and the new semantic IDs; everything else in the cache (the model-independent features of the baseline run) is untouched, so every model is scored on exactly the same pairs and features.

### Train, validation, test: ordinary same-account pairs

Pairwise accuracy on likes among posts of one audience account. Train: 700 of the accounts the model was fitted on, before the cutoff. Validation: held-out accounts, before the cutoff. Test: after the cutoff (683,152 pairs). Means of two seeds; `runs/<name>/splits.json` has each seed.

| Model | Train | Validation | Test | Test, each seed | Rank correlation train / validation / test | Likes NLL train / validation / test | Views pairs, test |
|---|---|---|---|---|---|---|---|
| bge-small (control) | 62.9% | 61.6% | **60.0%** | 59.8, 60.1 | 0.330 / 0.310 / 0.298 | 2.907 / 2.991 / 2.965 | 62.1% |
| 3-large, 1024 | 65.4% | 64.5% | **62.3%** | 62.4, 62.2 | 0.389 / 0.355 / 0.361 | 2.872 / 2.963 / 2.941 | 64.0% |
| 3-small, 1024 | 65.1% | 63.5% | **61.4%** | 61.3, 61.4 | 0.378 / 0.335 / 0.344 | 2.878 / 2.971 / 2.948 | 63.3% |
| bge-small + adapter | 62.3% | 61.6% | **59.7%** | 59.8, 59.7 | 0.345 / 0.312 / 0.303 | 2.907 / 2.989 / 2.965 | 61.7% |
| 3-large, 1024 + adapter | 65.3% | 64.4% | **61.9%** | 62.0, 61.8 | 0.434 / 0.347 / 0.359 | 2.859 / 2.963 / 2.938 | 63.7% |
| control and 3-large together (4 models) | 64.1% | 63.1% | **61.1%** | 59.8, 60.1, 62.4, 62.2 | 0.360 / 0.333 / 0.329 | 2.889 / 2.977 / 2.953 | 63.0% |

The train-to-validation gap is similar for every arm (0.7 to 1.6 points) and so is the validation-to-test drop (1.6 to 2.4 points, the period shift every track has seen); only the adapter arms show more fitting to their own accounts, in rank correlation (see step 3). The gain is the same size on held-out accounts before the cutoff (+2.9) and after it (+2.3): it is not an artefact of one split.

### Same-topic benchmark (3,530 pairs; 2,852 with a 25%+ gap) and launch pairs (850)

Two-seed score ensembles. "Confident" is stated confidence of 60% or more, scale cross-fitted over accounts (launch pairs: cross-fitted on the launch pairs). "Surest 40%" is accuracy on the 40% of pairs with the largest score gap, which needs no scale.

Same-topic benchmark, all 3,530 pairs:

| Model | Real gap at least | Pairs | Correct | Confident share (pairs) | Correct there | Surest 40% correct |
|---|---|---|---|---|---|---|
| bge-small (control) | 1.25x | 2,852 | 59.5% (57.3 to 61.7) | 27% (756) | 67.5% | 65.7% |
| bge-small (control) | 2.0x | 2,030 | 60.5% (57.3 to 63.2) | 25% (517) | 69.6% | 66.5% |
| bge-small (control) | 3.0x | 1,377 | 62.7% (59.5 to 65.9) | 27% (368) | 73.1% | 71.0% |
| bge-small (control) | 5.0x | 880 | 65.9% (62.0 to 70.2) | 28% (249) | 79.5% | 77.0% |
| 3-large, 1024 | 1.25x | 2,852 | 61.7% (59.1 to 64.1) | 34% (967) | 69.0% | 69.0% |
| 3-large, 1024 | 2.0x | 2,030 | 63.5% (60.9 to 65.9) | 32% (653) | 72.1% | 70.8% |
| 3-large, 1024 | 3.0x | 1,377 | 65.6% (62.2 to 68.9) | 33% (461) | 75.7% | 75.3% |
| 3-large, 1024 | 5.0x | 880 | 68.0% (63.4 to 72.0) | 32% (286) | 80.1% | 80.4% |
| 3-small, 1024 | 1.25x | 2,852 | 61.2% (58.7 to 63.6) | 35% (990) | 70.0% | 70.1% |
| 3-small, 1024 | 2.0x | 2,030 | 62.8% (59.6 to 65.8) | 33% (676) | 72.3% | 72.5% |
| 3-small, 1024 | 3.0x | 1,377 | 64.6% (61.2 to 68.0) | 34% (465) | 76.3% | 76.0% |
| 3-small, 1024 | 5.0x | 880 | 66.4% (61.8 to 70.7) | 34% (298) | 81.2% | 80.1% |
| bge-small + adapter | 1.25x | 2,852 | 59.5% (57.5 to 61.4) | 26% (736) | 66.4% | 66.3% |
| bge-small + adapter | 2.0x | 2,030 | 61.2% (58.9 to 63.5) | 25% (504) | 68.8% | 69.0% |
| bge-small + adapter | 3.0x | 1,377 | 64.3% (61.1 to 67.0) | 26% (352) | 72.7% | 73.7% |
| bge-small + adapter | 5.0x | 880 | 66.2% (62.6 to 69.6) | 27% (235) | 80.4% | 80.1% |
| 3-large, 1024 + adapter | 1.25x | 2,852 | 61.6% (59.5 to 63.7) | 28% (796) | 69.0% | 66.2% |
| 3-large, 1024 + adapter | 2.0x | 2,030 | 62.9% (60.4 to 65.2) | 27% (550) | 72.0% | 69.1% |
| 3-large, 1024 + adapter | 3.0x | 1,377 | 65.1% (61.7 to 68.6) | 27% (378) | 75.1% | 72.1% |
| 3-large, 1024 + adapter | 5.0x | 880 | 67.6% (63.4 to 71.5) | 29% (251) | 78.1% | 75.6% |
| control and 3-large together (4 models) | 1.25x | 2,852 | 61.4% (58.6 to 64.2) | 34% (960) | 70.0% | 69.1% |
| control and 3-large together (4 models) | 2.0x | 2,030 | 62.8% (59.6 to 65.7) | 33% (665) | 71.7% | 70.7% |
| control and 3-large together (4 models) | 3.0x | 1,377 | 65.2% (61.5 to 68.8) | 34% (464) | 76.3% | 75.5% |
| control and 3-large together (4 models) | 5.0x | 880 | 68.1% (63.5 to 72.7) | 34% (299) | 81.9% | 80.4% |

Same-topic benchmark, test-period pairs only:

| Model | Real gap at least | Pairs | Correct | Confident share (pairs) | Correct there | Surest 40% correct |
|---|---|---|---|---|---|---|
| bge-small (control) | 1.25x | 1,660 | 60.1% (57.8 to 62.6) | 28% (467) | 65.3% | 64.3% |
| bge-small (control) | 2.0x | 1,117 | 60.8% (57.6 to 63.8) | 28% (308) | 68.2% | 64.7% |
| bge-small (control) | 3.0x | 733 | 64.3% (60.1 to 68.5) | 29% (214) | 71.0% | 69.3% |
| bge-small (control) | 5.0x | 440 | 68.9% (63.1 to 74.4) | 30% (133) | 79.7% | 78.4% |
| 3-large, 1024 | 1.25x | 1,660 | 61.6% (59.0 to 64.3) | 34% (569) | 67.5% | 67.3% |
| 3-large, 1024 | 2.0x | 1,117 | 63.0% (60.0 to 66.2) | 32% (363) | 70.5% | 70.2% |
| 3-large, 1024 | 3.0x | 733 | 66.0% (61.6 to 69.9) | 35% (258) | 74.0% | 74.4% |
| 3-large, 1024 | 5.0x | 440 | 69.3% (63.8 to 74.8) | 34% (149) | 79.9% | 79.0% |
| 3-small, 1024 | 1.25x | 1,660 | 60.8% (58.1 to 63.8) | 36% (605) | 68.8% | 69.0% |
| 3-small, 1024 | 2.0x | 1,117 | 62.5% (59.8 to 65.5) | 36% (404) | 71.3% | 70.7% |
| 3-small, 1024 | 3.0x | 733 | 64.9% (59.7 to 68.9) | 38% (280) | 73.9% | 74.7% |
| 3-small, 1024 | 5.0x | 440 | 67.0% (60.9 to 72.4) | 39% (173) | 77.5% | 78.4% |
| bge-small + adapter | 1.25x | 1,660 | 59.9% (57.5 to 62.5) | 28% (460) | 66.7% | 66.3% |
| bge-small + adapter | 2.0x | 1,117 | 61.7% (58.3 to 64.7) | 26% (293) | 70.3% | 69.8% |
| bge-small + adapter | 3.0x | 733 | 66.4% (62.5 to 69.7) | 28% (203) | 74.9% | 74.1% |
| bge-small + adapter | 5.0x | 440 | 69.3% (64.6 to 73.7) | 30% (134) | 81.3% | 79.5% |
| 3-large, 1024 + adapter | 1.25x | 1,660 | 61.5% (59.3 to 64.5) | 30% (494) | 67.6% | 67.6% |
| 3-large, 1024 + adapter | 2.0x | 1,117 | 63.3% (60.3 to 66.0) | 29% (328) | 70.1% | 69.1% |
| 3-large, 1024 + adapter | 3.0x | 733 | 66.8% (62.3 to 70.6) | 32% (234) | 73.1% | 74.1% |
| 3-large, 1024 + adapter | 5.0x | 440 | 70.0% (64.3 to 74.8) | 33% (145) | 76.6% | 77.3% |
| control and 3-large together (4 models) | 1.25x | 1,660 | 61.4% (58.6 to 64.4) | 34% (570) | 67.0% | 67.9% |
| control and 3-large together (4 models) | 2.0x | 1,117 | 62.3% (59.1 to 65.8) | 34% (378) | 68.5% | 69.4% |
| control and 3-large together (4 models) | 3.0x | 733 | 65.8% (60.5 to 70.4) | 37% (270) | 73.0% | 74.4% |
| control and 3-large together (4 models) | 5.0x | 440 | 69.5% (63.4 to 75.1) | 38% (169) | 79.9% | 79.5% |

Launch pairs:

| Model | Real gap at least | Pairs | Correct | Confident share (pairs) | Correct there | Surest 40% correct |
|---|---|---|---|---|---|---|
| bge-small (control) | 1.25x | 850 | 63.4% (58.9 to 67.7) | 40% (341) | 72.7% | 74.4% |
| bge-small (control) | 2.0x | 568 | 66.2% (61.1 to 70.8) | 42% (240) | 77.1% | 78.0% |
| bge-small (control) | 3.0x | 401 | 68.8% (63.2 to 75.3) | 45% (180) | 78.3% | 78.1% |
| bge-small (control) | 5.0x | 240 | 70.0% (61.6 to 76.7) | 48% (115) | 75.7% | 72.9% |
| 3-large, 1024 | 1.25x | 850 | 67.2% (63.0 to 71.6) | 51% (435) | 77.0% | 77.1% |
| 3-large, 1024 | 2.0x | 568 | 72.0% (66.9 to 77.4) | 54% (308) | 81.8% | 82.4% |
| 3-large, 1024 | 3.0x | 401 | 74.3% (67.8 to 80.7) | 57% (229) | 83.4% | 85.6% |
| 3-large, 1024 | 5.0x | 240 | 74.2% (65.1 to 81.3) | 60% (144) | 82.6% | 83.3% |
| 3-small, 1024 | 1.25x | 850 | 65.1% (60.7 to 69.5) | 53% (449) | 74.8% | 77.4% |
| 3-small, 1024 | 2.0x | 568 | 69.0% (63.6 to 74.5) | 55% (311) | 78.5% | 80.2% |
| 3-small, 1024 | 3.0x | 401 | 69.8% (63.0 to 76.1) | 55% (220) | 80.9% | 80.6% |
| 3-small, 1024 | 5.0x | 240 | 68.8% (60.9 to 74.9) | 58% (139) | 79.1% | 75.0% |
| bge-small + adapter | 1.25x | 850 | 62.5% (57.8 to 67.8) | 32% (268) | 70.9% | 70.9% |
| bge-small + adapter | 2.0x | 568 | 64.8% (58.1 to 69.9) | 35% (197) | 73.1% | 73.6% |
| bge-small + adapter | 3.0x | 401 | 67.6% (62.0 to 73.7) | 37% (149) | 74.5% | 75.0% |
| bge-small + adapter | 5.0x | 240 | 67.9% (58.5 to 75.3) | 38% (91) | 70.3% | 71.9% |
| 3-large, 1024 + adapter | 1.25x | 850 | 64.8% (61.2 to 68.2) | 51% (431) | 74.7% | 76.8% |
| 3-large, 1024 + adapter | 2.0x | 568 | 67.4% (62.8 to 71.8) | 54% (307) | 77.5% | 79.3% |
| 3-large, 1024 + adapter | 3.0x | 401 | 71.3% (66.0 to 76.3) | 56% (225) | 81.8% | 83.1% |
| 3-large, 1024 + adapter | 5.0x | 240 | 72.9% (66.2 to 78.8) | 58% (139) | 80.6% | 83.3% |
| control and 3-large together (4 models) | 1.25x | 850 | 65.4% (61.5 to 69.6) | 50% (422) | 75.1% | 77.9% |
| control and 3-large together (4 models) | 2.0x | 568 | 69.9% (64.9 to 74.5) | 52% (294) | 78.9% | 82.4% |
| control and 3-large together (4 models) | 3.0x | 401 | 71.6% (65.7 to 77.7) | 54% (215) | 80.9% | 84.4% |
| control and 3-large together (4 models) | 5.0x | 240 | 72.5% (63.7 to 79.5) | 57% (138) | 79.0% | 81.2% |

Paired differences to the control, accounts resampled:

| A minus B | Test set | Real gap at least | Pairs | Correct, points | Confident share, points | Correct among confident, points |
|---|---|---|---|---|---|---|
| 3-large, 1024 minus bge-small (control) | benchmark | 1.25x | 2,852 | +2.2 (+0.4 to +3.9) | +7.4 (+5.3 to +9.9) | +1.5 (-1.3 to +4.2) |
| 3-large, 1024 minus bge-small (control) | benchmark | 2.0x | 2,030 | +3.0 (+1.2 to +5.1) | +6.7 (+4.3 to +9.0) | +2.5 (-1.2 to +6.7) |
| 3-large, 1024 minus bge-small (control) | benchmark test period | 1.25x | 1,660 | +1.4 (-0.8 to +3.5) | +6.1 (+3.5 to +9.3) | +2.2 (-1.8 to +5.8) |
| 3-large, 1024 minus bge-small (control) | benchmark test period | 2.0x | 1,117 | +2.2 (-0.3 to +4.8) | +4.9 (+1.9 to +8.6) | +2.3 (-2.6 to +8.3) |
| 3-large, 1024 minus bge-small (control) | launch | 1.25x | 850 | +3.8 (+0.1 to +7.6) | +11.1 (+6.9 to +16.1) | +4.3 (-0.7 to +9.5) |
| 3-large, 1024 minus bge-small (control) | launch | 2.0x | 568 | +5.8 (+1.4 to +10.0) | +12.0 (+6.5 to +16.6) | +4.7 (-0.4 to +9.4) |
| 3-small, 1024 minus bge-small (control) | benchmark | 1.25x | 2,852 | +1.8 (-0.2 to +3.3) | +8.2 (+5.8 to +10.5) | +2.5 (-0.5 to +5.2) |
| 3-small, 1024 minus bge-small (control) | benchmark | 2.0x | 2,030 | +2.3 (+0.5 to +4.1) | +7.8 (+5.2 to +10.1) | +2.7 (-0.9 to +6.9) |
| 3-small, 1024 minus bge-small (control) | benchmark test period | 1.25x | 1,660 | +0.7 (-1.5 to +3.0) | +8.3 (+5.6 to +10.7) | +3.4 (+0.2 to +6.4) |
| 3-small, 1024 minus bge-small (control) | benchmark test period | 2.0x | 1,117 | +1.7 (-1.1 to +4.4) | +8.6 (+5.4 to +11.8) | +3.1 (-1.1 to +7.8) |
| 3-small, 1024 minus bge-small (control) | launch | 1.25x | 850 | +1.6 (-2.0 to +5.3) | +12.7 (+8.2 to +17.8) | +2.1 (-3.1 to +7.7) |
| 3-small, 1024 minus bge-small (control) | launch | 2.0x | 568 | +2.8 (-1.9 to +7.7) | +12.5 (+6.6 to +17.7) | +1.4 (-4.6 to +7.1) |
| bge-small + adapter minus bge-small (control) | benchmark | 1.25x | 2,852 | +0.1 (-1.8 to +1.9) | -0.7 (-2.1 to +0.8) | -1.0 (-4.2 to +2.2) |
| bge-small + adapter minus bge-small (control) | benchmark | 2.0x | 2,030 | +0.7 (-1.6 to +3.2) | -0.6 (-2.3 to +1.0) | -0.8 (-4.4 to +3.2) |
| bge-small + adapter minus bge-small (control) | benchmark test period | 1.25x | 1,660 | -0.2 (-2.5 to +2.1) | -0.4 (-2.2 to +1.7) | +1.4 (-1.6 to +4.7) |
| bge-small + adapter minus bge-small (control) | benchmark test period | 2.0x | 1,117 | +0.9 (-1.8 to +3.8) | -1.3 (-3.9 to +1.1) | +2.1 (-2.0 to +6.6) |
| bge-small + adapter minus bge-small (control) | launch | 1.25x | 850 | -0.9 (-3.7 to +2.1) | -8.6 (-12.4 to -4.4) | -1.8 (-6.2 to +3.2) |
| bge-small + adapter minus bge-small (control) | launch | 2.0x | 568 | -1.4 (-5.0 to +1.7) | -7.6 (-12.1 to -2.9) | -4.0 (-8.9 to +1.3) |
| 3-large, 1024 + adapter minus bge-small (control) | benchmark | 1.25x | 2,852 | +2.1 (+0.1 to +4.0) | +1.4 (-0.4 to +3.5) | +1.5 (-1.4 to +4.5) |
| 3-large, 1024 + adapter minus bge-small (control) | benchmark | 2.0x | 2,030 | +2.4 (+0.4 to +4.6) | +1.6 (-0.5 to +3.7) | +2.4 (-1.4 to +6.8) |
| 3-large, 1024 + adapter minus bge-small (control) | benchmark test period | 1.25x | 1,660 | +1.4 (-0.9 to +3.9) | +1.6 (-0.9 to +4.3) | +2.3 (-1.3 to +6.0) |
| 3-large, 1024 + adapter minus bge-small (control) | benchmark test period | 2.0x | 1,117 | +2.5 (-0.5 to +5.2) | +1.8 (-1.1 to +5.4) | +1.9 (-3.7 to +7.6) |
| 3-large, 1024 + adapter minus bge-small (control) | launch | 1.25x | 850 | +1.4 (-1.9 to +5.0) | +10.6 (+6.4 to +15.0) | +2.0 (-2.7 to +6.7) |
| 3-large, 1024 + adapter minus bge-small (control) | launch | 2.0x | 568 | +1.2 (-3.1 to +4.9) | +11.8 (+6.2 to +16.1) | +0.4 (-5.7 to +5.9) |
| control and 3-large together (4 models) minus bge-small (control) | benchmark | 1.25x | 2,852 | +1.9 (+0.5 to +3.2) | +7.2 (+5.7 to +8.7) | +2.5 (+0.4 to +4.5) |
| control and 3-large together (4 models) minus bge-small (control) | benchmark | 2.0x | 2,030 | +2.3 (+0.8 to +3.9) | +7.3 (+5.7 to +8.9) | +2.1 (-0.3 to +4.5) |
| control and 3-large together (4 models) minus bge-small (control) | benchmark test period | 1.25x | 1,660 | +1.3 (-0.2 to +3.1) | +6.2 (+4.4 to +8.4) | +1.7 (-1.0 to +4.2) |
| control and 3-large together (4 models) minus bge-small (control) | benchmark test period | 2.0x | 1,117 | +1.5 (-0.4 to +3.8) | +6.3 (+4.4 to +8.7) | +0.3 (-2.9 to +4.0) |
| control and 3-large together (4 models) minus bge-small (control) | launch | 1.25x | 850 | +2.0 (-0.5 to +4.6) | +9.5 (+6.3 to +13.3) | +2.4 (-0.5 to +5.5) |
| control and 3-large together (4 models) minus bge-small (control) | launch | 2.0x | 568 | +3.7 (+0.5 to +7.3) | +9.5 (+5.6 to +13.6) | +1.8 (-1.4 to +5.0) |
| 3-large, 1024 + adapter minus 3-large, 1024 | benchmark | 1.25x | 2,852 | -0.1 (-1.7 to +1.7) | -6.0 (-8.1 to -4.0) | -0.0 (-2.4 to +2.7) |
| 3-large, 1024 + adapter minus 3-large, 1024 | benchmark | 2.0x | 2,030 | -0.6 (-2.5 to +1.4) | -5.1 (-7.2 to -3.0) | -0.1 (-3.1 to +2.9) |
| 3-large, 1024 + adapter minus 3-large, 1024 | benchmark test period | 1.25x | 1,660 | -0.1 (-2.3 to +2.1) | -4.5 (-6.9 to -2.2) | +0.1 (-3.5 to +3.5) |
| 3-large, 1024 + adapter minus 3-large, 1024 | benchmark test period | 2.0x | 1,117 | +0.3 (-2.6 to +3.0) | -3.1 (-5.8 to -0.5) | -0.4 (-5.1 to +3.5) |
| 3-large, 1024 + adapter minus 3-large, 1024 | launch | 1.25x | 850 | -2.4 (-6.0 to +1.5) | -0.5 (-5.3 to +4.5) | -2.3 (-7.3 to +2.2) |
| 3-large, 1024 + adapter minus 3-large, 1024 | launch | 2.0x | 568 | -4.6 (-9.0 to -0.5) | -0.2 (-4.4 to +4.3) | -4.3 (-8.6 to -0.1) |
| 3-small, 1024 minus 3-large, 1024 | benchmark | 1.25x | 2,852 | -0.5 (-2.0 to +1.0) | +0.8 (-1.3 to +2.8) | +1.0 (-1.3 to +3.0) |
| 3-small, 1024 minus 3-large, 1024 | benchmark | 2.0x | 2,030 | -0.7 (-2.7 to +1.0) | +1.1 (-1.2 to +3.4) | +0.2 (-2.7 to +3.0) |
| 3-small, 1024 minus 3-large, 1024 | benchmark test period | 1.25x | 1,660 | -0.7 (-2.7 to +1.1) | +2.2 (-0.7 to +4.3) | +1.3 (-1.8 to +4.7) |
| 3-small, 1024 minus 3-large, 1024 | benchmark test period | 2.0x | 1,117 | -0.5 (-2.8 to +1.6) | +3.7 (+0.9 to +6.6) | +0.8 (-3.6 to +5.0) |
| 3-small, 1024 minus 3-large, 1024 | launch | 1.25x | 850 | -2.1 (-6.4 to +2.2) | +1.6 (-3.1 to +6.4) | -2.2 (-6.6 to +2.7) |
| 3-small, 1024 minus 3-large, 1024 | launch | 2.0x | 568 | -3.0 (-8.0 to +1.3) | +0.5 (-4.5 to +5.3) | -3.4 (-7.4 to +1.0) |
| control and 3-large together (4 models) minus 3-large, 1024 | benchmark | 1.25x | 2,852 | -0.3 (-1.6 to +0.8) | -0.2 (-1.8 to +0.9) | +1.0 (-0.7 to +2.8) |
| control and 3-large together (4 models) minus 3-large, 1024 | benchmark | 2.0x | 2,030 | -0.7 (-2.3 to +0.6) | +0.6 (-1.1 to +2.3) | -0.4 (-2.8 to +1.6) |
| control and 3-large together (4 models) minus 3-large, 1024 | benchmark test period | 1.25x | 1,660 | -0.2 (-1.6 to +1.3) | +0.1 (-2.1 to +2.0) | -0.5 (-3.0 to +2.2) |
| control and 3-large together (4 models) minus 3-large, 1024 | benchmark test period | 2.0x | 1,117 | -0.7 (-2.7 to +1.4) | +1.3 (-1.0 to +3.5) | -2.0 (-5.4 to +1.4) |
| control and 3-large together (4 models) minus 3-large, 1024 | launch | 1.25x | 850 | -1.8 (-4.3 to +0.6) | -1.5 (-4.8 to +1.8) | -1.9 (-5.1 to +1.7) |
| control and 3-large together (4 models) minus 3-large, 1024 | launch | 2.0x | 568 | -2.1 (-5.3 to +0.3) | -2.5 (-5.7 to +1.4) | -2.9 (-5.8 to +0.3) |

Reading:

1. The gain is on all three kinds of pair. It is largest on launch pairs (+3.8, +5.8 at 2x), where the text decides more, and smallest on the test-period half of the same-topic benchmark (+1.4, interval -0.8 to +3.5), which by itself would not be evidence.
2. Coverage of confident calls rises with accuracy, not at its expense: on 2x benchmark pairs from 25% to 32% of pairs with accuracy 69.6% to 72.1%.
3. Why this encoder helps when bge-base, gte-base and mpnet did not: not established here. Two candidates are that it reads the whole post (up to 8,000 characters; bge-small stops at 512 tokens) and that it is a far larger model. The first is not the reason: only 134 of the 2,852 benchmark pairs contain a post longer than 2,000 characters (about bge-small's window) and the gain there is +0.7 (-6.9 to +6.7); it is +1.7 (-0.7 to +3.8) on the 1,968 pairs where both posts are at most 500 characters and +3.9 (+0.6 to +7.4) on the 750 in between (`by_length.json`).

## Step 3. Supervised adapter on frozen embeddings

`adapter.py`; results `adapters/<name>/result.json` (every evaluation step, all splits), checkpoints `adapters/<name>/adapter.pt`.

**Model.** Linear(in, 512), GELU, dropout, Linear(512, 128), unit length: the adapted embedding; plus a linear scalar head on it. About 0.6M parameters on 1,024 inputs.

**Objective, strictly within account.** Each batch is 48 accounts x 24 posts of the training split. (a) Ranking: logistic loss on the head's difference for two posts of one account whose likes differ by 25% or more. (b) Contrastive, in the form the trunk uses the embedding (`HistoryKNN`): a post's outcome relative to its account's level is predicted as the softmax(cosine / 0.1)-weighted mean of the account's other posts in the batch, squared error. Nothing is compared across accounts, and the account's level is subtracted, so audience size carries no signal.

**Validation by account and by time.** Following the coordinating session's note that a rule which held on held-out accounts failed after the cutoff, four sets besides the test split: fit (training accounts, posts up to 60 days before the cutoff, 189,784 posts), held-out accounts in the same period, the fitted accounts in the last 60 days, and held-out accounts in the last 60 days. The checkpoint and the settings were chosen on the last (new accounts and later time); the test split (after the cutoff) was scored once per input at the end. The fitted period ends 60 days before the cutoff, so no adapter saw a post from the last 60 pre-cutoff days or later.

Settings tried on the bge-small + 3-large input, chosen on held-out accounts in the later period (mean of the two accuracies): the default above 59.6% / 58.4%; stronger regularisation (dropout 0.5, input dropout 0.3, weight decay 0.2) 59.4% / 57.9%; a purely linear projection 59.0% / 58.8%. All within noise (intervals about +/- 1.2); the default was kept. (The default ran 1,500 steps and the other two 1,000; all three peaked between steps 750 and 875.) Three more settings (wider, 64 dimensions, contrastive weight 3) were not run: the grid was stopped because it was starving the model trainings of memory and CPU.

Pair accuracy of the adapter alone, same-account pairs at most 30 days apart with a 25%+ like gap. "Head" ranks two posts by the scalar head; "nearest history" ranks them by the similarity-weighted outcome of the account's previous 48 posts. "Raw" columns are the unadapted embedding: a linear head fitted on the same pairs, and the same nearest-history prediction on raw cosine.

| Input | Set | Pairs | Head, adapted | Head, raw (linear) | Nearest history, adapted | Nearest history, raw |
|---|---|---|---|---|---|---|
| bge-small (384), step 1250 | Fitted accounts, fitted period | 141,493 | 59.5% (59.1 to 60.0) | 56.6% | 52.5% | 50.1% |
| bge-small (384), step 1250 | Held-out accounts, same period | 42,734 | 55.3% (54.1 to 56.5) | 54.1% | 55.6% | 55.0% |
| bge-small (384), step 1250 | Fitted accounts, last 60 days | 462,958 | 56.4% (56.0 to 56.8) | 55.2% | 55.9% | 56.5% |
| bge-small (384), step 1250 | Held-out accounts, last 60 days (selection) | 51,207 | 56.2% (55.1 to 57.4) | 55.0% | 56.3% | 55.7% |
| bge-small (384), step 1250 | Test split, after the cutoff | 269,373 | 57.2% (56.8 to 57.6) | 55.5% | 55.3% | 54.7% |
| 3-small (1536), step 750 | Fitted accounts, fitted period | 141,383 | 64.0% (63.6 to 64.4) | 61.6% | 56.6% | 51.0% |
| 3-small (1536), step 750 | Held-out accounts, same period | 42,620 | 57.3% (56.1 to 58.5) | 56.2% | 57.1% | 56.7% |
| 3-small (1536), step 750 | Fitted accounts, last 60 days | 462,826 | 59.0% (58.6 to 59.4) | 57.7% | 57.4% | 57.5% |
| 3-small (1536), step 750 | Held-out accounts, last 60 days (selection) | 51,184 | 58.5% (57.3 to 59.6) | 57.7% | 58.2% | 57.5% |
| 3-small (1536), step 750 | Test split, after the cutoff | 269,093 | 59.9% (59.4 to 60.2) | 58.1% | 56.9% | 55.6% |
| 3-large (1024), step 1000 | Fitted accounts, fitted period | 141,383 | 66.5% (66.1 to 66.9) | 61.0% | 57.5% | 50.7% |
| 3-large (1024), step 1000 | Held-out accounts, same period | 42,620 | 58.2% (57.2 to 59.3) | 57.0% | 57.3% | 56.4% |
| 3-large (1024), step 1000 | Fitted accounts, last 60 days | 462,826 | 59.9% (59.6 to 60.3) | 58.0% | 57.9% | 57.6% |
| 3-large (1024), step 1000 | Held-out accounts, last 60 days (selection) | 51,184 | 59.9% (59.0 to 60.9) | 58.4% | 58.6% | 58.1% |
| 3-large (1024), step 1000 | Test split, after the cutoff | 269,093 | 60.7% (60.3 to 61.1) | 59.0% | 57.5% | 56.1% |
| bge-small + 3-large 1024 (settings grid, no test), step 750 | Fitted accounts, fitted period | 141,506 | 63.8% (63.5 to 64.3) | 61.6% | 57.3% | 51.9% |
| bge-small + 3-large 1024 (settings grid, no test), step 750 | Held-out accounts, same period | 42,734 | 58.4% (57.3 to 59.7) | 56.7% | 56.9% | 56.4% |
| bge-small + 3-large 1024 (settings grid, no test), step 750 | Fitted accounts, last 60 days | 462,989 | 60.0% (59.6 to 60.4) | 57.8% | 57.9% | 57.6% |
| bge-small + 3-large 1024 (settings grid, no test), step 750 | Held-out accounts, last 60 days (selection) | 51,213 | 59.6% (58.4 to 60.9) | 57.9% | 58.4% | 57.6% |

Reading: the adapter is a real but small improvement of the representation (for the head 0.8 to 1.7 points on unseen accounts in a later period and 1.7 to 1.8 on the test split; for nearest-history 0.5 to 0.8 and 0.6 to 1.4), it overfits its training accounts quickly (head 60 to 66% on fitted pairs against 55 to 61% elsewhere; the selected checkpoints are from steps 750 to 1,250), and it generalises equally to new accounts and to a later period. The encoder is the larger lever: the raw 3-large embedding (59.0% on test) is ahead of the adapted bge-small (57.2%).

**The adapted embedding as model input.** The table's embedding becomes [original, adapted] (two unit blocks, 512 dimensions for bge-small, 1,152 for 3-large), semantic IDs retrained, two seeds each, compared with the same encoder without the adapter. Numbers are in the tables of step 2 (rows "+ adapter").

Result: nothing on bge-small (ordinary pairs 59.7% against 60.0%; benchmark +0.1, -1.8 to +1.9; launch pairs -0.9, -3.7 to +2.1) and nothing to slightly negative on 3-large (ordinary pairs 61.9% against 62.3%, both seeds below both seeds without it; benchmark -0.1, -1.7 to +1.7, with confident coverage down 6.0 points, -8.1 to -4.0; launch pairs -2.4, -6.0 to +1.5, and -4.6, -9.0 to -0.5, at 2x).

Why a representation that is better alone adds nothing inside the model, as far as this track can tell:

1. The trunk already does what the adapter does. It has an MLP on the embedding trained on the same outcomes with the account's history beside it, so a 128-dimension supervised summary of the same vector is not new information. The adapter's stand-alone gain (1 to 2 points over a linear head) is the gain of a small non-linear model over a linear one, which the trunk has anyway.
2. The adapter's output is more informative on the rows it was fitted on than on new rows (head 60 to 66% on fitted pairs, 55 to 61% elsewhere), and those are the trunk's training rows too. The trunk therefore learns to trust the adapted block more than it deserves on new posts: with the adapter the within-account rank correlation on fitted accounts rises from 0.389 to 0.434 while validation and test fall slightly (0.355 to 0.347, 0.361 to 0.359). Fitting the adapter on one part of the accounts and the trunk on another (or cross-fitting) would remove this, at the cost of training data; given point 1 it was not tried.
3. The nearest-history kernel, the one place where distance itself is used, improved by only 0.5 to 1.4 points with the adapted space.

## Serving cost per request

Measured on the table's texts: 138 tokens per post on average. A request embeds the drafts and, the first time an account is seen, its history (the benchmark uses the last 56 posts). Prices: 3-large $0.13, 3-small $0.02 per million tokens. Latency is the median of five calls from this machine through the proxy.

| Option | Extra cost, first request for an account (56 history posts + 6 drafts, about 8,600 tokens) | Extra cost, later requests (6 drafts, about 830 tokens) | Extra latency | Other |
|---|---|---|---|---|
| bge-small (today) | none | none | CPU encoder, about 0.5 s for 62 posts on 4 cores (114 posts/s measured in step 1) | |
| 3-large, 1024 | $0.0011 | $0.00011 | 0.45 s (62 texts), 0.31 s (6 texts), one call | post text leaves the machine; 4 KB per stored history vector at fp32 (2 KB fp16) |
| 3-small, 1024 | $0.00017 | $0.000017 | 0.34 s (62 texts), 0.26 s (6 texts) | same |
| bge-small + adapter | none | none | under 1 ms (one small matrix product) | no gain, not worth shipping |
| 3-large + adapter | as 3-large | as 3-large | as 3-large | no gain over 3-large |

At 10,000 first requests and 100,000 later requests a month, 3-large costs about $22 a month and 3-small about $3.40. The embeddings of an account's history should be cached by text, as `ApiEmbedder` does, so that only new posts and drafts are sent.

## What worked, what did not

- Worked: a large general-purpose API embedding in place of bge-small. +2.3 points on 683,152 ordinary pairs, +2.2 on the same-topic benchmark (+3.0 at 2x), +3.8 on launch pairs (+5.8 at 2x), confident coverage on 2x benchmark pairs 25% to 32% with accuracy up, for about a tenth of a cent per first request. It is the first change to the model's inputs in these tracks that moves the same-topic benchmark. Recommended: adopt `text-embedding-3-large` at 1,024 dimensions as the model's text input (checkpoints `runs/te3l1024_s1`, `runs/te3l1024_s2`, serving files `serving/te3l1024/`), then re-run the pair-loss fine-tune on top.
- Worked, cheaper: `text-embedding-3-small`. Roughly 60% of the gain on ordinary pairs (+1.4) and on the benchmark (+1.8), less on launch pairs (+1.6 against +3.8), at 15% of the price. At these prices the large model is the better choice.
- Did not work: the supervised adapter as a model input, on either encoder. Better alone, nothing inside the model, slightly harmful on the API embedding.
- Did not work: keeping the two bge-small models in the ensemble beside the two 3-large models. The four together are no better than the two 3-large models (benchmark 61.4% against 61.7%, launch 65.4% against 67.2%).
- Not run: 3,072 and 256 dimensions and bge-small + 3-large side by side in the full model; `--pca`; the pair-loss fine-tune on the new embedding; end-to-end fine-tuning of an encoder (not possible on an API model, and CPU-only here).

## Honest limits

- Two seeds per arm. Seed spread on ordinary pairs is 0.2 to 0.3 points, on the benchmark about 1.5 points. The benchmark and launch comparisons are two-seed ensembles against two-seed ensembles.
- The benchmark gain on the test-period pairs alone has an interval that includes zero (+1.4, -0.8 to +3.5; at 2x +2.2, -0.3 to +4.8). The claim rests on the three test sets agreeing and on the 683,152 ordinary pairs, not on that row.
- No setting of the model was tuned for the new embedding (history-similarity temperatures, learning rate, steps are those of bge-small), and none was chosen on any test set. The only choices made in this track are the adapter's settings and checkpoint, made on pre-cutoff held-out accounts.
- 1,024 dimensions was chosen before training from the probe (level with 3,072 beside the structure columns, a third of the storage). 3,072, 256 and bge-small + 3-large side by side were not trained; the probe suggests 256 would lose about a point and the concatenation might add something.
- The 3-large table keeps the bge-small neighbour lists and neighbour-derived columns (three numeric columns), as the encoder track did.
- The API model is a dependency that can change or be withdrawn; `text-embedding-3-large` has been stable since January 2024. The stores in the dataset make the training side reproducible without the API; serving is not.
- Post text (public posts, with the author's name and handle line as the trunk's encoder has always received it) is sent to OpenAI. A draft that has not been published would be sent too; that is a product decision.
- The adapter was trained once per input (one seed), with a three-point settings grid.
- The benchmark validates likes only.
- The pair-loss fine-tune of the pairs and replicate tracks (the one other lever known to work, on confident coverage) was not combined with the new embedding. It is the obvious next step.

## Files

In the dataset under `tracks/embed_api/`:

- `REPORT.md`, `patch.diff` (new files only, all under `virality/research/tracks/embed_api/`; no existing file was changed).
- `probe_api.json`, `probe_api.txt`, `probe_api_scores.parquet`: step 1.
- `store/te3large_3072/`, `store/te3small_1536/`: the embedding stores (fp16, native size, keyed by text hash, with `ledger.json`). 581,292 texts each.
- `runs/<name>/best.pt`, `metrics.json`, `splits.json` for the trained models; `serving/<variant>/`: what serving needs besides a checkpoint (`recipe.json`, `sid_codebook.npz`, and `adapter.pt` where used).
- `bench_out/<name>/shard_*.parquet` (per-pair benchmark scores), `launch/<name>.parquet` (launch post scores), `results.json`, `results.txt`, `results.md`, `by_length.json`.
- `adapters/<name>/result.json`, `adapter.pt`; `latency.json`.

To score a checkpoint of this track on the benchmark: build its table directory with `api_table.py` (or take `serving/<variant>/`), run `rebench.sh <variant>` to rewrite the four benchmark caches and the launch cache (texts already in the store cost nothing), then `score.sh <run> <variant>`.
