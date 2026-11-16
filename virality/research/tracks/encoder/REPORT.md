# Track "encoder": does a stronger text representation raise pair accuracy?

2026-11-16, one 4-core CPU machine, about 5.5 hours.

## Answer

No, not on the benchmark. Not with any encoder that runs on these machines.

- **General-purpose sentence encoders three times the size of bge-small (bge-base, gte-base, all-mpnet-base) add nothing**, in the cheap probe or in combination with bge-small.
- **Twitter-domain language models (twitter-roberta-base, bertweet-base) do carry extra signal**, and it shows where there is enough data to see it: a surrogate trained with bge-small + twitter-roberta side by side ranks two ordinary posts by one account 1.3 points better on likes (61.7% against 60.4%, two seeds each, 132,764 test pairs, same accounts, same everything else).
- **That gain does not reach the same-topic pair benchmark.** On 1,439 benchmark pairs with a 25%+ gap, the two-seed ensemble with the added encoder is 60.2% correct against 59.7% for bge-small on the same training data: +0.5, interval -2.3 to +3.3. On the test-period pairs alone it is -0.4. Twitter-roberta without bge-small is 3 points worse (interval -5.7 to -0.5).
- No checkpoint beats the baseline, so none is uploaded as a candidate. The 2x-gap target (75 to 80% on confident calls covering 40%) is as far away as before: best here is 69.4% on 25% of 2x pairs.

This is the same pattern as MODEL.md 9.4 point 2 (text-structure columns help ordinary pairs, not two versions of one message): what a better text input buys is knowledge of which topics and styles do well for an account, and a same-topic pair holds exactly that constant.

## Step 1. Probe: embedding difference, antisymmetric logistic model

`probe.py`, `embed_pairs.py`; results `probe_all.json`, per-pair scores `probe_all_scores.parquet`, printed table `probe_all.txt`.

Setup: all same-topic pairs as `pair_model.py` finds them (9,188: 7,158 before the cutoff, 2,030 in the test period, which are exactly the benchmark's 2,030 test pairs). Each post's rendered text (author line, text, quoted text, as the trunk's encoder sees it) is embedded; the model is a logistic regression without intercept on the standardised difference, every pair in both orders. Trained on 5,791 pre-cutoff pairs with a 25%+ like gap. Regularisation chosen by 5-fold cross-validation over pre-cutoff accounts. Tested on the 1,660 test-period pairs with a 25%+ gap. Confidence scale from the out-of-fold pre-cutoff scores. The last column is the paired difference to bge-small with accounts resampled.

Embedding difference alone:

| Encoder | Dim | Pre-cutoff, held-out accounts | Test 1.25x (1,660) | 2x (1,117) | 3x (733) | 5x (440) | Confident 60%+: pairs, correct | vs bge-small |
|---|---|---|---|---|---|---|---|---|
| bge-small (baseline) | 384 | 53.9% | 54.7% | 55.1% | 56.6% | 60.0% | 196 at 58.2% | |
| bge-base | 768 | 54.0% | 53.9% | 53.0% | 54.6% | 56.1% | 53 at 49.1% | -0.8 [-4.4, +2.6] |
| gte-base | 768 | 54.4% | 54.9% | 55.6% | 57.2% | 59.8% | 139 at 60.4% | +0.2 [-2.6, +3.0] |
| all-mpnet-base-v2 | 768 | 52.8% | 53.8% | 54.1% | 56.8% | 59.1% | 14 at 42.9% | -0.9 [-4.2, +2.3] |
| twitter-roberta-base, mean pooled | 768 | 58.6% | 55.2% | 55.0% | 56.6% | 57.0% | 316 at 59.2% | +0.5 [-2.3, +3.5] |
| bertweet-base, mean pooled | 768 | 58.2% | 55.1% | 56.3% | 61.3% | 61.4% | 378 at 57.7% | +0.4 [-1.9, +3.0] |
| bge-small + bge-base | 1152 | 54.3% | 54.8% | 54.4% | 57.2% | 59.8% | 161 at 49.1% | +0.1 [-1.9, +2.0] |
| bge-small + gte-base | 1152 | 54.9% | 54.9% | 56.2% | 58.9% | 63.6% | 180 at 58.9% | +0.2 [-2.2, +2.9] |
| bge-small + mpnet | 1152 | 54.3% | 56.1% | 56.6% | 58.7% | 62.7% | 159 at 57.2% | +1.4 [-0.9, +3.7] |
| **bge-small + twitter-roberta** | 1152 | 58.2% | **56.7%** | 57.0% | 59.8% | 60.7% | 315 at 59.0% | **+2.0 [-0.5, +4.5]** |
| bge-small + bertweet | 1152 | 58.5% | 55.2% | 56.4% | 62.8% | 63.9% | 364 at 60.2% | +0.5 [-2.0, +3.1] |
| twitter-roberta + bertweet | 1536 | 59.1% | 54.9% | 55.4% | 60.2% | 61.6% | 388 at 57.7% | +0.2 [-2.2, +3.0] |
| all three | 1920 | 59.5% | 55.4% | 56.0% | 61.4% | 63.4% | 397 at 58.9% | +0.7 [-1.7, +3.4] |

Embedding difference plus `pair_model.py`'s 50 structure, media and link differences:

| Encoder | Pre-cutoff, held-out accounts | Test 1.25x (1,660) | 2x (1,117) | 3x (733) | 5x (440) | Confident 60%+: pairs, correct | vs bge-small |
|---|---|---|---|---|---|---|---|
| structure columns only | 68.2% | 56.6% | 60.1% | 64.0% | 69.5% | 1,035 at 59.4% | |
| bge-small | 67.3% | 58.4% | 61.3% | 66.6% | 72.0% | 1,022 at 62.2% | |
| bge-base | 66.4% | 58.3% | 60.7% | 66.6% | 71.6% | 953 at 60.8% | -0.1 [-2.3, +2.2] |
| gte-base | 67.2% | 58.9% | 62.0% | 68.2% | 73.9% | 1,041 at 61.7% | +0.5 [-1.6, +2.5] |
| all-mpnet-base-v2 | 66.6% | 58.3% | 61.9% | 67.1% | 70.7% | 986 at 60.6% | -0.1 [-2.4, +2.0] |
| twitter-roberta | 67.6% | 59.6% | 62.6% | 66.0% | 69.3% | 980 at 62.7% | +1.2 [-0.7, +3.5] |
| bertweet | 67.7% | 58.2% | 61.7% | 67.3% | 71.1% | 1,009 at 61.4% | -0.2 [-1.9, +1.9] |
| **bge-small + twitter-roberta** | 67.4% | **60.2%** | 62.7% | 66.7% | 70.5% | 935 at 63.0% | **+1.9 [+0.2, +3.9]** |
| bge-small + gte-base | 66.7% | 58.8% | 62.3% | 67.8% | 74.1% | 1,005 at 62.9% | +0.4 [-1.2, +2.4] |
| bge-small + mpnet | 66.2% | 59.0% | 62.5% | 67.3% | 71.1% | 980 at 61.1% | +0.7 [-1.5, +2.8] |
| twitter-roberta + bertweet | 67.0% | 59.5% | 62.8% | 67.3% | 70.0% | 987 at 61.6% | +1.1 [-0.9, +3.4] |

Reading:

1. The text embedding is a weak input for this question whatever the encoder: 54 to 57% alone, against 56.6% for 50 structure columns. With structure columns everything sits at 58 to 60%, the level 9.4 reports for direct pair models (56 to 58%).
2. Size does not help. bge-base, gte-base and mpnet are within a point of bge-small, alone or concatenated.
3. Domain helps before the cutoff: the two Twitter models reach 58 to 59% on held-out pre-cutoff accounts where every sentence encoder is at 53 to 54%. On the test period most of that is gone (+0.4 to +0.5 alone). bge-small + twitter-roberta is the only row whose interval excludes zero, once, out of 24 comparisons, so by itself it is weak evidence.
4. raw post text instead of rendered text makes no difference for bge-small (55.0% against 55.2%, `probe0.json`).

Not run: e5-base-v2 (dropped to protect the retraining budget). The regularisation choice moves a row by about half a point (bge-small was 55.2% with a narrower grid).

## Step 2. Retrained surrogate with bge-small + twitter-roberta

Best candidate from step 1: twitter-roberta-base beside bge-small. Also trained: twitter-roberta alone.

**Budget.** twitter-roberta runs at 24 posts/s in fp32 on these machines, 465k training rows would take over 5 hours. Two things made it fit: bf16 autocast (the CPUs have AMX; 2.3x faster, cosine to fp32 above 0.9999) and a 256-token cap, giving 32 to 40 posts/s; and a random 25% of accounts (1,805 of 7,221, audience and general sampled separately; 113,717 rows: 75,158 train, 8,013 validation, 30,546 test), embedded in 45 minutes. A bge-small model trained on the same 25% is the control, so the comparison is encoder against encoder with everything else equal.

**What was changed** (`subset_table.py`, `embed_table.py`, `concat_table.py`): the table's post embedding and its semantic IDs (codebook retrained on the new embedding). twitter-roberta vectors are centred on the training-split mean and renormalised before use: mean-pooled language-model vectors share one large component (random-pair cosine 0.95 +/- 0.03, against 0.62 +/- 0.07 for bge-small), which would flatten the trunk's history-similarity weights (`HistoryKNN`, temperatures 0.05 and 0.2 on raw cosine). The concatenation is [bge-small, centred twitter-roberta] / sqrt 2, 1152-d. Neighbour lists and neighbour-derived columns were left as built with bge-small (the retrieval index; serving passes none). Training: `scripts/train_one.py --features ts46 --preset small --steps 4000`, unchanged.

**Ordinary same-account pairs, test period, the subset's audience accounts** (from each run's `metrics.json`; 132,764 like pairs, 3,999 view pairs):

| Model (25% of accounts) | Pairwise, likes | Within-account Spearman, likes | Pairwise, views | Within-account Spearman, views | 10x-followers AUROC |
|---|---|---|---|---|---|
| bge-small, seed 1 | 60.0% | 0.281 | 62.9% | 0.301 | 0.869 |
| bge-small, seed 2 | 60.7% | 0.274 | 61.0% | 0.246 | 0.870 |
| bge-small + twitter-roberta, seed 1 | 61.5% | 0.296 | 61.7% | 0.226 | 0.877 |
| bge-small + twitter-roberta, seed 2 | 61.8% | 0.302 | 63.0% | 0.265 | 0.916 |
| twitter-roberta alone, seed 1 | 61.2% | 0.300 | 62.3% | 0.307 | 0.922 |

Likes: +1.3 points and +0.02 Spearman, both seeds of the new encoder above both seeds of the control. Views: 3,999 pairs, no difference visible. (For reference the full-data ts46_s1 model has 59.8% on likes over all audience accounts, so the 25% subset is not a handicapped test bed.)

**The pair benchmark** (`run_step2.sh`, `bench_reembed.py`, `bench_table.py`; results `bench_results.json`, `bench_results.txt`, per-pair scores under `bench/`). Half of the benchmark was scored: every second pair of the fixed 3,530 (`--shard 0/2`), 1,765 pairs, 1,439 with a 25%+ gap, of which 848 in the test period and 591 from never-trained accounts before it. The serving path is `tac_serve_eval.py` unchanged, with the same encoder, centre and codebook as in training; the baseline's stored scores are compared on exactly these pairs. Check: re-scoring ts46_s1 here gives 60.7% against 60.9% from its stored scores (bf16 flips 1.4% of near-zero differences).

All 1,765 scored pairs. "Confident" is stated confidence of 60% or more, scale cross-fitted over accounts:

| Model | 1.25x: pairs, correct | confident share (pairs), correct | 2x | confident | 3x | confident | 5x | confident |
|---|---|---|---|---|---|---|---|---|
| Baseline: three full-data small models (README) | 1,439, 60.7% | 25% (355), 65.1% | 1,019, 62.3% | 24% (242), 68.6% | 685, 63.6% | 24% (162), 73.5% | 436, 65.4% | 26% (113), 79.6% |
| bge-small, 25% of accounts, 2 seeds | 1,439, 59.7% | 23% (336), 64.3% | 1,019, 61.4% | 22% (227), 67.0% | 685, 64.1% | 22% (152), 71.1% | 436, 66.3% | 24% (105), 76.2% |
| **bge-small + twitter-roberta, 25%, 2 seeds** | 1,439, 60.2% | 26% (370), 67.3% | 1,019, 62.4% | 25% (255), 69.4% | 685, 65.7% | 26% (176), 73.3% | 436, 68.6% | 28% (121), 75.2% |
| twitter-roberta alone, 25%, 1 seed | 1,439, 57.5% | 21% (307), 65.8% | 1,019, 58.0% | 21% (213), 67.6% | 685, 62.0% | 21% (145), 73.8% | 436, 65.6% | 24% (103), 80.6% |
| All four bge-small and concatenated models | 1,439, 61.0% | 26% (371), 68.2% | 1,019, 63.5% | 25% (256), 70.7% | 685, 66.4% | 26% (178), 74.2% | 436, 69.5% | 27% (119), 77.3% |

By stated confidence, pairs with a 25%+ gap (correct, pairs):

| Model | 50 to 55% | 55 to 60% | 60 to 70% | 70% and over |
|---|---|---|---|---|
| Baseline | 56.2% (744) | 66.2% (340) | 63.1% (279) | 72.4% (76) |
| bge-small, 25%, 2 seeds | 55.2% (756) | 65.1% (347) | 64.4% (250) | 64.0% (86) |
| bge-small + twitter-roberta, 25%, 2 seeds | 54.9% (738) | 64.0% (331) | 69.2% (276) | 61.7% (94) |
| twitter-roberta alone | 53.5% (811) | 59.8% (321) | 64.6% (237) | 70.0% (70) |

Test-period pairs only (848 with a 25%+ gap):

| Model | 1.25x: pairs, correct | confident share (pairs), correct | 2x | confident | 3x | 5x |
|---|---|---|---|---|---|---|
| Baseline | 848, 62.1% | 28% (235), 62.6% | 569, 63.6% | 27% (156), 66.7% | 365, 65.8% | 220, 69.1% |
| bge-small, 25%, 2 seeds | 848, 59.9% | 21% (174), 60.9% | 569, 62.2% | 20% (114), 60.5% | 365, 66.8% | 220, 68.6% |
| bge-small + twitter-roberta, 25%, 2 seeds | 848, 59.6% | 20% (171), 61.4% | 569, 61.2% | 20% (112), 61.6% | 365, 64.7% | 220, 66.4% |
| twitter-roberta alone | 848, 57.2% | 16% (133), 60.9% | 569, 56.6% | 15% (87), 60.9% | 365, 61.9% | 220, 65.5% |

Paired differences, accounts resampled:

| Comparison | 1.25x, all pairs (1,439) | 2x, all pairs (1,019) | 1.25x, test period (848) | 2x, test period (569) |
|---|---|---|---|---|
| concatenated minus bge-small (2 seeds each) | +0.5 [-2.3, +3.3] | +1.0 [-2.7, +3.8] | -0.4 [-3.1, +3.0] | -1.1 [-4.9, +3.3] |
| twitter-roberta alone minus bge-small (seed 1) | -3.0 [-5.7, -0.5] | -4.1 [-7.3, -0.8] | -2.8 [-6.1, +0.6] | -5.6 [-9.6, -1.9] |
| concatenated (25%) minus baseline | -0.6 [-3.4, +2.2] | +0.1 [-3.2, +2.7] | -2.6 [-6.4, +0.5] | -2.5 [-6.7, +1.8] |
| bge-small (25%) minus baseline | -1.0 [-3.4, +1.3] | -0.9 [-3.7, +1.7] | -2.2 [-5.6, +1.2] | -1.4 [-5.5, +2.3] |

Reading:

1. Adding twitter-roberta: no detectable change on the benchmark. The apparent +3 points among confident calls on all pairs (67.3% against 64.3%) comes entirely from the never-trained-account pairs before the cutoff; in the test period confident calls are 61.4% against 60.9%.
2. Replacing bge-small with twitter-roberta hurts. A masked-language model's mean-pooled vector tracks surface form and loses the semantic similarity to the account's own history that the trunk uses; bge-small must stay.
3. Training on 25% of accounts costs about a point on the benchmark and roughly 5 points of coverage among confident test-period calls (20% against 28%). That is the size of the "more data" lever in this range, and it is larger than anything the encoder did.

## Step 3. Fine-tuning the encoder's top layers

`finetune_encoder.py`: top two layers of bge-small plus a linear head, logistic loss on which of two posts by one account (training split, before 2026-04-23 00:00 UTC, at most 7 days apart, 25%+ like gap) got more likes. On CPU this runs at 10.7 s per step of 32 pairs; a 250-step run (8,000 pairs, one epoch) was started with the time left. See "Step 3 result" at the end of this file if it finished; otherwise it did not finish within the budget and the script is delivered untested beyond its first 50 steps. A useful run (tens of thousands of pairs, several epochs, a base-size Twitter model) needs a GPU.

## What worked, what did not

- Worked: Twitter-domain representation for ordinary within-account ranking on likes (+1.3 points, cheap: one extra frozen encoder). Worth carrying into the full-scale run as a second text input next to the semantic encoder, if ranking ordinary posts matters.
- Did not work: any encoder change for the same-topic pair benchmark. Bigger sentence encoders: nothing, anywhere.
- Useful by-product: bf16 autocast makes every CPU embedding pass here 2.3x faster with identical vectors for practical purposes (bge-small 46 to 110 posts/s). `bench_run.py` and `embed_table.py` use it; `xviral/embed/text_small.py` could.

## Honest limits

- The step 2 models train on 25% of accounts and are scored on half of the benchmark pairs. Intervals on encoder-against-encoder differences are about +/- 3 points, so a real effect of a point or two in either direction cannot be excluded. The within-account result (132,764 pairs) is far tighter than the benchmark one.
- Two seeds per arm (one for twitter-roberta alone). Seed-to-seed spread on the benchmark is about 1.5 points.
- Neighbour lists stayed in bge-small space, and three quarters of them point outside the 25% subset and were dropped; this is the same for every arm.
- The twitter-roberta semantic-ID codebooks have 24 to 28 dead codes at the last level and low perplexity; no attempt was made to tune them.
- Only encoders that run on a 4-core CPU were tried (up to 125M parameters). X's own multimodal encoder and anything instruction-tuned or larger than base size are untested.
- The benchmark validates likes only.

## One thing to check in the main pipeline

The table's train/validation split runs to 2026-04-23 13:04 UTC, while the benchmark's `--after 2026-04-23` starts at 00:00 UTC. 1,090 of the 334,827 train/validation rows (258 in the 25% subset) are from those 13 hours. No benchmark test post is among the training rows (checked: 0 of 2,030 pairs), so nothing here is affected, but the two cutoffs should be the same instant.

## Files

- `REPORT.md`, `patch.diff` (new files only, all under `virality/research/tracks/encoder/`; no existing file was changed).
- `probe_all.json`, `probe_all_scores.parquet`, `probe_all.txt`, `probe0.json`: step 1.
- `runs/<model>/metrics.json` and `best.pt` for the five subset models; `bench/<model>_0o2/tac.parquet` per-pair scores; `bench_results.json`, `bench_results.txt`: step 2.
- `sid/<variant>/sid_codebook.npz`, `sid/<variant>/center.npy`: what serving needs besides the checkpoint for the twitter-roberta models (`cat` = bge-small + twitter-roberta, `twrc` = twitter-roberta alone).

To score one of the concatenated checkpoints: build the bge-small feature cache with `bench_run.py` (any bge-small checkpoint), rewrite it with `bench_reembed.py --encoder cardiffnlp/twitter-roberta-base --max-tokens 256 --center sid/cat/center.npy --sid-codebook sid/cat/sid_codebook.npz --concat-cache`, then run `bench_run.py` with that cache and codebook. `run_step2.sh` does all of it.
