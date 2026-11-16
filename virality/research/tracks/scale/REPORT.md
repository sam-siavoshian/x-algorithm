# Track "scale": what more model, more training and more data would buy

Worker report, 2026-11-16. 16 training runs on 4 CPU cores, one at a time, all with the 46 text-structure columns, the bge-small encoder, seed 1,
batch 64, learning rate 1e-3 (the recipe of the baseline `ts46_s1`). Selection is on validation only; the benchmark and the launch pairs were scored once, at the end.

## Headline

**Nothing here is limited by model size or training length. The trunk is limited by data, and even data is flattening.**

- **Longer training does not help.** `small` peaks at 3,000 to 4,000 steps. At 8,000 and 16,000 steps validation gets worse (like pairs 61.6% → 60.7% → 59.8%), while the training loss keeps falling: overfitting.
- **A 5.4x larger model does not help.** Width 384 with 6 layers (8.2M parameters) scores the same as `small` (1.5M) on every measure. `nano` (0.27M) is 0.8 points behind on like pairs.
- **Data is the only curve with a slope, and it is halving.** 25% → 50% → 100% of training accounts: 58.6% → 59.6% → 60.0% on test like pairs (+0.9, then +0.4 per doubling). View pairs do not move beyond noise.
- **Longer history is the one lead.** 96 posts against 48 at 2,000 steps: +0.4 on test like pairs and +1.1 on view pairs, one seed, about the size of seed noise. Worth two more seeds on CPU; it needs no GPU.
- **Regularisation does not matter.** Dropout, weight decay, pair-loss weight, label smoothing, schedule: all within 0.3 points on validation like pairs.
- **Benchmark and launch pairs: no change.** Every new model is within 0.5 points of the baseline on the 3,530-pair benchmark and within 1 point on the 850 launch pairs, far inside the intervals.
- **Projection.** A 10x larger trunk trained 10x longer, on this table and this encoder: 0 points on ordinary pairs (range -0.5 to +0.5) and 0 on the benchmark (range -1.5 to +1.5). I would not pay for a GPU for trunk scale. No checkpoint here beats the baseline, so none is uploaded as a replacement.

## Method

- Code: `scripts/scale_run.py` (one curve point), `scripts/scale_queue.sh` (runs them one at a time), `scripts/scale_summary.py`, `scripts/scale_launch_check.py`; two new training options in `xviral/model` (label smoothing of histogram targets, constant schedule). All in `patch.diff`.
- Validation: the same 20,000 rows of held-out audience accounts in every run: 66,681 same-account like pairs and 18,292 view pairs. Validation loss is the mean likelihood loss of the six distributional heads.
- Test (reported, never selected on): audience test split, 95,357 rows after 2026-04-23: 683,152 like pairs and 24,481 view pairs, scored with the best-validation checkpoint of each run.
- Noise to read the tables against: from earlier runs of the same recipe, two seeds differ by about 0.3 points on test like pairs and about 1 point on test view pairs. I ran one seed per configuration, so differences smaller than that are not findings.
- Training rows: 302,696 from 6,334 accounts. 4,000 steps of 64 episodes is 0.85 passes over the anchors; with up to 6 candidates per episode each post is seen about 3 times.

## 1. Training length (`small`, all data)

Each row is a separate run with its own full schedule (warm-up 500, cosine to 5%).

| Steps | Best validation step | Validation loss at best | Val like pairs at best | Val view pairs at best | At the last step: loss, like pairs, view pairs | Test like pairs | Test view pairs |
|---|---|---|---|---|---|---|---|
| 2,000 | 2,000 | 2.9684 | 61.3% | 59.3% | 2.9684, 61.3%, 59.3% | 59.9% | 61.6% |
| 4,000 | 3,000 | 2.9519 | 61.5% | 59.2% | 2.9524, 61.6%, 59.0% | 60.0% | 61.3% |
| 8,000 | 4,000 | 2.9475 | 61.4% | 59.0% | 2.9791, 60.7%, 58.3% | 59.8% | 61.8% |
| 16,000 | 4,000 | 2.9474 | 61.5% | 58.9% | 3.1291, 59.8%, 56.6% | 59.8% | 61.8% |

Inside the 16,000-step run (like-head loss on training batches, then validation):

| Step | Training like loss | Validation like loss | Val like pairs | Val view pairs |
|---|---|---|---|---|
| 2,000 | 2.986 | 3.006 | 61.3% | 58.9% |
| 4,000 | 2.917 | 2.984 | 61.5% | 58.9% |
| 8,000 | 2.851 | 2.989 | 60.7% | 58.2% |
| 12,000 | 2.745 | 3.022 | 60.2% | 57.3% |
| 16,000 | 2.703 | 3.057 | 59.8% | 56.6% |

The test columns of the 8,000 and 16,000 runs are their step-4,000 checkpoints (early stopping), which is why they equal the 4,000-step run. Without early stopping, 16,000 steps costs 1.7 points on like pairs and 2.4 on view pairs.
The model is at its best after about one pass over the data. More steps only memorise.

## 2. Data (`small`, 4,000 steps)

A random share of training accounts, one draw, so the 25% accounts are a subset of the 50% accounts.

| Share of accounts | Accounts | Training rows | Best validation step | Validation loss | Val like pairs | Val view pairs | Test like pairs | Test view pairs | Test like-head loss |
|---|---|---|---|---|---|---|---|---|---|
| 25% | 1,584 | 75,916 | 1,000 | 3.0220 | 60.5% | 58.3% | 58.6% | 61.0% | 3.029 |
| 50% | 3,167 | 152,196 | 2,000 | 2.9814 | 61.0% | 58.3% | 59.6% | 61.5% | 2.997 |
| 100% | 6,334 | 302,696 | 3,000 | 2.9519 | 61.5% | 59.2% | 60.0% | 61.3% | 2.968 |

- **Like pairs:** +0.93 points for the first doubling, +0.42 for the second. If the gain keeps halving, all further data adds about +0.4 in total; if it follows a straight line in log data (the optimistic fit, +0.67 per doubling), 10x more data adds about +2.2. So 10x more accounts of the same kind: **+0.4 to +2.2 points on like pairs**, most likely near +1.
- **View pairs:** 61.0%, 61.5%, 61.3%. No slope beyond the 1-point seed noise. Only a third of rows carry views, so this curve is short of data to measure, not evidence that data does not matter for views.
- **Likelihood loss** falls steadily (3.029 → 2.997 → 2.968): more data keeps improving the predicted distributions even where ranking stalls.
- **The best step moves with the data** (1,000 → 2,000 → 3,000): more data is what would make longer training useful.
- Limits: the neighbour inputs still point at posts of the removed accounts, so the smaller shares are slightly flattered.

## 3. Size (4,000 steps, all data) and history length

Measured time per step on 4 threads (batch 64), which decided what could run:

| Configuration | Parameters | Seconds per step | 4,000 steps |
|---|---|---|---|
| nano (width 64, 2 layers, history 16) | 0.27M | 0.05 | 3 min |
| small (192, 3 layers, history 48) | 1.52M | 0.25 | 17 min |
| small, 1 thread / 2 threads | 1.52M | 0.56 / 0.37 | |
| small, history 96 | 1.52M | 0.53 | 35 min |
| width 256, 4 layers | 2.85M | 0.42 | 28 min |
| width 384, 6 layers | 8.2M | 0.95 (0.88 in the run) | 63 min |
| width 384, 6 layers, history 96 | 8.2M | 1.9 | 2.1 h |
| base (512, 8 layers, history 96, 8 candidates) | 17.8M | 4.2 | 4.7 h |

`base` at 4,000 steps needs 4.7 hours here, which is why the earlier attempt did not finish. Width 384 with 6 layers was the largest that fit.

| Model | Parameters | Best validation step | Validation loss | Val like pairs | Val view pairs | Test like pairs | Test view pairs | Test within-account Spearman, likes / views |
|---|---|---|---|---|---|---|---|---|
| nano | 0.27M | 4,000 | 2.9771 | 60.2% | 57.2% | 59.2% | 59.8% | 0.293 / 0.226 |
| small | 1.52M | 3,000 | 2.9519 | 61.5% | 59.2% | 60.0% | 61.3% | 0.297 / 0.268 |
| width 384, 6 layers | 8.2M | 4,000 | 2.9523 | 61.4% | 59.6% | 59.8% | 61.7% | 0.295 / 0.261 |

From 0.27M to 1.5M parameters: +0.8 points on like pairs and +1.5 on view pairs. From 1.5M to 8.2M: -0.2 and +0.4, both inside noise. The size curve is flat from `small` up.

**History 48 versus 96** (`small`, 2,000 steps, because of time):

| History | Seconds per step | Validation loss | Val like pairs | Val view pairs | Test like pairs | Test view pairs |
|---|---|---|---|---|---|---|
| 48 posts | 0.25 | 2.9684 | 61.26% | 59.26% | 59.90% | 61.59% |
| 96 posts | 0.53 | 2.9680 | 61.56% | 59.35% | 60.31% | 62.73% |

History 96 is ahead on every column: +0.3 on validation like pairs, +0.4 on test like pairs, +1.1 on test view pairs, at twice the cost per step.
Each gap is about the size of the seed noise (0.3 and 1 point), from one seed at 2,000 steps, so this is a lead and not a result. It is the only change in this track that points the same way on every measure, and it costs minutes on CPU to confirm with two more seeds at 4,000 steps.

## 4. Regularisation and optimisation (`small`, 2,000 steps, validation only)

| Change from the default | Validation loss | Val like pairs | Val view pairs |
|---|---|---|---|
| Default (dropout 0.1, weight decay 0.05, cosine, pair weight 1, no label smoothing) | 2.9684 | 61.26% | 59.26% |
| Dropout 0 | 2.9653 | 61.12% | 58.80% |
| Dropout 0.25 | 2.9767 | 61.25% | 59.00% |
| Weight decay 0.3 | 2.9699 | 61.25% | 58.95% |
| Pair loss off | 2.9691 | 61.01% | 58.71% |
| Pair loss weight 3 | 2.9744 | 61.21% | 59.14% |
| Label smoothing 0.1 on the histogram targets | 3.0162 | 61.12% | 59.72% |
| Constant learning rate after warm-up | 2.9707 | 61.31% | 58.87% |

- Like pairs range from 61.0% to 61.3%; view pairs from 58.7% to 59.7% on 18,292 pairs. Nothing separates from the default.
- The one consistent sign: turning the pair loss off is lowest on both (-0.25 and -0.55), so it stays on; weight 3 is no better than 1.
- Label smoothing worsens the likelihood by construction (the model is trained toward flatter targets) and its view-pair gain of +0.5 is one seed at noise level. Not adopted.
- No setting was carried forward, because none earned it. These were 2,000-step runs; a setting that only pays off late (dropout against overfitting at 16,000 steps) was not tested there, though the early-stopped result would cap its value at what 4,000 steps already gives.

## 5. Benchmark and launch pairs

Checkpoints: the best-validation checkpoint of each run. By validation loss the best two are the 8,000- and 16,000-step runs (both stopped at step 4,000, loss 2.9475 and 2.9474); the 4,000-step run and the 8.2M model are added because they answer the size question.

**3,530 same-topic pairs** (2,852 with likes 1.25x apart; intervals resample accounts; confidence cross-fitted as in `gap_table.py`). Baseline per-pair scores are the stored `bench2` ones.

| Model | Correct, all 2,852 | vs `ts46_s1`, points | 2x pairs (2,071): correct | Confident share at 2x (pairs) | Correct there | 3x: confident share, correct | 5x: confident share, correct |
|---|---|---|---|---|---|---|---|
| Baseline `ts46_s1` | 59.9% (57.4 to 62.1) | | 60.8% | 25% (516) | 70.5% (65 to 75) | 26%, 73.8% | 27%, 80.7% |
| Baseline ensemble (three models) | 60.0% (57.9 to 62.0) | | 61.3% | 25% (528) | 70.1% (66 to 74) | 27%, 74.3% | 28%, 81.2% |
| small, 4,000 steps | 59.4% (57.0 to 61.8) | -0.5 (-1.6 to +0.8) | 61.3% | 26% (546) | 72.0% (67 to 77) | 28%, 75.5% | 28%, 82.3% |
| small, 8,000 steps (stopped at 4,000) | 59.8% (56.9 to 62.4) | -0.1 (-1.4 to +1.3) | 61.4% | 27% (569) | 71.5% (66 to 76) | 28%, 75.1% | 29%, 82.1% |
| small, 16,000 steps (stopped at 4,000) | 59.7% (57.1 to 62.0) | -0.2 (-1.5 to +1.3) | 61.3% | 27% (560) | 72.0% (67 to 77) | 28%, 75.5% | 29%, 82.1% |
| width 384, 6 layers, 8.2M | 59.8% (57.8 to 61.9) | -0.1 (-1.5 to +1.4) | 61.2% | 28% (570) | 70.7% (65 to 75) | 28%, 73.6% | 29%, 79.5% |
| Ensemble: 4,000 + 8,000 + 8.2M | 59.7% (57.4 to 62.1) | -0.2 (-1.3 to +0.9) against the baseline ensemble | 61.5% | 28% (586) | 72.4% (68 to 77) | 29%, 75.6% | 30%, 82.8% |

The 8.2M model against `small` at 4,000 steps: +0.4 (-1.0 to +1.7). Confident calls of the new ensemble against the baseline ensemble: +1.6 (-0.2 to +3.5), not a detectable difference.

**850 launch pairs** (151 accounts; same pairs as the launch track; "top 26%" is the quarter of pairs with the largest predicted gap, the baseline's share of confident calls, so no confidence scale is needed):

| Model | All 850 | vs baseline ensemble, points | 2x pairs (568) | 3x pairs (401) | Top 26% (221 pairs) | Top 26% and 2x |
|---|---|---|---|---|---|---|
| Baseline ensemble | 63.6% (59.5 to 68.0) | | 66.0% | 67.8% | 73.3% (64 to 82) | 78.6% (69 to 87) |
| Baseline `ts46_s1` | 63.8% (59.4 to 68.2) | +0.1 (-2.4 to +2.7) | 66.5% | 68.1% | 72.9% | 76.2% |
| small, 4,000 steps | 63.5% (59.2 to 67.8) | -0.1 (-3.0 to +2.7) | 66.0% | 68.8% | 76.5% (68 to 85) | 79.3% |
| small, 8,000 steps (stopped at 4,000) | 64.6% (60.2 to 68.9) | +0.9 (-2.4 to +4.0) | 68.3% | 69.1% | 75.6% | 79.8% |
| small, 16,000 steps (stopped at 4,000) | 63.8% (59.0 to 68.3) | +0.1 (-3.0 to +3.0) | 67.8% | 68.8% | 76.0% | 79.4% |
| width 384, 6 layers, 8.2M | 64.0% (59.3 to 68.3) | +0.4 (-3.2 to +3.6) | 65.8% | 67.8% | 77.8% (70 to 85) | 81.1% (72 to 89) |
| Ensemble of the four new models | 63.9% (59.3 to 68.3) | +0.2 (-2.7 to +2.8) | 66.4% | 67.8% | 76.5% | 80.2% |

(The baseline reads 63.6% here against the launch report's 63.7%: one tied pair.) The top-quarter figures of the new models are 2 to 4 points above the baseline's, on 221 pairs with intervals of ±8 points: not a finding.

## 6. Projection: a 10x larger model trained 10x longer on a GPU

What the curves say, for the present table (303k training rows) and the present frozen encoder:

| Lever | Measured slope | Projected effect of 10x | Range I would bet on |
|---|---|---|---|
| Trunk size (1.5M → 8.2M measured) | -0.2 like pairs, +0.4 view pairs per 5.4x | 0 | -0.5 to +0.5 points on ordinary pairs |
| Training length (4,000 → 16,000 measured) | 0 with early stopping; -1.7 without | 0 | 0 with early stopping; -2 or worse without |
| Both together, ordinary same-account pairs | | **0 points** | **-0.5 to +0.5** |
| Both together, 3,530-pair benchmark | all new models -0.5 to -0.1 against `ts46_s1` | **0 points** | **-1.5 to +1.5** (the width of a single model's paired interval; a real effect smaller than that cannot be seen on this benchmark) |
| Both together, 850 launch pairs | +0.1 to +0.9 | 0 | -3 to +3 (interval of that set) |
| 10x more accounts of the same kind, `small` | +0.9 then +0.4 per doubling on like pairs | about +1 | +0.4 to +2.2 on like pairs; views unmeasurable here |

How sure: the size and length curves are flat over a 30x range of parameters and an 8x range of steps, each point one seed. I would put the chance that a 10x trunk trained 10x longer gains more than 1 point on ordinary like pairs, with nothing else changed, at under 10%. What I did not measure and so cannot rule out:

- a larger trunk on much more data (the best step moving with data share says the two interact);
- a larger trunk on top of a better or fine-tuned encoder (stage B). The encoder track found no gain from larger frozen encoders; a fine-tuned one is untested and is the only GPU use these curves leave open;
- the pairs track's ceiling: label noise caps benchmark-style pairs near 80% and the model sits at 60%, so the gap is real but it is not a capacity gap in the trunk.

**The GPU run, if the owner wants the direct check anyway.** 10x the model is the `base` preset (17.8M parameters, 12x `small`); 10x longer is 40,000 steps:

```
cd virality && .venv/bin/python scripts/train_one.py --table data/work/model/table --out data/runs/base_40k_s1 \
    --preset base --steps 40000 --features ts46 --seed 1 --device cuda
```

- Expected time: 0.5 to 1.5 GPU-hours on one 24 GB card, plus about 10 minutes for the benchmark on CPU (`bash scripts/bench_pairs.sh base_40k_s1 data/runs/base_40k_s1/best.pt`). **This is an estimate, not a measurement**: on CPU `base` takes 4.2 s per step, and I assumed a 100x to 300x speed-up; no GPU was available to time it.
- `train_one.py` validates only 6 times per run (every 6,667 steps here), and the best step is near 4,000, so it will keep an overfitted checkpoint. Use `scripts/scale_run.py` with `--evals 2000,4000,6000,8000,...` instead once it has a `--device` option (it is CPU-only as written), or run `--steps 6000`.
- `scripts/gpu_run.sh` as it stands runs the whole pipeline from raw downloads, with stage A at `PRESET=large`, 60,000 steps of batch 256 per seed for three seeds, then stage B on the 8B encoder: `CA_ANON_KEY=<key> PRESET=base SEEDS="1" SKIP_MEDIA=1 scripts/gpu_run.sh`. MODEL.md section 8 puts stage A at a few GPU-hours and stage B at tens of GPU-hours per seed on an 80 GB card. Stage A at 60,000 x 256 is 51 passes over the present training rows; on these curves everything after the first pass or two is wasted unless the corpus is several times larger.

**Recommendation.** Do not rent a GPU for a bigger or longer-trained trunk. If a GPU is rented, spend it on stage B (the fine-tuned encoder) with the `small` trunk, and on embedding a larger corpus; keep `small` at about one pass over the data with early stopping.

## What worked, what did not

- Worked: the measurement itself. Three flat curves and one shallow one, cheaply, with fixed validation rows so runs are comparable.
- Did not work (as improvements): longer training, a larger trunk, any regularisation setting. None beats the baseline on validation, test, the benchmark or the launch pairs.
- Open: history 96 (+0.4 on test like pairs, +1.1 on view pairs, one seed, at noise level). Needs two more seeds and a benchmark cache with 96 posts of history.

## Honest limits

- One seed per configuration. Differences under about 0.3 points (like pairs) or 1 point (view pairs) are noise, and I did not run a second seed to confirm that noise level; it comes from the earlier reports.
- The largest model is 8.2M parameters, 5.4x `small`. `base` (12x) was profiled, not trained. The projection to 10x extrapolates one step beyond the data.
- The regularisation sweep and the history comparison ran at 2,000 steps, not 4,000, to fit the time.
- The longer runs used the same peak learning rate (1e-3). A lower rate for the larger model or the longer run was not tried; the early-stopped results suggest it would land at the same place, but that is an inference.
- The data curve has three points, drawn once. Its extrapolation spans a 5x range (+0.4 to +2.2).
- The benchmark caches hold 56 posts of history per pair, so a history-96 model could not be benchmarked with its full window; it was not benchmarked.
- Everything is on the bge-small frozen encoder. With a different encoder the size curve could differ.
- GPU hours are estimated, not measured.

## Files in `tracks/scale/`

- `REPORT.md`, `patch.diff`.
- `summary.json` (every run: configuration, validation curve, best step, test metrics, seconds per step), `runs/<name>/result.json` (full detail per run).
- `bench_gap_table.json` and `.txt` (benchmark by gap and confidence band, baselines and new models), `bench/<model>/shard_*.parquet` (per-pair scores).
- `launch_pairs.json` and `.txt`, `launch_scores.parquet` (per-post predictions of the four new models on the 582 launch posts).
- No checkpoint is uploaded as an improvement: none beats the baseline.
