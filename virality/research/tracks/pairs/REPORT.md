# Track "pairs": more controlled pairs, used to train the surrogate itself

Worker report, 2026-11-16. All numbers are likes; the benchmark is the fixed 3,530 same-account, same-topic pairs of MODEL.md 9.4
(2,852 with like counts at least 25% apart). No benchmark pair was trained or tuned on.

## Headline

The hypothesis holds weakly: a small gain that is real at the level of the three-model ensemble, well short of the goal, and it does not come from the newly mined pairs.

- **Ensemble.** Three 46-column models each fine-tuned for 600 steps with the pair loss (weight 10): 61.4% of all pairs correct (58.9 to 64.0) against 60.0% for the baseline ensemble,
  +1.4 (+0.2 to +2.6) points paired, interval from resampling accounts. The same three models fine-tuned the same way without the pair loss score 60.0%; against that control the gain is +1.4 (+0.1 to +2.5).
- **Confident calls.** On pairs 2x or more apart: 34% of pairs at 71.8% correct, against 25% at 70.1% (baseline) and 25% at 68.6% (control).
  At 3x: 36% at 75.4%; at 5x: 39% at 82.1%. The goal is 40% coverage at 75 to 80% on 2x pairs: coverage moved a third of the way, accuracy barely.
- **Single model.** The best fine-tune of `ts46_s1` is 60.3% against 59.9%, +0.4 (-0.8 to +1.3): not detectable alone. The three members gain +0.4, +0.9 and +0.9 against their own starting points.
- **The mined pairs add nothing.** Fine-tuning on the 1,471 same-link-or-quote pairs alone does as well as on all 11,507. On mined pairs of held-out accounts accuracy stays at 54.3% before and after.
- **From scratch is worse.** Training from scratch with the pair loss gives 56.8% (-3.1 (-5.2 to -0.6) points). It memorised the training pairs
  (95.9% correct on them) and stayed at 53.9% on held-out pairs.
- **Why so little.** When one account posts the same text twice, the like counts differ by 2x or more 38% of the time. Pairs of one message reworded are barely more spread out than that,
  so their labels are mostly luck and their ceiling is about 72%. Benchmark-style pairs (two takes on one link) have a ceiling near 80% overall and 85% at 2x.
- **Checkpoints.** `runs/ft_w10`, `runs/ft_w10_s2`, `runs/ft_w10_v2full` together beat the baseline ensemble by the margin above. Caveats before shipping them are under "Honest limits".

## Step 1. Mining more controlled pairs

`scripts/evidence/mine_pairs.py`, `scripts/evidence/select_pairs.py`. Counts in `pair_counts.json`.

Embedding all 4.48M posts with bge-small takes many hours on four cores, so mining ran in two stages. First a word-overlap prefilter: within each account,
every pair of posts at most 14 days apart, both at least 6 words, with TF-IDF cosine of 0.25 or more over word 1- and 2-grams (links removed): 2,145,761 candidate pairs
over 321,490 posts. Then those posts were embedded with bge-small (plain text, links removed, no author line) and each pair got its embedding cosine. Pairs with no shared
words at all are therefore missed. The same-link and same-quoted-post pairs of the benchmark definition were added whatever their cosine.

Threshold, from reading eight random pairs per band (pairs whose posts sit in at most three candidate pairs):

| Cosine | What the sampled pairs are |
|---|---|
| 0.95 and over | 8 of 8 the same message reworded (a sentence added, a word changed, the same story from another outlet) |
| 0.90 to 0.95 | 6 of 8 the same message; 2 the same topic with different content |
| 0.85 to 0.90 | about half the same message; the rest one template with different content (market recaps, action alerts) |
| 0.80 to 0.85 | 2 of 8; mostly templates and replies |
| below 0.80 | same account voice or same topic at most |

So 0.90 and over is the tier "strict" (same message reworded) and 0.85 to 0.90 the tier "loose". In a second sample taken from the final training set, 10 of 12 strict pairs
were the same message reworded (the two misses were a weekly podcast title and a daily reading list), and about 6 of 12 loose pairs.
The benchmark's own pairs are looser than either tier: their median cosine is 0.70, they are two takes on one link or quoted post, not one message reworded.

| Stage | Pairs | Before 2026-04-23 | After | Straddling | Accounts | Likes 1.25x apart | 2x apart |
|---|---|---|---|---|---|---|---|
| Same link or quoted post, 48 hours (benchmark definition) | 9,193 | 7,158 | 2,030 | 5 | 1,494 | 7,456 | 5,593 |
| Same link or quoted post, 14 days | 11,960 | 9,273 | 2,664 | 23 | 1,866 | 9,644 | 7,006 |
| Cosine 0.85 and over, texts differ | 807,280 | 652,196 | 78,025 | 77,059 | 10,949 | 463,097 | 215,403 |
| Cosine 0.90 and over, texts differ | 537,168 | 403,338 | 69,899 | 63,931 | 8,320 | 291,100 | 138,557 |
| After length and reply filters (cosine 0.85 and over, or shared link or quote) | 788,117 | 633,961 | 77,177 | 76,979 | 11,477 | 454,811 | 209,939 |
| After dropping template posts (a post in more than 3 mined pairs) | 74,131 | 64,309 | 8,090 | 1,732 | 11,310 | 54,267 | 34,716 |
| After the cap of 40 pairs per account and period | 60,678 | 51,246 | 7,700 | 1,732 | 11,310 | 43,940 | 26,854 |
| of which shared link or quoted post | 6,449 | 4,530 | 1,898 | 21 | 1,656 | 5,137 | 3,577 |
| of which strict (0.90 and over) | 31,954 | 27,587 | 3,458 | 909 | 7,843 | 22,867 | 14,118 |
| of which loose (0.85 to 0.90) | 22,275 | 19,129 | 2,344 | 802 | 6,879 | 15,936 | 9,159 |
| Training pool: before the cutoff, no benchmark post, no account with a benchmark calibration pair | 45,786 | 45,786 | 0 | 0 | 9,949 | 32,664 | 19,193 |
| of which shared link or quoted post | 2,171 | 2,171 | 0 | 0 | 926 | 1,713 | 1,185 |
| of which strict | 25,232 | 25,232 | 0 | 0 | 6,957 | 17,869 | 10,573 |
| of which loose | 18,383 | 18,383 | 0 | 0 | 6,053 | 13,082 | 7,435 |

Reading the table: raw near-duplicates are plentiful (403,338 pairs at cosine 0.90 before the cutoff) but nine in ten come from posts that repeat a template dozens of times.
After removing those, 51,246 pre-cutoff pairs remain against 7,158 by the benchmark definition, about seven times more. The training pool is 45,786 pairs, 32,664 with likes 1.25x apart.

Excluded from training: every pair containing a benchmark post, and every account that has a benchmark calibration pair (their histories contain benchmark posts). That second rule
costs half of the pre-cutoff link-or-quote pairs (4,530 to 2,171), because the calibration half of the benchmark was drawn from exactly those accounts.

Reposted-own-content pairs (identical text) were mined and used in step 2 as the noise measurement; they carry no wording signal and were not trained on. Thread-opener variants were
not mined separately: the filtered corpus holds originals and quotes, and an opener reworded is already a near-duplicate pair.

**What was actually trained on.** The pair table is built through the serving path (step 3), which needs bge-small embeddings of each pair's history, about 60 posts per second on this machine.
The full pool needed 468,000 new embeddings (about two hours), so accounts were chosen within a budget of 105,000: all accounts with a link-or-quote pair first, then by pairs per post to embed.
Result: 13,518 pairs from 3,037 accounts, all with likes 1.25x apart: 11,507 for training (1,471 link-or-quote, 6,026 strict, 4,010 loose)
and 2,011 from 15% of the accounts held out (242, 1,093, 676). The selection favours accounts with short histories (median 28 posts of history, against 48 for link-or-quote pairs).

## Step 2. Label noise: the ceiling for any model

`scripts/evidence/label_noise.py`, `label_noise.json`. Same account, same normalised text (lower case, links removed), at least 6 words, same media state and kind, at most 14 days apart.
At serving time nothing distinguishes the two sides, so their spread is noise no model can predict.

| Identical-text pairs | Pairs | Likes within 1.25x | within 2x | within 3x | within 5x |
|---|---|---|---|---|---|
| All (3,504 accounts) | 29,757 | 35% | 61% | 81% | 91% |
| At most 20 per account | 16,096 | 33% | 62% | 79% | 89% |
| One per account | 3,504 | 31% | 62% | 77% | 87% |
| At most 48 hours apart | 18,974 | 35% | 59% | 80% | 90% |
| 2 to 14 days apart | 10,783 | 34% | 63% | 82% | 92% |
| After the cutoff | 1,929 | 39% | 61% | 80% | 91% |
| Larger like count 0 to 9 | 18,110 | 44% | 66% | 88% | 97% |
| Larger like count 10 to 99 | 8,057 | 24% | 58% | 75% | 87% |
| Larger like count 100 to 999 | 2,736 | 14% | 41% | 58% | 73% |
| Larger like count 1000 to up | 854 | 9% | 28% | 41% | 54% |

The same text posted twice lands within 2x only 62% of the time, and beyond 5x 11% of the time. The noise grows with the size of the post:
above 1,000 likes, identical posts are within 2x only 28% of the time. The later copy wins 45% of the time (the benchmark's 64% for the later version does not come from reposting).

Reworded pairs next to that (at most 20 per account):

| Pairs | Count | Likes within 2x | within 3x | Spread of log gap |
|---|---|---|---|---|
| Identical text | 16,096 | 62% | 79% | 1.08 |
| Texts differ, cosine 0.97 to 1.00 | 20,201 | 60% | 75% | 1.27 |
| Texts differ, cosine 0.93 to 0.97 | 22,032 | 63% | 79% | 1.14 |
| Texts differ, cosine 0.90 to 0.93 | 22,401 | 62% | 79% | 1.10 |
| Texts differ, cosine 0.85 to 0.90 | 35,521 | 60% | 78% | 1.12 |
| Texts differ, cosine 0.80 to 0.85 | 35,111 | 59% | 77% | 1.16 |
| Texts differ, cosine 0.70 to 0.80 | 39,568 | 56% | 74% | 1.20 |
| Texts differ, cosine 0.60 to 0.70 | 18,151 | 54% | 71% | 1.31 |
| Same link or quoted post, 48 hours | 4,669 | 42% | 62% | 1.51 |

Near-duplicate pairs are spread almost exactly like identical ones. Link-or-quote pairs are clearly wider: those two posts differ in more than wording.

**Ceiling by size of gap.** The observed log gap is modelled as effect plus noise, the noise drawn from the identical pairs and the effect from a Laplace distribution holding the
variance the pairs have beyond noise. The ceiling is how often the observed winner is the side with the better effect: the score of a model that knew every effect exactly.

| Real gap at least | Benchmark-style pairs (link or quote, 48 h): share, ceiling | Reworded, cosine 0.90 and over: share, ceiling | Cosine 0.85 to 0.90: share, ceiling |
|---|---|---|---|
| 1.25x | 84%, 80% | 78%, 71% | 73%, 64% |
| 2.0x | 56%, 85% | 44%, 75% | 39%, 67% |
| 3.0x | 38%, 87% | 26%, 75% | 22%, 65% |
| 5.0x | 23%, 87% | 14%, 73% | 11%, 63% |

With a normal instead of a Laplace effect the ceilings are 2 to 3 points higher. Two readings. On the benchmark's kind of pair a perfect model would score about 80% overall and 85% at 2x,
so the target (75 to 80% on confident calls) is under the ceiling, and the present 60% is far from it. On reworded-message pairs the ceiling is about 71 to 75% and does not rise with the gap:
a large gap between two wordings of one message is mostly luck. Those are the pairs this track added.
Caveat: the model assumes reworded pairs carry the same noise as identical reposts; if platforms treat exact reposts differently, the noise is misjudged in an unknown direction.

## Step 3. Fine-tuning the surrogate with a loss on the mined pairs

Code: `xviral/model/pairs.py`, `scripts/build_pair_table.py`, `scripts/train_pairs.py`, small changes in `xviral/model/train.py` and `config.py` (all in `patch.diff`).

- **Pair table.** Each pair goes through `DraftScorer.featurize` exactly as in the benchmark: the two posts as alternative drafts at the time of the earlier one, the account's earlier posts
  as history (48, the model's window), neither outcome visible. Training episodes are therefore the serving episode, not consecutive posts with their own features.
- **Loss.** RankNet on the difference of the like head's predicted mean for the two isolated candidates, weight proportional to the log like ratio (capped at 5x), pairs under 1.25x unused,
  added to the ordinary loss with weight `w_ctl`. One batch of 64 pairs per step next to the ordinary batch of 64 episodes. A margin variant exists (`--ctl-loss margin`) and was not run for lack of time.
- **Runs.** From `runs/ts46_s1/best.pt`, 600 steps, learning rate 2e-4, warm-up 50, weights 0 (control), 1, 3, 10, 30; one run on the link-or-quote pairs alone; weights 10 and 0 also from `ts46_s2` and `compare_v2/full`
  to form a fine-tuned and a control three-model ensemble. Checkpoint choice inside a run uses the held-out controlled pairs and the validation loss, never the benchmark.
- **Benchmark.** `tac_serve_eval.py` with a new `--pairs` option that scores the stored list of 3,530 pairs. My run of the unchanged `ts46_s1` reproduces the stored per-pair scores (correlation 0.99999, same accuracy).

| Model | Trained pairs | Benchmark, all pairs | vs start, points | 2x and over | Confident share | Correct there | vs start | Held-out pairs | Training pairs | Test pairwise likes | Test like NLL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ts46_s1 (start) | 0 | 59.9% | - | 60.9% | 27% | 67.1% | - | 54.3% | 55.8% | 0.5981 | 2.966 |
| control, weight 0 | 0 | 60.0% | +0.1 (-0.6 to +0.8) | 60.9% | 24% | 66.7% | -0.4 (-1.6 to +1.5) | 53.8% | 55.8% | 0.5983 | 2.976 |
| weight 1 | 11,507 | 59.4% | -0.5 (-1.3 to +0.2) | 60.8% | 29% | 66.5% | -0.5 (-1.9 to +1.1) | 54.5% | 57.8% | 0.5983 | 2.972 |
| weight 3 | 11,507 | 59.8% | -0.1 (-1.2 to +1.0) | 61.4% | 32% | 68.6% | +1.6 (-0.3 to +3.6) | 54.3% | 59.8% | 0.5980 | 2.979 |
| weight 10 | 11,507 | 60.3% | +0.4 (-0.8 to +1.3) | 61.7% | 31% | 68.4% | +1.3 (-1.0 to +3.8) | 54.3% | 58.2% | 0.5991 | 2.989 |
| weight 30 | 11,507 | 60.3% | +0.4 (-0.8 to +1.6) | 62.0% | 31% | 69.1% | +2.1 (-0.2 to +5.0) | 54.4% | 58.5% | 0.6000 | 3.001 |
| link-or-quote pairs only, weight 3 | 1,471 | 60.3% | +0.4 (-1.0 to +1.8) | 61.4% | 33% | 68.7% | +1.6 (-0.9 to +4.1) | 54.7% | 57.4% | 0.5981 | 3.008 |
| half the pairs, weight 1 | 5,595 | 59.5% | -0.4 (-1.1 to +0.4) | 60.6% | 27% | 67.2% | +0.2 (-1.2 to +2.0) | 54.3% | 57.6% | 0.5986 | 2.973 |
| from scratch, weight 1 (step 4) | 11,507 | 56.8% | -3.1 (-5.2 to -0.6) | 58.1% | 15% | 64.2% | -2.9 (-8.2 to +3.4) | 53.9% | 95.9% | 0.5917 | 2.970 |
| from ts46_s2: control, weight 0 | 0 | 59.5% | -0.4 (-1.8 to +1.1) | 60.0% | 25% | 67.3% | +0.2 (-2.4 to +3.0) | 54.2% | 55.9% | 0.6023 | 2.973 |
| from ts46_s2: weight 10 | 11,507 | 59.9% | -0.0 (-1.4 to +1.2) | 60.7% | 30% | 68.1% | +1.0 (-1.5 to +3.7) | 54.4% | 58.6% | 0.6021 | 2.989 |
| from compare_v2/full: control, weight 0 | 0 | 59.3% | -0.6 (-2.2 to +0.9) | 60.2% | 26% | 65.5% | -1.5 (-4.1 to +1.5) | 55.8% | 56.5% | 0.5984 | 2.970 |
| from compare_v2/full: weight 10 | 11,507 | 61.1% | +1.2 (-0.5 to +2.8) | 63.0% | 35% | 70.0% | +2.9 (+0.5 to +6.0) | 55.4% | 62.2% | 0.5980 | 2.990 |

Benchmark columns: 2,852 pairs; intervals are 95% from resampling accounts, paired against `ts46_s1` (for the last four rows that is not their own starting model; those comparisons follow below). "Confident" is stated confidence of 60% or more, scale cross-fitted over accounts.
Held-out pairs: 2,011 mined pairs from 444 accounts never used for the pair loss. Test columns: the ordinary audience test split (95,357 rows).

Against each model's own start and its own control (points, paired):

| Member | Weight 10 vs start | Control vs start | Weight 10 vs control | Same, among confident calls |
|---|---|---|---|---|
| ts46_s1 | +0.4 (-0.8 to +1.3) | +0.1 (-0.6 to +0.8) | +0.3 (-0.7 to +1.3) | +1.7 (-0.9 to +4.3) |
| ts46_s2 | +0.9 (-0.2 to +2.1) | +0.6 (-0.2 to +1.2) | +0.4 (-0.8 to +1.5) | +0.8 (-1.4 to +3.6) |
| compare_v2/full | +0.9 (-0.4 to +2.2) | -0.9 (-1.6 to -0.2) | +1.8 (+0.6 to +3.1) | +4.4 (+2.1 to +6.5) |
| Ensemble of the three | +1.4 (+0.2 to +2.6) | +0.1 (-0.4 to +0.7) | +1.4 (+0.1 to +2.5) | +2.3 (+0.2 to +4.6) |

What the tables say:

1. **No transfer to held-out mined pairs.** Accuracy on them is 54.3% before and 54.3% to 54.5% after, at every weight, while accuracy on the training pairs rises from 55.8% to 59.8%.
2. **A small gain on the benchmark, visible only when models are pooled.** No single fine-tune clears its interval against its start. The three together do, against the baseline ensemble and against the control ensemble, by about 1.4 points.
   Fine-tuning without the pair loss changes nothing (control ensemble equals the baseline to a tenth of a point), so the gain is the pair loss, not the extra steps.
3. **More confident calls, slightly more often right.** From weight 3 up, confident calls cover 31 to 33% of pairs instead of 27% for the single model, and 34% instead of 27% for the ensemble, at 1.5 to 2 points higher accuracy.
   The model separates the two versions further apart where it was already leaning the right way; it does not get many more pairs right.
4. **1,471 link-or-quote pairs do as much as all 11,507.** The 10,036 mined rewordings add nothing measurable on top. That matches step 2: their labels are close to noise.
   What helps is training on the benchmark's own kind of pair, scored the way the product scores, and only about 1,700 such pairs were available for training.
5. **Weight.** 1 does nothing; 3, 10 and 30 are alike on the benchmark. Weight 10 was used for the other two members.
6. **Ordinary metrics hold.** Test pairwise accuracy on likes stays at 0.598 to 0.600. The like NLL drifts up with the weight (2.966 at the start, 2.989 at weight 10, 3.001 at weight 30), a small loss of calibration.
   (The validation NLL printed at the start of each run, 2.942, is on a different 20,000-row subset than the in-run figure, 2.957; the control shows the in-run level is unchanged.)

By held-out tier (pairs 1.25x apart; start model, then weight 10):

| Tier | Held-out pairs | Start | Weight 10 | Link-or-quote only | From scratch |
|---|---|---|---|---|---|
| Same link or quoted post | 242 | 59.1% | 57.4% | 58.3% | 59.1% |
| Strict (cosine 0.90 and over) | 1,093 | 53.0% | 52.7% | 53.7% | 53.7% |
| Loose (0.85 to 0.90) | 676 | 54.6% | 55.6% | 55.2% | 52.4% |
| All | 2,011 | 54.3% | 54.3% | 54.7% | 53.9% |

The start model is at 53% on reworded-message pairs and 59% on link-or-quote pairs, against ceilings of about 71% and 80%.

### Next to the baseline: accuracy by size of real gap

| Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there (95% interval) |
|---|---|---|---|---|---|
| Baseline: three 46-column models | 1.25x | 2,851 | 59.9% | 27% (777) | 67.3% (63.2 to 71.1) |
| Baseline: three 46-column models | 2.0x | 2,071 | 61.3% | 25% (528) | 70.1% (65.8 to 74.2) |
| Baseline: three 46-column models | 3.0x | 1,390 | 63.2% | 27% (373) | 74.3% (69.0 to 79.3) |
| Baseline: three 46-column models | 5.0x | 880 | 66.5% | 28% (245) | 81.2% (74.4 to 86.4) |
| The same three, fine-tuned with the pair loss (weight 10) | 1.25x | 2,851 | 61.4% | 34% (983) | 69.3% (65.5 to 72.6) |
| The same three, fine-tuned with the pair loss (weight 10) | 2.0x | 2,071 | 62.8% | 34% (701) | 71.8% (66.7 to 76.1) |
| The same three, fine-tuned with the pair loss (weight 10) | 3.0x | 1,390 | 65.3% | 36% (499) | 75.4% (68.9 to 80.7) |
| The same three, fine-tuned with the pair loss (weight 10) | 5.0x | 880 | 69.1% | 39% (340) | 82.1% (75.9 to 87.7) |
| The same three, fine-tuned without it (control) | 1.25x | 2,851 | 60.0% | 26% (751) | 67.0% (62.5 to 70.7) |
| The same three, fine-tuned without it (control) | 2.0x | 2,071 | 61.2% | 25% (516) | 68.6% (63.1 to 73.9) |
| The same three, fine-tuned without it (control) | 3.0x | 1,390 | 63.2% | 26% (358) | 72.6% (65.2 to 78.3) |
| The same three, fine-tuned without it (control) | 5.0x | 880 | 66.8% | 27% (236) | 79.2% (71.8 to 84.9) |
| ts46_s1 alone | 1.25x | 2,851 | 59.9% | 26% (755) | 67.0% (62.9 to 70.6) |
| ts46_s1 alone | 2.0x | 2,071 | 60.9% | 25% (516) | 70.5% (65.1 to 75.0) |
| ts46_s1 alone | 3.0x | 1,390 | 62.2% | 26% (363) | 73.8% (67.0 to 79.7) |
| ts46_s1 alone | 5.0x | 880 | 65.2% | 27% (238) | 80.7% (74.6 to 85.8) |
| ts46_s1 fine-tuned, weight 10 | 1.25x | 2,851 | 60.3% | 31% (890) | 68.3% (64.9 to 71.7) |
| ts46_s1 fine-tuned, weight 10 | 2.0x | 2,071 | 61.7% | 30% (625) | 71.8% (66.5 to 76.0) |
| ts46_s1 fine-tuned, weight 10 | 3.0x | 1,390 | 64.2% | 31% (436) | 76.1% (70.0 to 81.9) |
| ts46_s1 fine-tuned, weight 10 | 5.0x | 880 | 67.0% | 34% (296) | 83.1% (76.2 to 88.0) |
| ts46_s1 fine-tuned, link-or-quote pairs only | 1.25x | 2,851 | 60.3% | 33% (935) | 68.7% (65.3 to 71.6) |
| ts46_s1 fine-tuned, link-or-quote pairs only | 2.0x | 2,071 | 61.4% | 32% (663) | 71.5% (66.9 to 75.8) |
| ts46_s1 fine-tuned, link-or-quote pairs only | 3.0x | 1,390 | 64.0% | 34% (466) | 76.4% (70.7 to 81.2) |
| ts46_s1 fine-tuned, link-or-quote pairs only | 5.0x | 880 | 66.7% | 36% (319) | 84.0% (78.5 to 88.1) |
| From scratch with the pair loss | 1.25x | 2,851 | 56.8% | 15% (419) | 64.2% (58.4 to 70.6) |
| From scratch with the pair loss | 2.0x | 2,071 | 58.1% | 13% (272) | 67.6% (60.1 to 73.7) |
| From scratch with the pair loss | 3.0x | 1,390 | 60.6% | 14% (188) | 74.5% (66.7 to 81.4) |
| From scratch with the pair loss | 5.0x | 880 | 62.0% | 14% (120) | 80.8% (71.6 to 87.9) |

The gap is measured on (likes + 1), so the pair counts at 2x and 3x (2,071 and 1,390) are slightly above the README's 2,030 and 1,377; the baseline rows otherwise match it.

By stated confidence, pairs 1.25x apart (correct share of pair count):

- Baseline ensemble: 0.50 to 0.55: 55.1% of 1412, 0.55 to 0.60: 61.6% of 662, 0.60 to 0.70: 66.3% of 563, 0.70 to 0.80: 72.1% of 172, 0.80 to 1.00: 61.9% of 42
- Fine-tuned ensemble, pair loss: 0.50 to 0.55: 56.3% of 1207, 0.55 to 0.60: 59.0% of 661, 0.60 to 0.70: 67.5% of 662, 0.70 to 0.80: 73.3% of 262, 0.80 to 1.00: 71.2% of 59
- Fine-tuned ensemble, control: 0.50 to 0.55: 55.0% of 1410, 0.55 to 0.60: 62.8% of 690, 0.60 to 0.70: 64.9% of 552, 0.70 to 0.80: 75.3% of 170, 0.80 to 1.00: 58.6% of 29
- ts46_s1: 0.50 to 0.55: 55.8% of 1454, 0.55 to 0.60: 60.7% of 642, 0.60 to 0.70: 64.2% of 553, 0.70 to 0.80: 76.9% of 160, 0.80 to 1.00: 66.7% of 42
- ts46_s1, weight 10: 0.50 to 0.55: 55.0% of 1304, 0.55 to 0.60: 59.8% of 657, 0.60 to 0.70: 66.9% of 625, 0.70 to 0.80: 71.3% of 216, 0.80 to 1.00: 73.5% of 49
- From scratch: 0.50 to 0.55: 54.1% of 1696, 0.55 to 0.60: 58.8% of 736, 0.60 to 0.70: 65.2% of 394, 0.70 to 0.80: 48.0% of 25, 0.80 to 1.00: - of 0

The bands above 70% hold 200 to 300 pairs, and the 80% band 40 or fewer: read those as plus or minus 6 to 15 points.

## Step 4. From scratch with the pair loss

Same recipe as `ts46_s1` (small preset, 46 columns, seed 1, 4,000 steps, learning rate 1e-3) with the pair loss at weight 1 from the first step.
Ordinary validation ends where the baseline does (pairwise likes 0.613, test like NLL 2.970), but the benchmark falls to 56.8%
(-3.1 (-5.2 to -0.6) points, below the 58.2 to 60.9% range of the ten existing single models) and confident calls shrink to 15% of pairs.
The model reaches 95.9% on its training pairs and 53.9% on held-out ones: with 4,000 steps of 64 pairs it sees each pair about 22 times and learns
them by heart, apparently at the expense of what the like head otherwise learns about content. One run, one seed; the direction is clear, the size is not.

## What worked, what did not

Worked: fine-tuning with the pair loss on serving-path episodes, modestly (+1.4 points for the ensemble, confident coverage from 27% to 34%); the noise measurement, which explains both the size of this result
and the earlier failure of the direct pair model; the mining recipe and its template filter as a way to find rewordings; scoring the benchmark from a stored pair list with reused embeddings (71 seconds per model once the caches exist).

Did not work: the mined rewordings as training signal (no effect beyond the 1,471 link-or-quote pairs); held-out mined pairs (no gain at any weight); the pair loss from scratch (3 points worse); weight 1 (no effect).

## Honest limits

- **The weight for the ensemble was chosen with the benchmark in view.** Held-out mined pairs could not separate the weights, and weights 1, 3 and 10 had been scored on the benchmark for `ts46_s1` when 10 was picked.
  The other two members were then fine-tuned once at that weight and scored once; their gains do not carry that choice. The ensemble intervals (+0.2 to +2.6, and +0.1 to +2.5 against the control) only just exclude zero.
- One seed per configuration. Single-model differences under about 1.5 points are inside the account-resampling intervals. `compare_v2/full` carries most of the ensemble's gain against control (+1.8); `ts46_s1` and `ts46_s2` add +0.3 and +0.4.
- Only 13,518 of the 32,664 usable pool pairs were trained on (encoder time), chosen toward accounts that were cheap to embed, so with shorter histories than the benchmark's accounts. Held-out accuracy did not differ
  between half the pairs and all of them (54.3% and 54.5%), which suggests the remaining 19,000 would not change the picture, but that was not run.
- Half of the pre-cutoff link-or-quote pairs were unavailable for training because their accounts hold the benchmark's calibration pairs. Since those pairs are what helped, a benchmark that kept only the 2,030 test-period pairs would free about 2,400 more.
- The margin loss, longer fine-tunes, a lower pair weight from scratch, and the pair loss on the ts0 models (which lead this benchmark by about 2 points) were not run.
- Post-cutoff mined pairs (3,458 strict, 2,344 loose) exist as a second test set for reworded messages and were not scored; they would need about 40 minutes of embedding.
- Mining can only find rewordings that share words; a paraphrase with none is missed.
- The ceiling rests on identical reposts being a fair sample of noise, and on a symmetric effect distribution.
- The like head's NLL on the ordinary test split rises by 0.02 at weight 10. The advice layer's confidence scale must be refitted for fine-tuned models (the cross-fitted scale drops from 2.95 to 1.90 for the ensemble).

- Tests: 113 pass, 1 skipped, 2 fail (`tests/test_serve.py::test_advice_gates` and `test_operators_change_one_thing`). Both fail on a missing file, `xviral/data/edit_evidence.json`, which is not in the checkout; neither touches the code changed here.

## What I would do next

1. Re-test the weight-10 fine-tune with fresh seeds and on the ts0 models before shipping it; the gain is small enough that one more ensemble could confirm or erase it.
2. Stop mining rewordings. The limit there is the label, not the count: a ceiling near 72% with almost no room between noise and signal.
3. Get more pairs of the benchmark's own kind, which are the ones that helped (training already used the 14-day version): free the calibration accounts, and drop the cap of 40 per account for this tier.
4. Reduce label noise at the source: the product's own randomized comparisons (research/11, section 7), or accounts that post one text many times, where the average over copies is a far cleaner label.

## Files in `tracks/pairs/`

- `REPORT.md`, `patch.diff` (code changes and new files against the checked-out branch).
- `results.json` (every run: benchmark, held-out pairs, validation, test), `final_gap_table.json` and `.txt` (benchmark by gap and confidence band, all models and ensembles), `dev_eval.json` (held-out and training pairs by tier), `baseline_gap_table.json`.
- `pair_counts.json`, `label_noise.json`, `label_noise_identical.parquet`, `benchmark_by_cosine.json`.
- `train_pool.parquet` (45,786 pairs with cosine and tier), `train_pairs.parquet` (the 13,518 trained and held out), `after_cutoff_pairs.parquet`, `benchmark_pairs.parquet`.
- `pair_table/` (the serving-path feature table of the 13,518 pairs, ready for `scripts/train_pairs.py`).
- `bench/cache_{0..3}.pkl` (feature caches for the 3,530 benchmark pairs: any new checkpoint scores in about a minute with `scripts/bench_pairs.sh`), `bench/<run>/shard_*.parquet` (per-pair scores of every run).
- `runs/<run>/best.pt` and `metrics.json` for each run. `ft_w10`, `ft_w10_s2`, `ft_w10_v2full` are the fine-tuned ensemble; `ft_w0*` the controls.
