# Track "replicate": a clean test of the pair-loss fine-tune

Worker report, 2026-11-16. All numbers are likes. Benchmark: the fixed 3,530 same-account, same-topic pairs (2,852 with likes 1.25x apart, 2,030 at 2x, 1,377 at 3x, 880 at 5x).
Intervals are 95%, from resampling accounts. Settings, outcomes and decision rules were written to `PREREG.md` before the runs they govern. No benchmark pair was trained or tuned on.

## Headline

- **The accuracy gain does not replicate.** Seven models that played no part in choosing the weight, each fine-tuned with the pairs track's exact recipe (weight 10, 600 steps, same pair table) next to its own control:
  the fine-tuned seven are 61.0% correct on 2,852 pairs, the control seven 61.2%, difference -0.1 points (-1.3 to +1.0). The mean of the seven per-member differences is +0.5 (-0.3 to +1.3).
  The pairs track's +1.4 came from its three members; with all ten pooled it is +0.2 (-0.9 to +1.2). The baseline's 60.0% rises to about 61% just by averaging seven or ten untouched models.
- **The gain in confident calls does replicate, in all seven members and on new pairs.** On pairs 2x apart the fine-tuned seven state 60% or more on 37% of pairs (757 of 2,030) and are right on 73.6% of them;
  the control seven on 28% (570) at 70.4%; the baseline three on 26% (523) at 70.2%. Share +9.2 points (+6.6 to +11.9), accuracy among confident calls +3.2 (+1.0 to +5.6).
  Without any confidence scale: on the surest 40% of 2x pairs the fine-tuned ten are right 72.5% of the time (812 pairs), the control ten 68.7%, +3.8 (+1.6 to +6.2); the baseline three 67.6%.
  The same shows on 2,805 development pairs from other accounts that no one had looked at: confident share at 2x 25% against 20%, correct confident calls per 100 pairs +3.8 (+2.5 to +5.3).
- **So what the fine-tune does:** it does not get more pairs right; it makes the size of the predicted gap a better guide to which calls are right, on pairs of the benchmark's kind (two posts on one link or quoted post).
- **It does not carry over to launch pairs.** On the 850 held-out launch pairs fine-tuned and control ensembles are equal in accuracy (63.2% and 63.5% for the tens), and on the surest 40% of 2x pairs the fine-tuned ten are 75.8% against 77.1% (-1.3, -5.4 to +2.6).
  With the confidence scale fitted on the benchmark, the fine-tuned ensembles speak on fewer launch pairs (42% of 2x pairs against 60%), because that scale almost halves after fine-tuning (3.30 to 1.81).
- **Nothing extends it.** Lower learning rate, 1,200 steps, flat weighting, training only on pairs 2x apart, larger ensembles and mixing fine-tuned with untouched members: none beats the recipe on the development set by the rule fixed in advance.
- **Agreement between members adds nothing to the calibrated probability.** The best rule on the development set (probability plus at least 80% of members on the same side) added 0.5 points of coverage there and removed 0.7 on the benchmark.
- **Against the goal** (confident calls 75 to 80% correct on 40% of 2x pairs): the fine-tuned ten give 37% at 73.4%, or 72.5% at exactly 40%. The baseline gives 26% at 70.2%. Closer, not there.

## Step 1. Clean replication

Recipe, read from the pairs track's `metrics.json` and fixed in `PREREG.md` before the first run: `scripts/train_pairs.py`, 600 steps, learning rate 2e-4, warm-up 50, batch 64 + 64 pairs, RankNet loss with gap weights,
pair table `tracks/pairs/pair_table` (11,507 training pairs, 2,011 held out for the script's own checkpoint choice), seed 0, pair weight 10 (treatment) or 0 (control). Each start model keeps its own feature set.
Fourteen checkpoints, each scored once on the benchmark with `scripts/bench_pairs.sh` (same caches as the pairs track). Start-model scores are the stored `bench2/` ones. Primary outcome named in advance:
fine-tuned seven minus control seven on all 2,852 pairs.

Per member (benchmark, 2,852 pairs; last two columns on 2,030 pairs 2x apart):

| Member (features) | Start | Control | Fine-tuned | Fine-tuned minus control, all 2,852 pairs | Same, 2,030 pairs 2x apart | Confident share at 2x: control, fine-tuned | Correct there: control, fine-tuned |
|---|---|---|---|---|---|---|---|
| ts0_s0 (ts0) | 60.9% | 61.1% | 60.6% | -0.5 (-1.8 to +0.7) | -0.1 (-1.6 to +1.4) | 28% (576), 30% (611) | 68.2%, 71.2% |
| ts0_s1 (ts0) | 60.0% | 60.3% | 60.9% | +0.5 (-1.0 to +2.0) | +1.2 (-0.5 to +3.0) | 25% (515), 32% (646) | 69.9%, 71.8% |
| ts20_s1 (ts20) | 58.7% | 59.0% | 59.5% | +0.5 (-0.6 to +1.6) | +0.5 (-0.9 to +1.8) | 20% (413), 29% (581) | 68.5%, 69.0% |
| ts20_s2 (ts20) | 58.2% | 58.9% | 59.5% | +0.6 (-0.7 to +2.0) | +1.4 (-0.2 to +3.0) | 24% (479), 32% (655) | 67.6%, 70.5% |
| cmp_full (ts20) | 60.0% | 59.9% | 61.1% | +1.2 (-0.1 to +2.4) | +1.6 (+0.0 to +3.1) | 26% (525), 36% (737) | 70.5%, 73.3% |
| cmp_aud (ts20) | 59.9% | 59.4% | 60.6% | +1.2 (-0.4 to +2.7) | +1.2 (-0.4 to +2.9) | 26% (535), 36% (721) | 71.2%, 71.8% |
| cmpv2_aud (ts46) | 59.3% | 59.8% | 59.9% | +0.1 (-1.1 to +1.3) | +0.6 (-0.9 to +2.0) | 22% (451), 32% (650) | 67.2%, 73.1% |

Mean of the seven per-member differences: +0.5 (-0.3 to +1.3) on 2,852 pairs; +0.9 (-0.0 to +1.8) on 2,030 at 2x; +1.3 (+0.2 to +2.6) on 1,377 at 3x; +1.3 (+0.0 to +2.5) on 880 at 5x.
Six of seven members move up against their control, none by more than its interval on all pairs. Every member's confident share rises (by 2 to 10 points).

Ensembles (score-level averages; confidence scale cross-fitted over accounts for each row, as `compare_runs.py` does):

| Ensemble | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there |
|---|---|---|---|---|---|
| baseline (3 untouched 46-column models) | 1.25x | 2,852 | 60.0% (57.9 to 61.9) | 27% (778) | 67.4% (63.4 to 70.9) |
| baseline (3 untouched 46-column models) | 2.0x | 2,030 | 61.4% (58.7 to 64.0) | 26% (523) | 70.2% (65.0 to 74.6) |
| baseline (3 untouched 46-column models) | 3.0x | 1,377 | 63.3% (60.2 to 66.3) | 27% (371) | 74.4% (68.6 to 79.6) |
| baseline (3 untouched 46-column models) | 5.0x | 880 | 66.5% (62.2 to 70.7) | 28% (245) | 81.2% (75.3 to 86.4) |
| start, 7 new members | 1.25x | 2,852 | 60.9% (58.4 to 63.2) | 30% (842) | 67.1% (63.4 to 70.6) |
| start, 7 new members | 2.0x | 2,030 | 62.7% (59.4 to 65.9) | 28% (572) | 70.8% (65.5 to 75.7) |
| start, 7 new members | 3.0x | 1,377 | 64.9% (61.2 to 68.6) | 29% (395) | 74.4% (67.8 to 80.4) |
| start, 7 new members | 5.0x | 880 | 67.7% (63.4 to 72.1) | 30% (260) | 81.9% (75.1 to 87.4) |
| control, 7 new members | 1.25x | 2,852 | 61.2% (58.8 to 63.5) | 29% (834) | 66.9% (63.0 to 70.6) |
| control, 7 new members | 2.0x | 2,030 | 62.7% (59.6 to 65.7) | 28% (570) | 70.4% (65.0 to 74.8) |
| control, 7 new members | 3.0x | 1,377 | 64.9% (61.4 to 68.4) | 29% (396) | 75.0% (68.5 to 80.9) |
| control, 7 new members | 5.0x | 880 | 67.2% (63.0 to 71.3) | 29% (258) | 82.9% (76.1 to 88.5) |
| fine-tuned, 7 new members | 1.25x | 2,852 | 61.0% (58.5 to 63.6) | 37% (1054) | 69.4% (65.3 to 73.1) |
| fine-tuned, 7 new members | 2.0x | 2,030 | 63.0% (59.6 to 66.3) | 37% (757) | 73.6% (68.4 to 77.6) |
| fine-tuned, 7 new members | 3.0x | 1,377 | 65.5% (61.4 to 69.8) | 40% (552) | 77.5% (71.6 to 82.8) |
| fine-tuned, 7 new members | 5.0x | 880 | 68.6% (63.3 to 73.9) | 43% (378) | 83.1% (76.2 to 88.3) |
| pairs track: control 3 | 1.25x | 2,852 | 60.0% (58.0 to 61.9) | 26% (752) | 67.0% (62.9 to 70.9) |
| pairs track: control 3 | 2.0x | 2,030 | 61.2% (58.6 to 63.8) | 25% (510) | 68.4% (63.0 to 73.3) |
| pairs track: control 3 | 3.0x | 1,377 | 63.5% (60.5 to 66.4) | 26% (356) | 72.8% (66.3 to 78.8) |
| pairs track: control 3 | 5.0x | 880 | 66.8% (62.7 to 70.8) | 27% (236) | 79.2% (72.2 to 85.4) |
| pairs track: fine-tuned 3 | 1.25x | 2,852 | 61.4% (58.9 to 63.7) | 35% (984) | 69.3% (65.6 to 72.7) |
| pairs track: fine-tuned 3 | 2.0x | 2,030 | 62.8% (59.6 to 65.8) | 34% (691) | 71.8% (66.8 to 76.0) |
| pairs track: fine-tuned 3 | 3.0x | 1,377 | 65.6% (61.5 to 69.4) | 36% (497) | 75.5% (69.2 to 80.8) |
| pairs track: fine-tuned 3 | 5.0x | 880 | 69.1% (64.3 to 73.7) | 39% (340) | 82.1% (75.8 to 87.3) |
| start, all 10 | 1.25x | 2,852 | 61.0% (58.7 to 63.3) | 29% (831) | 67.3% (63.5 to 70.8) |
| start, all 10 | 2.0x | 2,030 | 62.6% (59.6 to 65.5) | 28% (575) | 70.8% (65.7 to 75.2) |
| start, all 10 | 3.0x | 1,377 | 64.1% (60.6 to 67.4) | 29% (398) | 74.4% (68.2 to 79.9) |
| start, all 10 | 5.0x | 880 | 66.6% (62.3 to 70.8) | 30% (261) | 82.4% (76.1 to 87.5) |
| control, all 10 | 1.25x | 2,852 | 61.2% (58.8 to 63.5) | 29% (835) | 67.3% (63.6 to 70.7) |
| control, all 10 | 2.0x | 2,030 | 62.7% (59.6 to 65.7) | 28% (568) | 71.0% (65.9 to 75.5) |
| control, all 10 | 3.0x | 1,377 | 64.3% (60.8 to 67.7) | 29% (396) | 75.5% (69.5 to 81.0) |
| control, all 10 | 5.0x | 880 | 66.5% (62.1 to 70.7) | 29% (257) | 83.7% (77.7 to 88.7) |
| fine-tuned, all 10 | 1.25x | 2,852 | 61.4% (58.7 to 63.9) | 37% (1062) | 69.9% (66.0 to 73.3) |
| fine-tuned, all 10 | 2.0x | 2,030 | 63.0% (59.7 to 66.2) | 37% (758) | 73.5% (68.5 to 77.6) |
| fine-tuned, all 10 | 3.0x | 1,377 | 65.4% (61.4 to 69.6) | 40% (548) | 77.4% (71.4 to 82.6) |
| fine-tuned, all 10 | 5.0x | 880 | 68.8% (63.3 to 73.9) | 43% (376) | 82.7% (75.9 to 87.8) |
| control, ts0 (2) | 1.25x | 2,852 | 61.1% (59.3 to 62.9) | 29% (830) | 66.0% (62.0 to 69.5) |
| control, ts0 (2) | 2.0x | 2,030 | 62.7% (60.1 to 65.0) | 28% (574) | 69.2% (63.9 to 73.5) |
| control, ts0 (2) | 3.0x | 1,377 | 65.9% (62.8 to 68.9) | 29% (405) | 73.8% (67.6 to 79.2) |
| control, ts0 (2) | 5.0x | 880 | 69.3% (65.1 to 73.2) | 31% (275) | 79.6% (72.9 to 85.0) |
| fine-tuned, ts0 (2) | 1.25x | 2,852 | 61.9% (59.7 to 63.9) | 33% (939) | 68.5% (64.3 to 72.2) |
| fine-tuned, ts0 (2) | 2.0x | 2,030 | 63.1% (60.2 to 65.8) | 33% (668) | 72.2% (67.0 to 76.3) |
| fine-tuned, ts0 (2) | 3.0x | 1,377 | 66.4% (63.0 to 69.8) | 36% (489) | 76.9% (70.9 to 81.9) |
| fine-tuned, ts0 (2) | 5.0x | 880 | 70.3% (65.9 to 74.5) | 39% (339) | 81.4% (74.6 to 86.8) |
| control, ts20 (4) | 1.25x | 2,852 | 59.2% (56.8 to 61.5) | 27% (780) | 67.3% (63.3 to 71.1) |
| control, ts20 (4) | 2.0x | 2,030 | 60.3% (57.0 to 63.6) | 27% (542) | 70.3% (65.1 to 74.8) |
| control, ts20 (4) | 3.0x | 1,377 | 62.6% (58.6 to 66.5) | 27% (375) | 76.0% (70.0 to 81.3) |
| control, ts20 (4) | 5.0x | 880 | 64.7% (59.7 to 69.7) | 27% (240) | 83.8% (77.2 to 89.0) |
| fine-tuned, ts20 (4) | 1.25x | 2,852 | 60.3% (57.6 to 63.0) | 36% (1038) | 69.0% (64.7 to 72.8) |
| fine-tuned, ts20 (4) | 2.0x | 2,030 | 62.1% (58.4 to 65.7) | 37% (750) | 72.1% (67.0 to 76.5) |
| fine-tuned, ts20 (4) | 3.0x | 1,377 | 64.8% (60.3 to 69.3) | 40% (549) | 76.0% (69.7 to 81.3) |
| fine-tuned, ts20 (4) | 5.0x | 880 | 67.3% (61.8 to 72.8) | 43% (377) | 81.2% (74.0 to 86.9) |

Paired differences:

| A minus B | Real gap at least | Pairs | Correct, points | Confident share, points | Correct among confident, points | Correct confident calls per 100 pairs |
|---|---|---|---|---|---|---|
| fine-tuned, 7 new members minus control, 7 new members | 1.25x | 2,852 | -0.1 (-1.3 to +1.0) | +7.7 (+5.7 to +10.0) | +2.5 (+0.4 to +4.8) | +6.1 (+4.3 to +8.2) |
| fine-tuned, 7 new members minus control, 7 new members | 2.0x | 2,030 | +0.3 (-1.1 to +1.7) | +9.2 (+6.6 to +11.9) | +3.2 (+1.0 to +5.6) | +7.7 (+5.3 to +10.2) |
| fine-tuned, 7 new members minus control, 7 new members | 3.0x | 1,377 | +0.6 (-1.3 to +2.3) | +11.3 (+8.4 to +14.5) | +2.5 (-0.4 to +5.7) | +9.5 (+6.6 to +12.6) |
| fine-tuned, 7 new members minus control, 7 new members | 5.0x | 880 | +1.5 (-1.0 to +3.6) | +13.6 (+9.8 to +17.4) | +0.1 (-3.0 to +3.0) | +11.4 (+7.6 to +15.0) |
| fine-tuned, 7 new members minus start, 7 new members | 1.25x | 2,852 | +0.2 (-1.1 to +1.4) | +7.4 (+5.2 to +10.1) | +2.3 (+0.1 to +4.7) | +5.9 (+4.0 to +8.2) |
| fine-tuned, 7 new members minus start, 7 new members | 2.0x | 2,030 | +0.3 (-1.2 to +1.7) | +9.1 (+6.5 to +12.1) | +2.8 (+0.1 to +5.5) | +7.5 (+5.1 to +10.3) |
| fine-tuned, 7 new members minus start, 7 new members | 3.0x | 1,377 | +0.6 (-1.4 to +2.5) | +11.4 (+8.2 to +15.0) | +3.1 (-0.2 to +6.5) | +9.7 (+6.8 to +13.1) |
| fine-tuned, 7 new members minus start, 7 new members | 5.0x | 880 | +0.9 (-1.6 to +3.1) | +13.4 (+9.4 to +17.5) | +1.1 (-2.0 to +4.1) | +11.5 (+7.8 to +15.4) |
| control, 7 new members minus start, 7 new members | 1.25x | 2,852 | +0.3 (-0.3 to +0.9) | -0.3 (-1.2 to +0.7) | -0.2 (-1.2 to +0.9) | -0.2 (-0.9 to +0.5) |
| control, 7 new members minus start, 7 new members | 2.0x | 2,030 | +0.0 (-0.7 to +0.8) | -0.1 (-1.0 to +0.9) | -0.5 (-1.8 to +0.8) | -0.2 (-1.0 to +0.6) |
| control, 7 new members minus start, 7 new members | 3.0x | 1,377 | +0.0 (-0.8 to +0.8) | +0.1 (-0.9 to +1.1) | +0.6 (-1.1 to +2.4) | +0.2 (-0.6 to +1.0) |
| control, 7 new members minus start, 7 new members | 5.0x | 880 | -0.6 (-1.4 to +0.2) | -0.2 (-1.2 to +0.9) | +1.0 (-0.6 to +2.8) | +0.1 (-0.7 to +1.0) |
| pairs track: fine-tuned 3 minus pairs track: control 3 | 1.25x | 2,852 | +1.4 (+0.1 to +2.4) | +8.1 (+6.4 to +9.9) | +2.3 (+0.1 to +4.7) | +6.2 (+4.7 to +7.8) |
| pairs track: fine-tuned 3 minus pairs track: control 3 | 2.0x | 2,030 | +1.6 (+0.2 to +2.8) | +8.9 (+6.9 to +11.0) | +3.3 (+0.8 to +6.0) | +7.2 (+5.4 to +9.1) |
| pairs track: fine-tuned 3 minus pairs track: control 3 | 3.0x | 1,377 | +2.1 (+0.3 to +3.8) | +10.2 (+7.6 to +12.7) | +2.7 (-0.4 to +5.8) | +8.4 (+5.9 to +10.9) |
| pairs track: fine-tuned 3 minus pairs track: control 3 | 5.0x | 880 | +2.3 (+0.4 to +4.0) | +11.8 (+8.4 to +14.9) | +2.8 (-0.3 to +5.9) | +10.5 (+7.1 to +13.5) |
| fine-tuned, all 10 minus control, all 10 | 1.25x | 2,852 | +0.2 (-0.9 to +1.2) | +8.0 (+6.0 to +10.1) | +2.6 (+0.5 to +4.8) | +6.3 (+4.5 to +8.4) |
| fine-tuned, all 10 minus control, all 10 | 2.0x | 2,030 | +0.3 (-1.0 to +1.7) | +9.4 (+6.8 to +12.1) | +2.5 (-0.1 to +5.2) | +7.6 (+5.3 to +10.2) |
| fine-tuned, all 10 minus control, all 10 | 3.0x | 1,377 | +1.2 (-0.5 to +2.7) | +11.0 (+7.9 to +14.2) | +1.9 (-1.3 to +4.9) | +9.1 (+6.0 to +12.2) |
| fine-tuned, all 10 minus control, all 10 | 5.0x | 880 | +2.3 (+0.3 to +4.1) | +13.5 (+9.4 to +17.3) | -0.9 (-4.6 to +2.0) | +10.9 (+6.9 to +14.7) |
| fine-tuned, all 10 minus baseline (3 untouched 46-column models) | 1.25x | 2,852 | +1.4 (-0.2 to +2.9) | +10.0 (+7.6 to +12.7) | +2.5 (+0.2 to +5.0) | +7.6 (+5.6 to +10.0) |
| fine-tuned, all 10 minus baseline (3 untouched 46-column models) | 2.0x | 2,030 | +1.6 (-0.2 to +3.3) | +11.6 (+9.0 to +14.5) | +3.3 (+0.5 to +6.2) | +9.4 (+6.9 to +12.1) |
| fine-tuned, all 10 minus baseline (3 untouched 46-column models) | 3.0x | 1,377 | +2.1 (-0.4 to +4.2) | +12.9 (+9.7 to +16.3) | +3.0 (-0.0 to +5.7) | +10.7 (+7.7 to +14.1) |
| fine-tuned, all 10 minus baseline (3 untouched 46-column models) | 5.0x | 880 | +2.3 (-0.3 to +4.4) | +14.9 (+10.7 to +18.8) | +1.5 (-1.8 to +4.1) | +12.7 (+8.5 to +16.6) |
| fine-tuned, ts0 (2) minus control, ts0 (2) | 1.25x | 2,852 | +0.8 (-0.4 to +1.9) | +3.8 (+1.8 to +5.9) | +2.5 (+0.3 to +4.5) | +3.3 (+1.7 to +5.1) |
| fine-tuned, ts0 (2) minus control, ts0 (2) | 2.0x | 2,030 | +0.4 (-1.0 to +1.7) | +4.6 (+2.4 to +7.0) | +3.0 (+0.4 to +5.5) | +4.2 (+2.2 to +6.4) |
| fine-tuned, ts0 (2) minus control, ts0 (2) | 3.0x | 1,377 | +0.6 (-1.3 to +2.4) | +6.1 (+3.2 to +9.0) | +3.1 (-0.1 to +6.3) | +5.6 (+2.8 to +8.5) |
| fine-tuned, ts0 (2) minus control, ts0 (2) | 5.0x | 880 | +1.0 (-1.1 to +2.9) | +7.3 (+3.9 to +10.7) | +1.8 (-2.1 to +5.1) | +6.5 (+3.2 to +10.1) |
| fine-tuned, ts20 (4) minus control, ts20 (4) | 1.25x | 2,852 | +1.1 (-0.4 to +2.4) | +9.0 (+6.8 to +11.4) | +1.7 (-0.6 to +4.2) | +6.7 (+4.8 to +8.9) |
| fine-tuned, ts20 (4) minus control, ts20 (4) | 2.0x | 2,030 | +1.8 (+0.1 to +3.3) | +10.2 (+7.5 to +13.1) | +1.8 (-0.9 to +4.7) | +7.9 (+5.3 to +10.6) |
| fine-tuned, ts20 (4) minus control, ts20 (4) | 3.0x | 1,377 | +2.2 (+0.3 to +4.2) | +12.6 (+9.3 to +16.2) | -0.0 (-3.3 to +3.3) | +9.6 (+6.2 to +13.2) |
| fine-tuned, ts20 (4) minus control, ts20 (4) | 5.0x | 880 | +2.6 (+0.5 to +4.4) | +15.6 (+11.4 to +19.5) | -2.6 (-6.8 to +0.8) | +11.9 (+7.6 to +16.2) |

"Correct confident calls per 100 pairs" counts calls that are both stated at 60%+ and right; it rises when coverage rises without accuracy falling.

Surest 40% of the 2,030 pairs 2x apart, ranked by the size of the ensemble's score gap (no confidence scale involved; `cov40.md`):

| Ensemble | Surest 40% of pairs 2x apart | Correct there |
|---|---|---|
| fine-tuned 7 | 812 of 2030 | 72.5% (67.5 to 76.7) |
| control 7 | 812 of 2030 | 68.7% (64.1 to 72.9) |
| pairs track fine-tuned 3 | 812 of 2030 | 71.1% (66.4 to 75.1) |
| pairs track control 3 | 812 of 2030 | 66.9% (62.3 to 71.0) |
| baseline 3 | 812 of 2030 | 67.6% (63.0 to 71.7) |
| start 7 | 812 of 2030 | 69.2% (64.7 to 73.4) |
| fine-tuned 10 | 812 of 2030 | 72.5% (67.6 to 76.7) |
| control 10 | 812 of 2030 | 68.7% (64.0 to 73.0) |
| start 10 | 812 of 2030 | 69.6% (64.9 to 74.0) |
| fine-tuned 7 minus control 7 | | +3.8 (+1.5 to +6.1) |
| fine-tuned 10 minus control 10 | | +3.8 (+1.6 to +6.2) |
| pairs track fine-tuned 3 minus pairs track control 3 | | +4.2 (+2.1 to +6.4) |
| fine-tuned 10 minus baseline 3 | | +4.9 (+2.5 to +7.1) |
| fine-tuned 10 minus start 10 | | +3.0 (+0.7 to +5.4) |
| control 10 minus baseline 3 | | +1.1 (-0.6 to +2.6) |

By half of the benchmark (test = 2,030 pairs posted on or after the cutoff; calibration = 1,500 earlier pairs from accounts never trained on):

| Half | A minus B | Real gap at least | Pairs | Correct A | Correct B | Correct, points | Confident share, points | Correct among confident, points |
|---|---|---|---|---|---|---|---|---|
| test | fine-tuned, 7 new members minus control, 7 new members | 1.25x | 1,660 | 60.5% | 61.1% | -0.6 (-2.3 to +1.1) | +7.7 (+4.4 to +11.3) | +1.8 (-1.2 to +4.7) |
| test | fine-tuned, 7 new members minus control, 7 new members | 2.0x | 1,117 | 62.3% | 62.6% | -0.3 (-2.3 to +1.9) | +9.4 (+5.7 to +13.6) | +2.8 (-1.1 to +6.3) |
| test | fine-tuned, all 10 minus control, all 10 | 1.25x | 1,660 | 60.8% | 61.0% | -0.2 (-1.8 to +1.4) | +7.6 (+4.4 to +11.0) | +2.0 (-1.1 to +5.0) |
| test | fine-tuned, all 10 minus control, all 10 | 2.0x | 1,117 | 62.0% | 62.3% | -0.4 (-2.5 to +1.8) | +9.5 (+5.6 to +13.7) | +2.0 (-1.7 to +5.6) |
| test | pairs track: fine-tuned 3 minus pairs track: control 3 | 1.25x | 1,660 | 61.4% | 60.4% | +1.0 (-0.5 to +2.5) | +7.4 (+4.7 to +10.1) | +1.8 (-1.0 to +4.2) |
| test | pairs track: fine-tuned 3 minus pairs track: control 3 | 2.0x | 1,117 | 62.6% | 61.2% | +1.3 (-0.6 to +3.2) | +8.9 (+5.5 to +12.2) | +2.0 (-1.5 to +5.2) |
| calibration | fine-tuned, 7 new members minus control, 7 new members | 1.25x | 1,192 | 61.7% | 61.2% | +0.5 (-1.2 to +2.0) | +7.7 (+4.9 to +10.1) | +3.7 (+0.6 to +7.1) |
| calibration | fine-tuned, 7 new members minus control, 7 new members | 2.0x | 913 | 63.9% | 62.8% | +1.1 (-0.9 to +2.6) | +9.0 (+6.0 to +12.0) | +3.5 (+0.4 to +6.8) |
| calibration | fine-tuned, all 10 minus control, all 10 | 1.25x | 1,192 | 62.2% | 61.4% | +0.8 (-0.8 to +2.0) | +8.5 (+5.8 to +11.0) | +3.1 (-0.2 to +6.5) |
| calibration | fine-tuned, all 10 minus control, all 10 | 2.0x | 913 | 64.3% | 63.1% | +1.2 (-0.5 to +2.7) | +9.2 (+6.3 to +12.0) | +2.9 (-0.5 to +6.7) |
| calibration | pairs track: fine-tuned 3 minus pairs track: control 3 | 1.25x | 1,192 | 61.3% | 59.5% | +1.8 (+0.2 to +3.3) | +9.1 (+6.7 to +11.3) | +2.6 (-1.2 to +7.2) |
| calibration | pairs track: fine-tuned 3 minus pairs track: control 3 | 2.0x | 913 | 63.1% | 61.2% | +1.9 (+0.1 to +3.6) | +9.0 (+6.4 to +11.4) | +5.0 (+0.8 to +9.4) |

Reading:

1. On overall accuracy the result is a null with a tight interval: -0.1 (-1.3 to +1.0) for the seven, +0.2 (-0.9 to +1.2) for all ten. The pairs track's three are +1.4 (+0.1 to +2.4) in my recomputation, as reported there;
   they are the top of a distribution whose centre is about +0.5 per member and about zero once seven or more members are averaged.
2. Averaging explains the rest. The control seven (61.2%) and the ten untouched models (61.0%) already sit where the pairs track's fine-tuned three did (61.4%). Fine-tuned single models gain a little (+0.5), fine-tuned ensembles gain nothing over control ensembles:
   the pair loss and ensembling remove some of the same errors.
3. The confident-call gain is large next to its interval at every gap size, in every member, in both halves of the benchmark, and in both feature families (ts0: share +4.6 at 2x; ts20: +10.2). In the test half alone the accuracy among confident calls is +2.8 (-1.1 to +6.3) at 2x: same direction, not separately established.
4. Fine-tuning without the pair loss changes nothing (control seven minus start seven: +0.3, -0.3 to +0.9; confident share -0.3).
5. Ordinary metrics: test pairwise accuracy on likes is unchanged (0.588 to 0.600 per member, within 0.003 of the control); the like NLL is 0.015 to 0.025 higher than the control in every member, as the pairs track saw.

## Step 2. The 850 held-out launch pairs

`scripts/rep_score_launch.py` scores the launch track's 582 test-period launch posts with every checkpoint through the draft path (56 posts of history, as `scripts/launch/score_launch.py`); `scripts/rep_launch.py` rebuilds the 850 pairs
exactly as `model_check.py` does (my rebuild of the stored baseline gives 63.6%, 217 confident, 73.3%: the launch report's row). Confidence uses each ensemble's scale from the benchmark, not refitted on launch pairs.

| Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there |
|---|---|---|---|---|---|
| baseline as stored by the launch track, its scale (scale 1.52) | 1.25x | 850 | 63.6% (59.5 to 67.6) | 26% (217) | 73.3% (64.8 to 81.4) |
| baseline as stored by the launch track, its scale (scale 1.52) | 2.0x | 568 | 66.0% (60.8 to 71.4) | 28% (157) | 78.3% (68.8 to 86.9) |
| baseline 3 rescored (scale 2.95) | 1.25x | 850 | 63.9% (59.8 to 67.9) | 52% (441) | 71.9% (65.6 to 77.7) |
| baseline 3 rescored (scale 2.95) | 2.0x | 568 | 66.5% (61.3 to 71.9) | 55% (310) | 75.8% (69.0 to 82.3) |
| fine-tuned 7 (scale 1.72) | 1.25x | 850 | 63.4% (58.7 to 67.6) | 38% (327) | 70.3% (63.3 to 77.0) |
| fine-tuned 7 (scale 1.72) | 2.0x | 568 | 65.3% (60.0 to 70.4) | 41% (234) | 73.5% (65.8 to 81.1) |
| control 7 (scale 3.40) | 1.25x | 850 | 63.2% (58.7 to 67.4) | 59% (503) | 71.2% (65.7 to 76.4) |
| control 7 (scale 3.40) | 2.0x | 568 | 65.5% (60.6 to 70.7) | 62% (351) | 73.2% (67.6 to 79.4) |
| start 7 (scale 3.48) | 1.25x | 850 | 62.6% (57.9 to 67.3) | 58% (496) | 70.2% (64.4 to 75.5) |
| start 7 (scale 3.48) | 2.0x | 568 | 64.6% (59.2 to 70.0) | 61% (344) | 72.1% (66.2 to 78.4) |
| pairs track fine-tuned 3 (scale 1.90) | 1.25x | 850 | 64.5% (60.2 to 68.7) | 38% (319) | 72.4% (65.0 to 78.9) |
| pairs track fine-tuned 3 (scale 1.90) | 2.0x | 568 | 67.3% (62.3 to 72.1) | 40% (230) | 74.8% (66.4 to 82.6) |
| pairs track control 3 (scale 2.83) | 1.25x | 850 | 64.6% (60.7 to 68.4) | 54% (460) | 70.2% (64.2 to 76.1) |
| pairs track control 3 (scale 2.83) | 2.0x | 568 | 67.1% (62.4 to 71.9) | 57% (326) | 73.6% (67.1 to 80.5) |
| fine-tuned 10 (scale 1.81) | 1.25x | 850 | 63.2% (58.4 to 67.5) | 39% (335) | 71.6% (64.7 to 78.4) |
| fine-tuned 10 (scale 1.81) | 2.0x | 568 | 65.1% (59.7 to 70.5) | 42% (241) | 75.1% (67.7 to 82.3) |
| control 10 (scale 3.30) | 1.25x | 850 | 63.5% (59.2 to 67.7) | 58% (492) | 70.9% (65.5 to 76.2) |
| control 10 (scale 3.30) | 2.0x | 568 | 66.2% (61.2 to 71.2) | 60% (340) | 73.5% (67.6 to 80.0) |
| start 10 (scale 3.36) | 1.25x | 850 | 62.5% (58.1 to 66.8) | 57% (481) | 71.1% (65.4 to 76.5) |
| start 10 (scale 3.36) | 2.0x | 568 | 64.3% (59.1 to 69.5) | 59% (333) | 73.6% (68.0 to 79.7) |

Paired (benchmark scale):

| A minus B | Real gap at least | Pairs | Correct, points | Confident share, points | Correct among confident, points | Correct confident calls per 100 pairs |
|---|---|---|---|---|---|---|
| fine-tuned 7 minus control 7 | 1.25x | 850 | +0.2 (-1.7 to +2.2) | -20.7 (-24.2 to -17.1) | -0.8 (-4.6 to +2.6) | -15.1 (-17.9 to -12.0) |
| fine-tuned 7 minus control 7 | 2.0x | 568 | -0.2 (-2.8 to +2.5) | -20.6 (-24.7 to -16.2) | +0.3 (-3.8 to +4.4) | -15.0 (-18.6 to -11.1) |
| pairs track fine-tuned 3 minus pairs track control 3 | 1.25x | 850 | -0.1 (-2.0 to +1.6) | -16.6 (-20.0 to -13.0) | +2.2 (-1.7 to +5.5) | -10.8 (-13.7 to -7.9) |
| pairs track fine-tuned 3 minus pairs track control 3 | 2.0x | 568 | +0.2 (-1.7 to +1.9) | -16.9 (-20.8 to -12.3) | +1.2 (-3.0 to +4.8) | -12.0 (-15.2 to -8.5) |
| fine-tuned 10 minus control 10 | 1.25x | 850 | -0.4 (-2.1 to +1.4) | -18.5 (-21.8 to -14.6) | +0.7 (-3.0 to +4.2) | -12.8 (-15.7 to -9.6) |
| fine-tuned 10 minus control 10 | 2.0x | 568 | -1.1 (-3.5 to +1.3) | -17.4 (-21.4 to -13.1) | +1.6 (-2.0 to +5.2) | -12.1 (-15.6 to -8.6) |
| fine-tuned 10 minus start 10 | 1.25x | 850 | +0.7 (-1.3 to +2.6) | -17.2 (-20.5 to -13.5) | +0.5 (-3.5 to +4.1) | -12.0 (-15.0 to -9.0) |
| fine-tuned 10 minus start 10 | 2.0x | 568 | +0.9 (-1.5 to +3.2) | -16.2 (-20.3 to -11.9) | +1.5 (-2.6 to +5.6) | -11.3 (-14.9 to -7.7) |
| fine-tuned 10 minus baseline 3 rescored | 1.25x | 850 | -0.7 (-3.2 to +1.7) | -12.5 (-16.2 to -8.6) | -0.2 (-4.1 to +3.9) | -9.1 (-11.9 to -6.2) |
| fine-tuned 10 minus baseline 3 rescored | 2.0x | 568 | -1.4 (-4.7 to +1.5) | -12.1 (-16.6 to -7.5) | -0.7 (-5.3 to +4.2) | -9.5 (-13.2 to -5.9) |

Scale cross-fitted on the launch pairs themselves (pairs 2x apart), and the surest 40% of 2x pairs with no scale:

| Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there |
|---|---|---|---|---|---|
| baseline as stored by the launch track, its scale (scale 2.13, cross-fitted on launch pairs) | 2.0x | 568 | 66.0% (60.8 to 71.4) | 41% (231) | 77.1% (70.2 to 83.7) |
| baseline 3 rescored (scale 2.13, cross-fitted on launch pairs) | 2.0x | 568 | 66.5% (61.3 to 71.9) | 40% (226) | 78.3% (71.5 to 85.0) |
| fine-tuned 7 (scale 1.47, cross-fitted on launch pairs) | 2.0x | 568 | 65.3% (60.0 to 70.4) | 33% (189) | 72.5% (64.2 to 80.9) |
| control 7 (scale 2.01, cross-fitted on launch pairs) | 2.0x | 568 | 65.5% (60.6 to 70.7) | 40% (227) | 75.3% (67.9 to 82.7) |
| start 7 (scale 1.97, cross-fitted on launch pairs) | 2.0x | 568 | 64.6% (59.2 to 70.0) | 37% (208) | 76.0% (68.2 to 83.5) |
| pairs track fine-tuned 3 (scale 1.72, cross-fitted on launch pairs) | 2.0x | 568 | 67.3% (62.3 to 72.1) | 35% (201) | 76.1% (68.1 to 84.0) |
| pairs track control 3 (scale 2.04, cross-fitted on launch pairs) | 2.0x | 568 | 67.1% (62.4 to 71.9) | 42% (240) | 76.7% (69.5 to 83.7) |
| fine-tuned 10 (scale 1.57, cross-fitted on launch pairs) | 2.0x | 568 | 65.1% (59.7 to 70.5) | 35% (198) | 74.2% (65.9 to 82.3) |
| control 10 (scale 2.05, cross-fitted on launch pairs) | 2.0x | 568 | 66.2% (61.2 to 71.2) | 40% (230) | 76.1% (68.8 to 83.2) |
| start 10 (scale 2.04, cross-fitted on launch pairs) | 2.0x | 568 | 64.3% (59.1 to 69.5) | 38% (215) | 76.7% (69.3 to 84.2) |

| Model | Surest 40% of pairs 2x apart | Correct there |
|---|---|---|
| baseline as stored by the launch track, its scale | 227 of 568 | 77.5% (70.1 to 84.7) |
| baseline 3 rescored | 227 of 568 | 77.5% (70.3 to 84.7) |
| fine-tuned 7 | 227 of 568 | 73.6% (66.0 to 81.2) |
| control 7 | 227 of 568 | 76.2% (68.9 to 83.6) |
| start 7 | 227 of 568 | 77.1% (69.7 to 84.5) |
| pairs track fine-tuned 3 | 227 of 568 | 75.8% (67.5 to 83.5) |
| pairs track control 3 | 227 of 568 | 78.4% (70.6 to 85.7) |
| fine-tuned 10 | 227 of 568 | 75.8% (68.2 to 83.3) |
| control 10 | 227 of 568 | 77.1% (69.8 to 84.4) |
| start 10 | 227 of 568 | 77.1% (69.8 to 84.6) |
| fine-tuned 7 minus control 7 | | -2.6 (-6.9 to +1.4) |
| pairs track fine-tuned 3 minus pairs track control 3 | | -2.6 (-6.5 to +1.0) |
| fine-tuned 10 minus control 10 | | -1.3 (-5.4 to +2.6) |
| fine-tuned 10 minus start 10 | | -1.3 (-5.0 to +2.1) |
| fine-tuned 10 minus baseline 3 rescored | | -1.8 (-5.8 to +2.3) |

Reading: on launch pairs the fine-tune changes nothing that can be measured in accuracy (fine-tuned ten minus control ten: -0.4, -2.1 to +1.4 on 850 pairs), and its confident calls are not better (surest 40% of 2x pairs: -1.3, -5.4 to +2.6; the seven: -2.6, -6.9 to +1.4).
The large drop in confident share under the benchmark scale is a calibration effect, not a loss of skill: fine-tuning widens the score gaps on same-link pairs, the benchmark scale shrinks to match (3.30 to 1.81), and launch pairs, whose gaps were not widened as much, fall under 60%.
Two things follow. The benchmark scale is too timid for launch pairs in every model (the untouched baseline with its benchmark scale of 2.95 calls 55% of 2x launch pairs confident and is right on 75.8%). And a fine-tuned ensemble would need its confidence scale fitted per kind of pair, or it will go quiet on launch posts.

## Step 3. Trying to extend it

The accuracy gain did not replicate, so by the brief this step was optional; I ran it because the confident-call gain did, and that is the goal's own measure.

**Development set** (`scripts/rep_mine_dev.py`, `dev_pairs.parquet`): same account, same link or quoted post, both before the cutoff, at most 14 days apart, wordings differ, both at least 6 words, likes 1.25x apart, no benchmark post, no account in the pair table, at most 20 pairs per account:
2,805 pairs from 366 accounts (1,956 at 2x; 1,465 within 48 hours). Built through the serving path (`build_pair_table.py`, 48 posts of history) and scored with `scripts/rep_dev_scores.py`. 165 of the pairs are from accounts in the training table; 2,104 are from accounts that also have benchmark pairs (other posts).
Untouched models are 56% correct on it, against 60 to 61% on the benchmark (it includes pairs up to 14 days apart).

Selection rule, written before any variant was trained: correct confident calls per 100 pairs among development pairs 2x apart; a variant replaces the recipe only if its paired interval against the recipe excludes zero.

The development set first, as a second replication on pairs nobody had scored (`dev_step1.txt`):

| Ensemble | Pairs | Correct | At 2x: pairs | Correct | Confident share | Correct there |
|---|---|---|---|---|---|---|
| Baseline three | 2,805 | 56.4% | 1,956 | 58.5% | 18% | 71.7% |
| Start, ten | 2,805 | 56.1% | 1,956 | 58.1% | 19% | 72.5% |
| Control, ten | 2,805 | 55.4% | 1,956 | 57.1% | 20% | 72.2% |
| Fine-tuned, ten | 2,805 | 56.3% | 1,956 | 58.4% | 25% | 73.4% |
| Control, seven | 2,805 | 56.3% | 1,956 | 57.9% | 20% | 71.0% |
| Fine-tuned, seven | 2,805 | 56.4% | 1,956 | 58.9% | 25% | 72.6% |
| Fine-tuned, pairs track's three | 2,805 | 56.3% | 1,956 | 58.7% | 25% | 72.7% |
| Ten fine-tuned plus ten untouched | 2,805 | 56.2% | 1,956 | 58.2% | 25% | 72.1% |

Fine-tuned ten minus control ten: accuracy +0.9 (-0.2 to +1.9), at 2x +1.4 (+0.1 to +2.7); confident share at 2x +4.9 (+3.2 to +6.7); correct confident calls per 100 pairs at 2x +3.8 (+2.5 to +5.3). Against the untouched ten: accuracy +0.2 (-0.9 to +1.3), share +5.3 (+3.7 to +7.0).
Ensemble size and mixing: ten fine-tuned minus the three: +0.2 per 100 (-0.8 to +1.2); minus the seven: -0.1 (-0.6 to +0.4); adding the ten untouched members: -0.1 (-0.7 to +0.5). No difference.

Variants, each fine-tuned from `ts46_s1`, `ts20_s1`, `ts0_s0` and compared as three-member ensembles with the recipe's three (`dev_step3.txt`; 2,805 pairs, 1,956 at 2x):

| Variant | Correct, all pairs | Correct at 2x | Confident share at 2x | Correct there | Minus recipe: correct, all pairs | Minus recipe: correct confident calls per 100 pairs at 2x |
|---|---|---|---|---|---|---|
| Control (weight 0) | 55.9% | 58.0% | 18% | 72.5% | +0.5 (-0.6 to +1.7) | -3.9 (-5.3 to -2.5) |
| Recipe (weight 10, 600 steps, 2e-4, gap weights) | 55.4% | 57.5% | 23% | 71.4% | - | - |
| Learning rate 1e-4 | 56.1% | 58.5% | 23% | 71.3% | +0.7 (-0.0 to +1.5) | -0.4 (-1.2 to +0.5) |
| Flat weights | 55.4% | 57.6% | 22% | 71.9% | +0.0 (-0.6 to +0.6) | -0.9 (-1.7 to -0.0) |
| Only pairs 2x apart | 55.5% | 57.8% | 24% | 70.9% | +0.1 (-0.4 to +0.6) | +0.1 (-0.5 to +0.7) |
| 1,200 steps (script's pick, step 900) | 55.8% | 58.1% | 23% | 72.8% | +0.5 (-0.7 to +1.7) | +0.1 (-1.0 to +1.2) |
| 1,200 steps, last step | 55.7% | 58.0% | 23% | 73.1% | +0.3 (-1.0 to +1.7) | +0.1 (-1.1 to +1.2) |

No variant passes the rule. Flat weights are slightly worse; the others are the recipe within noise. The gain appears within the first 150 to 300 steps and does not grow with more.
**Chosen configuration: the recipe unchanged, all ten fine-tuned members.** Its benchmark score is the "fine-tuned, all 10" row of step 1 (each member scored once): 61.4% on 2,852 pairs; at 2x 63.0%, confident on 37% (758) at 73.5%.

## Step 4. Speak or stay silent

`scripts/rep_rules.py`, `scripts/rep_step4.py`. Seven rules, each a probability that the ensemble's pick is right, cross-fitted over accounts, speaking at 60% or more: the score gap alone; plus the share of members on the ensemble's side; the gap gated on 80% or 100% agreement;
the gap divided by its standard error across members; both; and the gap plus the weakest member's margin. Decision rule fixed beforehand: replace probability alone only if correct spoken calls per 100 development pairs at 2x rise with an interval excluding zero and accuracy among spoken calls does not fall.

Development set, ten fine-tuned members (1,956 pairs 2x apart; all members agree on 45% of pairs):

| Rule | Speaks on, 2x pairs (pairs) | Correct there | Correct at 40% coverage of 2x pairs | Speaks on, all pairs | Correct there | Log loss |
|---|---|---|---|---|---|---|
| prob | 25% (483 of 1956) | 73.1% (68.7 to 77.3) | 68.7% (64.7 to 72.6) of 782 | 26% (717 of 2805) | 68.9% (65.0 to 72.8) | 0.6706 |
| agree | 24% (469 of 1956) | 73.6% (69.2 to 78.0) | 68.4% (64.5 to 72.3) of 782 | 25% (694 of 2805) | 69.5% (65.4 to 73.5) | 0.6710 |
| agree_gate0.8 | 25% (492 of 1956) | 73.2% (68.7 to 77.4) | 68.7% (64.7 to 72.7) of 782 | 26% (727 of 2805) | 69.1% (65.2 to 72.9) | 0.6704 |
| agree_gate1.0 | 24% (470 of 1956) | 72.6% (68.2 to 76.9) | 67.5% (63.6 to 71.5) of 782 | 24% (685 of 2805) | 68.5% (64.4 to 72.5) | 0.6717 |
| tstat | 31% (612 of 1956) | 69.9% (65.7 to 74.1) | 67.4% (63.6 to 71.1) of 782 | 31% (866 of 2805) | 65.9% (61.9 to 70.0) | 0.6764 |
| both | 25% (485 of 1956) | 73.2% (68.9 to 77.4) | 68.4% (64.5 to 72.4) of 782 | 25% (715 of 2805) | 69.1% (65.1 to 73.1) | 0.6708 |
| floor | 24% (473 of 1956) | 73.4% (69.0 to 77.6) | 69.1% (65.2 to 73.0) of 782 | 25% (701 of 2805) | 69.0% (65.0 to 73.0) | 0.6710 |
agree minus prob, 2x pairs: share -0.7 (-1.2 to -0.3), correct among spoken +0.5 (-0.4 to +1.4), correct spoken calls per 100 pairs -0.4 (-0.8 to -0.1)
agree_gate0.8 minus prob, 2x pairs: share +0.5 (+0.1 to +0.8), correct among spoken +0.1 (-0.6 to +0.7), correct spoken calls per 100 pairs +0.4 (+0.1 to +0.6)
agree_gate1.0 minus prob, 2x pairs: share -0.7 (-1.4 to +0.1), correct among spoken -0.5 (-1.9 to +0.9), correct spoken calls per 100 pairs -0.6 (-1.1 to -0.1)
tstat minus prob, 2x pairs: share +6.6 (+4.6 to +8.6), correct among spoken -3.2 (-6.3 to +0.0), correct spoken calls per 100 pairs +3.8 (+2.3 to +5.3)
both minus prob, 2x pairs: share +0.1 (-0.3 to +0.5), correct among spoken +0.1 (-0.7 to +0.9), correct spoken calls per 100 pairs +0.1 (-0.2 to +0.4)
floor minus prob, 2x pairs: share -0.5 (-0.9 to -0.2), correct among spoken +0.3 (-0.3 to +0.9), correct spoken calls per 100 pairs -0.3 (-0.6 to -0.1)

Only "gate at 80% agreement" passes the rule, by +0.4 calls per 100 pairs. The standard-error rule speaks more often (31%) but is less often right (69.9%) and worse calibrated (log loss 0.676 against 0.671): it trades accuracy for coverage, which a lower threshold on the probability does as well.

The chosen rule on the benchmark, once (2,030 pairs 2x apart):

| Ensemble | Rule | Speaks on, 2x pairs | Correct there | Correct at 40% coverage of 2x pairs (812) |
|---|---|---|---|---|
| Fine-tuned ten | probability alone | 37% (760) | 73.4% (68.4 to 77.5) | 72.4% (67.4 to 76.6) |
| Fine-tuned ten | probability, at least 80% of members agree | 37% (746) | 73.7% (68.7 to 77.9) | 72.3% (67.3 to 76.4) |
| Control ten | probability alone | 28% (563) | 70.5% (65.6 to 75.0) | 68.6% (64.0 to 72.7) |
| Untouched ten | probability alone | 28% (574) | 70.7% (65.7 to 75.2) | 69.7% (65.0 to 74.1) |
| Baseline three | probability alone | 26% (521) | 70.2% (65.2 to 74.6) | 67.1% (62.6 to 71.2) |

Gate minus probability alone, fine-tuned ten: share -0.7 (-1.1 to -0.4), correct among spoken +0.3 (-0.2 to +0.8), correct spoken calls per 100 pairs -0.4 (-0.7 to -0.2). The small development-set advantage did not carry over.
Agreement is already contained in the size of the mean gap: when ten members disagree the mean is small and the probability is under 60% anyway. (These rows use a maximum-likelihood fit, the step 1 rows a grid; counts differ by two or three pairs.)

Coverage of 40% is available only by speaking at a lower stated probability, and then the fine-tuned ten are right 72.4% of the time on the benchmark and 68.7% on the development set. The goal's 75 to 80% at 40% is not reached by any rule tried.

## What worked, what did not

Worked: the control design (weight 0 next to every fine-tune), which separates the pair loss from extra steps and from averaging; the confident-call gain of the pair-loss fine-tune, now seen in ten of ten members, on two halves of the benchmark and on a fresh development set;
the development set as a selection tool (its paired intervals on "correct confident calls per 100 pairs" are about plus or minus 1.2).

Did not work: the overall accuracy gain (not there once seven clean members are pooled); any transfer to launch pairs; every extension of the recipe; ensemble agreement as a second signal.

## Honest limits

- One fine-tune seed per member. The seven members are not independent of each other: they share the training table and, in pairs, a feature set.
- The confidence scale is cross-fitted on the benchmark's own pairs, as in every earlier table. The "surest 40%" rows avoid it and agree.
- The development set is not the benchmark's population: pre-cutoff, up to 14 days apart, 75% of its pairs from accounts that also have benchmark pairs (different posts), and scored with the training-path score (predicted mean, 48 posts of history), not the product's median with 56. It selects between fine-tune variants; it is not a second benchmark.
- Step 3 variants were compared as three-member ensembles; a difference under about 1 call per 100 pairs would be missed. Weight, loss type (margin) and pair-table content were not varied.
- The chosen configuration's benchmark numbers come from step 1's runs. Nothing was selected on the benchmark, but the reader sees ten members and several sub-ensembles; the pre-registered primary comparison is the seven against their controls.
- Launch pairs: 850 pairs from 151 accounts; intervals of plus or minus 2 points on accuracy differences and 4 to 5 on confident calls. A real effect of 1 to 2 points either way cannot be excluded.
- The like NLL on the ordinary test split is 0.015 to 0.025 worse after the fine-tune, and the fine-tuned ensemble's confidence scale must be refitted (1.81 against 3.30), per kind of pair.

## What I would do next

1. Stop describing the pair-loss fine-tune as an accuracy gain. Ship it, if at all, as a confidence improvement for same-link and same-quote comparisons only, with its own scale, and keep untouched members for everything else.
2. The cheap gain on the table is ensemble size: seven to ten untouched models give 61.0 to 61.2% against 60.0% for three, at no training cost beyond what exists.
3. Coverage at 75%+ needs the model to get more pairs right, not a better silence rule: every rule here only reorders the same 61 to 63%.
4. Fit the confidence scale per pair type. On launch pairs the benchmark scale is too timid for every model.

## Files in `tracks/replicate/`

- `REPORT.md`, `PREREG.md` (settings and decision rules, with the time each part was written), `patch.diff` (`scripts/rep_*`, and `--ctl-min-ratio` added to `scripts/train_pairs.py`).
- `step1.md`, `step1.json` (every member and ensemble by gap, all paired differences, by benchmark half), `cov40.md`, `cov40.json`.
- `launch.md`, `launch.json`, `launch_scores/<model>.parquet` (predicted likes for the 582 launch posts, 30 checkpoints; `feature_cache.pkl` scores another checkpoint in 15 seconds).
- `dev_pairs.parquet`, `dev_table/` (serving-path pair table of the 2,805 development pairs), `dev_scores/<model>.npy`, `dev_step1.*`, `dev_step3.*`.
- `step4_dev_ft10.*`, `step4_bench_*.txt|json`.
- `bench/rep_w{10,0}_<member>/shard_*.parquet` (per-pair benchmark scores of the fourteen new checkpoints).
- `runs/rep_w10_<member>/best.pt` (fine-tuned), `runs/rep_w0_<member>/best.pt` (controls), `runs/v_<variant>_<member>/best.pt` (step 3), each with `metrics.json`.
  Members: ts0_s0, ts0_s1, ts20_s1, ts20_s2, cmp_full (`compare/full`), cmp_aud (`compare/audience_only_training`), cmpv2_aud (`compare_v2/audience_only_training`).
  The final ensemble is these seven `rep_w10_*` plus the pairs track's `ft_w10`, `ft_w10_s2`, `ft_w10_v2full`. It beats the baseline on confident calls, not on overall accuracy beyond what ten untouched models give.

## Reproduce

From `virality/`, after `scripts/worker_job.sh`-style setup, with the dataset's `work/`, `runs/`, `bench2/`, `tracks/pairs/`, `tracks/launch/` under `data/` (same relative paths):

```
mkdir -p data/pairs data/bench3 data/rep
ln -s $PWD/data/tracks/pairs/pair_table data/pairs/table_full
ln -s $PWD/data/tracks/pairs/train_pairs.parquet $PWD/data/tracks/pairs/benchmark_pairs.parquet data/pairs/
for i in 0 1 2 3; do ln -s $PWD/data/tracks/pairs/bench/cache_$i.pkl data/bench3/; done

# step 1: fourteen fine-tunes (pair weight 10 and 0 side by side) and their benchmark scores; about 20 minutes per model on 4 cores
bash scripts/rep_queue.sh ts0_s0:ts0:data/runs/ts0_s0/best.pt ts0_s1:ts0:data/runs/ts0_s1/best.pt ts20_s1:ts20:data/runs/ts20_s1/best.pt ts20_s2:ts20:data/runs/ts20_s2/best.pt \
  cmp_full:ts20:data/work/model/compare/full/best.pt cmp_aud:ts20:data/work/model/compare/audience_only_training/best.pt cmpv2_aud:ts46:data/work/model/compare_v2/audience_only_training/best.pt
# one fine-tune, spelled out (rep_queue.sh runs exactly this, then scripts/bench_pairs.sh <name> <out>/best.pt 4):
python scripts/train_pairs.py --table data/work/model/table --pair-table data/pairs/table_full --pair-info data/pairs/train_pairs.parquet \
  --out data/runs/rep_w10_ts0_s0 --features ts0 --init data/runs/ts0_s0/best.pt --steps 600 --lr 2e-4 --w-ctl 10      # control: --w-ctl 0
python scripts/rep_step1.py            # data/rep/step1.md, step1.json
python scripts/rep_cov40.py            # data/rep/cov40.md

# development pairs and their table (about 60 minutes, mostly the encoder)
python scripts/rep_mine_dev.py --out data/rep/dev_pairs.parquet
python scripts/build_pair_table.py --pairs data/rep/dev_pairs.parquet --posts data/work/filtered/posts.parquet --checkpoint data/runs/ts46_s1/best.pt \
  --table data/work/model/table --sid-codebook data/work/emb/sid_codebook.npz --history 48 --out data/rep/dev_table
bash scripts/rep_side.sh               # development scores and launch-post scores for every checkpoint present
bash scripts/rep_launch_run.sh         # step 2 table
python scripts/rep_dev.py "ft10=$(cat data/rep/ft10.txt)" "ctl10=$(cat data/rep/ctl10.txt)" --versus "ft10|ctl10"

# step 3 variants (the arguments are in PREREG.md; rep_queue3.sh takes "name|features|checkpoint|extra train_pairs args")
bash scripts/rep_queue3.sh "v_lr1e4_ts46_s1|ts46|data/runs/ts46_s1/best.pt|--lr 1e-4" "v_flat_ts46_s1|ts46|data/runs/ts46_s1/best.pt|--ctl-weight flat" \
  "v_min2_ts46_s1|ts46|data/runs/ts46_s1/best.pt|--ctl-min-ratio 2" "v_s1200_ts46_s1|ts46|data/runs/ts46_s1/best.pt|--steps 1200"   # and the same for ts20_s1 (ts20), ts0_s0 (ts0)

# step 4
python scripts/rep_step4.py --set dev --members $(cat data/rep/ft10.txt)
python scripts/rep_step4.py --set bench --members $(cat data/rep/ft10.txt) --rules prob,agree_gate0.8
```

Member names in `ft10.txt`: `ft_<member>` = `data/runs/rep_w10_<member>`, `pairs_ft_<member>` = the pairs track's `ft_w10*` runs (`scripts/rep_models.sh` holds the mapping).
