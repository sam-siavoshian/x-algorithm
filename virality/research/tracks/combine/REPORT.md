# Track "combine": one production model set from the three gains

Worker report, 2026-11-19, one 4-core CPU machine, about three and a half hours. Likes unless it says views. Intervals are 95%, from
resampling accounts. Rules were written to `PREREG.md` before any result and decisions to `DECISIONS.md` with the time,
each before the test scores it governs were computed. Spend: 9.09 US dollars of OpenAI (limit 25).

## Answer

1. **The gains add up in the forecasting model.** Training on the broad table with the OpenAI embedding (two seeds, "A")
   keeps the embedding's gain on the tech audience and the broad data's gain on unseen accounts, both in full:

   | Test set | Pairs | Current product (bge-small, 3 models) | bge-small + broad (C) | OpenAI + original (B) | **OpenAI + broad (A)** |
   |---|---|---|---|---|---|
   | Ordinary tech-audience like pairs | 683,152 | 60.2% | 60.1% | 62.7% | **62.8%** |
   | Generalization test bed, like pairs, as a customer | 368,970 | 59.7% | 63.3% | 63.3% | **66.4%** |
   | Benchmark, pairs 1.25x apart | 2,852 | 60.0% | 59.3% | 61.7% | **61.7%** |
   | Benchmark, 2x apart and both posts 3+ likes | 1,086 | 60.2% | 61.0% | 62.2% | **64.4%** |
   | Launch pairs | 850 | 63.6% | 63.8% | 67.2% | **67.5%** |
   | Full timelines, accounts inside 1.5x of the forecast level, as a customer | 490 | 78% | 91% | 75% | **90%** |

2. **One surprise on validation, and what was done about it.** On launch pairs from before the cutoff, B is ahead of A by
   3.1 points (+1.7 to +4.6): broad training data costs the ranking of an archive account's launch posts, with either
   encoder. By the rule fixed beforehand the bundle therefore uses A for forecasts and same-topic comparisons and B for
   comparisons of two launch posts (four checkpoints, one encoder, no extra embedding cost). After the cutoff the two are
   level on launch accuracy (67.5% and 67.2%); B keeps more confident calls (51% of pairs against 46%).
3. **The pair-loss fine-tune is left out.** On validation it added no significant confident coverage on top of the new
   embedding (+1.0 correct confident calls per 100 development pairs 2x apart, -0.5 to +2.4). The untouched pair already
   states 60%+ on 28% of those pairs; the bge-small models stated it on 18%.
4. **Final system against the current product**, strict slices (wording differs, both posts after the judge's knowledge
   cutoff, both posts 3+ likes), pairs 2x apart, each scored once:

   | | Pairs | Current product: all calls; speaks on; right when it speaks | **New bundle** |
   |---|---|---|---|
   | Launch, no judge | 487 | 68.0%; 31%; 81.9% | **74.3%; 52%; 84.1% (78.1 to 89.0)** |
   | Launch, `gpt-6.1-sol` | 487 | 83.6%; 92%; 85.7% (81.4 to 89.2) | **83.4%; 89%; 88.2% (84.0 to 91.4)** |
   | Launch, `gpt-6-luna` | 426 | 78.2%; 86%; 80.9% | **78.4%; 84%; 83.5% (79.3 to 87.8)** |
   | Benchmark, no judge | 673 | 59.1%; 37%; 63.5% | **64.3%; 46%; 70.7% (65.7 to 75.8)** |
   | Benchmark, `gpt-6.1-sol` | 673 | 67.9%; 62%; 74.2% (69.4 to 78.7) | **69.2%; 30%; 80.8% (74.5 to 87.1)** |
   | Benchmark, `gpt-6-luna` | 630 | 61.7%; 46%; 68.2% | **64.8%; 51%; 71.8% (66.2 to 76.9)** |

   Without a judge and with the cheap judge the bundle is better on every row. With `gpt-6.1-sol` the share of all calls
   that are right is unchanged on launch pairs and 1.3 points higher on the benchmark: that judge carries most of the
   number. The verdicts are more often right (launch 88% against 86%; benchmark 81% against 74%), and on the benchmark the
   refitted rule voices far fewer of them (30% of pairs against 62%).
5. **Serving works with the OpenAI embedding**: encoder read from the checkpoints, vectors cached per account, cost in
   every response, a clear failure and a fallback model set. First request 1.1 to 1.8 s and 0.0003 to 0.0008 dollars;
   repeat request 0.5 s and nothing.
6. **Delivered**: the bundle (`bundle/`, one `bundle.json`), `MODEL_CARD.md`, 11 new tests, `patch.diff`.

## Step 1. The surrogate with the new embedding on the broad table

Code: `scripts/combine/embed_broad.py`, `build_table.py`, `predict_rows.py`, `eval_rows.py`, `score_sets.sh`,
`pair_sets.py`, `pair_eval.py`; training with the general track's `scripts/general/train_general.py`, unchanged.

**Embedding.** The broad table (1,154,680 rows) has 1,131,753 distinct rendered texts; 673,443 were not in the store.
Estimate from a 2,000-text sample of the missing texts: 76.3 tokens per text, 51.9M tokens, 6.75 dollars. Actual: 6.78
dollars. The table is the general track's with only the post embedding (first 1,024 dimensions, renormalised) and the
semantic IDs replaced; IDs use the embedding track's codebook, so one codebook serves the models of both tables.
Neighbour lists stay as built with bge-small, as in the embedding track. The checkpoints store the encoder's name.

**Arms.** A: new, `te3l_broad_s1`, `_s2`, 6,000 steps, checkpoint by the general track's rule (step 6,000 for both).
B: `embed_api/runs/te3l1024_s1`, `_s2` (same embedding, original 465,675-row table, 4,000 steps). C:
`general/runs/broad_s1`, `_s2` (bge-small, broad table, 6,000 steps). B and C are the stored models; B was scored on the
new table (its ordinary test pairs reproduce the embedding track: 62.7% as a two-seed ensemble, 62.4% and 62.2% per seed
there), C on the general table (60.1% and 63.3%, the general report's numbers). The general track showed that its
trainer reproduces the original trainer's control.

**Overfitting.** Training curves of A, likes loss at each evaluation point (seed 1; seed 2 within 0.006 everywhere):

| Step | Train | Validation, tech | Validation, broad | Test, tech | Test bed |
|---|---|---|---|---|---|
| 750 | 2.740 | 3.082 | 2.776 | 2.943 | 2.723 |
| 2,250 | 2.660 | 3.028 | 2.717 | 2.869 | 2.639 |
| 3,750 | 2.635 | 3.003 | 2.696 | 2.849 | 2.627 |
| 5,250 | 2.617 | 2.995 | 2.688 | 2.837 | 2.616 |
| 6,000 | 2.614 | 2.994 | 2.687 | 2.835 | 2.616 |

All five fall together to the last step; train and test bed are equal at the end.

**Ordinary same-account pairs** (two posts of one account within 7 days, outcomes 0.1 apart in log10). Columns: all
pairs / real gap 2x or more / both posts 3+ likes (views: both 100+ views). Differences in brackets are against C,
paired on accounts.

| Set | Outcome | Pairs (all / 2x / 3+) | C: bge-small + broad | B: OpenAI + original | A: OpenAI + broad |
|---|---|---|---|---|---|
| Train (700 tech accounts) | likes | 170,015 / 121,619 / 107,674 | 60.9 / 63.5 / 62.2 | 66.0 / 69.7 / 67.5 | 63.5 / 66.7 / 64.9 |
| Validation, tech accounts | likes | 71,368 / 51,895 / 55,746 | 61.9 / 64.9 / 62.5 | 64.9 / 68.3 / 65.8 | 64.5 (+2.6, +1.9 to +3.2) / 68.0 / 65.2 |
| Validation, broad accounts, as a customer | likes | 241,429 / 167,816 / 177,267 | 63.5 / 66.9 / 64.1 | 62.9 / 66.2 / 63.2 | 66.8 (+3.2, +2.7 to +3.8) / 70.7 / 67.2 |
| **Test, tech audience** | likes | 683,152 / 503,864 / 508,959 | 60.1 / 62.0 / 61.0 | 62.7 (+2.7, +2.1 to +3.2) / 65.2 / 64.0 | **62.8 (+2.8, +2.4 to +3.2) / 65.3 / 64.1** |
| **Test, tech audience** | views | 24,481 / 16,262 / 20,968 | 62.0 / 64.6 / 61.3 | 64.3 (+2.3, +1.2 to +3.2) / 68.0 / 63.3 | 63.1 (+1.1, -0.2 to +3.1) / 66.5 / 62.0 |
| **Test bed, as a customer** | likes | 368,970 / 262,522 / 257,682 | 63.3 / 66.1 / 63.8 | 63.3 (0.0, -0.6 to +0.6) / 66.3 / 64.2 | **66.4 (+3.1, +2.6 to +3.6) / 69.8 / 67.6** |
| **Test bed, as a customer** | views | 124,669 / 77,248 / 103,930 | 62.4 / 65.0 / 61.8 | 63.4 (+1.0, +0.3 to +1.7) / 66.7 / 62.9 | **65.3 (+2.8, +2.1 to +3.6) / 68.9 / 65.0** |

A against B on the tech test like pairs: +0.1 (-0.3 to +0.5). Likes loss, train / tech validation / tech test / test
bed: A 2.902 / 2.963 / 2.938 / 2.577; B 2.873 / 2.967 / 2.943 / 2.667; C 2.932 / 2.987 / 2.960 / 2.598. B fits its
own training accounts 2.5 points better than A and is no better on any held-out set: the original table lets a model
memorise its accounts.

Test bed by slice, like pairs, as a customer (A; difference to C; B):

| Slice | Pairs | A | A minus C | B |
|---|---|---|---|---|
| twitter100m full timelines | 247,210 | 68.1% | +2.9 (+2.4 to +3.4) | 64.4% |
| 2026 view corpus | 109,725 | 63.0% | +3.6 (+2.9 to +4.5) | 61.8% |
| Brand accounts | 12,035 | 61.2% | +3.3 (+1.2 to +5.2) | 55.3% |
| Accounts under 5,000 followers | 46,007 | 64.7% | +2.1 (+1.0 to +3.1) | 61.7% |
| Under 1,000 followers (twitter100m) | 14,497 | 60.9% | | 58.9% |
| 100,000 followers and over (twitter100m) | 89,109 | 69.9% | | 66.2% |

**Forecast level, scored as a customer** (full timeline, source unknown): median account's predicted median over actual
median; share of accounts inside 1.5x.

| Accounts | C | B | A |
|---|---|---|---|
| All full timelines (490) | 1.06x; 91% | 0.98x; 75% | 1.03x (1.00 to 1.07); 90% |
| Brand timelines (36) | 1.10x; 89% | 1.47x; 50% | 1.11x (1.03 to 1.22); 86% |
| 100,000 followers and over (86) | 0.99x; 91% | 0.65x; 47% | 0.92x (0.85 to 0.98); 85% |
| Under 1,000 followers (92) | 1.17x; 92% | 1.16x; 88% | 1.17x (1.10 to 1.19); 90% |
| Whole test bed (922) | 1.06x; 87% | 1.02x; 78% | 1.07x (1.04 to 1.09); 86% |

The honest level is the broad data's doing and survives the change of encoder. A is slightly low at 100,000 followers
and over (0.92x, where C was 0.99x). Under 1,000 followers every model is 1.17x high.

**Pairs through the serving path** (56 posts of history, median of the like histogram). Each model set's confidence
scale is fitted on pre-cutoff pairs of the same kind and applied unchanged: same-topic A 3.43, B 3.03, C 2.68, current
2.72; launch A 2.09, B 2.32, C 1.54, current 1.84. "Confident" = stated 60% or more.

Validation (before the cutoff; confidence cross-fitted over accounts):

| Set | Slice | Pairs | Current | C | B | A |
|---|---|---|---|---|---|---|
| Development pairs, same-topic | all 1.25x apart | 2,805 | 56.4% | 56.1% | 59.4% | 59.7% (57.4 to 61.9) |
| Development pairs | 2x: all calls; confident share, correct | 1,956 | 58.5%; 18%, 71.7% | 56.7%; 18%, 71.7% | 61.4%; 23%, 70.0% | 60.9%; 28%, 71.7% |
| Development pairs | 2x: surest 40% | 782 | 66.2% | 65.9% | 68.2% | 70.7% |
| Launch pairs before the cutoff | all 1.25x apart | 3,655 | 63.8% | 62.2% | **68.0%** | 64.9% (62.6 to 67.0) |
| Launch pairs before the cutoff | accounts no model trained on | 1,054 | 66.4% | 64.1% | **68.2%** | 63.7% |
| Launch pairs before the cutoff | later-time, other-accounts cell | 858 | 67.2% | 65.7% | **69.0%** | 66.8% |

B minus A on pre-cutoff launch pairs: +3.1 (+1.7 to +4.6); on accounts not trained on +4.6 (+2.0 to +7.1); within 30
days +3.3 (+0.3 to +6.2). Current minus C: +1.6 (+0.3 to +2.9). Neither set calls the later post more often than the
other (53% and 55%), so it is not posting order. This is the decision recorded in `DECISIONS.md`: B decides launch
comparisons.

Test (scored after the decision):

| Set | Slice | Pairs | Current | C | B | A |
|---|---|---|---|---|---|---|
| Benchmark | all 1.25x apart | 2,852 | 60.0% (57.9 to 62.0) | 59.3% | 61.7% (59.3 to 64.0) | **61.7% (59.3 to 63.9)** |
| Benchmark | 2x apart | 2,030 | 61.4% | 60.3% | 63.5% | 63.3% (60.5 to 65.7) |
| Benchmark | both 3+ likes, 1.25x apart | 1,782 | 59.2% | 59.0% | 60.1% | 61.6% (58.9 to 64.3) |
| Benchmark | both 3+ likes, 2x apart | 1,086 | 60.2% | 61.0% | 62.2% | **64.4% (61.2 to 67.5)** |
| Benchmark | 2x: confident share, correct | 2,030 | 24%, 70.4% | 26%, 70.1% | 30%, 72.6% | **32%, 73.7% (69.9 to 77.2)** |
| Benchmark | 2x: surest 40% | 812 | 67.6% | 67.2% | 70.8% | **72.7% (69.1 to 76.0)** |
| Launch | all 1.25x apart | 850 | 63.6% (59.4 to 67.9) | 63.8% | 67.2% (63.2 to 71.3) | 67.5% (63.5 to 71.3) |
| Launch | 2x apart | 568 | 66.0% | 65.3% | 72.0% | 70.4% (65.5 to 75.0) |
| Launch | both 3+ likes, 1.25x apart | 809 | 64.0% | 64.2% | 67.2% | 68.0% (64.2 to 71.9) |
| Launch | both 3+ likes, 2x apart | 535 | 66.9% | 66.4% | 72.7% | 71.6% (67.1 to 76.4) |
| Launch | 2x: confident share, correct | 568 | 35%, 77.8% | 36%, 77.8% | **54%, 82.0% (75.8 to 87.6)** | 49%, 80.9% |
| Launch | 2x: surest 40% | 227 | 77.5% | 76.7% | 82.4% | 83.7% (76.0 to 90.4) |

Paired differences on the test sets: A minus current, benchmark +1.8 (-0.3 to +3.5), at 2x with both posts 3+ likes
+4.1 (+1.5 to +6.9), correct confident calls per 100 pairs 2x apart +6.6 (+5.0 to +8.4); launch +3.9 (0.0 to +7.9),
correct confident calls per 100 pairs 2x apart +12.3 (+7.4 to +17.0). A minus C: benchmark +2.5 (+0.7 to +4.2), launch
+3.8 (-0.5 to +7.9). A minus B: benchmark 0.0 (-1.7 to +1.9), at 2x with both 3+ likes +2.2 (-0.1 to +5.0); launch +0.4
(-2.6 to +3.0), confident share -5.5 (-9.2 to -1.6).

Reading: the embedding is worth 2.5 to 3 points on the tech audience with either table; the broad table is worth 3
points on unseen accounts with either encoder and nothing on the tech audience; together they are worth both. The
validation lead of B on launch pairs is not there after the cutoff in accuracy and is there in confident coverage.

## Step 2. Pair-loss fine-tune

`scripts/combine/pair_table_te3l.py` (the pairs track's pair table with the new embedding, its semantic IDs and the two
source inputs; texts rendered from the corpus by post id and checked equal to the stored bge-small vectors on 300 rows,
cosine 1.0000), `scripts/combine/finetune.sh` (the replicate track's recipe: `train_pairs.py`, 600 steps, learning rate
2e-4, warm-up 50, seed 0, weight 10 next to weight 0). Four runs, 16 minutes each.

Validation, two-seed ensembles:

| Development pairs 2x apart (1,956) | Fine-tuned | Control (weight 0) | Untouched A | Fine-tuned minus control |
|---|---|---|---|---|
| All calls | 61.2% | 60.9% | 60.9% | +0.4 (-1.0 to +1.9) |
| Stated 60%+: share, correct | 29%, 73.3% | 29%, 71.9% | 28%, 71.7% | share +0.8 (-1.1 to +2.4) |
| Correct confident calls per 100 pairs | 21.5 | 20.5 | 19.8 | +1.0 (-0.5 to +2.4) |
| Surest 40% | 70.8% | 69.8% | 70.7% | +1.0 (-0.9 to +2.8) |
| All development pairs (2,805), correct | 59.5% | 59.8% | 59.7% | -0.3 (-1.6 to +0.9) |
| Launch pairs before the cutoff (3,655), correct | fine-tuned +1.6 (+0.6 to +2.5) over control | | | |

The fixed test (correct confident calls per 100 development pairs 2x apart must rise with an interval excluding zero)
fails. **Left out of the bundle.**

The same tables on the test sets, as the brief asks (scored after the decision, used for nothing):

| Test set | Fine-tuned | Control | Fine-tuned minus control |
|---|---|---|---|
| Ordinary tech like pairs (683,152): all / 2x / 3+ | 62.8 / 65.2 / 64.0 | 62.8 / 65.2 / 64.0 | 0.0 (-0.2 to +0.2) |
| Ordinary tech view pairs (24,481) | 63.2% | 63.1% | +0.1 (-0.4 to +0.7) |
| Test bed like pairs as a customer (368,970): all / 2x / 3+ | 66.3 / 69.7 / 67.4 | 66.4 / 69.8 / 67.6 | -0.1 (-0.3 to 0.0) |
| Likes loss, tech test / test bed | 2.960 / 2.606 | 2.939 / 2.579 | worse by 0.02 to 0.03 |
| Full timelines inside 1.5x, as a customer | 89% | 91% | |
| Benchmark, 1.25x apart (2,852) | 63.7% (61.2 to 66.3) | 62.6% | +1.2 (-0.5 to +2.7) |
| Benchmark, 2x apart (2,030) | | | +2.1 (+0.4 to +3.9) |
| Benchmark, 2x apart, both 3+ likes (1,086) | 65.0% | 64.8% | +0.2 (-1.8 to +1.9) |
| Benchmark, correct confident calls per 100 pairs 2x apart | | | +4.2 (+2.4 to +6.2) |
| Launch, 1.25x apart (850) | 68.1% | 66.8% | +1.3 (-0.7 to +3.1) |
| Launch, confident share on pairs 2x apart | | | +7.0 (+3.5 to +10.9) |

On the benchmark the fine-tune did add confident calls, as it did for the bge-small models; on the development pairs it
did not. Development pairs are up to 14 days apart and benchmark pairs at most 48 hours, and every earlier track saw the
two disagree in the same direction. The fine-tune stays out because the validation said so; it is the first thing to
re-test with a validation set built like the benchmark.

## Step 3. Rules for the chosen set, fitted before the cutoff

**Confidence scales** (`scripts/combine/fit_selective.py`, `xviral/data/selective_rule.json` entry `te3l_broad2`): one
scale per kind, on all pre-cutoff pairs 1.25x apart. Same-topic 3.43 on A's gap (2,805 pairs, 366 accounts); launch
2.32 on B's gap (3,655 pairs, 628 accounts). Time-and-account split check (fitted on the fit cell, read on the later-time
and other-accounts cell), pairs 2x apart:

| Kind | | Train: surest 40%; stated 60%+ share, correct | Validation | Test (above) |
|---|---|---|---|---|
| Same-topic (A) | scale on the fit cell 4.03 | 69.8%; 27%, 74.6% | 64.1%; 40%, 64.1% (382 pairs) | 72.7%; 32%, 73.7% |
| Launch (B) | scale on the fit cell 2.20 | 81.9%; 65%, 78.0% | 87.7%; 66%, 82.6% (649 pairs) | 82.4%; 54%, 82.0% |

The launch scale holds from validation to test. The same-topic validation cell is the hardest slice of the development
pairs for every model (the judge track saw the same), and the benchmark comes out above it.

**Post-type level correction** (`scripts/combine/fit_type_correction.py`, `type_correction.json`): ridge fit on 3,990
own announcements of 1,499 accounts before the cutoff. Weights in log10 likes: funding +0.243, open source +0.147,
launch +0.090, hiring +0.044, demo +0.035, milestone +0.035, feature +0.017 (the current product's: launch +0.133,
funding +0.339). Fitted on half the accounts and read on the other half: mean error +0.075 to +0.002, mean absolute
error 0.379 to 0.373; launch +0.092 to +0.005; funding +0.263 to +0.037. Kept. On the 576 test launch posts the forecast
is 1.31x the account's usual without it and 1.63x with it; actual 2.19x (current product 1.10x and 1.50x).

**Judge blend rule** (`scripts/judge_product/fit_rule.py` and `test_eval.py`, unchanged except that the working folder,
the surrogate's scales and the output file can be set; `scripts/combine/judge_dir.py` puts the new gaps beside the stored
judge answers). No new judge calls. Run on the current product's gap, the same code reproduces the shipped rule exactly
(weights 2.017/0.042, 1.198/0.486, 1.578/0.135, 1.692/0.847) and the judge_product report's test numbers.

| Judge | Kind | Weight on the surrogate's gap | Weight on the judge | Clip | Speak | Fitted on |
|---|---|---|---|---|---|---|
| `gpt-6-luna` | same-topic | 2.705 | 0.037 | 14 | 60% and the two agree | 2,413 pairs |
| `gpt-6-luna` | launch | 1.876 | 0.120 | 14 | 60% | 3,567 pairs |
| `gpt-6.1-sol` | same-topic | 1.783 | 0.458 | 1.5 | 70% | 809 pairs |
| `gpt-6.1-sol` | launch | 1.662 | 0.805 | 4 | 60% | 596 pairs |

Train and validation side by side (fit cell, validation cell; pairs 1.25x apart): same-topic with luna, blend 61.7% and
61.1% (surrogate 57.5% and 60.4%); same-topic with sol, 59.5% and 60.3% (148 and 335 pairs); launch with luna, 73.6% and
68.2%; launch with sol, 79.2% and 76.2% (surrogate 67.4% and 72.8%). The same-topic speak rule with sol changed from "60%
and (agree, or the judge alone at 70%)" to "70%": on the 206 validation pairs 2x apart it is now the rule with the most
correct spoken calls among those at 75% or better. It rests on few pairs, as it did before.

**Final system on the test sets, once** (`llm_new/jp_test.md`; the current product from the same script,
`llm_base3/jp_test.md`). Cells: all calls; speaks on; right when it speaks.

| Test set, judge | Slice | Real gap | Pairs | Current product | New bundle | New: blend minus its surrogate, all calls |
|---|---|---|---|---|---|---|
| Launch, sol | all | 1.25x | 850 | 75.6%; 88%; 78.5% | 75.6% (70.7 to 80.0); 86%; 80.4% | +8.5 (+4.2 to +12.6) |
| Launch, sol | all | 2x | 568 | 80.8%; 89%; 84.2% | 80.5%; 87%; 86.0% (81.3 to 90.5) | +8.5 (+3.5 to +13.3) |
| Launch, sol | wording differs, after the cutoff | 2x | 511 | 82.8%; 91%; 85.3% | 82.8%; 88%; 88.0% (84.1 to 91.2) | +8.8 (+3.5 to +14.5) |
| Launch, sol | strict | 2x | 487 | 83.6%; 92%; 85.7% (81.4 to 89.2) | **83.4%; 89%; 88.2% (84.0 to 91.4)** | +9.0 (+3.0 to +14.5) |
| Launch, luna | all | 1.25x | 850 | 71.2%; 84%; 73.6% | 70.5%; 82%; 75.2% | +3.3 (-1.0 to +7.4) |
| Launch, luna | strict | 2x | 426 | 78.2%; 86%; 80.9% | 78.4%; 84%; 83.5% (79.3 to 87.8) | +3.8 (-2.2 to +10.0) |
| Benchmark, sol | all | 1.25x | 2,852 | 68.2%; 50%; 75.6% | 68.3% (66.1 to 70.3); 19%; 81.3% | +6.6 (+4.3 to +8.8) |
| Benchmark, sol | wording differs | 2x | 1,647 | 69.6%; 55%; 77.7% | 70.3%; 24%; 84.8% (80.2 to 88.9) | +5.4 (+3.0 to +7.9) |
| Benchmark, sol | wording differs, after the cutoff | 2x | 953 | 67.7%; 58%; 76.1% | 68.9%; 26%; 81.9% (76.5 to 87.0) | +3.5 (+0.4 to +6.2) |
| Benchmark, sol | strict | 2x | 673 | 67.9%; 62%; 74.2% (69.4 to 78.7) | **69.2%; 30%; 80.8% (74.5 to 87.1)** | +4.9 (+1.0 to +8.6) |
| Benchmark, luna | all | 1.25x | 2,852 | 64.4%; 41%; 72.8% | 66.0%; 43%; 72.7% | +4.2 (+2.4 to +6.3) |
| Benchmark, luna | wording differs, after the cutoff | 2x | 890 | 63.8%; 44%; 70.6% | 65.8%; 48%; 72.8% (68.2 to 77.2) | +1.2 (-1.2 to +3.7) |
| Benchmark, luna | strict | 2x | 630 | 61.7%; 46%; 68.2% | 64.8%; 51%; 71.8% (66.2 to 76.9) | +1.3 (-1.7 to +4.0) |

"Strict" = near-identical pairs removed, both posts after the judge's knowledge cutoff, both posts 3+ likes. The
surrogate alone on the strict slices: launch 74.3%, 52% at 84.1% (current 68.0%, 31% at 81.9%); benchmark 64.3%, 46% at
70.7% (current 59.1%, 37% at 63.5%).

Against the project goal (confident calls 75 to 80% right on at least 40% of pairs 2x apart): met on launch pairs by the
surrogate alone (52% at 84%) and by every blend; on the strict benchmark slice the sol blend is at 81% on 30% of pairs,
the luna blend at 72% on 51%, the surrogate alone at 71% on 46%. Not met there.

Why the strong judge adds less to the new surrogate (launch +8.5 where it added +11.8 to the old one): the better
surrogate already gets right part of what the judge used to add. The two are still far from the same opinion.

## Step 4. Serving

Code: `xviral/embed/api.py` (new), `xviral/serve.py`, `xviral/advise.py`, `xviral/app/service.py`, `server.py`,
`index.html`, `xviral/model/ensemble.py`; reference in `docs/launch_advisor_api.md`, section "The text embedding".

- **Encoder from the checkpoint.** `table_meta["encoder"]` holds the name (`openai:text-embedding-3-large@1024`; absent
  = bge-small, which is every older checkpoint; the embedding track's recipe form is read too). `DraftScorer` builds the
  encoder from it; members of an ensemble must agree; an encoder of another size, or `XVIRAL_ENCODER` naming another one,
  is refused at start-up.
- **Client.** `OpenAIEmbedder`: batches of 256 texts, batches in parallel, 20 s timeout, two retries on 408, 409, 429,
  5xx and connection errors, none on a refused request; any failure raises one `EmbeddingError` and returns no vectors.
  Usage (calls, texts, tokens, dollars, seconds) is kept per request and in total.
- **Cache per account.** Vectors are kept by rendered text and grouped by account; when more than 512 accounts are held
  the least recently used account's vectors go. A repeat request embeds only texts it has not seen.
- **Failure and fallback.** `Service(..., fallback=)`: a second model set with a local encoder. It answers the whole
  request when the provider fails (report: `embedding.fallback_used`, `fallback_reason`, `error`, `primary_attempt`) and
  when the request does not allow its text to be sent. Without a fallback: 503 "Nothing was scored" for a provider
  failure, 422 with the reason for a request that may not be sent. Never a report on partial vectors.
- **Cost in the response.** `embedding` = encoder, provider, model, texts needed, sent and from cache, calls, tokens,
  dollars, milliseconds, what was sent; `cost_usd` = `{embedding, judge, total}`; the request log line carries the same
  without text.
- **Members by kind.** `DraftScorer.kind_scorers = {"launch": scorer}`: the launch members read the features and vectors
  the forecast members computed (no embedding call, about 0.1 s of model time) and their gap decides launch comparisons;
  every pair in the report says which members decided it.
- **Stub.** `StubEmbedder`: deterministic vectors from a hash of the text, counts calls and cost, can fail or delay;
  `XVIRAL_EMBED_STUB=1` swaps it in for a dry run. Every test uses it or a fake HTTP session.
- **One config.** `service.load_bundle(folder)`, or `XVIRAL_BUNDLE`.

**What is sent to the provider.** For the embedding: the rendered text of each version and of the account's most recent
posts the model reads (at most 48): the author line (display name and handle) and the post text with link wrappers
removed. No like or view counts, no follower numbers, no media, no times. Once per text. For the judge, unchanged: the
versions, handle and follower count, and up to 30 past posts with like counts. Posts that came with a request are sent to
neither without `share_past_posts: true`; `external_embedding: false` keeps every text of a request away from the
embedding provider. Before this change a request with `judge: false` sent nothing anywhere; with an API embedding that is
only true together with `external_embedding: false` (or uploaded posts without `share_past_posts`), and the page and the
API reference now say so. An unpublished draft is sent to the embedding provider by default: that is a product decision
this track made visible and switchable, not one it can make.

**Latency and cost, real run** (`scripts/combine/serve_check.py`, through the HTTP layer, 4 cores, three corpus accounts
drawn with a fixed seed, public posts only; `serve_check/summary.json`, `serve_check_sol/summary.json`):

| Request | Seconds | Texts sent | Tokens | Embedding cost, dollars | Embedding call, ms |
|---|---|---|---|---|---|
| First for an account, two versions (53, 63, 300 past posts supplied; 47 to 49 read) | 1.82, 1.16, 1.05 | 49, 50, 51 | 2,574, 5,979, 3,443 | 0.00034, 0.00078, 0.00045 | 1,249, 668, 433 |
| The same request again | 0.48, 0.45, 0.65 | 0 | 0 | 0 | 0 |
| Same account, one version edited | 0.69, 0.77, 1.20 | 1 | 41, 80, 122 | 0.000005 to 0.000016 | 254, 270, 586 |
| Six versions, account seen before | 1.35 | 5 | 220 | 0.00003 | 821 |
| Past posts may not be sent: fallback models (bge-small encoded locally) | 2.45, 2.19, 1.55 | 0 | 0 | 0 | |
| First request with `gpt-6.1-sol` as judge | 2.91, 4.12, 3.23 | 49, 50, 51 | | 0.0003 to 0.0008, plus judge 0.008 to 0.013 | |
| Six versions with `gpt-6.1-sol` (15 pairs, 30 calls) | 7.60 | 5 | | judge 0.126 | |

The launch members add 60 to 180 ms of model time. The live call returns the training vectors: 40 table texts embedded
again through the client, cosine 0.9996 to 1.0000. The bundle's serving path reproduces the evaluated gaps of both member
sets to 1e-7 on 40 benchmark units (`scripts/combine/parity.py`).

## Step 5. The bundle

`bundle/` in the dataset (51 MB): `bundle.json`, four checkpoints, `sid_codebook.npz`, `selective_rule.json`,
`type_correction.json`, `judge_rule.json`, `fallback/` (the three bge-small models with their codebook, scales,
correction and judge rule), `service.env`, `MODEL_CARD.md`. Also in the package: `xviral/data/selective_rule.json` has
the new entry `te3l_broad2` (the default stays `base3`), `xviral/data/judge_rule_te3l_broad2.json` is the new judge rule.

Left out, as the brief asks for steps that did not improve on validation: the pair-loss fine-tune.

Tests: `tests/test_embed_api.py`, 11 tests (client, retries and errors, encoder from metadata, per-account cache, cost
and what was sent for a first and a repeat request, provider failure with and without fallback, text not sent without
permission, account warming, the bundle loader, members by kind, the HTTP layer). Two existing tests changed with the
behaviour they describe (the encoder setting's default; the report now has an `embedding` block). Whole suite: 170
passed, 1 skipped.

## What worked, what did not

Worked: the three gains in one model, additive (test bed 59.7% to 66.4%, tech pairs 60.2% to 62.8%, benchmark confident
calls 24% at 70% to 32% at 74%, launch 35% at 78% to 54% at 82%, honest level kept); reusing the store (9 dollars for
everything); the fixed-rule discipline, which caught a real regression on validation (broad data and launch pairs) before
any test score existed; rule fits reproduced exactly from stored judge answers.

Did not work, or not as hoped: the pair-loss fine-tune on validation; the validation lead of the launch members did not
show after the cutoff in accuracy; with the strongest judge the final all-calls accuracy barely moves, because the judge
already supplied what the better surrogate now supplies; the refitted same-topic rule for that judge speaks on 30% of
pairs, under the 40% the goal asks for.

## Honest limits

- Two seeds per arm. Seed spread on the benchmark is about 1.5 points per model; all pair results are two-seed ensembles.
- The split of members by kind of comparison was chosen after seeing the validation table (it is finer than the rule's
  "B for the tech audience"), and the test sets do not show B more accurate than A on launch pairs. Using A for
  everything would be simpler and, on these test sets, level in accuracy with fewer confident launch calls.
- B's own test numbers are from the same test sets the embedding track reported; nothing about B was chosen here.
- Launch pairs are two different launch posts, a median of 55 days apart, of archive accounts. Which members should decide
  a launch comparison for an account outside the archive is untested.
- The same-topic judge rule for `gpt-6.1-sol` rests on 809 development pairs; its speak rule on 206.
- The fine-tune's test tables were computed after it was dropped; they are reported, not used.
- No setting of the trunk was tuned for the embedding or the table (steps, learning rate, history kernel as before).
- Neighbour lists in the table are still bge-small's; serving passes none.
- The fallback path was exercised with the stub (failure) and for real (no permission); a real provider outage was not
  observed.
- The served account cache is in memory and per process.
- Views: the tech-audience view pairs did not improve over B (63.1% against 64.3%, -1.1, -2.2 to +0.4).

## Tests

`pytest tests`: 170 passed, 1 skipped (`pytest.txt`), 11 of them new (`tests/test_embed_api.py`). One earlier run made
while four model jobs held every core failed a timing assertion of an existing judge test (twelve stub calls of 0.1 s
within 0.6 s); it passes on an idle machine.

## Spend

| Item | Texts | Tokens | US dollars |
|---|---|---|---|
| Estimate sample (2,000 texts) | 2,000 | 152,647 | 0.02 |
| Broad table, texts not in the store | 671,443 | 52.1M | 6.78 |
| Serving-path caches of the development and pre-cutoff launch pairs | 88,169 | 6.5M | 0.84 |
| Pair table for the fine-tune | 99,958 | 9.7M | 1.26 |
| Serving checks (embedding) | about 520 | about 45,000 | 0.006 |
| Real run with `gpt-6.1-sol` as judge (48 calls) | | 95,938 in, 2,192 out | 0.19 |
| **Total** | | 68.5M embedding tokens | **9.09** |

No other chat-model call was made; the judge rules were refitted from stored answers. The store in the dataset
(`tracks/embed_api/store/te3large_3072/`) now holds 1,442,862 texts; the 58 new shard files (5.3 GB) are uploaded.

## Files in `tracks/combine/`

- `REPORT.md`, `MODEL_CARD.md`, `PREREG.md`, `DECISIONS.md`, `patch.diff` (against branch head e5791b6; new files included), `pytest.txt`.
- `bundle/`: the production bundle.
- `runs/te3l_broad_s1`, `_s2` (`best.pt`, `curves.json`); `runs/ft_w10_*`, `ft_w0_*` (the fine-tuned and control models, `best.pt`, `metrics.json`).
- `table/`: `emb.npy` (1,154,680 x 1,024, fp16), `sid.npy`, `meta.json`, `recipe.json`, `sid_codebook.npz`; with the general track's table files they are the training table.
- `pair_table/`: `emb.npy`, `sid.npy`, `cat.npy`, `meta.json` of the re-embedded pair table (the other files are the pairs track's).
- `caches/te3l/`: the thirteen serving-path feature caches with the new embedding.
- `sel/`: per-unit serving-path scores of every model on the four pair sets; `pairs/`: the four pair tables with every member's gap.
- `pred/`, `pred_bge/`: per-row predictions on train, validation, test and test-bed rows.
- `eval/`: `rows_step1.*`, `rows_step2.*`, `pairs_step1.*`, `pairs_step2.*`, `val_step1.*`, `val_step2.*` (every table above with intervals).
- `llm_new/`, `llm_base3/`: `jp_fit.md`, `jp_test.md` and their JSON for the new bundle and the current product; `judge_rule.json`, `judge_rule_fitted_frozen.json`, `selective_rule.json`, `type_correction.json`.
- `serve_check/`, `serve_check_sol/`: the real runs.

## Reproduce

From `virality/`, after the set-up of `scripts/worker_job.sh` and the downloads at the top of this track's `dl.py` list
(general table, the store, the embedding and general checkpoints, the pair caches, the judge outputs):

```bash
python scripts/combine/embed_broad.py --estimate; python scripts/combine/embed_broad.py --max-dollars 9
python scripts/combine/build_table.py
for s in 1 2; do python scripts/general/train_general.py --table data/combine/table_te3l_broad --out data/combine/runs/te3l_broad_s$s --seed $s --roles orig,expand,train --steps 6000; done
bash scripts/combine/rewrite_caches.sh 19.6
bash scripts/combine/score_sets.sh te3l te3l_broad "dev launch_pre" te3l_broad_s1=... te3l_broad_s2=...     # and te3l_orig, bge
python scripts/combine/predict_rows.py --table data/combine/table_te3l_broad --run ... ; python scripts/combine/eval_rows.py --model ... --base ...
python scripts/combine/pair_sets.py --scores data/combine/sel/te3l --scores data/combine/sel/bge; python scripts/combine/pair_eval.py --ens ... --sets dev,launchpre
python scripts/combine/pair_table_te3l.py --max-dollars 18.8; bash scripts/combine/finetune.sh 3 te3l_broad_s1=... te3l_broad_s2=...
python scripts/combine/fit_selective.py --name te3l_broad2 --members te3l_broad_s1,te3l_broad_s2 --launch-members te3l1024_s1,te3l1024_s2 --out data/combine/selective_rule.json
python scripts/combine/fit_type_correction.py --table data/combine/table_te3l_broad --run a.pt,b.pt --name te3l_broad2 --out data/combine/type_correction.json
python scripts/combine/judge_dir.py --out data/combine/llm_new/ --members ... --launch-members ...
JP_DIR=data/combine/llm_new/ JP_SURR_SCALE='{"same_topic": 3.433, "launch": 2.3222}' python scripts/judge_product/fit_rule.py
JP_DIR=data/combine/llm_new/ JP_SURR_SCALE='...' JP_RULE_OUT=data/combine/judge_rule.json python scripts/judge_product/test_eval.py
python scripts/combine/make_bundle.py --name te3l_broad2 --checkpoint ... --launch-checkpoint ... --selective ... --type-correction ... --judge-rule ... --validation data/combine/validation.json
python scripts/combine/parity.py; python scripts/combine/serve_check.py --bundle data/combine/bundle [--judge gpt-6.1-sol]
pytest tests/test_embed_api.py tests/test_app.py tests/test_serve.py tests/test_judge.py tests/test_selective.py
```
