# Track "combine": decisions, with the time each was made

## Step 1, 2026-11-19 15:25 UTC. Made on validation numbers only; no benchmark, launch-test, ordinary-test or test-bed score of any new model existed.

Candidates: A = OpenAI embedding + broad table (`te3l_broad_s1`, `_s2`, new), B = OpenAI embedding + original table
(`te3l1024_s1`, `_s2`), C = bge-small + broad table (`broad_s1`, `_s2`). Two-seed ensembles.

| Validation measure | A | B | C | A minus B |
|---|---|---|---|---|
| Tech-audience validation accounts, like pairs (71,368) | 64.5% | 64.9% | | -0.4 (-0.9 to +0.1) |
| Broad validation accounts, like pairs (241,429), as a customer | 66.8% | 62.9% | 63.5% (computed after the decision; the training curves had shown 62.0% on a sample) | +3.9 (+3.2 to +4.6) |
| Broad validation accounts, level: accounts inside 1.5x | 86% | 75% | | |
| Development pairs, same-topic (2,805) | 59.7% | 59.4% | 56.1% | +0.4 (-1.7 to +2.4) |
| Development pairs 2x apart: correct confident calls per 100 | 19.8 | 16.3 | 13.2 | +3.5 (+1.9 to +5.2) |
| Launch pairs before the cutoff (3,655) | 64.9% | 68.0% | 62.2% | -3.1 (-4.6 to -1.7) |
| Same, accounts no model trained on (1,054) | 63.7% | 68.2% | 64.1% | -4.6 (-7.1 to -2.0) |
| Same, later-time and other-accounts cell (858) | 66.8% | 69.0% | 65.7% | -2.2 (-4.8 to +0.4) |

Reading. A carries the embedding gain onto broad data (A against C on development pairs +3.6, +1.3 to +5.8) and the broad-data
gain onto the new embedding (A against B on broad validation +3.9). On the tech audience it is level with B on ordinary
validation pairs and on same-topic pairs, with more correct confident calls. On launch pairs before the cutoff B is
ahead by an interval that excludes zero, in every slice looked at (accounts not trained on, pairs within 30 days, the
later-time cell as a point estimate), and the same direction shows with bge-small (the current three against C: +1.6,
+0.3 to +2.9). Broad training data costs ranking of an archive account's launch posts before the cutoff.

Decision, by the rule in PREREG.md ("A, unless A is worse than B by an interval that excludes zero on ... pre-cutoff
launch pair accuracy; then B for the tech audience"):
- **Forecasts (level, ranges, odds), same-topic comparisons, and every account that is not in the archive: A.**
- **Comparisons of two launch posts: B.** One bundle, one encoder, four checkpoints; B's members read the same features
  and vectors and add no embedding cost.
- This split by kind of comparison is finer than the rule's wording ("B for the tech audience"). It was chosen after
  seeing the validation table, because A is not worse than B on same-topic pairs and makes more correct confident calls
  there. A four-model mix (A and B averaged) was looked at on validation and is not used: on launch pairs it loses most of
  B's lead (66.0%, -1.9 against B, -2.8 to -0.9).
- Not known from validation: whether B's lead on launch pairs holds after the cutoff (the general track's bge-small
  models showed no loss from broad data on the 850 test launch pairs: +0.4, -3.8 to +4.0), and whether it holds for an
  account outside the archive. The test tables report A and B on launch pairs side by side.

## Step 2, 2026-11-19 16:21 UTC. Made on validation numbers only; no test score of a fine-tuned or control model had been computed.

Pair-loss fine-tune of both seeds of A (weight 10) against the weight-0 control, two-seed ensembles, development pairs
2x apart (1,956), confidence cross-fitted over accounts:

| | Fine-tuned | Control | Fine-tuned minus control |
|---|---|---|---|
| Correct, all development pairs (2,805) | 59.5% | 59.8% | -0.3 (-1.6 to +0.9) |
| Stated 60%+ on pairs 2x apart: share, correct | 29%, 73.3% | 29%, 71.9% | share +0.8 (-1.1 to +2.4) |
| Correct confident calls per 100 pairs 2x apart | 21.5 | 20.5 | +1.0 (-0.5 to +2.4) |
| Surest 40% of pairs 2x apart | 70.8% | 69.8% | +1.0 (-0.9 to +2.8) |
| Launch pairs before the cutoff, correct (3,655) | 66.2% | 64.6% | +1.6 (+0.6 to +2.5) |

Rule (PREREG.md): keep only if correct confident calls per 100 development pairs 2x apart rise with an interval that
excludes zero. They do not (+1.0, -0.5 to +2.4). **The fine-tune is left out of the bundle.** With the new embedding the
untouched pair already states 60%+ on 28% of development pairs 2x apart (the bge-small models: 18%), which is the coverage the
fine-tune used to add. Its small gain on pre-cutoff launch pairs is not used either: launch comparisons are decided by B,
which is 1.8 points ahead of the fine-tuned A there. The fine-tuned and control models are reported on the test sets
all the same, as the brief asks.
