# Track "combine": rules fixed before results

Written 2026-11-19 14:08 UTC, while the two new models were at steps 1,600 and 1,100 of 6,000 and before any
serving-path score of any new model existed. The only numbers of the new models seen at this point are the training
curves' first evaluation (step 750: tech validation pairs 62.0%, broad validation 62.7%).

## Splits

- Test, never used for a choice: ordinary test rows (posts from 2026-04-23 on), the 3,530 benchmark pairs, the 850
  launch pairs, the generalization test bed. Each is scored once per configuration that the brief asks to be reported.
- Validation, used for every choice: tech-audience validation accounts (`val_aud`), the general track's validation
  accounts (`val_broad`), the 2,805 development pairs and the 3,655 launch pairs from before the cutoff (`dev`,
  `launchpre`), both through the serving path. Inside `dev` and `launchpre` the judge track's cells are kept: fit = both
  posts before T and account half A; validation = later post on or after T and account half B (T = 2025-10-01 and
  2026-01-01).
- Order of work: validation numbers first; the decisions below are then written to `DECISIONS.md` with the time; only
  after that are benchmark and launch test scores computed for the configurations to be reported.

## Step 1: which model set

Candidates: (A) OpenAI embedding + broad table (new, two seeds), (B) OpenAI embedding + original table
(`embed_api/runs/te3l1024_s1,_s2`), (C) bge-small + broad table (`general/runs/broad_s1,_s2`). Checkpoint of (A): the
general track's rule (lowest mean of tech and broad validation loss among eight evaluation points), unchanged.
Choice: A, unless on validation A is worse than B by an interval that excludes zero on tech validation pairs or on
development or pre-cutoff launch pair accuracy (then B for the tech audience, and the report says so), or A is no better
than C on broad validation pairs (then the embedding adds nothing on broad data, and the report says so).

## Step 2: pair-loss fine-tune

Recipe exactly as the replicate track: `scripts/train_pairs.py`, 600 steps, learning rate 2e-4, warm-up 50, weight 10
(treatment) and 0 (control), seed 0, the pairs track's pair table re-embedded. Applied to both seeds of the chosen pair.
Kept only if, on the development pairs (same-topic) with the confidence scale cross-fitted over accounts,
correct confident calls per 100 pairs 2x apart rise against the weight-0 control with an interval excluding zero, and
accuracy on all pairs does not fall (point estimate not below -0.5 points and interval including zero); and on the
pre-cutoff launch pairs accuracy does not fall by the same test. One model set serves both kinds of comparison, so a
gain on same-topic pairs bought with a loss on launch pairs is not kept.

## Step 3: rules for the chosen set (pre-cutoff pairs only)

- Confidence scales: one per kind, maximum likelihood on all pre-cutoff pairs 1.25x apart of that kind
  (`sel_models.fit_scale`). Reported next to a time-split check (fitted on the fit cell, read on the validation cell).
- Post-type level correction: the posttype track's ridge fit (ridge 20, seven type columns) on own announcements before
  the cutoff; kept if the mean error on held-out accounts moves toward zero.
- Judge blend rule: `scripts/judge_product/fit_rule.py` unchanged (clip by validation log-loss, speak rule by its fixed
  criterion, weights refitted on all clean development pairs), run on the new surrogate's gap. No new judge calls.
- Final report: `scripts/judge_product/test_eval.py` unchanged, once, for the new set; the same script on the current
  product's gap reproduces the judge_product report and is the comparison.

## Leave-out rule

A step that does not improve its validation measure is reported and left out of the bundle.
