# Track "selective": settings and decision rules, written before any deferral model was fitted or scored

Written 2026-11-19, after rescoring the benchmark pairs with full member outputs (scores only reproduce the stored
ones; no accuracy, confidence or deferral number had been computed on the benchmark or the launch pairs by this track).

## Sets

- **Training and validation (all before 2026-04-23):** the replicate track's 2,805 development pairs, rescored through
  the product's serving path (56 posts of history, median of the like histogram), and pre-cutoff launch pairs built
  like the launch track's test pairs (same account, both launch-type, likes 1.25x apart, at most 30 per account,
  shuffle seed 1) from launch-type posts before the cutoff. Validation is 5-fold over accounts, repeated 3 times.
- **Test, scored once per final configuration:** the 3,530 benchmark pairs (both halves; the 1,500 "calibration"
  pairs are not trained on in the primary configuration) and the 850 launch pairs.

## The call and what the deferral model predicts

- The call is the sign of the ensemble's mean score gap (log10 of predicted median likes). Ensembles: fine-tuned ten
  (ft10) for same-topic pairs, untouched ten (start10) for launch pairs, as the replicate track recommended; the
  baseline three is reported next to them with the logistic rule.
- Target: the call is right. Pairs under 1.25x are unused. Weights, one of: flat; proportional to the log like
  ratio capped at 5x; pairs 2x apart only. Chosen on validation.
- Inputs: only what exists before posting, and nothing that tells which version was posted first. Features are
  oriented by the call (picked version, other version), never by posting order. For launch pairs (two posts scored
  at their own times) account and history features enter only as the mean of the two posts.

## Methods compared

1. Current rule: one logistic scale on the size of the ensemble's gap, fitted on the training pairs.
2. Regularised logistic regression on a short list of features (linear).
3. Gradient-boosted trees (LightGBM), shallow and strongly regularised, monotone in the gap, Platt-calibrated on out-of-fold predictions.
4. Small neural network (one hidden layer, five seeds averaged), calibrated the same way.
5. Risk control on top of the best score: threshold chosen on the training pairs so that the error rate among spoken
   2x pairs is at most 25% with a finite-sample guarantee (fixed-sequence binomial test, delta 0.1), and the plain
   conformal-risk-control threshold on the monotone loss "spoken and wrong".

## Selection rule

- Validation metric: mean accuracy of the surest 20, 30, 40 and 50% of validation pairs 2x apart (out-of-fold,
  ranked within each repeat), i.e. the area under the risk-coverage curve over the range the goal cares about.
- Hyperparameters and the weighting are chosen by that metric. The "deferral model" is the best of methods 2 to 4.
- It replaces the logistic rule in the product only if its paired difference against the logistic rule on the
  validation metric has an account-resampled 95% interval above zero. Whatever the outcome, the chosen deferral
  model and the logistic rule are both scored on the test sets once, and both are reported.
- Per pair type: a separate model (or scale) for launch pairs is compared on launch validation pairs with the pooled
  one; the separate one is used only if it is better there.

## Reporting

Accuracy at 20, 30, 40, 50% coverage (surest pairs first), for all pairs 1.25x apart and for pairs 2x apart, with
95% intervals from resampling accounts; the same at the product's threshold (stated probability 60%+); by benchmark
half, by follower tier, and for benchmark accounts with and without development pairs. Goal check: at 40% coverage of
2x pairs, is accuracy 75% or more.

## Declared secondary configuration

"More same-type data": the same method trained on development pairs plus the benchmark's 1,500 pre-cutoff
calibration pairs, tested on the 2,030 post-cutoff test pairs only. It answers whether more pairs of the benchmark's
own kind would help; it is reported separately and is not the headline.
