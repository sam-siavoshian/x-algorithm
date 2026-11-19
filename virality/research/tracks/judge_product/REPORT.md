# Track "judge_product": the reading model, hardened and shipped

Worker report, 2026-11-19. All numbers are likes. Intervals are 95%, from resampling accounts. "Surrogate" is the product's
forecasting ensemble (the baseline three). "Judge" is a language model that reads the two posts and 30 of the account's
earlier posts with their like counts, asked in both orders. "Blend" is the fitted rule that combines the two.
Every choice (weights, clip, speak rule, history length, prompt) was made on pairs from before 2026-04-23; the benchmark
and the launch test pairs were scored once per final configuration, with the rule frozen in a file beforehand.
Spend: 22.09 US dollars on OpenAI (limit 30), 5.05 on the Publik endpoint (limit 20).

## Headline

1. **With `gpt-6.1-sol` as second opinion the project's goal is met on launch pairs with room to spare and reached at its lower edge on the benchmark
   once every flattering pair is removed.** Goal: confident calls 75 to 80% right on pairs 2x apart, covering at least 40% of them, overall accuracy up.
   One fixed rule per kind of pair, fitted on development pairs, nothing rescaled on a test set:

   | Test set (pairs 2x apart) | Pairs | Surrogate alone: all calls; speaks on; right when it speaks | Surrogate + `gpt-6-luna` | Surrogate + `gpt-6.1-sol` |
   |---|---|---|---|---|
   | Benchmark, all | 2,030 | 61.4%; 24%; 70.3% (64.9 to 75.4) | 67.8%; 41%; 77.0% (72.8 to 81.4) | 71.1%; 50%; 79.6% (76.5 to 82.7) |
   | Benchmark, near-identical pairs removed | 1,647 | 61.6%; 28%; 69.1% | 65.5%; 44%; 74.3% (69.7 to 78.6) | 69.6%; 55%; 77.7% (74.1 to 81.0) |
   | Benchmark, near-identical removed and both posts after the judge's knowledge cutoff | 890 (luna), 953 (sol) | 60.7%; 31%; 66.2% / 61.6%; 31%; 67.3% | 63.8%; 44%; 70.6% (65.1 to 75.5) | **67.7%; 58%; 76.1% (72.3 to 79.7)** |
   | Launch pairs, all (every pair is after 2026-04-23) | 568 | 66.5%; 35%; 78.2% (70.3 to 86.1) | 76.8%; 86%; 79.2% (74.2 to 83.6) | 80.8%; 89%; 84.2% (79.6 to 88.3) |
   | Launch pairs, near-identical removed and after the judge's cutoff | 442 (luna), 511 (sol) | 67.6%; 30%; 80.6% / 67.5%; 32%; 81.0% | 77.1%; 86%; 80.1% (75.0 to 84.7) | **82.8%; 91%; 85.3% (81.1 to 88.9)** |

   | Benchmark, as the row above and both posts with 3+ likes (10+ likes) | 673 (521) with sol | 59.1%; 37%; 63.5% (58.7%; 40%; 63.0%) | 630 pairs: 61.7%; 46%; 68.2% | **67.9%; 62%; 74.2% (69.4 to 78.7)** (67.6%; 62%; 73.1%, 67.3 to 78.7) |
   | Launch, as the launch row above and both posts with 3+ likes | 487 with sol | 68.0%; 31%; 81.9% | | 83.6%; 92%; 85.7% (81.4 to 89.2) |

   Read the benchmark rows downward: each removes something that flatters the number (corrected reposts, posts the model may know, pairs with a post at 0 to 2 likes). On the strictest row the sol blend
   speaks on 62% of pairs 2x apart and is right 74.2% of the time, against 37% at 63.5% for the surrogate: the coverage half of the goal is passed by a wide margin and the accuracy half sits at its lower edge (73 to 76% depending on the row; intervals 67 to 80).
   On launch pairs every row is at 84 to 86%.

   Overall accuracy (pairs 1.25x apart): benchmark 60.0% to 68.2% with sol (+8.2, +5.8 to +10.4; 2,852 pairs), 64.4% with luna (+4.5, +2.3 to +6.6);
   launch pairs 63.9% to 75.6% with sol (+11.8, +6.5 to +17.1; 850 pairs), 71.2% with luna (+7.3, +2.2 to +12.1).
   On the clean, post-cutoff benchmark slice: sol 60.6% to 66.1% (+5.5, +2.7 to +8.2; 1,478 pairs); luna 59.9% to 61.5% (+1.5, -0.9 to +3.8; 1,383 pairs).

2. **The earlier sample result for the stronger model replicates on the pairs it had not seen.** On the 1,771 benchmark pairs scored here for the first time
   `gpt-6.1-sol` alone is right 68.2% against 58.8% for the surrogate (+9.4, +6.2 to +12.5); on the 1,081 scored by the earlier track it was 66.5% against 61.9%.
   Launch: 75.6% against 60.7% on the 328 new pairs (+14.9, +7.7 to +22.7); 74.0% against 65.9% on the earlier 522.

3. **The cheap model is not enough once the two artefacts are removed; the strong one is.** `gpt-6-luna` adds 1.5 points on the clean post-cutoff benchmark
   slice and its confident calls are 70.6% right there (the surrogate alone: 66.2% on fewer pairs). On launch pairs it widens coverage (30% to 86% of 2x pairs) at the
   surrogate's accuracy. `gpt-6.1-sol` adds accuracy everywhere it was measured.

4. **No open model on the Publik endpoint matches `gpt-6.1-sol`; two match `gpt-6-luna`.** Development validation pairs (later period, held-out accounts), judge alone,
   same pairs: launch `sol` 74.9 to 75.3%, `Qwen3.5-397B` 66.2%, `DeepSeek-V4.1-Flash` 67.1%, `luna` 64.1%, surrogate 68.5%; same-topic `Qwen` 63.6%, `DeepSeek` 59.7%, `sol` 59.1 to 60.7%, `luna` 58.3 to 59.1%.
   Neither is cheaper than luna (0.39 and 2.09 dollars per 1,000 comparisons against 0.29). `gpt-oss-120b` costs 0.27 but takes about 40 seconds per call. `Qwen3.5`, whose weights were published before every test pair, was scored on the test sets as a leakage-free check: it adds +2.7 (-1.1 to +6.8) on 700 benchmark pairs and +4.2 (-0.3 to +8.4) on the 850 launch pairs, luna's size of gain.

5. **More history and a prompt that states the measured launch evidence do not help.** `gpt-6.1-sol`, same pairs: 30 posts 67.2% (same-topic) and 80.2% (launch);
   60 posts -0.7 (-3.6 to +1.9) and +1.2 (-1.1 to +3.2); 100 posts -0.2 and -0.5; evidence in the prompt -0.8 (-3.1 to +1.5). Fewer does hurt: 10 posts -3.6, none -6.8.

6. **It ships.** `xviral/judge.py` (judge class for the OpenAI and Publik endpoints, stub, blend rule), `xviral/data/judge_rule.json`, and the advisor, service,
   server, page and API reference changed so that every comparison of two versions uses surrogate plus judge when `XVIRAL_JUDGE_MODEL` is set and the surrogate alone
   otherwise or on any failure. The report names the provider and the model, says what was sent, which of the two opinions preferred which version, and states the
   accuracy measured for the configuration in use. Opt-out per request; uploaded posts are never sent without the request's permission.
   14 new tests, the 19 existing ones pass. Real run through the HTTP layer on three corpus accounts: one pair adds 2.8 to 3.7 seconds and 0.008 to 0.013 dollars with
   `gpt-6.1-sol` (0.4 to 1.0 seconds and 0.0004 to 0.0006 dollars with `gpt-6-luna`); six versions (30 calls) add 7.6 seconds and 0.13 dollars (1.6 seconds, 0.006 dollars).

## A. Evaluation

Code: `scripts/judge_product/` (`prep.py`, `run.py`, `jp.py`, `fit_rule.py`, `test_eval.py`, `extras.py`, `e2e.py`). The prompts are built by `xviral/judge.py`,
the module the product uses; they are character for character the earlier track's (checked on 1,136 requests).

### Data and split

| Set | Role | Pairs 1.25x apart | Near-identical (text similarity 0.9+) | Accounts |
|---|---|---|---|---|
| Development pairs, same-topic (replicate track; before 2026-04-23) | choices and fitting, same-topic rule | 2,805 | 392 (later copy wins 87%) | 366 |
| Launch pairs before the cutoff (selective track) | choices and fitting, launch rule | 3,655 | 88 | 628 |
| Benchmark | test, same-topic rule | 2,852 | 405 (later copy wins 95%) | 770 |
| Launch pairs after the cutoff (launch track) | test, launch rule | 850 | 5 | 151 |

Development pairs are split two ways at once. *Fit cell*: both posts before a date T and the account in hash half A. *Validation cell*: the later post on or after T and
the account in half B, so validation is later in time and on other accounts. T is 2025-10-01 for same-topic pairs (fit 692 pairs, validation 609) and 2026-01-01 for
launch pairs (1,020 and 858). Near-identical pairs are removed before anything is fitted or chosen.

The surrogate input is its gap in log10 predicted likes through the serving path (`tracks/selective/feat_*_base3.parquet`), the unit the product computes, for all four
sets; it reproduces the known accuracies (56.3%, 63.8%, 60.0%, 63.9%). The judge input is the order-averaged log-odds, clipped. Neither is rescaled per set.

Judge runs: `gpt-6-luna` on all development and test pairs; `gpt-6.1-sol` on 929 same-topic development pairs (809 clean; 335 in the validation cell), 600 launch
development pairs (300 per cell), and now every test pair; the open models on 300 + 300 validation-cell pairs.

### 1. `gpt-6.1-sol` on the pairs it had not seen

The earlier track scored a random 1,081 benchmark pairs and 522 launch pairs. The remaining 1,771 and 328 were scored here (three batches, nine minutes, 9.38 dollars).
Judge alone, no rule:

| Set | Pairs | Count | Surrogate | `gpt-6.1-sol` alone | Difference |
|---|---|---|---|---|---|
| Benchmark | scored by the earlier track | 1,081 | 61.9% | 66.5% (63.0 to 69.2) | +4.6 (+1.0 to +7.9) |
| Benchmark | scored here | 1,771 | 58.8% | 68.2% (65.5 to 70.9) | +9.4 (+6.2 to +12.5) |
| Benchmark | all | 2,852 | 60.0% | 67.6% (65.2 to 69.8) | +7.6 (+5.0 to +10.1) |
| Launch | scored by the earlier track | 522 | 65.9% | 74.0% (68.6 to 79.0) | +8.1 (+1.1 to +14.1) |
| Launch | scored here | 328 | 60.7% | 75.6% (69.4 to 81.7) | +14.9 (+7.7 to +22.7) |
| Launch | all | 850 | 63.9% | 74.6% (69.5 to 79.5) | +10.8 (+4.9 to +16.6) |

### 2. Open models on the Publik endpoint

Same prompt, same 30 posts of history, both orders. `DeepSeek-V4.1-Flash` and `Qwen3.5-397B` return log-probabilities for a one-letter answer with reasoning off
(the mode `gpt-6-luna` uses); `gpt-oss-120b` always reasons and returns no log-probabilities, so it answers `{"p_a": x}` like `gpt-6.1-sol`. A 25-pair cost and latency test preceded each run.
Development validation cell, near-identical removed, pairs that `gpt-6.1-sol` also scored (`jp_extras.md`):

| Kind | Model | Pairs | Judge alone | Same pairs: sol | luna | Surrogate | Blend with the surrogate, fitted on half the accounts and read on the other | Minus sol | Minus luna |
|---|---|---|---|---|---|---|---|---|---|
| Same-topic | `Qwen/Qwen3.5-397B-A17B` | 258 | 63.6% (56.6 to 71.5) | 60.7% | 58.3% | 56.6% | 64.0% | +2.9 (-4.5 to +11.0) | +5.2 (-0.0 to +10.8) |
| Same-topic | `deepseek-ai/DeepSeek-V4.1-Flash` | 236 | 59.7% (54.1 to 65.8) | 59.1% | 59.1% | 55.1% | 50.0% | +0.6 (-5.0 to +7.3) | +0.6 (-4.4 to +6.3) |
| Same-topic | `gpt-6-luna` | 335 | 58.4% (52.1 to 64.2) | 59.1% | | 54.0% | 58.2% | -0.7 (-6.8 to +5.2) | |
| Same-topic | `gpt-6.1-sol` | 335 | 59.1% (53.6 to 64.8) | | 58.4% | 54.0% | 59.7% | | +0.7 (-5.2 to +6.8) |
| Launch | `Qwen/Qwen3.5-397B-A17B` | 293 | 66.2% (60.4 to 71.4) | 74.9% | 63.8% | 67.9% | 71.3% | -8.7 (-15.6 to -2.4) | +2.4 (-2.6 to +8.0) |
| Launch | `deepseek-ai/DeepSeek-V4.1-Flash` | 298 | 67.1% (61.4 to 72.2) | 75.3% | 64.1% | 68.5% | 68.5% | -8.2 (-13.4 to -3.1) | +3.0 (-3.2 to +9.0) |
| Launch | `gpt-6-luna` | 298 | 64.1% (58.0 to 70.7) | 75.3% | | 68.5% | 71.1% | -11.2 (-18.3 to -5.1) | |
| Launch | `gpt-6.1-sol` | 298 | 75.3% (69.1 to 81.5) | | 64.1% | 68.5% | 76.8% | | +11.2 (+5.1 to +18.3) |
| Same-topic | `openai/gpt-oss-120b` | 140 | 55.7% (46.9 to 63.7) | 60.0% | 59.3% | 55.7% | 52.9% | -4.3 (-11.6 to +3.4) | -3.6 (-12.9 to +6.9) |
| Launch | `openai/gpt-oss-120b` | 148 | 59.8% (52.2 to 67.0) | 72.3% | 61.5% | 72.3% | 77.0% | -12.5 (-21.4 to -3.7) | -1.7 (-10.2 to +7.7) |

Cost, speed and what each model can know (`jp_extras.md`; price lists of 2026-11-19; one comparison = both orders with 30 posts of history):

| Model | Endpoint | US dollars per 1,000 comparisons | Median seconds per call (90th percentile) | Knowledge cutoff | Source |
|---|---|---|---|---|---|
| `gpt-6-luna` | OpenAI | 0.29 | 0.8 (1.8) | 2026-05-18 | OpenAI model page (earlier track) |
| `gpt-6.1-sol` | OpenAI | 7.85 (3.9 through the batch interface) | about 3 for a pair in the product (end-to-end run) | 2026-04-30 | OpenAI model page (earlier track) |
| `openai/gpt-oss-120b` | Publik | 0.27 | 38.5 (55.2) | 2024-06 | stated in the model's chat template on Hugging Face; weights published 2025-08-04 |
| `deepseek-ai/DeepSeek-V4.1-Flash` | Publik | 0.39 | 1.2 (5.6) | not stated; weights published 2026-09-10 | Hugging Face model card and repository date |
| `Qwen/Qwen3.5-397B-A17B` | Publik | 2.09 | 1.2 (5.6) | not stated; weights published 2026-02-16, which bounds it | Hugging Face model card and repository date |

Reading. Against the brief's test ("matches luna or sol at lower cost"): `DeepSeek-V4.1-Flash` matches luna within noise at 1.3 times its price; `Qwen3.5` matches luna and may be
better on same-topic pairs (+5.2, interval touches zero) at 7 times its price and a quarter of sol's; neither matches sol on launch pairs (-8). `gpt-oss-120b` is the cheapest
and far too slow for a request. So no open model replaces either OpenAI model in the product on these numbers. `DeepSeek`'s weights are dated after most test pairs, so a
test-set score for it would carry the leakage doubt at full strength; it was not run. The Publik key allows 60 requests a minute in total, which also rules it out for a
product request of six versions (30 calls) under any load.

**A model that cannot have seen the test posts.** `Qwen3.5-397B`'s weights were published on 2026-02-16, before the earliest test post (2026-04-23), so nothing it says about a test pair can be memory.
Its rule was fitted on its 300 + 300 development validation pairs and frozen (`judge_rule_fitted_frozen_with_open_models.json`) before its test answers were read. Scored once on all 850 launch pairs and a random 700 of the
1,660 benchmark pairs with both posts on or after 2026-04-23 (the key's rate limit set the sample size):

| Test set | Real gap at least | Pairs | Surrogate: correct; speaks on; right | Qwen alone | Blend: correct; speaks on; right | Blend minus surrogate |
|---|---|---|---|---|---|---|
| Benchmark sample | 1.25x | 700 | 62.3%; 32%; 65.6% | 64.6% (61.0 to 68.2) | 65.0%; 56%; 72.1% (67.1 to 76.6) | +2.7 (-1.1 to +6.8) |
| Benchmark sample | 2x | 462 | 63.6%; 33%; 66.4% | 69.3% | 69.3%; 58%; 76.3% (69.7 to 82.2) | +5.6 (+0.2 to +10.9) |
| Launch | 1.25x | 850 | 63.9%; 32%; 74.5% | 63.8% (59.4 to 67.9) | 68.1%; 72%; 72.7% (67.4 to 77.9) | +4.2 (-0.3 to +8.4) |
| Launch | 2x | 568 | 66.5%; 35%; 78.2% | 68.7% | 73.2%; 74%; 78.5% (71.7 to 84.4) | +6.7 (+1.1 to +11.7) |

So a reader with no possible knowledge of the outcomes still adds to the surrogate, by about what `gpt-6-luna` adds (launch +7.3, clean benchmark +1.5 to +2.2), and far less than `gpt-6.1-sol` (+11.8 and +5.5 to +6.7).
That bounds the leakage worry from one side: the direction of the gain does not depend on memory. It does not prove that sol's larger gain is free of it; for that the after-cutoff slices in section 3 are the evidence.
Because it was scored, `Qwen/Qwen3.5-397B-A17B` is in the shipped rule file with its own validated statement and can be run through the Publik endpoint; it is not the recommended configuration.

### 3. Leakage

**Near-identical pairs.** 405 of the 2,852 benchmark pairs are corrected reposts (the later copy wins 95%). Every table in `jp_test.md` is issued for all pairs, without them,
and for them alone. The blend gains 17 to 18 points on those 405 pairs with either judge and 2.2 (luna) or 6.7 (sol) on the other 2,447, so the plain benchmark numbers overstate
what a user can act on. Baseline table re-issued with the README's method (confidence scale cross-fitted inside the slice; `jp_extras.md` has ten models and 3x, 5x as well):

| Slice | Real gap at least | Pairs | Surrogate correct | Confident (60%+) share | Correct there |
|---|---|---|---|---|---|
| All pairs (the README's table) | 1.25x | 2,852 | 60.0% (57.9 to 62.0) | 27% (777) | 67.4% |
| All pairs | 2x | 2,030 | 61.4% (58.7 to 64.1) | 26% (521) | 70.2% |
| Without the 405 near-identical pairs | 1.25x | 2,447 | 60.0% (58.1 to 62.0) | 28% (692) | 66.6% (62.6 to 70.5) |
| Without the 405 near-identical pairs | 2x | 1,647 | 61.6% (58.9 to 64.0) | 29% (470) | 68.9% (63.8 to 74.1) |
| Wording differs, both posts on or after 2026-04-23 | 2x | 982 | 61.7% (58.4 to 64.6) | 30% (296) | 66.6% (61.2 to 72.1) |
| Wording differs, after 2026-04-30 (sol's cutoff) | 2x | 953 | 61.6% (58.4 to 64.9) | 27% (254) | 65.7% (60.2 to 72.6) |
| Wording differs, after 2026-05-18 (luna's cutoff) | 2x | 890 | 60.7% (57.2 to 64.1) | 25% (226) | 65.0% (58.6 to 71.6) |
| Wording differs, both posts before 2026-04-23 | 2x | 665 | 61.5% (57.5 to 65.8) | 27% (178) | 73.0% (65.0 to 79.5) |

The surrogate's own numbers barely move when the reposts go (it cannot see which copy was corrected); its confident calls are 4 to 7 points worse after the cutoff than before, as the selective track found.

**Knowledge cutoff.** Benchmark, fixed rule, pairs 2x apart, wording differs (`jp_test.md` has every gap):

| Judge | Slice | Pairs | Surrogate: all calls; speaks on; right | Blend: all calls; speaks on; right | Blend minus surrogate, all calls |
|---|---|---|---|---|---|
| luna | before 2026-05-18 | 757 | 62.7%; 25%; 73.4% | 67.4%; 43%; 78.7% (73.7 to 83.2) | +4.6 (+0.9 to +7.7) |
| luna | after 2026-05-18 | 890 | 60.7%; 31%; 66.2% | 63.8%; 44%; 70.6% (65.1 to 75.5) | +3.1 (+0.1 to +6.4) |
| sol | before 2026-04-30 | 694 | 61.7%; 24%; 72.3% | 72.3%; 51%; 80.2% (75.1 to 84.2) | +10.7 (+5.9 to +14.9) |
| sol | after 2026-04-30 | 953 | 61.6%; 31%; 67.3% | 67.7%; 58%; 76.1% (72.3 to 79.7) | +6.1 (+2.8 to +9.6) |

The gain is smaller after each cutoff (sol +10.7 before, +6.1 after), and so is the surrogate's own accuracy on confident calls, so the two halves differ for a reader of any kind.
The product only ever sees posts written after the cutoff, so the "after" rows are the ones it may quote, and the rule file quotes those.

**Launch pairs, in numbers.** 850 pairs, earliest post 2026-04-23, latest 2026-11-06: every pair is after the test cutoff. Both posts are after `gpt-6.1-sol`'s cutoff in 777 pairs
and after `gpt-6-luna`'s in 663. 5 pairs are near-identical. Blend minus surrogate on all 850: sol +11.8, luna +7.3; on the clean post-cutoff slice: sol +12.7 (+6.6 to +18.5; 772 pairs), luna +6.5 (-0.1 to +12.2; 659 pairs).
The 73 launch pairs before sol's cutoff show no gain (63.0% against 63.0%), which is the opposite of what leakage would do.

### 4. More history; the measured launch evidence in the prompt

`gpt-6.1-sol`, development pairs, the same pairs in every row (`jp_extras.md`):

| Set | Variant | Pairs | Correct | 2x apart | Minus 30 posts |
|---|---|---|---|---|---|
| Same-topic (293 pairs of the fit half) | no history | 293 | 60.4% | 61.9% | -6.8 (-10.5 to -3.1) |
| Same-topic | 10 posts | 293 | 63.7% | 66.8% | -3.6 (-7.1 to -0.5) |
| Same-topic | 30 posts (scored on the test sets) | 293 | 67.2% (61.5 to 72.2) | 69.3% | |
| Same-topic | 60 posts | 293 | 66.6% | 70.8% | -0.7 (-3.6 to +1.9) |
| Same-topic | 100 posts | 293 | 67.1% | 71.1% | -0.2 (-3.4 to +2.7) |
| Launch before the cutoff (300 pairs of the fit cell) | 30 posts | 300 | 80.2% (74.7 to 85.7) | 87.7% | |
| Launch | 60 posts | 300 | 81.3% | 88.4% | +1.2 (-1.1 to +3.2) |
| Launch | 100 posts | 300 | 79.7% | 87.7% | -0.5 (-2.3 to +1.3) |
| Launch | 30 posts + the measured launch evidence (ten graded traits with win rates) in the prompt | 300 | 79.3% | 85.4% | -0.8 (-3.1 to +1.5) |

Among accounts that have 100 or more earlier posts the picture is the same (same-topic, 209 pairs: 65.1%, 64.1%, 64.4% for 30, 60, 100 posts). The gain from history saturates by 30 posts,
and the model does not need to be told the launch traits. The product keeps 30 posts and the original prompt, which is also the configuration every test number belongs to.
(The evidence table was measured on launch posts that include these development pairs, so that variant had every advantage.)

### 5. The product rule

Per kind and per judge: `p = sigmoid(w_s * gap + w_j * clip(judge log-odds, -c, +c))`, no intercept, fitted by logistic regression on pairs in both orientations.
The clip and the speak rule were chosen with the model fitted on the fit cell and read on the validation cell; the shipped weights are then refitted on every clean
development pair the judge scored (`fit_rule.py`, `jp_fit.md`). Speak-rule candidates: stated probability at or above 60, 65, 70%; 60% and the two opinions agree; agree alone;
60% and (agree, or the judge alone at 70%+). Criterion fixed before reading validation: among rules at 75% or better on validation pairs 2x apart, the one with the most
correct spoken calls per 100 such pairs (the plain 60% rule wins ties within 1); if none reaches 75%, the most accurate rule that speaks on 40% or more.

Train and validation side by side (model fitted on the fit cell; pairs 1.25x apart, and 2x apart in brackets):

| Kind, judge | | Fit cell (train) | Validation cell |
|---|---|---|---|
| Same-topic, luna (598 / 548 pairs) | surrogate | 60.5% [63.1%] | 53.8% [55.6%] |
| | judge | 60.5% [62.9%] | 59.2% [63.5%] |
| | blend | 62.2% [65.5%] | 60.0% (55.5 to 64.2) [64.1%] |
| Same-topic, sol (148 / 335) | surrogate | 56.8% [60.6%] | 54.0% [55.3%] |
| | judge | 62.5% [63.6%] | 59.1% [58.0%] |
| | blend | 59.5% [63.6%] | 57.9% (52.0 to 62.9) [58.7%] |
| Launch, luna (1,008 / 848) | surrogate | 65.6% [68.7%] | 67.6% [71.2%] |
| | judge | 70.7% [75.7%] | 61.9% [64.6%] |
| | blend | 71.6% [76.8%] | 68.0% (63.5 to 72.5) [72.0%] |
| Launch, sol (298 / 298) | surrogate | 67.1% [69.7%] | 68.5% [73.4%] |
| | judge | 80.0% [87.7%] | 75.3% [80.3%] |
| | blend | 80.5% [87.2%] | 76.2% (70.6 to 81.8) [81.7%] |

Speak rules on the validation cell, pairs 2x apart (share spoken; right when spoken):

| Kind, judge (pairs 2x apart) | Surrogate alone, 60% | Blend 60% | 65% | 70% | 60% and agree | Agree | 60% and (agree or judge 70%+) | Chosen |
|---|---|---|---|---|---|---|---|---|
| Same-topic, luna (340) | 29%; 66.0% | 58%; 67.2% | 43%; 67.6% | 30%; 70.9% | 46%; 68.2% (59.5 to 75.9) | 61%; 65.5% | 46%; 68.2% | 60% and agree |
| Same-topic, sol (206) | 29%; 62.7% | 50%; 62.7% | 37%; 65.8% | 22%; 73.9% | 43%; 64.0% | 64%; 59.8% | 45%; 64.1% (52.8 to 75.6) | 60% and (agree or judge 70%+) |
| Launch, luna (642) | 57%; 79.9% | 80%; 77.0% (71.7 to 82.2) | 69%; 79.4% | 58%; 81.9% | 56%; 83.2% | 59%; 80.6% | 72%; 77.1% | 60% |
| Launch, sol (218) | 56%; 82.0% | 90%; 84.3% (78.2 to 89.6) | 85%; 87.0% | 78%; 89.4% | 62%; 92.6% | 65%; 90.8% | 86%; 84.5% | 60% |

Shipped rule (`xviral/data/judge_rule.json`):

| Judge | Kind | Weight on the surrogate's gap (per log10) | Weight on the judge's log-odds | Clip | Speak | Fitted on |
|---|---|---|---|---|---|---|
| `gpt-6-luna` | same-topic | 2.017 | 0.042 | 14 | 60% and the two agree | 2,413 pairs, 355 accounts |
| `gpt-6-luna` | launch | 1.578 | 0.135 | 14 | 60% | 3,567 pairs, 616 accounts |
| `gpt-6.1-sol` | same-topic | 1.198 | 0.486 | 4 | 60% and (agree, or the judge alone at 70%+) | 809 pairs |
| `gpt-6.1-sol` | launch | 1.692 | 0.847 | 4 | 60% | 596 pairs |

(The luna weights look small because its log-odds run to plus or minus 14; sol's stay within 4.)

**Test, once** (`test_eval.py`, `jp_test.md`; the rule file was frozen as `judge_rule_fitted_frozen_luna_sol.json` first). Pairs with wording that differs, by size of the real gap:

| Test set, judge | Real gap at least | Pairs | Surrogate: correct | speaks on | right when it speaks | Blend: correct | speaks on | right when it speaks | Blend minus surrogate |
|---|---|---|---|---|---|---|---|---|---|
| Benchmark, sol | 1.25x | 2,447 | 60.0% | 28% | 66.7% | 66.7% (64.2 to 68.9) | 53% | 74.0% (71.0 to 76.5) | +6.7 (+4.5 to +8.8) |
| Benchmark, sol | 2x | 1,647 | 61.6% | 28% | 69.1% | 69.6% (66.6 to 72.7) | 55% (52 to 58) | 77.7% (74.1 to 81.0) | +8.0 (+5.2 to +10.6) |
| Benchmark, sol | 3x | 1,044 | 64.2% | 31% | 73.0% | 73.1% | 58% | 81.7% (77.8 to 85.1) | +8.9 (+5.7 to +12.0) |
| Benchmark, sol | 5x | 603 | 68.7% | 36% | 80.0% | 76.3% | 59% | 84.9% (80.3 to 88.8) | +7.6 (+3.8 to +11.3) |
| Benchmark, sol, after its cutoff | 1.25x | 1,478 | 60.6% | 31% | 64.9% | 66.1% (63.7 to 68.5) | 55% | 73.3% (70.6 to 76.2) | +5.5 (+2.7 to +8.2) |
| Benchmark, sol, after its cutoff | 2x | 953 | 61.6% | 31% | 67.3% | 67.7% (64.3 to 70.9) | 58% (55 to 61) | 76.1% (72.3 to 79.7) | +6.1 (+2.8 to +9.6) |
| Benchmark, sol, after its cutoff | 3x | 589 | 64.7% | 36% | 70.1% | 72.5% | 63% | 80.5% (75.6 to 85.2) | +7.8 (+4.0 to +12.1) |
| Benchmark, sol, after its cutoff | 5x | 323 | 70.3% | 42% | 77.8% | 74.0% | 65% | 82.9% (77.0 to 88.3) | +3.7 (-2.3 to +10.0) |
| Benchmark, luna | 1.25x | 2,447 | 60.0% | 28% | 66.7% | 62.2% (60.2 to 64.1) | 43% | 70.5% (67.4 to 73.4) | +2.2 (+0.5 to +3.9) |
| Benchmark, luna | 2x | 1,647 | 61.6% | 28% | 69.1% | 65.5% | 44% (41 to 47) | 74.3% (69.7 to 78.6) | +3.8 (+1.6 to +6.0) |
| Benchmark, luna, after its cutoff | 1.25x | 1,383 | 59.9% | 31% | 64.3% | 61.5% (59.2 to 63.9) | 43% | 67.9% (64.1 to 71.6) | +1.5 (-0.9 to +3.8) |
| Benchmark, luna, after its cutoff | 2x | 890 | 60.7% | 31% | 66.2% | 63.8% | 44% (41 to 49) | 70.6% (65.1 to 75.5) | +3.1 (+0.1 to +6.4) |
| Launch, sol | 1.25x | 845 | 64.1% | 32% | 74.5% | 75.7% (70.8 to 80.5) | 88% | 78.5% (73.4 to 83.0) | +11.6 (+6.1 to +17.2) |
| Launch, sol | 2x | 563 | 67.0% | 35% | 78.2% | 81.0% (75.7 to 85.5) | 90% (86 to 93) | 84.1% (79.0 to 88.3) | +14.0 (+8.5 to +19.8) |
| Launch, sol | 3x | 396 | 68.9% | 37% | 78.2% | 84.3% | 91% | 86.7% (81.3 to 91.2) | +15.4 (+9.6 to +21.4) |
| Launch, sol | 5x | 236 | 70.8% | 40% | 73.7% | 86.4% | 92% | 88.1% (80.8 to 94.3) | +15.7 (+8.7 to +23.4) |
| Launch, sol, after its cutoff | 2x | 511 | 67.5% | 32% | 81.0% | 82.8% (78.8 to 86.6) | 91% (87 to 94) | 85.3% (81.1 to 88.9) | +15.3 (+9.6 to +20.8) |
| Launch, sol, posts within 30 days of each other | 2x | 188 | 63.8% | 31% | 72.4% | 81.4% (74.0 to 88.1) | 87% (82 to 92) | 86.0% (78.7 to 92.5) | +17.6 (+9.4 to +25.0) |
| Launch, luna | 1.25x | 845 | 64.1% | 32% | 74.5% | 71.4% (67.2 to 75.2) | 84% | 73.7% (69.3 to 77.9) | +7.2 (+1.8 to +12.1) |
| Launch, luna | 2x | 563 | 67.0% | 35% | 78.2% | 77.1% | 87% (84 to 89) | 79.5% (74.3 to 84.1) | +10.1 (+4.6 to +15.4) |
| Launch, luna, after its cutoff | 2x | 442 | 67.6% | 30% | 80.6% | 77.1% (72.1 to 81.6) | 86% (83 to 90) | 80.1% (75.0 to 84.7) | +9.5 (+2.3 to +15.8) |
| Launch, luna, posts within 30 days of each other | 2x | 188 | 63.8% | 31% | 72.4% | 73.4% | 83% | 78.2% (69.9 to 85.7) | +9.6 (+1.0 to +17.3) |

Other slices (wording differs, 2x apart, sol; `jp_test.md`): benchmark under 50k followers 53% of 1,304 pairs at 77.7%, 50k and over 66% of 297 at 79.0%; launch under 50k followers 85% of 160 at 83.8%,
50k and over 92% of 307 at 85.5%; launch with under 30 earlier posts 91% of 93 at 83.5% (surrogate alone 29% at 59.3%). Luna on launch pairs under 50k followers gains nothing significant (+1.8, -9.2 to +11.2 on 220 pairs), as the earlier track saw.

**Split by like count** (asked for by the coordinating session: 944 of the benchmark's 2,030 pairs 2x apart contain a post with 0 to 2 likes, where the surrogate's confident calls are far more often right). Wording differs; `jp_test.md` has every gap and the launch rows:

| Benchmark slice, judge | Real gap at least | Pairs | Surrogate: correct; speaks on; right | Blend: correct; speaks on; right when it speaks | Blend minus surrogate |
|---|---|---|---|---|---|
| A post with 0 to 2 likes, sol | 2x | 613 | 64.4%; 13%; 87.8% | 69.5%; 44%; 82.0% (76.6 to 86.7) | +5.1 (-0.3 to +10.2) |
| Both posts 3+ likes, sol | 1.25x | 1,725 | 59.1%; 35%; 63.8% | 66.3%; 57%; 72.5% (69.5 to 75.5) | +7.1 (+4.5 to +9.4) |
| Both posts 3+ likes, sol | 2x | 1,034 | 60.0%; 37%; 65.1% | 69.7%; 61%; 75.9% (71.8 to 79.9) | +9.8 (+6.5 to +12.8) |
| Both posts 10+ likes, sol | 2x | 755 | 59.3%; 41%; 64.2% | 69.4%; 61%; 75.3% (69.7 to 79.8) | +10.1 (+6.1 to +13.5) |
| Both posts 3+ likes, after sol's cutoff | 2x | 673 | 59.1%; 37%; 63.5% | 67.9%; 62%; 74.2% (69.4 to 78.7) | +8.8 (+4.8 to +12.7) |
| Both posts 10+ likes, after sol's cutoff | 2x | 521 | 58.7%; 40%; 63.0% | 67.6%; 62%; 73.1% (67.3 to 78.7) | +8.8 (+4.1 to +13.5) |
| A post with 0 to 2 likes, luna | 2x | 613 | 64.4%; 13%; 87.8% | 68.2%; 39%; 79.7% | +3.8 (-0.4 to +7.7) |
| Both posts 3+ likes, luna | 2x | 1,034 | 60.0%; 37%; 65.1% | 63.8%; 47%; 71.6% (66.2 to 77.2) | +3.9 (+1.3 to +6.5) |
| Both posts 10+ likes, luna | 2x | 755 | 59.3%; 41%; 64.2% | 63.3%; 48%; 69.6% | +4.0 (+0.9 to +6.8) |
| Both posts 3+ likes, after luna's cutoff | 2x | 630 | 58.4%; 37%; 62.4% | 61.7%; 46%; 68.2% (62.1 to 74.2) | +3.3 (-0.1 to +7.1) |
| Both posts 10+ likes, after luna's cutoff | 2x | 491 | 58.2%; 40%; 62.1% | 60.5%; 48%; 67.4% | +2.2 (-1.5 to +6.9) |

The judge's gain is not carried by near-zero posts: with sol it is larger where both posts have 3 or more likes (+9.8 at 2x) than where one has 0 to 2 (+5.1). What the near-zero pairs do carry is the level:
confident calls are 82% right there and 74 to 76% elsewhere. Launch pairs hold few such pairs (38 of 563 at 2x); with both posts at 3+ likes and after sol's cutoff the blend speaks on 92% of 487 pairs at 85.7% (81.4 to 89.2); at 10+ likes, all dates, 91% of 502 at 85.1%; luna 86% of 502 at 80.8%.

**Which opinions agreed** (test, wording differs, pairs 1.25x apart):

| Test set, judge | Same version preferred | Right when they agree | When they disagree: surrogate right, judge right, blend right | Rule speaks on agreeing pairs (right) | on disagreeing pairs (right) |
|---|---|---|---|---|---|
| Benchmark, luna | 62% | 67.0% (64.4 to 69.6) | 48.0%, 52.0%, 54.0% | 69% (70.5%) | 0% by rule |
| Benchmark, sol | 63% | 70.3% (67.4 to 73.1); 73.7% at 2x | 42.3%, 57.7%, 60.9% | 78% (74.5%) | 11% (67.7%) |
| Launch, luna | 57% | 80.3% (75.6 to 84.8); 86.0% at 2x | 41.2%, 58.8%, 58.8% | 94% (80.9%) | 73% (60.7%) |
| Launch, sol | 59% | 82.6% (77.5 to 87.2); 87.5% at 2x | 36.8%, 63.2%, 65.9% | 97% (83.1%) | 76% (69.9%) |

On launch pairs the rule also speaks when the two disagree, and those calls are right 61% (luna) and 70% (sol) of the time. The product report says for every pair
whether they agreed, so a reader can discount a verdict the surrogate did not share.

**Stated probability against outcome** (test, wording differs): benchmark, sol: 50 to 55% stated: 54% right of 369; 55 to 60%: 58% of 547; 60 to 70%: 69% of 1,004; 70 to 80%: 79% of 416; 80 to 90%: 84% of 106.
Launch, sol: 60 to 70%: 62% of 136; 70 to 80%: 67% of 200; 80 to 90%: 88% of 218; 90%+: 92% of 189. Launch, luna: 70 to 80%: 69% of 181; 80 to 90%: 78% of 307; 90%+: 88% of 95.
The benchmark is close to the diagonal; on launch pairs the 70 to 80% band is a few points optimistic for both judges.

Development validation did not predict the same-topic test result: the chosen sol rule gave 45% coverage at 64% on validation pairs 2x apart and gives 55 to 58% at 76 to 78% on the benchmark.
Development pairs are up to 14 days apart and the validation cell is the surrogate's hardest (53.8%); benchmark pairs are at most 48 hours apart. The selective track saw the same direction
of difference for the surrogate alone. The launch rule's validation (90% at 84.3%) and test (89 to 91% at 84 to 85%) agree.

## B. Product

### 6. `xviral/judge.py` and the integration

- `Judge(model, endpoint=None, ...)`: one OpenAI-compatible chat call per order of a pair; `compare_many(account, history, pairs)` runs every call of a batch in parallel
  threads and returns per pair the order-averaged log-odds, the two raw answers, tokens, dollars and seconds. Endpoints `openai` (`api.openai.com`, `OPENAI_API_KEY`) and
  `publik` (`publikhq.com/api/v1`, `PUBLIK_API_KEY`); either key may be injected by a proxy. The model table holds each model's mode (log-probabilities or JSON), reasoning
  setting and price; an unknown model defaults to JSON mode. Per-call timeout (45 s), retries on 408, 409, 429 and 5xx and on connection errors with backoff, none on a
  refused request. `judge.spend` accumulates requests, tokens and dollars.
- `StubJudge`: never leaves the process; deterministic; has a position bias that asking both orders removes; can be given a function, a delay or failures. Used by every test.
- `JudgeBatch`: the calls of one request, started in the background so that they overlap with the surrogate's scoring; `results()` waits at most `XVIRAL_JUDGE_DEADLINE`
  seconds (40) and never raises.
- `JudgeRule`: loads `xviral/data/judge_rule.json`; `decide(gap, judge_logodds, kind, model)` returns the blended probability, whether to speak, each opinion's own
  pick and probability, and whether they agree. Checked equal to the research code on all 7,404 test decisions (probability to 1e-9, speak or silent exactly).
- `Advisor(..., judge=, judge_rule=, judge_deadline=)`: `start_judging`, `_compare(base, alt, target, judged)`, `compare(base, alt, judged)`, `advise(..., judged=, use_judge=, share_history=)`.
  With a judge result for a pair the fitted rule replaces both the single confidence scale and the "every member agrees" gate (the rule was validated without that gate).
  The kind of pair is `launch` when both versions read as a launch by the text rules, else `same_topic`.
- `Service.analyze`: all pairs of versions go to the judge in one batch started before the surrogate scores; `compare_versions(reports, judged)` and the "against version 1"
  rows use the results; proposed rewrites are judged against the draft inside `advise`. Single defined edits (attach an image, move the link, one point per line, remove
  hashtags) and the spacing check are not sent: they are near-identical texts, which the judge was not validated on (and which is where the benchmark's artefact sits).
- Settings: `XVIRAL_JUDGE_MODEL` (unset = no judge), `XVIRAL_JUDGE_ENDPOINT`, `XVIRAL_JUDGE_DEADLINE`, `XVIRAL_JUDGE_TIMEOUT`, `XVIRAL_JUDGE_HISTORY`, `XVIRAL_JUDGE_CONCURRENCY`.
  A model the rule file does not know is named in the report and not used.
- Fallback: a pair whose two calls did not both answer (error, unreadable answer, deadline) is decided by the surrogate exactly as before; the request never fails
  because of the judge. With `judge: false`, with no judge configured, or when every call fails, the comparison, the version table and the pick are identical to the report without a judge (tested).

What the report says (`docs/launch_advisor_api.md`, section "The second opinion"):
- `judge`: `{configured, used, provider, model, why_not, sent: {to, model, versions, rewrites, account_line, past_posts, past_posts_source, past_posts_fields}, pairs, pairs_answered, calls, cost_usd, added_ms, error}`.
- every entry of `comparison.pairs`, every `versions[i].vs_version_1` and every suggested rewrite: `decided_by`, and `opinions` = which version the forecasting model preferred,
  which the language model preferred, each one's own probability, and `agree`; `comparison.opinions.statement` counts the agreeing pairs; the verdict sentence ends with
  "The forecasting model and the language model both prefer it." or says which one did.
- `method.validation`: the accuracy measured for the configuration that decided the comparisons (the rule file's statement for this model and each kind of pair that
  occurred); the surrogate's own record moves to `method.forecast_model_validation`; `method.comparisons_decided_by`.
- `timing_ms.judge_wait`: the time the request waited for the judge after the surrogate had finished.

Latency and cost per request, real run (`e2e/summary.json`, `e2e_luna/summary.json`; 4 CPU cores, models warm, 30 posts of history sent):

| Request | Judge | Calls | Seconds without the judge | Seconds with | Added | Judge cost, US dollars |
|---|---|---|---|---|---|---|
| Two versions, account 1 | `gpt-6.1-sol` | 2 | 0.37 | 3.17 | 2.80 | 0.0077 |
| Two versions, account 2 | `gpt-6.1-sol` | 2 | 0.39 | 4.09 | 3.70 | 0.0127 |
| Two versions, account 3 | `gpt-6.1-sol` | 2 | 0.47 | 3.70 | 3.23 | 0.0092 |
| Six versions (15 pairs) | `gpt-6.1-sol` | 30 | 0.50 | 8.15 | 7.65 | 0.1298 |
| Two versions, accounts 1 to 3 | `gpt-6-luna` | 2 | 0.39 to 0.59 | 0.89 to 1.43 | 0.43 to 1.04 | 0.0004 to 0.0006 |
| Six versions (15 pairs) | `gpt-6-luna` | 30 | 0.53 | 2.10 | 1.61 | 0.0055 |

### 7. Privacy and honesty

- The judge sends the versions (text, kind of media, link domain), the handle and follower count, and up to 30 past posts from before the planned posting time (text, kind of media, link domain, like count). No views, no files.
- `judge: false` in the request: nothing is sent. The page has the checkbox, ticked by default, with a sentence saying what is sent and to whom the report will attribute it.
- `past_posts` supplied with a request (typed in, or read from an X analytics export: the server cannot tell the two apart, so both are treated as private) are never sent
  unless the request sets `share_past_posts: true`; without it such a request gets no second opinion at all and the report says why. Posts from the server's corpus of public posts are sent.
  The page has the second checkbox, unticked by default.
- The provider and the model are named in `judge`, in the page's "How far to trust this" card, and the request log line carries `judge` = `{used, model, pairs, cost_usd, added_ms, why_not}` and never text.
- The method note comes from the rule file's `validated` block, which `test_eval.py` writes from the slice that matches product use (wording differs, both posts after the
  model's knowledge cutoff). For `gpt-6.1-sol` on launch pairs it reads: "With gpt-6.1-sol as second opinion, checked on 772 real pairs of launch posts by the same account that differ in wording and were posted after the language
  model's knowledge cutoff (2026-04-30): the two models together picked the better post 77% of the time (72 to 81), the forecasting model alone 64%. Where the real gap was 2x or more
  (511 pairs) a verdict was given on 91% of pairs and was right 85% of the time (81 to 89); the forecasting model alone: 32% of pairs, 81%. These are two different launch posts weeks or
  months apart, not two drafts of one launch; no set of same-day launch drafts with outcomes exists." Only models scored on the test sets are in the shipped file. A call that failed or timed out is reported with `used: false`, the error, and the `sent` block, because the text left the server all the same.

### 8. Tests and the real run

`tests/test_judge.py`, 14 tests, the language model always the stub: rendering and prompt (the measured prompt, braces included); history selection; answer parsing for both modes and for
an endpoint that returns only the chosen token; retries, timeouts, no retry on a refusal, cost with cached tokens, both endpoints' request bodies, settings from the environment;
the stub's determinism, order balance and concurrency (12 calls of 0.1 s in under 0.6 s), the deadline; the rule's arithmetic by hand, clip, symmetry, agreement rule, caller's stricter
threshold; the shipped file (monotone, validated block present for every model and kind); the service with both opinions (the blended number equals the rule applied to the surrogate's gap
and the judge's log-odds; opinions named; the method note is the rule file's statement for the kind that occurred); six versions in one batch of 30 concurrent calls with no single edit sent;
launch and same-topic kinds; opt-out; uploaded posts kept on the server without `share_past_posts` and sent (likes, not views) with it; an account with no history; only posts from before the planned posting time are sent; total failure and deadline
(the report equals the one with no judge to 1e-9); partial failure; an unknown model; rewrites judged against the draft; HTTP fields, schema and log line without text.
`pytest tests/test_judge.py tests/test_app.py tests/test_serve.py tests/test_selective.py`: 33 passed.

Real end-to-end run, through the HTTP layer with the three production checkpoints (`scripts/judge_product/e2e.py`, seed fixed before any call): three accounts drawn from the launch test pairs
(real gap 2x or more, 30 or more earlier posts), two real launch posts of the account as versions 1 and 2, the account's public posts from before the earlier one as `past_posts` with
`share_past_posts`, the earlier post's time as `when`. Three accounts show that the pipeline runs; they measure nothing. Full responses in `e2e/account_*.json`.

| Account | Followers | Real likes (version 1, 2) | Surrogate alone | With `gpt-6.1-sol` | Opinions | Added seconds | Cost |
|---|---|---|---|---|---|---|---|
| @cognition | 188,336 | 4,274 and 274 | version 2 predicted worse (33% that it is better); pick 1 | "Version 1 is predicted to do better than version 2 (95% likely, about 1.8x the likes). The forecasting model and the language model both prefer it." | agree | 2.80 | 0.0077 |
| @Alibaba_Qwen | 306,741 | 22,751 and 8,323 | too close to call (42%) | "Version 1 is predicted to do better than version 2 (74% likely, about 1.3x the likes). The forecasting model and the language model both prefer it." | agree | 3.70 | 0.0127 |
| @jessyka_boat | 1,397 | 12 and 5 | too close to call (47%) | "Version 1 is predicted to do better than version 2 (77% likely, about 1.1x the likes). The forecasting model and the language model both prefer it." | agree | 3.23 | 0.0092 |

The `judge` block of the third response, as returned:

```json
{"configured": true, "used": true, "provider": "OpenAI", "model": "gpt-6.1-sol", "why_not": null,
 "sent": {"to": "OpenAI", "model": "gpt-6.1-sol", "versions": 2, "rewrites": 0, "account_line": "handle and follower count", "past_posts": 30,
          "past_posts_source": "posts you supplied", "past_posts_fields": "text, kind of media, link domain, like count"},
 "pairs": 1, "pairs_answered": 1, "calls": 2, "cost_usd": 0.009222, "added_ms": 3229, "error": null}
```
and its pair: `{"a": 1, "b": 2, "p_b_better": 0.229, "status": "predicted_worse", "decided_by": "forecasting model and language model", "kind": "launch", "p_b_better_forecast_model_only": 0.474,
"opinions": {"forecast_model_prefers": 1, "forecast_model_p_b_better": 0.482, "language_model_prefers": 1, "language_model_p_b_better": 0.227, "agree": true}}`.

Six real launch posts of the first account as six versions (real likes 4,274, 1,061, 2,430, 6,804, 522, 1,581): 15 pairs, 30 calls, all answered, 7.6 seconds added, 0.13 dollars.
Verdict: "Too close to call between versions 3 and 4. Versions 1, 2, 5 and 6 are each predicted to do worse than another version." (the real best is version 4); "The forecasting model and the
language model were both asked about 15 of 15 pairs; they preferred the same version in 6." With `gpt-6-luna` the three two-version verdicts are the same version 1 (86%, 75%, 70%).
The page was driven in a browser against a live server with `gpt-6-luna` (handle path, the server's corpus): verdict sentence, the two "prefers" columns in the pair table and the trust card render with no script error.

## Tokens and dollars

| Run | Endpoint | Requests | Input tokens | Cached | Output tokens | USD |
|---|---|---|---|---|---|---|
| bench__n30__Qwen--Qwen3.5-397B-A17B__logprob | publik | 1,406 | 3,639,247 | 299,904 | 3,124 | 1.578 |
| bench__n30__gpt-6.1-sol__json-low (batch) | openai | 3,380 | 7,192,700 | 7,780 | 143,220 | 7.901 |
| dev__n100__gpt-6.1-sol__json-low (batch) | openai | 586 | 2,525,408 | 0 | 24,728 | 2.649 |
| dev__n30__Qwen--Qwen3.5-397B-A17B__logprob | publik | 860 | 1,665,141 | 209,088 | 1,538 | 0.706 |
| dev__n30__deepseek-ai--DeepSeek-V4.1-Flash__logprob | publik | 1,343 | 1,699,876 | 927,616 | 1,664 | 0.161 |
| dev__n30__gpt-6.1-sol__json-low (batch) | openai | 455 | 976,606 | 0 | 18,705 | 1.070 |
| dev__n30__openai--gpt-oss-120b__json-default | publik | 1,109 | 1,110,192 | 299,773 | 222,538 | 0.073 |
| dev__n60__gpt-6.1-sol__json-low (batch) | openai | 586 | 1,695,694 | 0 | 24,398 | 1.818 |
| e2e gpt-6-luna | openai | 36 | 67,928 | 0 | 144 | 0.007 |
| e2e gpt-6.1-sol | openai | 36 | 69,404 | 0 | 2,063 | 0.159 |
| launch__n30__Qwen--Qwen3.5-397B-A17B__logprob | publik | 1,710 | 4,676,443 | 1,146,816 | 3,826 | 1.852 |
| launch__n30__gpt-6.1-sol__json-low (batch) | openai | 582 | 1,349,571 | 0 | 26,514 | 1.482 |
| launchpre__n100__gpt-6.1-sol__json-low (batch) | openai | 600 | 1,441,866 | 114,427 | 27,006 | 1.468 |
| launchpre__n30__Qwen--Qwen3.5-397B-A17B__logprob | publik | 600 | 1,077,959 | 82,368 | 1,316 | 0.470 |
| launchpre__n30__deepseek-ai--DeepSeek-V4.1-Flash__logprob | publik | 600 | 1,051,222 | 576,512 | 1,304 | 0.099 |
| launchpre__n30__gpt-6-luna__logprob | openai | 7,310 | 10,741,402 | 218,011 | 29,240 | 1.069 |
| launchpre__n30__gpt-6.1-sol__json-low (batch) | openai | 1,200 | 1,818,020 | 146,924 | 52,053 | 1.939 |
| launchpre__n30__openai--gpt-oss-120b__json-default | publik | 300 | 489,415 | 139,852 | 118,545 | 0.036 |
| launchpre__n30_ev__gpt-6.1-sol__json-low (batch) | openai | 600 | 1,082,170 | 0 | 27,916 | 1.222 |
| launchpre__n60__gpt-6.1-sol__json-low (batch) | openai | 600 | 1,163,900 | 0 | 26,881 | 1.298 |
| **Total, publik** | | 7,928 | | | | **4.98** |
| **Total, openai** | | 15,971 | | | | **22.08** |

The Publik balance header moved from 999.839 to 994.788 dollars over the session: 5.05 spent, which includes the capability probes (one of them let `Qwen3.5` reason for 2,000 tokens twice, about 0.012 dollars) and the requests refused by the rate limit (free).

OpenAI prices as in the earlier track (per million tokens: `gpt-6-luna` 0.10 input, 0.01 cached, 0.50 output; `gpt-6.1-sol` 2.00, 0.10, 10.00, half through the batch interface).
Publik prices from `GET /api/v1/models` (input, cached input, output): `gpt-oss-120b` 0.037, 0.0185, 0.17; `DeepSeek-V4.1-Flash` 0.20, 0.006, 0.60; `Qwen3.5-397B` 0.45, 0.22, 3.00.
Dollars are computed from the token counts each response reported. Everything sent was a public post already in the corpus, with its like count, except the two example
drafts from `docs/launch_advisor_api.md` used once for the page check.

## Honest limits

- Launch pairs are two different launch posts of one account, a median of 55 days apart, not two drafts of one launch. The judge mostly reads which announcement is the bigger news.
  The closest slice to "two versions" is launch pairs within 30 days (188 pairs 2x apart: 87% spoken at 86.0% with sol), still different announcements. No set of same-slot launch drafts with outcomes exists.
- The same-topic rule for `gpt-6.1-sol` rests on 809 development pairs, 148 of them in the strict fit cell; its speak rule was chosen on 206 validation pairs 2x apart where no candidate reached 75%.
  The test result is far better than validation said. Treat the benchmark numbers as the measurement and the rule's exact shape as loosely determined.
- One candidate was dropped from the speak-rule search after seeing validation: "55% and above" won the fixed criterion for sol on launch pairs (95% coverage at 82%). It was removed because the product
  never recommends below a stated 60%. That is a change after looking, in the conservative direction.
- The launch development pairs have a median of 17 earlier posts, the launch test pairs 79. The judge was no worse with little history (sol on launch test pairs with under 30 earlier posts: 83.5% when it speaks),
  so the product applies the rule whatever the history length, including none; with none it is the "no history" configuration, which on same-topic development pairs was 6.8 points worse than 30 posts.
  A request whose uploaded posts may not be sent therefore gets no second opinion rather than a weaker, unvalidated one.
- Knowledge cutoffs for the OpenAI models are the dates on their model pages. For `Qwen3.5` and `DeepSeek-V4.1-Flash` no cutoff is published; only the dates the weights appeared are known.
- The near-identical threshold (text similarity 0.9) is the earlier track's, untuned; the 0.5 to 0.9 band may still hold corrected reposts.
- `gpt-6.1-sol` is not deterministic; each pair was asked once per order. 1 to 3% of pairs come back exactly undecided and count as half right.
- In the product the scorer and the judge calls run inside the service's one-analysis-at-a-time lock, so a request with a judge holds the lock for 3 to 8 seconds with `gpt-6.1-sol`.
  The judge's wait could move outside the lock; it was left as is.
- `compare_versions` still uses the same-topic confidence scale for the surrogate-only path even when both versions are launches (an existing inconsistency with the "against version 1" rows,
  which use the launch scale). The judge path uses the right kind. Not changed, to keep reports without a judge identical to before.
- When the judge misses the deadline its calls are not cancelled: they finish in the background (and are billed), while the report's `cost_usd` for that request shows only answered pairs. `Judge.spend` has the true total.
- `predicted_ratio` in the pair table and in `vs_version_1` stays the forecasting model's ratio also when the language model turned the verdict; the verdict sentence drops the ratio in that case.
- An independent read of the product code found one path that could fail a request (a malformed link in a draft) and three places where the `judge` block misdescribed a failure; all are fixed and tested. It found no path that sends posts against the opt-out or without `share_past_posts`.
- The Publik key is limited to 60 requests a minute. Early runs of three models in parallel lost about 1,400 requests to that limit before the runner was given a rate.
- The e2e accounts and all validation are English-language startup and company accounts in the corpus; nothing here says how the judge does elsewhere.

## What I would do next

1. Turn it on with `gpt-6.1-sol` for launch comparisons first: that is where the gain is largest (64% to 77% overall, 91% of 2x pairs at 85%) and the cost is one cent per pair.
2. Log every judged comparison (probabilities, opinions, model) with the eventual outcome of the version that was posted: that is the only calibration set of real same-slot drafts.
3. Score `gpt-6.1-sol` on 300 more same-topic development pairs in the fit cell (about 1.2 dollars) and refit the same-topic speak rule; today it rests on 148.
4. Use the judge as a labeller for the surrogate (the earlier track's idea): the 3,655 pre-cutoff launch pairs now have luna scores, 600 have sol scores.
5. Move the judge's wait outside the service lock before any load.

## Files in `tracks/judge_product/`

- `REPORT.md`, `patch.diff` (against branch head 3a642ba; new files included), `judge_rule.json` (the shipped rule with its validated numbers: `gpt-6.1-sol`, `gpt-6-luna`, `Qwen/Qwen3.5-397B-A17B`).
- `jp_fit.md`, `jp_fit.json` (development: clip search, train and validation, every speak rule, calibration, agreement, by history); `jp_test.md`, `jp_test.json` (test: every slice and gap);
  `jp_extras.md`, `jp_extras.json` (history and prompt variants, old and new pairs, open models, baseline re-issued, per-call latency and price, dollars).
- `judge_rule_fitted_frozen_luna_sol.json`, `judge_rule_fitted_frozen_with_open_models.json` (the rules as frozen before each test scoring).
- `out/<set>__<config>__<model>__<mode>.jsonl` (every request of this track: pair, order, probability, tokens, seconds; the earlier track's files are under `tracks/llm_judge/out/`),
  `jp_pairs_*.parquet` (pair lists with the surrogate's gap, split cell, times, followers, history size), `jp_textsim_*.parquet`, `sample_*.json` (the key lists of every sample), `batches.json`, `spend.jsonl`.
- `e2e/`, `e2e_luna/` (requests and full responses of the real run), `page.png` (the page with a second opinion).

## Reproduce

From `virality/`, with the dataset's `bench2/`, `tracks/replicate/`, `tracks/launch/`, `tracks/pairs/benchmark_pairs.parquet`, `tracks/selective/feat_*_base3.parquet`, `tracks/llm_judge/` and
`work/filtered/posts.parquet` under `data/` (`tracks/llm_judge/*` also copied to `data/llm/`):

```
python scripts/llm_judge/prep.py                     # items of dev, bench, launch (the earlier track's)
python scripts/judge_product/prep.py                 # launch development items, serving-path gaps, split cells, samples
R="python scripts/judge_product/run.py"
python scripts/llm_judge/batch.py submit --set launch; python scripts/llm_judge/batch.py submit --set bench --clear --every 2 --offset 0; python scripts/llm_judge/batch.py submit --set bench --clear --every 2 --offset 1
$R submit --set launchpre --config n30 --keys data/llm/sample_launchpre_sol.json; $R submit --set dev --config n30 --keys data/llm/sample_dev_val_sol.json
$R submit --set dev --config n60 --half A --every 4; $R submit --set dev --config n100 --half A --every 4
for c in n60 n100 n30_ev; do $R submit --set launchpre --config $c --keys data/llm/sample_launchpre_cfg.json; done
$R collect                                           # repeat until every batch is collected (--cancel-after 10 for one stuck at 99%)
$R sync --set launchpre --config n30 --model gpt-6-luna
bash scripts/judge_product/publik_dev.sh; bash scripts/judge_product/publik_oss.sh 8 & bash scripts/judge_product/publik_qwen.sh 45
python scripts/judge_product/fit_rule.py             # development only; writes data/llm/judge_rule_fitted.json
python scripts/judge_product/test_eval.py            # test sets, once; writes xviral/data/judge_rule.json
python scripts/judge_product/extras.py
python scripts/judge_product/e2e.py; python scripts/judge_product/e2e.py --model gpt-6-luna --out data/llm/e2e_luna     # needs the three checkpoints and sid_codebook.npz
pytest tests/test_judge.py tests/test_app.py tests/test_serve.py tests/test_selective.py
```
