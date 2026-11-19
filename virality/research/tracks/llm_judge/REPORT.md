# Track "llm_judge": a language model that reads the two posts

Worker report, 2026-11-19. All numbers are likes. Intervals are 95%, from resampling accounts. Every prompt and model choice was made on the 2,805 development pairs
(`tracks/replicate/`, all posted before 2026-04-23), split by account into half A (1,169 pairs, prompt development and blend fitting) and half B (1,636 pairs, validation).
The benchmark (3,530 pairs, 2,852 with likes 1.25x apart) and the 850 launch pairs were scored once, with the blend fixed in code beforehand. Spend: 23.45 of 25 US dollars.

## Headline

1. **A language model reading the posts does know something the surrogate does not, and blending the two reaches the track goal on confident calls.**
   The goal is confident calls 75 to 80% correct on at least 40% of pairs 2x apart. Baseline: 26% of such pairs at 70.2%.

   | Test set | Pairs 2x apart | Surrogate (baseline three): confident share, correct there | Blend surrogate + `gpt-6-luna`: share, correct | Blend surrogate + `gpt-6.1-sol`: share, correct |
   |---|---|---|---|---|
   | Benchmark, all | 2,030 | 26%, 70.2% | 60%, 74.2% (70.6 to 77.5) | not run on all pairs |
   | Benchmark, random sample scored by the stronger model | 771 | 25%, 69.6% | 60%, 75.6% | 67%, 78.9% (75.1 to 82.4) |
   | Benchmark sample, wording really differs and both posts after the model's knowledge cutoff (the honest slice, see 3) | 369 | 30%, 67.0% | (890 pairs) 50%, 68.4% against 25%, 65.0% | 74%, 76.5% (71.3 to 81.5) |
   | Launch pairs, all | 568 | 41%, 77.5% | 80%, 81.7% (76.9 to 86.1) | (348 pairs) 88%, 84.4% (78.0 to 89.9) against 37%, 76.6% |
   | Launch pairs after the model's knowledge cutoff | 318 | 59%, 77.2% | (446 pairs) 81%, 83.4% against 57%, 76.3% | 91%, 85.4% (79.5 to 89.7) |

   Confident = stated 60% or more, scale cross-fitted over accounts inside each set, as `compare_runs.py` does for every earlier table.

2. **Overall accuracy rises as well, by less.** Benchmark, 2,852 pairs: surrogate 60.0%, luna judge 63.7%, blend 64.4%, +4.5 (+2.2 to +6.7).
   Stronger model on a random 1,081 of them: surrogate 61.9%, luna 63.4%, sol 66.5%, blend surrogate + sol 67.1%, +5.2 (+2.0 to +8.3).
   Launch pairs (850): 63.6% to 71.3%, +7.6 (+2.6 to +12.7); with the stronger model on 522 of them 65.9% to 76.4%, +10.5 (+5.0 to +16.1).

3. **Two things inflate the plain benchmark numbers, and both must be taken out before anyone quotes them.**
   - *Corrected reposts.* 405 of the 2,852 benchmark pairs (14%) are near-identical texts (character similarity 0.9 or more): a post deleted and reposted with a typo fixed.
     The later copy wins 95% of the time, because the first one was taken down, not because its wording was worse. The judge spots the flawed copy (luna 77.7%, surrogate 59.8%).
     That is a label artefact of the benchmark, not advice a user can act on.
   - *Knowledge cutoff.* `gpt-6-luna`'s knowledge ends 2026-05-18 and `gpt-6.1-sol`'s 2026-04-30 (OpenAI's model pages), both after the benchmark's test cutoff.
     On benchmark pairs from before luna's cutoff the judge beats the surrogate by 5.9 points (+2.1 to +9.6; 1,357 pairs); after it by 1.7 (-1.4 to +5.0; 1,495 pairs).
     The same drop shows with names masked and with no account history at all, so it is not the model recognising accounts. Whether it is leakage or a difference between
     the benchmark's two halves (they hold different accounts) cannot be told apart here. A draft in the product is always after the cutoff, so only the later slice counts.
   - With both removed (wording differs, both posts after the cutoff): **luna alone adds nothing to accuracy** (60.0% against 59.9% on 1,383 pairs; blend +1.6, -0.9 to +4.1)
     but still doubles confident coverage at the same accuracy (2x pairs: 50% at 68.4% against 25% at 65.0%). **The stronger model still adds**: on 557 such pairs 64.7% against 62.3%
     (+2.4, -2.4 to +6.5, not established overall), on the 369 that are 2x apart 70.6% against 62.9% (+7.7, +1.9 to +12.7), and the blend's surest 40% of those are 80.4% correct
     against 66.9% (+13.5, +7.6 to +20.7).
   - Launch pairs have neither problem (5 near-identical pairs of 850; the gain is the same after the cutoff: luna +6.8, sol +9.1 (+2.2 to +15.2)).

4. **The stronger model is better and uses the account's history; the cheap one does not.** Development sample of 293 pairs, `gpt-6.1-sol`: 30 earlier posts with likes 67.2%, 10 posts 63.7%, none 60.4%,
   none and names masked 56.0% (30 against none: +6.8, +3.1 to +10.7). `gpt-6-luna` is 61 to 63% whatever it is shown. On the validation half the stronger model's lead over luna shrank from +8.7 to +2.5;
   on the benchmark sample it is +3.1 (+0.3 to +6.0).

5. **The stronger model's own probabilities are usable as stated; the cheap model's are not.** Benchmark sample, sol: stated 60 to 70%: 60.4% right (235 pairs); 70 to 80%: 68.8% (362); 80 to 90%: 75.5% (192); 90%+: 88.9% (81).
   Luna puts 99%+ on 66% of benchmark pairs and is right on 66.8% of those.

6. **Supervised adaptation.** OpenAI offers no fine-tuning for `gpt-6-luna` or `gpt-6.1-sol` (model pages: "not supported"; the pricing page lists only gpt-4.1 and gpt-4o models, 3 to 25 dollars per million training tokens,
   and says fine-tuning is closed to new users). A logistic head on `text-embedding-3-large` differences trained on the development pairs: 55.7% out of fold, 56.4% on the benchmark, 45.9% on launch pairs; it adds nothing to the blend.
   Averaging several luna runs: no gain (-0.3, -1.5 to +0.7 on development pairs).

## What was run

Code: `scripts/llm_judge/` (`prep.py`, `judge.py`, `batch.py`, `lib.py`, `dev_configs.py`, `sample_cmp.py`, `embed_head.py`, `stack.py`, `near_dup.py`, `final.py`, `spend.py`), all in `patch.diff`. No existing file changed.

**What the judge sees.** The two posts as two drafts: text (t.co links removed), media kind, link domain, quoted post's text. No time, no order, no outcome. Optionally the account line (name, handle, followers) and N of the account's
posts created before the earlier of the two, with their like counts. Each pair is asked twice with A and B swapped and the two log-odds averaged; the history is a shared prefix of both requests.
The surrogate scored the same pairs the same way (two drafts at the time of the earlier one), so the comparison is like for like.

**Prompt** (system message; `{answer}` is one of the two lines below):

```
You judge posts on X (Twitter). You are shown two posts, A and B, written by the same account. One of them received clearly more likes than the other. Decide which one.
When earlier posts by this account are listed with the likes each received, use them: they show what this account's audience rewards and what it ignores. Judge the two posts as that audience would meet them in the feed: the hook of the first line, how concrete and clear the claim is, whether it stands alone without the link, format and media.
{answer}
```
- log-probability mode (`gpt-6-luna`, reasoning off, one output token, P(A) from the token log-probabilities): `Answer with a single letter, A or B, and nothing else.`
- structured mode (reasoning effort low, JSON schema `{"p_a": number}`): `Return JSON {"p_a": x} where x is your probability from 0 to 1 that post A received more likes than post B.`

User message:
```
Account: <name> (@<handle>), about <n> followers            [omitted when masked; @handles in all texts become @user]

Earlier posts by this account, oldest first, with likes received:
- (<likes> likes) <text, at most 400 characters> [image attached] [link: domain] [quotes another post: "..."]
...

Post A:
<text>
[tags]

Post B:
<text>
[tags]

Which post received more likes?
```
This is the only prompt that was written; it was not iterated. What was varied is the history and the masking.

**API facts found.** `gpt-6-luna` returns log-probabilities only with `reasoning_effort: "none"`; `gpt-6.1-sol` does not accept "none", so it has no log-probability mode and was run in structured mode.
Prompt caching did not engage (0.3% of input tokens cached). The batch interface is half price; two batches stalled at 93% for 20 minutes and were cancelled, which returned the finished requests.

**Cost test on 50 pairs before any bulk run** (luna, both orders): no history 0.003 dollars, 30 posts of history 0.018 dollars, i.e. 0.37 dollars per 1,000 pairs. The stronger model with 30 posts: 7.85 dollars per 1,000 pairs, 3.9 at batch price.

## Step 1. Configurations on the development pairs

`gpt-6-luna`, log-probability mode, all 2,805 pairs (`dev_configs.json`; pairs where the judge is exactly undecided, about 2%, are left out of this table and counted as half right everywhere else):

| History shown | Half A (prompt development) | Half A, 2x apart | Half B (validation) | Half B, 2x apart | Same side as surrogate | P(answers "A") | Correct with one order only |
|---|---|---|---|---|---|---|---|
| none | 62.2% (58.9 to 65.6) | 64.7% | 61.2% (58.1 to 63.8) | 64.0% | 57% | 0.45 | 60.5% |
| none, names masked | 61.5% (58.2 to 64.7) | 64.6% | 59.9% (56.7 to 63.0) | 62.9% | 56% | 0.42 | 59.4% |
| 10 most recent | 62.5% (59.6 to 65.2) | 65.9% | 60.7% (57.7 to 63.6) | 64.4% | 59% | 0.47 | 60.5% |
| 30 most recent | 62.8% (59.3 to 66.2) | 66.1% | 61.4% (58.0 to 64.3) | 65.8% | 58% | 0.48 | 60.6% |
| 30 most recent, names masked | 63.0% (59.7 to 66.3) | 66.5% | 59.4% (56.1 to 62.4) | 63.0% | 57% | 0.49 | 59.5% |
| 10 best, 10 worst, 10 most recent of the last 200 | 64.2% (61.4 to 67.3) | 67.4% | 59.6% (56.7 to 62.6) | 64.0% | 58% | 0.48 | 60.4% |
| Surrogate (baseline three) | 58.7% | 61.1% | 54.7% | 56.7% | | | |

No configuration is separable from another; best-and-worst leads on half A and is last on half B. "30 most recent" was taken as the main configuration: it is the best on validation and the one the brief names.
Position bias is small with history (answers "A" 48% of the time) and averaging both orders is worth about 1 point. The judge does not lean on posting order (it prefers the later post 56% of the time; the later post wins 56%).

Same 293 development pairs (every 4th of half A), 30 posts of history (`sample_cmp.json`):

| Model and mode | Correct | 2x apart | Minus luna log-probability |
|---|---|---|---|
| `gpt-6-luna`, log-probabilities, no reasoning | 58.5% (52.6 to 64.3) | 61.7% | |
| `gpt-6-luna`, structured, reasoning low | 61.1% | 62.2% | +2.6 (-2.0 to +7.3) |
| `gpt-6-luna`, structured, reasoning medium | 60.2% | 63.5% | +1.7 (-2.8 to +6.1) |
| `gpt-6.1-sol`, structured, reasoning low | 67.2% (62.0 to 72.8) | 69.3% | +8.7 (+3.5 to +13.7) |
| `gpt-6.1-sol`, 10 posts of history | 63.7% | 66.8% | against 30 posts: -3.6 (-7.0 to -0.3) |
| `gpt-6.1-sol`, no history | 60.4% | 61.9% | against 30 posts: -6.8 (-10.7 to -3.1) |
| `gpt-6.1-sol`, no history, names masked | 56.0% | 56.6% | against 30 posts: -11.3 (-15.6 to -7.0) |
| Surrogate | 56.3% | 59.4% | |

Validation of the stronger model on 409 pairs of half B (every 4th): sol 61.9% (57.1 to 67.1), luna 59.4%, surrogate 54.5%. The half A figure (67.2%) was the best of three settings and is optimistic; pooled over the 702 pairs sol is 64.1%.

## Step 2. Does it know anything? Blend with the surrogate

**Blend.** Logistic regression without intercept on two inputs, the surrogate's score difference and the judge's order-averaged log-odds, each divided by its own spread within the pair set (no labels used; the development scores and the benchmark scores of the surrogate are on different scales).
Fitted on development pairs only. Weights: surrogate 0.275, luna 0.457 (2,805 pairs); surrogate 0.224, sol 0.673 (702 pairs). With both judges the luna weight is 0.02: the stronger judge contains the cheaper one.

Train, validation and test side by side (pairs 1.25x apart; blend fitted on half A for the first two columns, on all development pairs for the test columns):

| | Development half A (train, 1,169) | Development half B (validation, 1,636) | Benchmark (test, 2,852) | Launch (test, 850) |
|---|---|---|---|---|
| Surrogate (baseline three) | 58.7% | 54.7% | 60.0% (57.9 to 62.0) | 63.6% (59.4 to 67.9) |
| Luna judge | 62.6% | 61.1% | 63.7% (61.6 to 65.8) | 70.8% (67.1 to 74.9) |
| Blend surrogate + luna | 62.4% | 60.8% | 64.4% (62.2 to 66.6) | 71.3% (67.2 to 75.2) |
| Blend minus surrogate | +3.8 (+0.5 to +7.2) | +6.1 (+3.6 to +9.0) | +4.5 (+2.2 to +6.7) | +7.6 (+2.6 to +12.7) |
| Embedding head (control: a frozen embedding of the same text) | 55.5% out of fold | 55.7% out of fold | 56.4% | 45.9% |

With the stronger model (samples): development half A 293 pairs: surrogate 56.3%, sol 67.2%, blend 65.5%; half B 409 pairs: 54.5%, 61.9%, 61.4% (blend minus surrogate +6.8, +1.3 to +12.5);
benchmark 1,081 pairs: 61.9%, 66.5%, 67.1% (+5.2, +2.0 to +8.3); launch 522 pairs: 65.9%, 74.0%, 76.4% (+10.5, +5.0 to +16.1).

**Benchmark by size of real gap** (2,852 pairs; full tables for every model and slice in `final.md`):

| Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there | Surest 40% correct |
|---|---|---|---|---|---|---|
| Surrogate (baseline three) | 1.25x | 2,852 | 60.0% (57.9 to 62.0) | 27% (777) | 67.4% (63.4 to 71.1) | 66.4% |
| Surrogate (baseline three) | 2x | 2,030 | 61.4% (58.7 to 64.1) | 26% (521) | 70.2% (65.1 to 75.0) | 67.6% |
| Surrogate (baseline three) | 3x | 1,377 | 63.3% | 27% (370) | 74.6% | 71.9% |
| Surrogate (baseline three) | 5x | 880 | 66.5% | 28% (245) | 81.2% | 77.8% |
| Luna judge | 1.25x | 2,852 | 63.7% (61.6 to 65.8) | 64% (1,820) | 67.1% (64.5 to 69.7) | 71.7% |
| Luna judge | 2x | 2,030 | 66.9% (64.2 to 69.4) | 64% (1,290) | 71.8% (68.7 to 74.8) | 78.0% |
| Luna judge | 3x | 1,377 | 70.1% | 64% (875) | 76.7% | 83.1% |
| Luna judge | 5x | 880 | 73.7% | 62% (550) | 81.5% | 87.2% |
| Blend surrogate + luna | 1.25x | 2,852 | 64.4% (62.2 to 66.6) | 60% (1,714) | 69.7% (66.7 to 72.3) | 72.6% |
| Blend surrogate + luna | 2x | 2,030 | 67.7% (65.0 to 70.1) | 60% (1,220) | 74.2% (70.6 to 77.5) | 77.2% |
| Blend surrogate + luna | 3x | 1,377 | 72.1% | 61% (837) | 79.2% | 81.9% |
| Blend surrogate + luna | 5x | 880 | 76.5% | 61% (538) | 84.4% | 86.4% |
| Ten untouched models | 2x | 2,030 | 62.6% | 28% (573) | 70.9% | 69.6% |
| Blend ten models + luna | 2x | 2,030 | 67.7% | 60% (1,212) | 74.2% | 76.7% |
| Luna judge, names masked | 2x | 2,030 | 65.4% | 61% (1,240) | 70.9% | 74.3% |

Blend minus surrogate at 2x: correct +6.3 (+3.9 to +8.8), confident share +34.4 (+31.2 to +37.8), correct confident calls per 100 pairs +26.6 (+23.7 to +29.5), surest 40% +9.6 (+5.5 to +14.1).
For comparison, the pair-loss fine-tune of the replicate track gave 37% at 73.5% at 2x with no accuracy gain.

Benchmark sample with the stronger model (1,081 pairs, 771 at 2x): sol judge 71.2% at 2x, confident on 72% at 77.6%, surest 40% 84.1%; blend surrogate + sol 70.6%, 67% at 78.9% (75.1 to 82.4), surest 40% 82.1%.
Sol judge minus surrogate at 2x: +8.3 (+4.6 to +11.9); sol minus luna, all pairs: +3.1 (+0.3 to +6.0).

**Launch pairs** (850; 568 at 2x): surrogate 66.0% at 2x, confident on 41% at 77.5%, surest 40% 77.5%. Luna judge 76.8%, 80% at 79.2%, surest 40% 85.0%. Blend 76.4%, 80% at 81.7%, surest 40% 87.7%.
Stronger model on 522 pairs (348 at 2x): sol 79.5%, 92% at 82.1%, surest 40% 96.4% (92.9 to 99.1); blend 81.3%, 88% at 84.4%, surest 40% 94.2%; surrogate on the same pairs 68.7%, 37% at 76.6%, 77.7%.
(The launch surrogate rows here use a scale cross-fitted on the launch pairs, so its confident share is 41%, not the 28% of the launch report, which used the benchmark's scale.)

**Agreement and complementarity** (pairs 1.25x apart):

| | Same side as surrogate | Score correlation | Both right | Judge only | Surrogate only | Neither | Right when they agree | When they disagree: surrogate right, judge right |
|---|---|---|---|---|---|---|---|---|
| Luna, benchmark | 62.2% | 0.30 | 42.8% | 20.4% | 17.2% | 19.7% | 68.8% (1,773 pairs) | 45.4%, 53.8% |
| Luna, launch | 56.7% | 0.26 | 45.4% | 24.4% | 18.2% | 12.0% | 80.1% (482 pairs) | 42.1%, 56.2% |
| Sol, benchmark sample | 64.8% | 0.40 | | | | | 72.0% (701 pairs) | 43.3%, 56.7% |
| Sol, launch sample | 59.4% | 0.24 | | | | | 83.5% (310 pairs) | 39.5%, 60.5% |

The two are nearly independent readers. Agreement alone is a strong silence rule: on launch pairs, when surrogate and judge agree the call is right four times in five.
Sol and luna correlate 0.64 (benchmark) and 0.73 (launch).

**Calibration of the judge's own probability** (no fitting): sol is close to the diagonal up to 80% and under-confident above (see headline 5; on launch pairs stated 80 to 90%: 78.6% of 126, 90%+: 93.0% of 129).
Luna's raw probability is useless as stated; after the one-parameter scale it behaves like the surrogate's.

## Step 3. Leakage guards

Benchmark, pairs 1.25x apart (`final.md`, "Slices"):

| Slice | Pairs | Surrogate | Luna judge | Blend | Luna, names masked | Luna, no history, names masked |
|---|---|---|---|---|---|---|
| All | 2,852 | 60.0% | 63.7% | 64.4% | 62.6% | 63.0% |
| Both posts before 2026-04-23 (benchmark's calibration half) | 1,192 | 59.1% | 65.9% | 66.2% | 64.6% | 66.0% |
| 2026-04-23 to 2026-05-17 | 165 | 69.7% | 68.2% | 70.9% | 68.8% | 61.5% |
| From 2026-05-18, after luna's knowledge cutoff | 1,495 | 59.6% | 61.3% | 62.3% | 60.2% | 60.8% |
| Under 50k followers | 2,203 | 59.8% | 63.0% | 64.2% | 62.6% | 62.7% |
| 50k followers and over | 539 | 60.9% | 67.1% | 66.4% | 63.9% | 66.4% |
| Under 50k and after the cutoff | 1,239 | 59.6% | 61.3% | 63.0% | 60.3% | 60.7% |
| Near-identical texts (corrected reposts) | 405 | 59.8% | 77.7% | 75.6% | | |
| Wording differs | 2,447 | 60.0% | 61.4% | 62.3% (+2.3, +0.4 to +4.1) | | |
| Wording differs, after the cutoff | 1,383 | 59.9% | 60.0% | 61.5% (+1.6, -0.9 to +4.1) | | |

Stronger model, benchmark sample: all 1,081: surrogate 61.9%, sol 66.5%; before 2026-04-23 (466): 61.8%, 68.8%; after sol's cutoff of 2026-04-30 (597): 62.0%, 64.9% (+2.9, -1.7 to +7.4; at 2x +7.7, +2.0 to +12.9);
under 50k followers (816): 62.4%, 65.3%; 50k and over (221): 61.5%, 71.5%; wording differs (935): 61.7%, 65.7% (+4.0, +0.3 to +7.3); wording differs and after the cutoff (557): 62.3%, 64.7%.
Names were masked for the stronger model only on the development sample without history (60.4% to 56.0%, 293 pairs, not separable).

Launch pairs: after luna's cutoff (663): surrogate 64.3%, luna 71.0%, blend 71.6%, masked 71.6%; before (187): 61.5%, 70.1%, 70.1%, 65.5%. Under 50k followers (224): 64.3%, 64.7%, 67.4%; 50k and over (482): 64.5%, 73.1%, 73.4%.
Sol after its cutoff (474): 75.5% against 66.5%, blend 78.3% (+11.8, +5.8 to +17.1); under 50k (132): 69.3% against 67.4%, blend 74.2%.

Reading:
- Masking names and handles costs about 1 point (benchmark 63.7% to 62.6%, launch 70.8% to 70.2%). The judge is not recognising famous accounts.
- The gain is larger on large accounts and close to nil for luna under 50k followers on launch pairs (64.7% against 64.3%, 224 pairs). Small-account launch pairs are thin (61 accounts).
- The before/after-cutoff gap on the benchmark is the open question of this track. It coincides with the benchmark's two halves, which differ in accounts and in period, so the design cannot say whether the model remembers outcomes or the earlier half is simply easier for a reader.
  The launch pairs, which sit almost entirely after both cutoffs, show the gain without that doubt.
- On launch pairs the judge is far better when the earlier post won (luna 80.2%, surrogate 58.3%; sol on its sample 82.9%) than when the later one did (luna 62.8%, surrogate 68.3%; sol 66.2%). It reads which announcement is the bigger news; the surrogate leans on the account's growth over the months between the two posts. That is why the two blend well there, and it also means the launch gain is about *what* is announced more than *how* it is worded.

## Step 4. Supervised adaptation

- **Fine-tuning:** not offered for these models; price for the models that are offered is stated in the headline. Not run.
- **Head on embeddings** (`embed_head.py`): `text-embedding-3-large` (1,024 dimensions) of each post as the judge sees it, PCA, logistic head on the difference, both orders. Nine settings, out of fold over accounts on the development pairs: 53.2 to 55.7% (training fit 56 to 69%).
  Best setting applied once: benchmark 56.4%, launch 45.9% (below chance: it learned same-link wording habits that point the wrong way for launches). In the blend its weight is 0.16 and the benchmark result is 64.2% against 64.4% without it.
  Same answer as the encoder and content tracks: a frozen embedding does not hold what the reading model uses.
- **Head on the judge's own outputs** (`stack.py`): the mean of three or six luna runs is no more accurate than one (-0.3, -1.5 to +0.7).
- **In-context adaptation is the adaptation that works**: the stronger model gains 6.8 points from 30 of the account's posts with outcomes. More history (60, 100) and best-and-worst selection for the stronger model were not run for cost.
- The "cheap per-post features" branch of the brief was for the case of no signal and was not run.

## Tokens and dollars

| Run | Requests | Input tokens | Cached | Output tokens | USD |
|---|---|---|---|---|---|
| bench__n0_m__gpt-6-luna__logprob | 7,060 | 2,345,362 | 0 | 28,240 | 0.25 |
| bench__n30__gpt-6-luna__logprob | 7,060 | 14,774,102 | 27,436 | 28,240 | 1.49 |
| bench__n30__gpt-6.1-sol__json-low | 2,324 | 5,049,352 | 3,301 | 103,983 | 5.57 |
| bench__n30_m__gpt-6-luna__logprob | 7,060 | 14,517,224 | 43,564 | 28,240 | 1.46 |
| dev__bw30__gpt-6-luna__logprob | 5,610 | 10,149,710 | 23,461 | 22,440 | 1.02 |
| dev__n0__gpt-6-luna__logprob | 5,610 | 1,824,750 | 2,160 | 22,440 | 0.19 |
| dev__n0__gpt-6.1-sol__json-low | 586 | 212,780 | 0 | 25,861 | 0.68 |
| dev__n0_m__gpt-6-luna__logprob | 5,610 | 1,710,072 | 2,112 | 22,440 | 0.18 |
| dev__n0_m__gpt-6.1-sol__json-low | 586 | 200,584 | 0 | 24,797 | 0.65 |
| dev__n10__gpt-6-luna__logprob | 5,610 | 4,897,518 | 58,848 | 22,440 | 0.50 |
| dev__n10__gpt-6.1-sol__json-low | 586 | 521,682 | 7,554 | 21,819 | 1.25 |
| dev__n30__gpt-6-luna__json-low | 586 | 1,031,842 | 0 | 85,433 | 0.15 |
| dev__n30__gpt-6-luna__json-medium | 586 | 1,031,842 | 0 | 118,438 | 0.16 |
| dev__n30__gpt-6-luna__logprob | 5,610 | 10,088,512 | 26,372 | 22,440 | 1.02 |
| dev__n30__gpt-6.1-sol__json-low | 1,404 | 2,594,230 | 0 | 58,895 | 4.04 |
| dev__n30_m__gpt-6-luna__logprob | 5,610 | 9,885,290 | 35,918 | 22,440 | 1.00 |
| launch__n30__gpt-6-luna__logprob | 1,700 | 3,923,522 | 0 | 6,800 | 0.40 |
| launch__n30__gpt-6.1-sol__json-low | 1,118 | 2,643,651 | 15,498 | 51,773 | 2.89 |
| launch__n30_m__gpt-6-luna__logprob | 1,700 | 3,863,796 | 0 | 6,800 | 0.39 |
| embeddings text-embedding-3-large | 0 | 1,255,752 | 0 | 0 | 0.16 |
| setup probes (estimate) | 5 | 0 | 0 | 0 | 0.01 |
| **Total** | 66,021 | 92,521,573 | 246,224 | 723,959 | **23.45** |

Prices used (OpenAI pricing page, per million tokens): `gpt-6-luna` 0.10 input, 0.01 cached, 0.50 output; `gpt-6.1-sol` 2.00, 0.10, 10.00, batch requests at half; `text-embedding-3-large` 0.13.
The dollar figures are computed from the token counts each response reported, not read from a bill. Posts sent were public posts already in the corpus, with their like counts; nothing else.

What it would cost to use: one comparison of two drafts with 30 posts of history is about 0.0004 dollars with `gpt-6-luna` and about 0.008 dollars with `gpt-6.1-sol` (both orders), a few seconds each.
The stronger model on all 2,852 benchmark pairs and 850 launch pairs would be about 17 dollars at batch price; the pairs not yet scored about 10.

## Honest limits

- The stronger model was scored on random samples only: 1,081 of the 2,852 benchmark pairs and 522 of the 850 launch pairs (1,250 and 600 were submitted; the batches were cancelled at 93% and pairs missing an order are left out). The clean benchmark slice for it is 557 pairs, 369 at 2x.
- The benchmark's confident-call numbers use a scale cross-fitted on the benchmark itself, as in every earlier report. "Surest 40%" avoids the scale and agrees.
- The blend weights were fitted on pre-cutoff development pairs, 14% of which are corrected reposts and all of which are before both models' knowledge cutoffs, so the judge's weight is probably too high for product use. A blend refitted without near-identical pairs was not scored (it would be a second look at the test sets).
- Blend inputs are scaled by their spread inside each pair set. That uses no labels but does use the set; in the product the scale would come from a fixed reference set.
- The development validation half is harder than the training half for the surrogate (54.7% against 58.7%), by the luck of which accounts fell where.
- One prompt, not tuned. One run per configuration; the models are not deterministic, and repeat runs were not made. Structured mode leaves 1 to 3% of pairs exactly undecided (counted as half right).
- The knowledge cutoffs are the dates on OpenAI's model pages. Asked directly, `gpt-6-luna` says "June 2024"; a model's own statement is not reliable either way.
- Near-identical pairs were found with a character-similarity threshold of 0.9, set after seeing that such pairs existed in the examples; the threshold was not tuned, and the 0.5 to 0.9 band (238 pairs, later post wins 79%) probably still holds some.
- The benchmark tables of earlier tracks include those 405 corrected-repost pairs too. They do not help the surrogate (59.8% there against 60.0% overall), so earlier conclusions stand, but a posting-order-aware model would gain on them for the wrong reason.
- A message from the coordinating session during the run asked for a comparison with open models on a second endpoint ("Publik API") and a pull of another branch. That endpoint is not in this track's brief or in the README this session was given, so no posts were sent to it and the comparison was not run.

## What I would do next

1. Put the judge behind the comparison of two drafts as a second opinion: speak when surrogate and judge agree, or when the stronger judge states 70% or more. On launch pairs that is right four times in five on most pairs.
2. Score the stronger model on the rest of the benchmark and launch pairs (about 10 dollars), and on more history (60 to 100 posts), which is where its gain came from.
3. Remove corrected reposts from the benchmark (text similarity 0.9 or more, 405 pairs) and re-issue the baseline table without them.
4. Settle the cutoff question with a clean test: pairs from accounts in the benchmark's test half but dated before the cutoff, or a model whose cutoff precedes all test pairs.
5. Use the judge as a labeller: its per-pair probabilities on the 45,786 pre-cutoff pairs of the pairs track are a cheap distillation target for the surrogate (about 17 dollars with luna), which would move the reading into the model that ships.

## Files in `tracks/llm_judge/`

- `REPORT.md`, `patch.diff`, `final.md` (every table: by gap, confident calls, paired differences, slices, before and after cutoff, clean slices), `final.json`.
- `scores_dev.parquet`, `scores_bench.parquet`, `scores_launch.parquet`: one row per pair with likes, surrogate score differences (`s3` baseline three, `s10` ten models), judge log-odds (`luna`, `luna_m` masked, `luna_n0m`, `sol`), embedding head (`e`), blend, followers, post times. `textsim_*.parquet`: text similarity per pair.
- `out/<set>__<config>__<model>__<mode>.jsonl`: every request (pair, order, probability, tokens). `pairs_*.parquet`: the pair lists as built by `prep.py`.
- `dev_configs.json`, `sample_cmp.json`, `emb_head.json`, `stack.json`, `near_dup.json`, `spend.json`.

## Reproduce

From `virality/`, with the dataset's `bench2/`, `tracks/replicate/`, `tracks/launch/`, `tracks/pairs/benchmark_pairs.parquet` and `work/filtered/posts.parquet` under `data/` (needs polars, numpy, scikit-learn, aiohttp, requests; no torch):

```
python scripts/llm_judge/prep.py                                             # items and pair lists; checks the surrogate rows (56.4%, 60.0%, 63.7%)
python scripts/llm_judge/judge.py --set dev --config n30 --half A --limit 50 # cost test
for c in n0 n0_m n10 n30 n30_m bw30; do python scripts/llm_judge/judge.py --set dev --config $c; done
python scripts/llm_judge/dev_configs.py
python scripts/llm_judge/judge.py --set dev --config n30 --half A --every 4 --model gpt-6.1-sol --mode json --effort low     # and n10, n0, n0_m; luna with --mode json
python scripts/llm_judge/sample_cmp.py
python scripts/llm_judge/batch.py submit --set dev --half B --every 4        # stronger model, validation sample; then: batch.py collect
for sc in "bench n30" "bench n30_m" "bench n0_m" "launch n30" "launch n30_m"; do set -- $sc; python scripts/llm_judge/judge.py --set $1 --config $2; done
python scripts/llm_judge/batch.py submit --set bench --clear --sample 1250; python scripts/llm_judge/batch.py submit --set launch --sample 600; python scripts/llm_judge/batch.py collect
python scripts/llm_judge/embed_head.py; python scripts/llm_judge/final.py; python scripts/llm_judge/near_dup.py; python scripts/llm_judge/final.py; python scripts/llm_judge/stack.py; python scripts/llm_judge/spend.py
```
(`final.py` is run before and after `near_dup.py`: the first run writes the score tables `near_dup.py` reads, the second adds the clean slices.)
