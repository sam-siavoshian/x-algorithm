# Model card: Launch advisor production bundle `te3l_broad2`

Made 2026-11-19 by the "combine" track. Everything here is about likes unless it says views. Intervals are 95%, from
resampling accounts. "Test" means posts from 2026-04-23 on (or accounts no model trained on); every choice was made on
earlier data. Method and every table: `research/tracks/combine/REPORT.md`.

## What is in the bundle

| File | What it is |
|---|---|
| `bundle.json` | The one config. `xviral.app.service.load_bundle(folder)` or `XVIRAL_BUNDLE=<folder>` loads everything below. |
| `checkpoints/te3l_broad_s1.pt`, `te3l_broad_s2.pt` | **Forecast members.** Two seeds, 1.65M parameters each, trained on the broad table with the OpenAI embedding. They produce every forecast (ranges, odds, "x times your usual") and decide same-topic comparisons. |
| `checkpoints/te3l1024_s1.pt`, `te3l1024_s2.pt` | **Launch comparison members.** Two seeds, same size and encoder, trained on the original table (the embedding track's). They decide only comparisons of two versions that both read as a launch. |
| `sid_codebook.npz` | Semantic-ID codebook for the embedding (6 levels of 256), shared by the four checkpoints. |
| `selective_rule.json` | Confidence scales: same-topic 3.43 (on the forecast members' gap), launch 2.32 (on the launch members' gap). |
| `type_correction.json` | Post-type level correction for the forecast members (launch +0.090, funding +0.243, open source +0.147 in log10 likes; four more types under +0.05). |
| `judge_rule.json` | Blend of the forecasting model with a language model, for `gpt-6.1-sol` and `gpt-6-luna`, per kind of comparison, with the validated statement the report quotes. |
| `fallback/` | The three bge-small models in production before this bundle, with their own codebook, scales, correction and judge rule. Local encoder. They answer when the embedding provider fails or a request does not allow its text to be sent. |
| `service.env` | The environment variables for the service. |
| `MODEL_CARD.md` | This file. |

Encoder: OpenAI `text-embedding-3-large`, first 1,024 dimensions (the API's `dimensions` parameter), stored in each
checkpoint's metadata as `openai:text-embedding-3-large@1024`. The service refuses to serve a checkpoint with another
encoder.

## Data

| | Forecast members | Launch comparison members |
|---|---|---|
| Training rows | 824,056 posts of 12,307 accounts | 302,696 posts of 6,334 accounts |
| Sources | Community Archive (tech audience, 302,696 rows); twitter100m full timelines, 2020 to April 2023 (388,495 rows: company and founder accounts, and accounts across five follower tiers); the 2026 view corpus (106,641 rows); brand timelines (26,224 rows) | Community Archive only |
| Extra inputs | sampling regime and data source of each row (source hidden for 15% of accounts, so "unknown" is a trained value) | none |
| Cutoff | no post from 2026-04-23 on | same |
| Steps, checkpoint | 6,000 steps; the checkpoint with the lowest mean of tech and broad validation loss (step 6,000 for both seeds) | 4,000 steps |
| Validation | 32,131 tech-audience rows and 45,106 rows of 538 broad accounts, all before the cutoff | 32,131 tech-audience rows |

Test sets, never used for a choice: 683,152 ordinary same-account like pairs after the cutoff (2,846 accounts); the
3,530 same-topic benchmark pairs (770 accounts); the 850 launch pairs (151 accounts); the generalization test bed (924
accounts no baseline model trained on: 454 full timelines, 434 accounts of the 2026 corpus, 36 brands; 368,970 like pairs).
Pairs used for fitting rules: 2,805 same-topic development pairs (366 accounts) and 3,655 launch pairs (628 accounts),
all before the cutoff.

Posts are public posts. Embedding the 673,443 training texts that were not yet stored cost 6.80 US dollars.

## Validated numbers

### Forecast members alone (no language model)

| Measure | Pairs | This bundle | Before (three bge-small models) |
|---|---|---|---|
| Ordinary same-account like pairs, tech audience, after the cutoff | 683,152 | 62.8% (train 63.5%, validation 64.5%) | 60.2% |
| Same, real gap 2x or more | 503,864 | 65.3% | |
| Same, both posts 3+ likes | 508,959 | 64.1% | |
| Ordinary view pairs, tech audience, after the cutoff | 24,481 | 63.1% | 62.8% |
| Generalization test bed, like pairs, scored as a customer (source unknown) | 368,970 | 66.4%, +3.1 (+2.6 to +3.6) over the bge-small broad models | 59.7% |
| Test bed, 2x or more / both 3+ likes | 262,522 / 257,682 | 69.8% / 67.6% | |
| Test bed, view pairs | 124,669 | 65.3% | 61.7% |
| Test bed slices, like pairs: full timelines / 2026 corpus / brands / accounts under 5k followers | 247,210 / 109,725 / 12,035 / 46,007 | 68.1% / 63.0% / 61.2% / 64.7% | 60.7% / 58.3% / 53.7% / 59.3% |
| Same-topic benchmark, all pairs 1.25x apart | 2,852 | 61.7% (59.3 to 63.9) | 60.0% (57.9 to 62.0) |
| Benchmark, 2x apart | 2,030 | 63.3% (60.5 to 65.7) | 61.4% |
| Benchmark, 2x apart and both posts 3+ likes | 1,086 | 64.4% (61.2 to 67.5), +4.1 (+1.5 to +6.9) | 60.2% |
| Benchmark, stated 60%+ on pairs 2x apart: share, correct | 2,030 | 32%, 73.7% (69.9 to 77.2) | 24%, 70.4% |
| Benchmark, surest 40% of pairs 2x apart | 812 | 72.7% (69.1 to 76.0) | 67.6% |
| Benchmark, wording differs and both posts after 2026-04-30, 2x apart: all calls; stated 60%+ | 953 | 65.5%; 40% at 73.1% (68.1 to 77.9) | 61.6%; 31% at 67.3% |
| Launch pairs (launch members), all 1.25x apart | 850 | 67.2% (63.2 to 71.3) | 63.6% (59.4 to 67.9) |
| Launch pairs 2x apart: all calls; stated 60%+: share, correct | 568 | 72.0%; 54%, 82.0% (75.8 to 87.6) | 66.0%; 35%, 77.8% |
| Launch pairs 2x apart, both posts 3+ likes | 535 | 72.7% (67.4 to 77.9); 54% at 81.9% | 66.9%; 34% at 77.9% |

Confidence scales were fitted on pre-cutoff pairs and not rescaled on any test set. Stated probability against outcome
on the benchmark: stated 50 to 55%: 54% right (1,267 pairs); 55 to 60%: 63% (643); 60 to 70%: 67% (629); 70 to 80%: 79%
(233); 80 to 90%: 75% (69). On launch pairs: 60 to 70%: 76% (239); 70 to 80%: 80% (132); 80 to 90%: 80% (46).

### Forecast level (how far the predicted median is from the actual median, per account)

Scored as a customer: full timeline, source unknown.

| Accounts | Median account | Median absolute factor | Accounts inside 1.5x | Before |
|---|---|---|---|---|
| All full timelines (490) | 1.03x (1.00 to 1.07) | 1.18x | 90% | 1.03x; 78% |
| twitter100m full timelines (454) | 1.02x (1.00 to 1.06) | 1.18x | 90% | 1.00x; 79% |
| Brand timelines (36) | 1.11x (1.03 to 1.22) | 1.20x | 86% | 1.41x; 56% |
| 100,000 followers and over (86) | 0.92x (0.85 to 0.98) | 1.20x | 85% | 0.71x; 59% |
| Under 1,000 followers (92) | 1.17x (1.10 to 1.19) | 1.20x | 90% | 1.19x; 89% |
| 2026 view corpus, a search sample (433) | 1.09x (1.06 to 1.12) | 1.19x | 82% | 1.06x; 80% |

Launch posts: on 576 test launch posts the forecast is 1.31x the account's usual and 1.63x with the post-type correction;
they actually got 2.19x. Before this bundle: 1.10x, and 1.50x with its correction.

### With a language model as second opinion

One fixed rule per kind and per model, fitted on pre-cutoff pairs from stored judge answers, scored on the test sets
once. "Speaks" = gives a verdict.

| Test set, pairs 2x apart | Pairs | `gpt-6.1-sol`: all calls; speaks on; right when it speaks | Before, same judge | `gpt-6-luna` | Before, same judge |
|---|---|---|---|---|---|
| Launch, all | 568 | 80.5%; 87%; 86.0% (81.3 to 90.5) | 80.8%; 89%; 84.2% | 76.6%; 84%; 81.5% (76.4 to 86.0) | 76.8%; 86%; 79.2% |
| Launch, wording differs, after the judge's cutoff, both posts 3+ likes | 487 (sol), 426 (luna) | **83.4%; 89%; 88.2% (84.0 to 91.4)** | 83.6%; 92%; 85.7% (81.4 to 89.2) | 78.4%; 84%; 83.5% (79.3 to 87.8) | 78.2%; 86%; 80.9% |
| Benchmark, all | 2,030 | 71.6%; 20%; 85.0% (80.3 to 89.2) | 71.1%; 50%; 79.6% | 69.3%; 43%; 77.3% (73.3 to 80.7) | 67.8%; 41%; 77.0% |
| Benchmark, wording differs, after the judge's cutoff, both posts 3+ likes | 673 (sol), 630 (luna) | **69.2%; 30%; 80.8% (74.5 to 87.1)** | 67.9%; 62%; 74.2% (69.4 to 78.7) | 64.8%; 51%; 71.8% (66.2 to 76.9) | 61.7%; 46%; 68.2% |

All pairs 1.25x apart: launch with sol 75.6% (70.7 to 80.0), before 75.6%; benchmark with sol 68.3% (66.1 to 70.3),
before 68.2%; launch with luna 70.5%, before 71.2%; benchmark with luna 66.0%, before 64.4%.

Read it this way. With `gpt-6.1-sol` the share of all calls that are right is the same as before on launch pairs and one
point higher on the benchmark: the language model carries most of that number. What changed is which calls are voiced.
On launch pairs the verdicts are right 88% of the time instead of 86% on 89% of pairs instead of 92%. On same-topic pairs
the refitted rule voices only calls stated at 70% or more: 30% of pairs at 81% right, where the previous rule voiced 62%
at 74%. With `gpt-6-luna` the benchmark rows are better (all calls on the strict slice 64.8% against 61.7%) and the launch
rows are level in all calls with more accurate verdicts (83.5% against 80.9%). With no language model the bundle is better
on every row.

## Serving

- What is sent to OpenAI for the embedding: the rendered text (author name and handle line, then the post text) of each
  version and of the account's most recent posts the model reads (at most 48). No counts, no follower numbers, no media.
  Each text once: vectors are cached per account.
- Not sent: posts supplied with a request unless `share_past_posts: true`; anything when `external_embedding: false`.
  Such requests are answered by the fallback models and the report says so.
- Provider failure: the fallback models answer the whole request, marked in the report; without a fallback the request
  fails with 503. A report is never built on missing vectors.
- Measured on 4 cores, three corpus accounts (`serve_check/summary.json`): first request for an account 1.1 to 1.8 s,
  49 to 51 texts sent, 2,600 to 6,000 tokens, 0.0003 to 0.0008 US dollars; the same request again 0.45 to 0.65 s, nothing
  sent, 0 dollars; one edited version 0.7 to 1.2 s, one text, 0.00001 dollars; six versions 1.35 s. The fallback path
  1.5 to 2.5 s. With `gpt-6.1-sol` as judge: 2.9 to 4.1 s and 0.008 to 0.014 dollars for two versions, 7.6 s and 0.13
  dollars for six.
- The served vectors equal the training vectors (cosine 0.9996 to 1.0 on 40 texts), and the bundle's serving path
  reproduces the evaluated gaps to 1e-7 (`scripts/combine/parity.py`).

## Known weaknesses

- **The luck ceiling.** One account reposting identical text lands within 2x only 62% of the time. No model can be right
  on much more than about 80% of benchmark-style pairs, or 85% at 2x. Launch posts are still forecast low (1.63x of usual
  against 2.19x actual): launches did better after the cutoff than before it, and that cannot be learned from earlier data.
- **Accounts under 1,000 followers.** The forecast is 1.17x high for the median such account (their median post has one
  or two likes) and ranking is weakest there: 60.9% of like pairs on the test bed (68.1% on all full timelines), 60.8% on
  the 309 benchmark pairs.
- **Very large accounts on same-topic pairs.** 134 benchmark pairs at 100,000 followers and over: confident calls are 66%
  right, not 74%.
- **Launch pairs are not drafts.** They are two different launch posts of one account, a median of 55 days apart. No set
  of same-day launch drafts with outcomes exists, so the launch numbers are an upper reading of what a draft comparison
  can expect.
- **The choice of launch members rests on validation that the test did not confirm in accuracy.** Before the cutoff the
  launch members were 3.1 points ahead of the forecast members on launch pairs (+1.7 to +4.6); after it the two are level
  (67.2% and 67.5%), and the launch members keep only more confident calls (51% of pairs against 46%, correct confident
  calls per 100 pairs +4.9, +1.5 to +7.9). For an account outside the archive the choice is untested.
- **The same-topic rule with `gpt-6.1-sol` is strict and loosely determined.** It was fitted on 809 development pairs and
  its speak rule chosen on 206; it speaks on 30% of benchmark pairs 2x apart. The goal of 40% coverage at 75% or better is
  met on launch pairs and, with this judge, not on the benchmark (30% at 81%).
- **Views.** 63.1% on tech-audience view pairs, no better than before within the interval; the launch members were 64.3%.
- **Time.** For the earlier models stated confidence meant 4 to 7 points less after the cutoff than before it. For this
  bundle the benchmark's post-cutoff slice is 1 point below the whole on confident calls (73.1% against 73.9% at 2x), and the
  launch scale held (82.6% on the later-time validation cell, 82.0% on the test pairs). The next months are not measured.
- **A provider dependency.** The embedding is an API model. The store in the dataset reproduces training without the
  API; serving needs it or falls back to the bge-small models, whose numbers are the "before" column.
- **Not in the bundle:** the pair-loss fine-tune. It did not pass its validation test (+1.0 correct confident calls per
  100 development pairs 2x apart, -0.5 to +2.4). On the benchmark, scored afterwards, the fine-tuned pair made 4.2 more
  correct confident calls per 100 pairs 2x apart than its control (+2.4 to +6.2). That is a lead for a follow-up with a
  validation set built like the benchmark, not a result this bundle may claim.
- English-language tech, startup and company accounts. Nothing here says how the models do elsewhere.

## Load it

```python
from xviral.app import service
svc = service.load_bundle("bundle/")                 # OPENAI_API_KEY in the environment, or a key-injecting proxy
report = svc.analyze({"handle": "yourhandle", "followers": 12000, "versions": [{"text": "..."}, {"text": "..."}],
                      "past_posts": [...], "share_past_posts": True})
report["embedding"], report["cost_usd"], report["comparison"]
```

Service: `set -a; . bundle/service.env; set +a; uvicorn xviral.app.server:app --port 8000`.
Dry run without any provider: `XVIRAL_EMBED_STUB=1` (the numbers then mean nothing).
