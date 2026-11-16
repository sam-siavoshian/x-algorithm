# Track "marketing": outside knowledge graded, then tested on our launch posts

2026-11-16. No model was trained; this track delivers evidence and input traits.

## Headline

- 214 outside claims graded (`research_records.json`, reviewed in `MARKETING_RESEARCH.md`): 27 controlled, 48 large
  observational, 18 small observational, 78 anecdote, 23 platform code, 16 platform statement. 43 are specific to AI
  companies and none of those is a controlled study of X posts; 74 of the 82 launch-playbook records are single cases.
- 70 hypotheses written from them (`hypotheses.json`); 49 tested with new traits, 13 answered from the launch track's
  results, 8 not readable from the corpus. Verdicts: 9 confirmed, 4 contradicted, 31 tested and not confirmed, 26
  untestable here.
- New effects that pass the launch track's rule (likes against the account's own usual, same account, order balanced):

| Trait | Verdict, grade | Win rate | Likes ratio | Adjusted | Pairs / accounts |
|---|---|---|---|---|---|
| Open source / open weights wording | confirmed, B | 58.8% (51.2 to 66.2) | 1.33x (1.08 to 1.67) | 1.28x (1.03 to 1.60) | 1,129 / 102 |
| "First / best / most capable" claim | confirmed, A | 61.4% (56.0 to 66.9) | 1.53x (1.29 to 1.81) | 1.30x (1.10 to 1.52) | 1,500 / 137 |
| Nothing posted in the previous 24 hours | confirmed, B | 57.5% (51.9 to 62.4) | 1.43x (1.18 to 1.71) | n/a | 1,510 / 140 |
| Model release (a topic, not a lever) | confirmed, A | 61.2% (54.5 to 67.5) | 1.46x (1.24 to 1.71) | 1.24x (1.05 to 1.44) | 1,768 / 148 |
| Credits, discount, deal or prize offer | contradicted, C (pooled) | 27.5% (17.5 to 42.6) | 0.55x (0.37 to 0.84) | 0.48x (0.35 to 0.70) | 211 / 27 |
| Another launch 1 to 7 days earlier | contradicted, B (fragile) | 44.3% (38.1 to 50.0) | 0.79x (0.68 to 0.90) | 0.80x (0.70 to 0.91) | 1,403 / 95 |

- Outside claims our data does not support: weekday morning Pacific time (seven comparisons, none passes; 8 to 10am
  slot 53.3%, 48.9 to 57.6, likes ratio 1.13x, 0.98 to 1.28); a call to action (52.6%, 45.6 to 59.5); "free" (50.1%,
  39.4 to 59.8); a waitlist (51.3%, 36.6 to 68.2); a teaser in the 72 hours before (54.9%, 41.1 to 67.8); a benchmark
  claim in the text (54.5%, 48.7 to 60.5); an availability line (49.5%, 44.6 to 55.1).
- Outside claims contradicted using the launch track's results: "open on the pain point" (announcement first wins
  72.5% for the post text) and "text beats video" (video wins 65.4% inside accounts).
- Coordinated first-hour amplification cannot be tested: launches quoted by another corpus account within an hour did
  1.76x (1.54 to 2.02), with a dose pattern, but launches that take off get quoted sooner. Same-team quote in the
  first hour against later the same day: 45.5% (32.0 to 64.3), 169 pairs, 17 accounts.

## What was done

1. **Review.** Three parallel literature passes (peer-reviewed work; X's code, statements and vendor studies; launch
   playbooks), each claim recorded with source, link, sample, account control, grade, and whether the figure was
   re-checked on the source page (17 were not; marked).
2. **Traits.** 24 text traits (`scripts/marketing/traits.py`), Pacific-time clock traits with daylight saving,
   teaser / quiet-before / launch-in-previous-week / follow-up traits from the account's other posts, and quote posts
   by other corpus accounts by delay (`build_traits.py`). Seven samples per text trait were read before any outcome.
3. **Tests.** `effects_mk.py` copies the launch track's three estimators unchanged and runs them on nine sets of
   posts, likes and reposts. `build_marketing_evidence.py` applies the launch track's rule. The launch table was
   rebuilt from the corpus and matches the published `launch_posts.parquet` on all 8,902 rows.
4. **Extras.** `context_extra.py`: how follow-up and teaser posts themselves do, quote coverage, first-hour dose.

## What did not work or could not be done

- Video length: `duration_ms` is empty for all 852 video launch posts.
- Views: not attempted (205 launch posts with views; the launch track already found nothing statable).
- Thin traits (under 200 pairs or 25 accounts even pooled): engagement-bait requests (10 posts), asking for support
  (13), price stated (16), named customers (23), investors named (39), backstory (26), analogy pitch (10), Product
  Hunt mention (8), several images against one (49).
- Regular-expression precision is uneven: about half of the matches for credits / discounts, named customers,
  backstory, "coming soon" and "backed by" were correct when read. The credits result should be re-checked with a
  better label before it is shown to users.
- The amplification question needs data the corpus lacks: timestamps of all quotes and reposts, not only those by
  corpus accounts, and ideally accounts that arrange quotes for some launches and not others.

## Honest limits

- Observational, likes only, mostly technology and AI accounts over 10,000 followers.
- Open source, "most capable" and model release describe what is launched as much as how it is written; they are
  inputs the model needs, not proven edits.
- About 45 new comparisons; one or two passing by chance would not be surprising. The fragile one is "another launch
  in the previous week" (interval ends at 50.0%, absent after April 2026 and for accounts over 100,000 followers).
- The outside review was done by web search in one day; vendor pages were sometimes read through search summaries.

## Suggested next steps

1. Add the tier-1 traits of `model_traits.json` as model inputs together with post type (the "posttype" track):
   open-source wording, superlative claim, credits offer, hours since previous post, launches in the previous 7 days,
   model release.
2. Wire into the advisor only: "leave 24 hours before the launch post" (with the 2-hour rule), and, as information
   not advice, "open-source releases get about 1.3x". Do not show timing-of-day, call-to-action, waitlist or teaser
   advice; the corpus gives no support for them.
3. Collect quote and repost timestamps for launch posts to settle first-hour amplification; fill video durations.

## Files

| File | Content |
|---|---|
| `MARKETING_RESEARCH.md` | graded review with links, the tests, contradictions, traits for the model |
| `research_records.json` | 214 outside claims: source, url, sample, account control, grade, AI-specific, verified |
| `hypotheses.json` | 70 hypotheses: claim, sources, outside grade, trait, verdict, win rate, interval, ratios, pairs, accounts |
| `marketing_evidence.json` | launch_evidence.json shape, one entry per trait (`supported` true for those that survive), strict set, periods, tiers, reposts, model releases only, `_context_extra`, `_meta` |
| `model_traits.json` | traits the model should receive, in three tiers, and those not to add |
| `effects_mk.json`, `effects_mk.log`, `context_extra.json` | every estimate |
| `marketing_table_launch_posts.parquet` | launch-type posts with every trait and outcome |
| `patch.diff` | `scripts/marketing/*.py` and this folder |

Reproduce from `virality/` with `data/work/filtered/posts.parquet` and `data/launch/accounts.parquet` in place:
`scripts/launch/build_table.py`, then under `scripts/marketing/`: `build_traits.py`, `effects_mk.py`,
`context_extra.py`, `build_marketing_evidence.py`, `build_review.py <dir with the three research json files>`.
