# Product track: launch advisor

2026-11-16. Code: `patch.diff` (applies cleanly to the branch head; `git apply --binary patch.diff` from the
repository root). Nothing was committed or pushed. Tests: 126 pass, 1 skipped (it was skipped before), including 10
new tests in `tests/test_app.py`. Screenshots in `screenshots/`. A sample report as JSON (`sample_report.json`) and as
the downloadable page (`sample_shared_report.html`). Timings: `speed.json`.

## 1. Speed

### Where the time went

A request takes `history lookup → features → text embedding → model passes → advisor's second pass`. Profiled with the
three production checkpoints (ts46_s1, ts46_s2, compare_v2/full) on 4 cores:

| Stage, original path | Account with 300 posts, first request | Repeat request |
|---|---|---|
| Text embedding | 31.9 s (every history post, one batch of 256 padded to its longest post) | 0.01 s (the scorer already cached by text) |
| Features | 0.35 s (whole feature chain run once per draft) | 0.35 s |
| Model passes | 0.10 to 0.23 s (ensemble run once for the mixture, then again for each member) | same |
| History lookup | 0.2 s (parquet scan) | 0.2 s |
| Advisor's second pass (spacing) | one more `score` call, only when the draft is under 2 hours after the previous post | |

The encoder is the cost: 96% of the first request. The model reads only the newest 48 history posts (`max_history`),
but every post of up to 300 was embedded.

### What changed (`xviral/serve.py`, `DraftScorer.fast`, on in the service)

1. **Embed only what the model can read.** The drafts and the newest `max_history` history rows (ties on the timestamp
   kept). Older rows reach the model only through trailing features, which use no embedding. Their embedding stays
   zero, which nothing reads.
2. **Smaller encoder batches** (16, sorted by length) so a short post is not padded to the longest one in the history.
3. **All drafts and operator variants featurised in one pass.** Each extra draft gets a copy of the history under an
   author ID of its own, so the per-author feature functions keep every draft isolated. The frame is checked equal,
   row for row and column for column, to the per-draft frame (test).
4. **Each ensemble member runs once.** The mixture is built from the members' own outputs (`Ensemble.combine`), and
   each member's report reuses its output; the summary is computed once per episode, not once per candidate.
5. **Caches across requests.** Per-account history from the corpus and its own-record facts (LRU of 256 accounts,
   invalidated when the corpus file changes); embeddings by text (already in the scorer); `text_metrics` (X's
   weighted length, a pure-Python loop per character) memoised by text.
6. **Warm-up.** `GET /api/account` embeds an account's readable history when the handle field loses focus, so for a
   known account the first analysis usually does not wait for the encoder.

### Result

12 corpus accounts drawn at random (seed 1) among those with a follower count and at least 100 posts. Same process,
same requests, three versions unless noted, posting time fixed. Seconds:

| Account | Posts | Original, first | Fast, first | Original, repeat | Fast, repeat | Original, 6 versions | Fast, 6 versions | Largest relative difference |
|---|---|---|---|---|---|---|---|---|
| ntvuganda | 100 | 4.0 | 1.3 | 0.5 | 0.2 | 0.8 | 0.4 | 4.3e-06 |
| yoheinakajima | 145 | 17.6 | 2.6 | 0.5 | 0.3 | 0.8 | 0.4 | 5.5e-06 |
| EylonALevy | 100 | 12.9 | 2.5 | 0.6 | 0.3 | 0.8 | 0.4 | 3.3e-06 |
| ludwigABAP | 299 | 30.2 | 1.6 | 0.6 | 0.3 | 0.8 | 0.4 | 2.5e-05 |
| LastingCzardd | 300 | 32.1 | 3.9 | 1.0 | 0.8 | 1.5 | 1.0 | 3.3e-06 |
| gabriel1 | 300 | 26.9 | 2.2 | 0.5 | 0.2 | 1.0 | 0.4 | 6.8e-06 |
| xsphi | 100 | 7.6 | 1.5 | 0.5 | 0.3 | 0.7 | 0.3 | 7.6e-06 |
| davidmanheim | 203 | 24.4 | 1.7 | 0.6 | 0.3 | 1.0 | 0.3 | 1.6e-06 |
| Elohim_Gadol | 100 | 10.8 | 2.6 | 0.7 | 0.3 | 0.9 | 0.4 | 1.4e-05 |
| ID_AA_Carmack | 165 | 20.6 | 2.7 | 0.7 | 0.3 | 0.9 | 0.4 | 1.1e-06 |
| theo | 300 | 30.3 | 1.4 | 0.7 | 0.3 | 0.9 | 0.4 | 1.1e-06 |
| UnderwaterBepis | 300 | 28.8 | 2.4 | 0.6 | 0.3 | 0.7 | 0.4 | 2.2e-06 |
| **Median** | | **22.5** | **2.3** | **0.6** | **0.3** | **0.9** | **0.4** | |
| **Worst** | | **32.1** | **3.9** | **1.0** | **0.8** | **1.5** | **1.0** | **2.5e-05** |

(The `text_metrics` cache also helps the original path in this table, since both run in one process; before it,
LastingCzardd, an account with 4,000-character posts, took 5.2 s first and 1.6 s on repeat on the fast path.)

**Equality.** In all 24 compared reports (first request and the six-version request per account), every status,
verdict, pick, checklist line, rule and string is identical; the largest relative difference in any number is 2.5e-5.
That difference comes from the encoder batch size (floating-point accumulation order, then a float16 cast); with the
same encoder batches the fast path is exact. The tests prove it with a deterministic stand-in encoder:
`test_fast_path_gives_the_same_scores` (one model and a two-member ensemble; 1, 2 and 6 drafts; history longer and
shorter than the model's window; batched feature frame equal to the per-draft frame) and
`test_fast_service_reports_match_and_skip_the_encoder` (whole reports, including a pasted history, a new account and
the advisor's spacing pass; the repeat request calls the encoder zero times). Reproduce the timing table with
`scripts/product/bench_speed.py`.

Measured in the browser through the page: 0.8 to 1.9 s from clicking Analyze to the rendered report for @supabase
with three versions and a 1.6 MB video upload.

## 2. Launch planner

All of it is computed from the evidence files; drop a grade from `launch_evidence.json` and the line disappears
(tested).

- **(a) Launch-day plan.** Shown when any version reads as a launch (the `post_type` rules), or when forced with
  `launch: true`. Seven lines, strongest evidence first: names the product in the first line (A), three or more lines
  (A-), a video or GIF (B, with both comparisons: against no media and against an image, and the note that an image is
  not established as better than none), opens with the announcement (B), an original post (B), no external link (B),
  at least 2 hours after the previous post (C, pooled across all accounts, with the within-account 0.71x for launches
  under 2 hours). Each line shows grade, win rate with interval, pairs and accounts, and the likes ratio, and a ✓ / ✗
  per version. The 2-hour line is checked only when a posting time is given; without one it shows "–" and says why.
  The number check ("a number in the first line", the weakest finding) stays in the per-version checklist but is not
  one of the plan's lines, which are the seven named in the brief.
- **(b) Up to six versions compared.** Every pair is scored in the same pass and judged by the same gates as any
  recommendation (calibrated probability ≥ 60% and all ensemble members agreeing). A winner is named only when it
  clears the gates against every other version; otherwise "too close to call", naming the versions still in contention
  and those predicted worse than another. Version 1 can win. Every pair is listed under the table.
- **(c) Benchmarks.** Each version's forecast against the account's own usual (median of recent posts, as a ratio), and,
  for a launch, what launch posts did for accounts of the same size: a new table `xviral/data/launch_by_size.json`
  (`scripts/launch/by_size.py`), built from the launch track's `launch_posts.parquet` with its yardstick and account
  resampling. Tiers: under 1,000 followers (14 posts, too few: the page says so), 1k to 10k: 2.2x the account's usual
  (1.7 to 3.1), 119 posts / 56 accounts; 10k to 100k: 2.5x (2.1 to 3.0), 560 / 161; 100k+: 2.1x (1.9 to 2.4), 692 / 83.
  The lift times the account's usual is drawn as a band on the chart next to the forecast ranges, and the page states
  that the forecast does not include this lift.
- **(d) Export.** "Download JSON" (the report itself) and "Download page" (a static copy: same styles, no script, no
  external request, dark mode kept), plus print to PDF. Reports are not stored on the server, so there is no hosted
  share link; that needs storage and accounts (section 5).

## 3. Media inputs

`xviral/app/media.py`. A version can carry a file (multipart `media_1` … `media_6`). The server probes it in memory:
image format, width, height, whether it is animated (an animated GIF or WebP counts as video); video container,
width, height, duration and audio track via `ffprobe` (a temporary file deleted at once). The scorer is told "image" or
"video" as before, and the measured width, height and duration go into the media record, so X's rule that a video must
be longer than 10 seconds for the out-of-network video index is now checked on uploaded videos. A ticked box gives no
length, and the report now says the length is unknown instead of reporting the rule as failed (a fix). Because duration
and size are model inputs, an uploaded file can move the forecast relative to a ticked box (seen in the screenshots:
version 2 went from 81% to 77% once its 34-second length was known).

**The plug-in interface** for the media tracks: one function `(MediaInfo, bytes) -> MediaUnderstanding | None`, set
with `XVIRAL_MEDIA_ANALYZER=pkg.module:function`. Its `attributes` (vocabulary in `media.ATTRIBUTE_VOCABULARY`:
product_screenshot, product_demo, talking_head, screen_recording, …, plus booleans such as has_captions and shows_face)
appear in the report, and `attributes` and `embedding` reach the scorer in the draft as `media_attributes` and
`media_embedding`, ignored by `DraftScorer` today. A failing plug-in leaves the version scored on its kind alone
(tested). Documented in `docs/launch_advisor_api.md`.

## 4. Accounts we do not know

"Import X analytics CSV" on the page posts the file to `POST /api/import/analytics`, which reads it with
`xviral/sources/analytics.py` (both known layouts), drops replies and rows without text, keeps the newest 500, and
returns `past_posts` with a summary (rows, kept, skipped, date span, the handle in the post links, which the page fills
in). Nothing is stored. Tested end to end, including an unrecognised file. Screenshots 3 and 4: 113 posts imported
(7 replies left out), then a report for the account.

## 5. Product basics

- **Validation and errors.** Pydantic models with `extra="forbid"`; every error is JSON with a readable `detail` shown
  as is by the page, for example "version 1, text: is too long (at most 4000 characters)", "at most six versions can
  be compared", "an X handle has at most 15 letters, digits or underscores", "follower count is required for an
  account we have no posts for", "version 1: the attached file is not an image … or a video … we can read". 413 for
  oversize bodies, 503 when not configured, and a JSON 500 with the request ID for anything unexpected.
- **Logging.** One JSON line per API request: ID, path, status, ms, versions, character counts, media kinds,
  whether a handle was given, past-post count, history source and size, time per stage. No draft text unless
  `XVIRAL_LOG_TEXT=1` (tested by searching the captured log for the draft).
- **Rate limit.** Token bucket per caller (`XVIRAL_RATE_PER_MIN`, default 30; burst 10); 429 with `Retry-After`.
- **Health.** `/api/health` (liveness, load error) and `/api/ready` (503 until loaded; used by the Docker health check).
- **Docker.** `virality/Dockerfile`, built from the repository root because xviral reads X's code by path
  (`phoenix/reference` for post rendering, `home-mixer/params/param.rs` and `vm-ranker/params.rs` for weights). CPU
  torch, ffmpeg, the encoder downloaded at build time, non-root user, health check, models mounted. Verified here:
  built, ran, healthy, analysed a request and probed an uploaded video inside the container. Image 4.1 GB. In this
  sandbox the build needed a mirror for the base image (Docker Hub answered 429) and the session's proxy CA, added
  only in a throwaway copy of the Dockerfile; the shipped file has neither.
- **Configuration.** Environment variables only, listed with defaults in `server.Settings` and in the API doc.
- **API documentation.** `docs/launch_advisor_api.md` (endpoints, every field, errors, privacy, media interface,
  billing seam), and the live OpenAPI schema at `/api/docs`.
- **Accounts and billing: the seam.** `xviral/app/accounts.py`: every API call goes through
  `Gate.authorize(...) -> Caller` and `Gate.record_usage(caller, Usage)` (counts, never text). The default gate admits
  everyone keyed by IP, or checks `XVIRAL_API_KEYS`; a billing layer is a class with those two methods loaded with
  `XVIRAL_GATE`, raising `Denied(401|402|403, message)`. The rate limiter already keys on `Caller.id`.

## 6. Design

The page was rebuilt (`xviral/app/index.html`, no dependencies). Order follows the decision: verdict first, then the
versions table with each version's standing, likes, range, views, ratio to the account's usual and launch checks met,
then one chart (each version's 80% range and median on a log scale, the account's usual as a dashed line, launch posts
for accounts of its size as a band), then the launch plan as a grid of checks by version, then single changes, X's
rules, the account's own record, and how far to trust it. Colour carries only state (green better, red worse, amber
warning), always with a word. Light, dark (system or toggle, remembered) and print styles; phone layout with no
horizontal scroll (checked at 390 px); keyboard focus rings; hover tooltips on the chart.

New sentences in the report, by kind: verdicts ("Version 2 is predicted to do better than every other version (77%
likely or more against each; …)", "Too close to call between versions 2 and 3. Version 1 is predicted to do worse
than another version."); forecasts ("Typically 113 likes and …"); measured effects (each plan line, the size tier
line); a rule from X's code (the video-length line). The rest is labels, instructions for the form and descriptions of
method. No generated advice was added.

## Fixes along the way

- `xviral/data/` was never in the repository: virality's `.gitignore` rule `data/` also matched it, so
  `edit_evidence.json` was missing (the pairs track logged two test failures from this). The rule is now `/data/`.
  `edit_evidence.json` was rebuilt with the repository's own scripts (`tac_pairs.py`, then `build_edit_evidence.py`)
  on `work/filtered/posts.parquet`; the values match research/11 (image or video 64.9%, three or more lines 61.5%,
  link not first 61.3%, 1.5x longer 58.7%). Its `spacing` entry (0.72x views, 0.68 to 0.76, 465 accounts) was taken
  from `research/11_within_account_effects.json`, the source the advisor's wording cites. `launch_evidence.json` was
  copied from `tracks/launch/`.
- An unhandled error returned plain text; now JSON with the request ID.
- A test of mine compared two reports scored at "now" a few milliseconds apart; it failed once under load. Every
  request in it now carries its posting time.

## Limits

- The first request for an account the corpus does not hold still runs the encoder on up to 48 history posts: 1.3 to
  3.9 s here, longer for long-form accounts. The warm-up call hides it for corpus accounts once the handle is typed.
- Analyses are serialised in one process; the rate limiter and caches are in memory. Scale is more processes or
  machines, with sticky routing by handle for cache hits.
- The forecast for a launch post is still about half of what launch posts get (launch track: the model expects 1.03x
  the account's usual; launches get 2.19x). The page shows the measured lift beside the forecast and says the
  forecast does not include it; correcting the forecast is the posttype track's job.
- The size benchmark uses the largest follower count seen for each account, is on likes only, and has no row for
  accounts under 1,000 followers (14 posts).
- Shareable page is a download, not a hosted link.
- The media seam is in place but nothing plugs into it yet; `DraftScorer` ignores media attributes and embeddings
  until a model trained with them exists.
