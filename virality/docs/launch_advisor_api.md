# Launch advisor API

The web service in `xviral/app/`. The page at `/` uses only this API. A live OpenAPI schema is served at `/api/docs`.

Every sentence in a report is one of: a forecast, a verdict validated on real pairs of posts, a rule from X's code, or
an effect measured on real posts, with its numbers. Nothing is written by a language model.

## Run

```bash
pip install -e ".[app]" sentence-transformers
XVIRAL_CHECKPOINTS=a/best.pt,b/best.pt,c/best.pt XVIRAL_SCALES=research/15_pair_confidence.json \
XVIRAL_POSTS=work/hygiene/posts.parquet XVIRAL_SID=work/emb/sid_codebook.npz XVIRAL_PRELOAD=1 \
uvicorn xviral.app.server:app --port 8000
```

Or with Docker (models and corpus mounted at `/data`):

```bash
docker build -f virality/Dockerfile -t launch-advisor .   # from the repository root; --build-arg BASE=<mirror>/python:3.12-slim if Docker Hub is rate-limited
docker run -p 8000:8000 -v $PWD/data:/data \
  -e XVIRAL_CHECKPOINTS=/data/runs/ts46_s1/best.pt,/data/runs/ts46_s2/best.pt,/data/work/model/compare_v2/full/best.pt \
  -e XVIRAL_POSTS=/data/work/hygiene/posts.parquet -e XVIRAL_SID=/data/work/emb/sid_codebook.npz launch-advisor
```

Run one process per machine (`--workers 1`). Analyses are serialised inside the process because the scorer holds the
model and its caches; a 4-core machine answers a typical request in well under a second once an account is warm.
Scale out with more machines behind a load balancer, with sticky routing by handle if you want cache hits.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `XVIRAL_CHECKPOINTS` | (required) | comma-separated `best.pt` files; several are scored as an ensemble |
| `XVIRAL_SCALES` | (required) | pair-confidence scales, `research/15_pair_confidence.json` |
| `XVIRAL_POSTS` | none | corpus parquet; accounts found there need no follower count or past posts |
| `XVIRAL_SID` | none | semantic-ID codebook the checkpoints were trained with |
| `XVIRAL_BUNDLE` | none | a `bundle.json` or its folder: one config for a whole model set (checkpoints, encoder, codebook, confidence scales, selective rule, post-type correction, judge rule, fallback set). Replaces `XVIRAL_CHECKPOINTS`, `XVIRAL_SCALES`, `XVIRAL_SID`, `XVIRAL_TYPE_CORRECTION` |
| `XVIRAL_ENCODER` | from the checkpoints | optional. The text encoder is read from the checkpoints (each stores the name of the encoder it was trained with: `BAAI/bge-small-en-v1.5`, or `openai:text-embedding-3-large@1024`). When set it must be that encoder; a mismatch is refused at start-up |
| `OPENAI_API_KEY` | none | key for the OpenAI embedding (and an OpenAI judge) when the checkpoints use it; not needed behind a proxy that injects the key |
| `XVIRAL_EMBED_TIMEOUT`, `XVIRAL_EMBED_RETRIES`, `XVIRAL_EMBED_BATCH`, `XVIRAL_EMBED_CONCURRENCY` | 20, 2, 256, 4 | API embedding: seconds per call, retries on 408/409/429/5xx and connection errors, texts per call, parallel calls |
| `XVIRAL_EMBED_STUB` | off | `1`: replace an API encoder with the built-in stub (dry runs and tests; nothing is sent, the numbers mean nothing) |
| `XVIRAL_FALLBACK_CHECKPOINTS`, `XVIRAL_FALLBACK_SCALES`, `XVIRAL_FALLBACK_SID`, `XVIRAL_FALLBACK_TYPE_CORRECTION` | none | a second model set with a local encoder (the bge-small models). It answers when the embedding provider fails and when a request does not allow its text to be sent. A bundle names its own fallback |
| `XVIRAL_SELECTIVE_RULE`, `XVIRAL_SELECTIVE_ENSEMBLE`, `XVIRAL_JUDGE_RULE` | shipped files | the confidence scales and the judge blend rule fitted for the checkpoints in use, when no bundle is given. The shipped defaults belong to the three bge-small models |
| `XVIRAL_DEVICE` | `cpu` | torch device |
| `XVIRAL_TORCH_THREADS` | torch default | intra-op threads |
| `XVIRAL_PRELOAD` | off | `1`: load models at start-up; `/api/ready` turns 200 when done |
| `XVIRAL_RATE_PER_MIN` | 30 | analyses per caller per minute (token bucket); `0` turns the limit off |
| `XVIRAL_RATE_BURST` | 10 | analyses a caller may make at once |
| `XVIRAL_MAX_UPLOAD_MB` | 64 | per media file |
| `XVIRAL_MAX_BODY_MB` | 200 | whole request |
| `XVIRAL_MEDIA_ANALYZER` | none | `package.module:function`, the media-understanding plug-in (below) |
| `XVIRAL_API_KEYS` | none | comma-separated keys; when set, the API needs `Authorization: Bearer <key>` |
| `XVIRAL_GATE` | none | `package.module:make_gate`, the accounts-and-billing plug-in (below) |
| `XVIRAL_LOG_TEXT` | off | `1`: include draft text in request logs |
| `XVIRAL_LOG_LEVEL` | `INFO` | |
| `XVIRAL_JUDGE_MODEL` | none | a language model as second opinion on version comparisons (below): `gpt-6.1-sol`, `gpt-6-luna`, or a Publik `vendor/model` name. Unset: no model is called |
| `XVIRAL_JUDGE_ENDPOINT` | by model name | `openai` (`api.openai.com`, key in `OPENAI_API_KEY`) or `publik` (`publikhq.com/api/v1`, key in `PUBLIK_API_KEY`) |
| `XVIRAL_JUDGE_DEADLINE` | 40 | seconds a request waits for the judge; after that the forecasting model decides alone |
| `XVIRAL_JUDGE_TIMEOUT`, `XVIRAL_JUDGE_HISTORY`, `XVIRAL_JUDGE_CONCURRENCY` | 45, 30, 16 | seconds per call; past posts sent; parallel calls |

## Endpoints

### `POST /api/analyze`

JSON body, or `multipart/form-data` with the JSON in the field `request` and media files in `media_1` to `media_6`
(the number is the version).

```json
{
  "handle": "supabase",
  "followers": null,
  "versions": [
    {"text": "Introducing Vector Buckets\n\nBillions of embeddings.\nQuery with SQL.", "video": true},
    {"text": "Vector Buckets are live. https://supabase.com/blog", "image": true}
  ],
  "past_posts": [],
  "when": "2026-11-18T16:00:00Z",
  "launch": null,
  "judge": true,
  "share_past_posts": false
}
```

| Field | Rules |
|---|---|
| `handle` | optional; up to 15 letters, digits or underscores, `@` allowed |
| `followers` | 0 to 10⁹; required when the corpus has no posts for the handle |
| `versions` | 1 to 6; `text` up to 4,000 characters; `image` / `video` say what will be attached when no file is uploaded |
| `past_posts` | up to 500 of `{text, posted_at, likes, views, replies, reposts, image, video}`; used instead of the corpus |
| `when` | ISO 8601; enables the 2-hour spacing check and scores the drafts at that time |
| `launch` | `true` / `false` forces the launch plan on or off; `null` shows it when a version reads as a launch |
| `judge` | default `true`: allow the second opinion of a language model when the server has one (see "The second opinion"); `false` sends nothing |
| `share_past_posts` | default `false`: allow `past_posts` to be sent to the language model's provider |

The report (`report_version` 2):

| Key | What it is |
|---|---|
| `account` | handle, followers, `history_posts`, `history_source` (`our corpus of public posts`, `posts you supplied`, `none`), `new_account` |
| `versions[]` | per version: `likes_median`, `likes_range` (10th to 90th percentile), `views_median`, `views_range`, `type`, `checklist`, `media` (kind and probed properties), `vs_version_1`, `plan_checks_met` |
| `comparison` | every pair of versions (`pairs[]`: `p_b_better`, `predicted_ratio_b_over_a`, `members_agree`, `status`); `status` is `clear_winner`, `too_close` or `one_version`; `winner`, `contenders`, `eliminated`, `statement` |
| `pick` | the verdict for the page: `version` (or null when too close to call), `confidence`, `statement` |
| `launch_plan` | when a launch: `items[]` with `rule`, `grade` (A, A-, B, C), `met` per version (`true`, `false`, `null` when it cannot be checked), `measured` (the sentence with its numbers), `numbers`, `notes`; `met_by_version`, `grades`, `basis` |
| `benchmarks` | `own_usual` (the account's median likes and views, and each version's forecast against it); `launch_for_size` (what launch posts did for accounts in the same follower tier, applied to the account's usual); `launch_type` |
| `odds` | version 1: chance of the account's own top 10%, of reaching 1x, 10x, 100x followers in views |
| `changes[]` | single edits of version 1 scored against it, each with `status` and its measured evidence |
| `rules[]` | X's code: distribution checks that warn or fail |
| `record[]` | differences in the account's own history that are large enough to state |
| `judge` | the second opinion: whether a language model was used, its provider and model, what was sent, calls, cost, added time, or why it was not used |
| `method` | thresholds and the validation behind the probabilities: `validation` is the accuracy measured for the configuration that decided this report's comparisons; `comparisons_decided_by` |
| `timing_ms` | where the time went |

A version is named the winner only when it clears the gates against every other version: a calibrated probability of
at least 60% that it does better, and every model in the ensemble agreeing on the direction.

### `POST /api/import/analytics`

`multipart/form-data` with `file` (the per-post CSV from analytics.x.com, Content, Export data) and optionally
`handle`. Returns `{past_posts, summary}`: `summary` gives the rows read, posts kept, replies and text-less rows left
out, older posts beyond 500 left out, the date span and the handle in the post links. Nothing is stored; send
`past_posts` with `/api/analyze`. Both X export layouts that `xviral/sources/analytics.py` knows are accepted.

### `GET /api/account?handle=...`

Whether the corpus has the account, how many posts, its last post and follower count. As a side effect the account's
recent posts are embedded, so the following analysis does not wait for the encoder. The page calls it when the
handle field loses focus. Not rate-limited.

### `GET /api/health`, `GET /api/ready`

Liveness (always 200, with `loaded` and any `load_error`) and readiness (503 until the models are loaded). Neither
is logged or rate-limited, and neither needs a key.

## Errors

Every error is JSON with a readable `detail` string, which the page shows as is; validation errors add `errors[]`.

| Status | When |
|---|---|
| 401 | `XVIRAL_API_KEYS` (or a gate) requires a key and none or a wrong one was sent |
| 402, 403 | reserved for a billing gate (out of credit, plan does not include it) |
| 413 | body over `XVIRAL_MAX_BODY_MB`, CSV over 20 MB |
| 422 | invalid input, for example `version 2, text: is too long (at most 4000 characters)`, `follower count is required for an account we have no posts for`, `version 1: the attached file is not an image (JPEG, PNG, WebP, GIF) or a video (MP4, MOV, WebM) we can read` |
| 429 | rate limit; `Retry-After` gives the seconds |
| 503 | not configured, or models failed to load; or the embedding provider did not answer and no fallback model set is configured (`Retry-After: 5`; nothing was scored, and the detail says the texts had been sent) |

## The second opinion (`judge`)

With `XVIRAL_JUDGE_MODEL` set, every comparison of two versions whose wording differs (the versions against each other,
proposed rewrites against version 1) is decided by the forecasting model together with a language model that reads both
versions and the account's 30 most recent posts with their like counts. The two are blended by a rule fitted on
development pairs (`xviral/data/judge_rule.json`; `research/tracks/judge_product/REPORT.md`). Single defined edits (attach
an image, move the link) and the spacing check stay with the forecasting model.

- **What is sent** to the provider: the versions (text, kind of media, link domain), the handle and follower count, and
  up to 30 past posts (text, kind of media, link domain, like count). Nothing else: no view counts, no files.
- **Opt out per request**: `"judge": false`. Nothing is sent and the report is the one a server without a judge gives.
- **Past posts that came with the request** (`past_posts`: typed in, or read from an X analytics export) are never sent
  unless the request sets `"share_past_posts": true`. Without it such a request is decided by the forecasting model
  alone. Posts from the server's own corpus of public posts are sent without that flag.
- **The report says what happened**: `judge` = `{configured, used, provider, model, why_not, sent: {to, model, versions,
  rewrites, account_line, past_posts, past_posts_source, past_posts_fields}, pairs, pairs_answered, calls, cost_usd,
  added_ms, error}`. Each entry of `comparison.pairs` and each `versions[i].vs_version_1` carries `decided_by` and, when
  both were asked, `opinions` = which version the forecasting model preferred, which the language model preferred, and
  whether they agree; `comparison.opinions.statement` sums that up and the verdict sentence says it for the winner.
  `method.validation` then states the accuracy measured for this model and each kind of pair that occurred (the same
  text is in `method.judge_validation`); the forecasting model's own record is in `method.forecast_model_validation`.
  Past posts sent are the 30 most recent from before `when`.
- **Failure**: a call that errors or exceeds the deadline leaves that pair to the forecasting model; the request never
  fails because of the judge. Calls of one request run in parallel while the forecasting model scores.
- **Cost and time**, measured with `gpt-6.1-sol` (research report, end-to-end run): two versions, 2 calls, about 3 s
  and 0.01 dollars; six versions, 30 calls, about 8 s and 0.13 dollars.

## The text embedding (`embedding`, `cost_usd`)

A model set is served with the encoder it was trained with, named in its checkpoints. With the production bundle
(`research/tracks/combine/REPORT.md`) that is OpenAI `text-embedding-3-large`, first 1,024 dimensions.

**What is sent to the provider, and when.** The rendered text of every post the model reads: the author line (display
name and handle) followed by the post text with link wrappers removed, for each version in the request and for the
account's most recent posts (at most the model's history length, 48 in the bundle). No like or view counts, no follower
numbers, no media files, no posting times. A text is sent once: vectors are cached in memory per account (512 accounts),
so a repeat request for an account sends only versions it has not seen. `GET /api/account` (the page calls it when the
handle field loses focus) embeds a corpus account's recent posts ahead of the first report.

**When it is not sent.** Posts that came with the request (`past_posts`, typed in or from an X analytics export) are
sent only with `share_past_posts: true`, the same switch that governs the judge. `external_embedding: false` keeps every
text of the request on the server. In both cases the request is answered by the fallback model set (local bge-small
encoder) when the server has one, and the report says so; a server without a fallback refuses the request with 422 and
the reason. Posts from the server's corpus of public posts are sent.

**When the provider fails.** Each call has a timeout and retries. If vectors still do not arrive, nothing is scored with
partial or substitute vectors: the fallback model set answers the whole request (`embedding.fallback_used: true`,
`fallback_reason`, `error`, and `primary_attempt` stating that the texts had been sent), or, without a fallback, the
request fails with 503 and a message that says nothing was scored.

Report fields:
- `model_set`: the name of the model set that produced the report (the bundle's, or the fallback's).
- `embedding`: `{encoder, provider, model, dimensions, external, texts, texts_sent, texts_from_cache, calls, tokens,
  cost_usd, ms, sent: {to, model, what} | null, fallback_used, fallback_reason, primary_encoder, error}`. `sent` is null
  when no text left the server for the embedding.
- `cost_usd`: `{embedding, judge, total}` for this request, in US dollars, from the token counts the providers returned.
- `method.validation`, `method.launch_validation`: the numbers measured for the model set that answered.

**Members by kind of comparison.** A bundle may name other checkpoints for one kind of comparison
(`comparison_members.launch` in `bundle.json`). They share the encoder and read the features and vectors the main
members computed, so they add no embedding call and a few milliseconds of model time (`timing_ms.model_launch`). A
comparison of two versions that both read as a launch is then decided from their predicted gap (with the launch
confidence scale and the launch judge rule); every entry of `comparison.pairs` says which members decided it
(`decided_by_members`: `forecast` or `launch`). Forecasts, ranges, odds and `predicted_ratio` always come from the main
members.

The request log line carries `embedding` = `{encoder, sent, texts_sent, cost_usd, ms, fallback_used, fallback_reason}`,
never text. `GET /api/health` reports `model_set`, `embedding` and whether a fallback set is loaded.

## Privacy and logging

With a judge configured, drafts and past posts leave the server as described above unless the request opts out; the
request log line then also carries `judge` = `{used, model, pairs, cost_usd, added_ms, why_not}`, never text.


Draft text is never written to disk or to the logs unless `XVIRAL_LOG_TEXT=1`. Each API request logs one JSON line:
request ID, path, status, milliseconds, number of versions, character counts, media kinds, whether a handle was given,
number of past posts, history source and size, and the time per stage. Uploaded media are held in memory for the
request; a video is written to a temporary file for `ffprobe` and deleted at once. An imported CSV is written to a
temporary directory for parsing and deleted. Reports are not stored: "Download page" and "Download JSON" on the page
create the files in the browser. The downloaded page is static HTML with no script and no external requests.

## Media: the interface for media understanding

Today an uploaded file is probed (`xviral/app/media.py`): kind, MIME type, size, width and height; for a video the
duration and whether it has audio; an animated GIF or WebP counts as a video. The scorer is told "image" or "video",
as with the tick boxes, and the measured size and length go into the media record that X's rules and the feature
code read: an uploaded 34-second video passes X's rule that a video must be longer than 10 seconds for the
out-of-network video index, while a ticked box with no file is reported as "length not known".

A media-understanding model plugs in as one function:

```python
from xviral.app.media import MediaInfo, MediaUnderstanding

def analyze(info: MediaInfo, data: bytes) -> MediaUnderstanding | None:
    # info.kind is "image" or "video"; data is the file's bytes, in memory
    return MediaUnderstanding(
        attributes={"content": "product_demo", "has_captions": True, "shows_face": False},
        embedding=vector,           # numpy array from the media encoder (e.g. xviral/embed/siglip.py)
        extra={},                   # transcript, per-frame features, anything else
    )
```

Set `XVIRAL_MEDIA_ANALYZER=yourpkg.media:analyze`. It is called once per uploaded file. A failure or `None` leaves the
version scored on its kind alone. What it returns:

- `attributes` appear in the report under `versions[i].media.attributes` (and `embedding_dim`, never the vector);
- `attributes` and `embedding` reach the scorer in the draft as `media_attributes` and `media_embedding`, next to
  `media`. `DraftScorer` ignores them today; a model trained with media inputs reads them in
  `xviral/serve.py` (`_draft_row`, `featurize`) with no change to the service, the API or the page;
- attribute names follow `media.ATTRIBUTE_VOCABULARY` (`content`: product_screenshot, product_demo, talking_head,
  screen_recording, animation_or_motion_graphic, chart_or_benchmark, text_card, logo_or_brand, photo_of_people, meme,
  other; booleans has_text_overlay, has_captions, has_speech, has_music, shows_product_ui, shows_face; `first_frame`
  for video), so that effects measured on typed media can be keyed on the same names and added to the checklist the way
  the launch evidence is.

## Accounts and billing: the seam

Out of scope today. Every API call passes through `xviral/app/accounts.py`: `Gate.authorize(headers, client_ip,
endpoint) -> Caller` before any work, and `Gate.record_usage(caller, Usage)` after a report (versions, media files,
history size, milliseconds; never text). The default gate admits everyone, keyed by IP for the rate limit, or checks
`XVIRAL_API_KEYS`. A billing layer is a class with these two methods, loaded with `XVIRAL_GATE`; it raises
`accounts.Denied(status, message)` to refuse (401, 402, 403). The rate limiter counts per `Caller.id`, so per-account
limits follow automatically once the gate returns account IDs.

## Speed

Measured on 4 CPU cores with the three production checkpoints (`research/tracks/product/REPORT.md`): an account with
300 posts went from 32.9 s to 1.5 s for the first request and 0.35 s after that, with identical numbers.
