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
| `XVIRAL_ENCODER` | `BAAI/bge-small-en-v1.5` | text encoder; must match the checkpoints |
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
  "launch": null
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
| `method` | thresholds and the validation behind the probabilities |
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
| 503 | not configured, or models failed to load |

## Privacy and logging

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
