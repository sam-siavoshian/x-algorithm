# Track "scrape": the owner's home-timeline scrape, validated, converted and tested as training data

2026-11-17, one 4-core CPU machine. Everything is in the private dataset under `tracks/scrape/`. Code changes are in
`patch.diff` (nothing committed). The scrape was read and processed on this machine and in the private dataset only;
nothing from it was sent to any outside service (no OpenAI calls; 0 tokens, 0 dollars).

## Headline

- **The counts in this file are not outcomes.** Each post was saved once, at the moment X's For You feed served it:
  a median of 4.4 hours after posting, 97.7% under 72 hours. The same posts read again later by the Community Archive
  (10,609 of them) ended with 22x the likes when captured in the first hour, 3.1x at 3 to 6 hours, 1.5x at 12 to 24
  hours. The pipeline's maturity rule keeps 6,351 of 271,226 posts.
- **It is also a selected sample**, in the other direction: inside 853 accounts, the posts the feed showed have 3.4x
  (2.9 to 4.0) the final likes of the account's other archived posts.
- **Converted and validated.** `xviral/sources/home_scrape.py` reads it into the post schema (271,226 posts, 51,490
  accounts; ids exact, timestamps agree with ids in 100% of rows, 2 duplicate rows). The id fix was needed: through a
  float only 1,075 of 271,228 ids survive. The picture description goes to `media[0].alt`. The brief's "descriptions
  on about 150,000 media posts" is 58,112 in the data: the other description vectors are one placeholder vector.
  The description is X's own (Grok) annotation, read from the API response; the embedding encoders are not documented
  anywhere and could not be identified.
- **Adding it to training does not help, and with the sampling input it hurts the existing view pairs.** Two seeds
  each, control under identical conditions:

| | Control | + scrape accounts with 20+ posts, no input | the same, with sampling and capture-age input | + all posts as accounts down to one post, with the input |
|---|---|---|---|---|
| Existing view test pairs (24,481 pairs, 851 accounts) | 62.05% | 61.53%, -0.52 (-1.3 to +0.7) | 60.38%, -1.67 (-2.6 to -0.1) | 60.27%, -1.78 (-2.9 to -0.1) |
| the same, train / validation | 68.2% / 59.3% | 65.3% / 59.7% | 67.5% / 58.9% | 65.3% / 59.0% |
| Existing like test pairs (683,152) | 59.97% | -0.10 (-0.3 to +0.1) | -0.34 (-0.6 to -0.1) | -0.07 (-0.3 to +0.2) |
| Benchmark, 2,851 same-topic pairs | 59.5% | -0.2 (-1.7 to +0.9) | +0.6 (-1.1 to +2.2) | +0.6 (-0.7 to +1.8) |
| Benchmark, confident calls on 2x pairs: share, correct | 25%, 69.6% | 26%, 71.7% | 27%, 70.6% | 27%, 69.9% |
| Launch pairs (850) | 63.4% | +0.1 (-2.6 to +2.7) | -2.4 (-5.0 to +0.2) | -0.8 (-3.4 to +1.5) |
| Held-out scrape accounts, view pairs (288,055 pairs, 1,791 accounts) | 50.6% | 55.5% | 70.6% | 70.8% |
| the same, both posts captured 12 h+ (19,567 pairs, 658 accounts) | 50.9% | 53.2% | 53.0% | 53.5% |
| Forecast level on held-out scrape posts captured 12 h+ (predicted / actual views) | 0.26x to 0.29x | 0.44x to 0.45x | 0.83x to 0.86x | 1.08x |
| Forecast level on existing view test rows | 1.11x to 1.12x | 1.12x to 1.13x | 1.08x to 1.11x | 1.10x to 1.12x |

  Differences in points against the control with a 95% interval from resampling accounts. The 70% on held-out scrape
  accounts is the model reading the capture age from its input; on pairs captured a day old it is 53%.
  The drop on the existing view pairs comes from the young counts: a fifth variant that trains only on scrape posts
  captured 12 hours or more after posting (48,545 rows) is level with the control everywhere (existing view pairs
  62.13%, +0.08 (-0.6 to +1.2); benchmark +0.3 (-1.0 to +1.5); launch pairs -2.0 (-4.4 to +0.2)). Level, not better.
- **Level bias.** The source is too high by about 3x to 4x against the archive once its counts are a day old (more
  for small accounts, more against a full timeline) and too low by 3x to 10x as read. A sampling input that also
  carries the capture age fixes the forecast level on this source's own rows (0.26x to 1.08x) and leaves forecasts
  for existing accounts where they were. It does not make the source evidence about unselected posts, and it buys no
  accuracy.
- **Images.** Across accounts, with followers, capture age and the account's level controlled, the image vector adds
  a little to the text vector (residual variance explained 2.6% to 3.2%; 7.4% to 9.1% without the account level).
  Inside accounts it adds nothing to ranking on views (-0.02 points, -0.3 to +0.3, 258,579 pairs) and +0.4 on likes.
  None of the images track's kind effects reappears on views here (photo against none 0.97x, chart 0.99x, text card
  0.99x, post or chat screenshot 1.06x); the file cannot settle them, because every post in it was already selected
  by the ranker.
- **Recommendation: do not add it to the production training set.** Use it for what it holds: early counts against
  final counts for 10,609 posts, late-2025 follower counts for 594 training accounts that have none, and a list of
  1,922 audience accounts and 1,532 startup-list handles the ranker was distributing. **Re-read the 271,226 post ids
  once**: that alone turns the file into mature view labels with an early reading attached. Then whole timelines of
  the startup list, and keep every sighting instead of the first (section 7).
- No checkpoint beats the control; none is put forward.

## 1. What the scrape is

Files: `scrape/converted/` in the dataset. Code: `scripts/scrape/describe.py` (every number below is in `describe.json`),
`scripts/scrape/levels.py` (`levels.json`), `scripts/scrape/probe_encoders.py` (`encoder_probe.json`).

### 1.1 How it was collected (from `Scraper.py`)

- Endpoint: X's `HomeTimeline` GraphQL query (the "For You" feed), polled in a loop by logged-in accounts, with
  `includePromotedContent: False`.
- Only top-level timeline entries whose result is a `Tweet` are kept. Conversation modules (a post shown with its
  replies) are skipped, which is why there are no replies.
- A post id is saved once, the first time it is served (`seen_ids`), with the counts it had at that moment (`saved_at`).
  There is no second reading of any post.
- The text is `legacy.full_text`: posts longer than 280 characters are cut and end in a t.co link (34,313 posts are
  270+ characters and end in a link). The full text of long posts (`note_tweet`) was not saved.
- Saved per post: likes, view count, the author's followers, following, post count, media count and blue-check flag,
  media URLs, and `post_image_description`. Not saved: author id, reply, repost, quote and bookmark counts, language,
  bio, quoted-post id, whether the post is a quote, video duration.
- `post_image_description` is read from X's own response (`item.get("post_image_description")`, with the request
  feature `responsive_web_grok_image_annotation_enabled` on). It is X's Grok annotation of the picture, not the output
  of a captioning model run by the owner.
- `Scraper.py` computes no embedding and no derived label. The three embedding files, `is_viral`, `engagement_rate` and
  the text flags come from a later script that is not in the folder.

### 1.2 The embedding files: what can be established

| File | Shape | Rows with a vector | Established |
|---|---|---|---|
| `text_emb.npy` | 271,228 x 768, float16 | all | unit norm; mean cosine between random rows 0.28 |
| `imgdesc_emb.npy` | 271,146 x 768, float16 (compact: only rows where the mask is set) | 271,146 | unit norm; **only 58,112 rows have a real description. The other 213,034 vectors are one identical vector**, the embedding of the missing-value placeholder (the description column holds the string "NaN" there). Same encoder family as the text file is likely (same width, same norm, a post's text is closer to its own description than to another's: cosine 0.36 against 0.18) but not proven |
| `img_emb.npy` | 150,370 x 1024, float16 (compact) | 150,370 (152,241 posts carry a media URL) | unit norm; mean cosine between random rows 0.42. Covers photos and the first frame of videos. 1,024 wide, so it is not in the text space |

The encoders are not named anywhere in the folder. A check of eight open 768-wide text encoders against the stored
vectors was written (`probe_encoders.py candidates`) but the session's permission check refused to run it, and it was
not worked around. So: the text and description encoders are unknown 768-wide models with normalised output; the image
encoder is an unknown 1,024-wide model. Consequences: new posts cannot be embedded into these spaces, so the files are
usable for analysis inside the scrape only (section 5), never as model inputs at serving time.

The brief says "image descriptions on about 150,000 media posts". The data has an image **vector** for 150,370 posts
and a **description** for 58,112 (57,146 of them on a post with a media URL: 55,987 photos, 1,134 GIFs, 25 videos).
Videos have no description.

### 1.3 Integrity

| Check | Result |
|---|---|
| Ids | `id_str` is a string of digits in all 271,228 rows. `util.core.to_int` returns the exact id for every row. Read through a float, only 1,075 of 271,228 ids survive unchanged, so the fix mattered. The `tweet_id` column of the converted file is a float column, all NaN; the reader ignores it. The new reader casts string to integer directly and refuses a file whose `id_str` is not a string |
| Time | `created_at` equals the time encoded in the id to the second in 100% of rows |
| Duplicated ids | 2 ids appear twice (identical rows, same `saved_at`); 271,226 posts after removal |
| Handles | 51,505 distinct `screen_name` values, 51,490 after lower-casing (15 handles appear in two spellings) |
| Repeated text | 1,717 groups of one account posting the same text (4,693 posts); 1,215 texts posted by more than one account (3,012 posts) |
| Reposts | 6,085 rows start with "RT @" and 5,785 of them have 0 views: reposts served in the feed. Dropped by the kind filter |
| Counts | views present on every row (6,479 are 0, almost all reposts); likes above views in 313 rows (0.12%) |
| Dates | 99.3% created between 2025-10-01 and 2025-12-02; 1,767 older (back to 2016), none on or after the 2026-04-23 cutoff. Saved between 2025-10-08 and 2025-12-03, a median of 7,600 posts a day over 38 days |

### 1.4 Overlap with the existing corpus

| | Posts | Accounts |
|---|---|---|
| Scrape | 271,226 | 51,490 |
| Same post id in `work/filtered/posts.parquet` (all Community Archive rows) | 10,609 | |
| Same post id in the training rows | 4,410 | |
| Handle in the filtered corpus | 103,309 | 7,533 |
| Handle in the training rows | 47,559 | 2,275 (532 of them with 20+ scrape posts) |
| Handle of a benchmark account | 4,379 | 258 |
| Handle in `startup_accounts.parquet` | 20,163 | 1,532 (621 company, 911 founder); 690 with 5+ posts, 244 with 20+ |

Posts per account: median 1; 22,219 accounts have 2 or more, 9,469 have 5+, 5,022 have 10+, 2,520 have 20+ (146,256
posts), 865 have 50+, 297 have 100+.

### 1.5 The snapshot delay: counts were read while the post was still spreading

`saved_at` minus the post time: 10% under 16 minutes, median **4.4 hours**, 75% under 11.7 hours, 90% under 20 hours.
28.7% of posts were read in their first hour, 57.4% within 6 hours, 94.3% within 24 hours, **97.7% within 72 hours**.
The pipeline's maturity rule (counts read at least 72 hours after posting) keeps 6,351 of 271,226 posts.

How immature: 10,609 of these posts were read again later by the Community Archive. Final likes over likes at capture,
same post (posts with 5+ final likes, archive reading at least 72 hours after posting):

| Age at capture | Posts | Final / captured likes, median | Quartiles | Within 1.25x of final |
|---|---|---|---|---|
| under 1 h | 3,344 | 21.7x | 9.3x to 62.8x | 0% |
| 1 to 3 h | 1,656 | 5.2x | 3.1x to 11.7x | 0.4% |
| 3 to 6 h | 1,624 | 3.1x | 2.1x to 5.3x | 2% |
| 6 to 12 h | 1,843 | 2.2x | 1.6x to 3.7x | 6% |
| 12 to 24 h | 1,594 | 1.5x | 1.2x to 2.3x | 27% |
| 24 to 72 h | 326 | 1.12x | 1.03x to 1.39x | 64% |

Views have no second reading anywhere, but the same shape shows inside accounts: for accounts with 10+ posts, views
over the account's own median are 0.58x for posts captured in the first hour, 1.06x at 1 to 3 hours, 1.36x at 3 to 6,
1.63x at 6 to 12, 2.16x at 12 to 24 and 2.68x at 24 to 48 hours.

What this means: a count in this file is "the count at the moment X served the post", a quantity set mostly by how old
the post was. Across the file the age at capture and the account alone explain 73% of the variance of log views
(section 5). The count is not the 24-hour or final outcome the model forecasts, and no rule converts one into the
other: at 3 to 6 hours the final count is anywhere from 2x to 5x the captured one for the middle half of posts.

Posts captured two days or more after posting are a different thing again: 6,934 posts, of which 4,819 (1,691
accounts) have under 0.3 likes per 100 views against a file median of 2.2, with a median of 95,916 views. The accounts
are NFL, Fortune, SoFi, Lowe's, FTMO, Savage X Fenty and similar: paid distribution that reached the feed although
promoted content was switched off in the request. They are 1.8% of the file and should be dropped from any use.

### 1.6 Views against followers

| Followers | Posts | Accounts | Median views | Median likes | Views per follower, 10% / median / 90% | Likes per view, median |
|---|---|---|---|---|---|---|
| under 100 | 5,146 | 3,652 | 189 | 5 | 0.36 / 5.2 / 361 | 3.2% |
| 100 to 1k | 28,926 | 11,270 | 226 | 7 | 0.045 / 0.52 / 25 | 3.1% |
| 1k to 10k | 80,256 | 18,093 | 446 | 12 | 0.011 / 0.14 / 5.7 | 2.6% |
| 10k to 100k | 77,398 | 12,834 | 1,512 | 26 | 0.005 / 0.053 / 3.1 | 1.9% |
| 100k to 1M | 49,741 | 5,308 | 12,164 | 144 | 0.002 / 0.050 / 1.5 | 1.7% |
| 1M and over | 29,757 | 1,451 | 157,309 | 3,665 | 0.001 / 0.045 / 0.36 | 1.9% |

Log views on log followers: slope 0.62, correlation 0.57, residual spread 1.09 in log10 (a factor of 12 either way).
The spread inside a tier is 300x to 1,000x between the 10th and 90th percentile, most of it the capture age. Account
followers: median 4,191, a tenth under 163, a tenth over 154,000. 89% of posts are by blue-check accounts. The scrape
does contain small accounts: 34,072 posts by 14,922 accounts under 1,000 followers and 114,328 by accounts under
10,000, nearly all as one or two posts per account.

### 1.7 Topics, post types, language

- Language (the pipeline's detector): 98.8% English (268,005 posts).
- Topic rules of `xviral/text/tech.py` on posts: strict tech pattern 11.1%, broad tech pattern 13.6%, off-audience
  pattern (crypto, trading, politics) 8.4% (crypto 4.3%, politics 4.6%). So about one post in nine is on the product's
  audience by the strict rule; the rest is sport, entertainment, news, lifestyle and general chatter.
- Accounts by the pipeline's audience rule (strict tech share at least 0.15 and not below the off-audience share):

| Accounts with at least | Accounts | Posts | Pass the audience rule | Their posts |
|---|---|---|---|---|
| 1 post | 51,490 | 271,226 | 7,034 | 71,511 |
| 5 posts | 9,469 | 208,911 | 2,621 | 63,168 |
| 10 posts | 5,022 | 180,014 | 1,524 | 56,217 |
| 20 posts | 2,520 | 146,256 | 810 | 46,489 |

  With one post per account the rule is a coin flip on that post; from 5 posts up it means something. The tech and
  startup audience with a usable history is therefore about 810 accounts and 46,000 posts (20+), or 2,621 accounts and
  63,000 posts (5+): roughly a quarter of the file.
- Accounts on the startup list: 1,532 (20,163 posts); 244 of them have 20+ posts.
- Post types (`launch_labels.py`, own announcements only): 2,815 posts by 1,968 accounts: launch 1,388, feature 596,
  demo 296, hiring 239, funding 168, milestone 102, open-source 26. Own launch posts by accounts on the startup list:
  399. Own launch posts from accounts under 10,000 followers: 689 (245 under 1,000), all with views, though read early.
  Accounts with two or more own launch posts: 172, so same-account launch pairs are few.
- Media: 152,241 posts have media (first item: 101,861 photo, 47,596 video, 2,784 GIF); 118,985 have none.

## 2. The source reader and the converted source

`xviral/sources/home_scrape.py` (new; source name `home_scrape`, added to `schema.SOURCES`), tests in
`tests/test_home_scrape.py` (4 pass). Converted file: `tracks/scrape/source/posts.parquet` (271,226 posts, post schema),
with `reader_report.json`, `describe.json`, `post_meta.parquet` (language, topic flags, post type per post) and
`accounts.parquet` (posts, tech share, off-audience share, followers per handle).

The existing `sources/user_scrape.py` was written before the file arrived, for a raw JSONL of unknown layout and for
"a customer's own timeline" (the model's `user_scrape` regime is `full_timeline`). This file is neither, so it gets its
own reader and its own source name instead of being read as a full timeline:

| Field | How it is filled |
|---|---|
| `post_id` | `id_str` cast string to integer; refused if the column is not a string; asserted equal to the string after the cast |
| `created_at_ms`, `created_at_src_ms` | time in the id; `created_at` parsed. Guard: must agree to the second in 99% of rows (100%) |
| `snapshot_at_ms`, `snapshot_kind` | `saved_at`, `exact` |
| `author_id` | none in the file: `synthetic_author_id(handle)`, the same rule as the 2026 view corpus, so `unify.resolve_handles` maps it to the real id when another source knows the handle |
| `views`, `likes` | `impressions`, `likes`. Guard: likes at most views in 99% of rows (99.88%). Replies, reposts, quotes, bookmarks: null |
| `media` | one item per URL; type from the URL path (`amplify_video_thumb` and `ext_tw_video_thumb` are videos, `tweet_video_thumb` a GIF, else photo); no size, no duration, no video file |
| image description | `media[0].alt`, the schema's slot for a text description of the media; the placeholder string "NaN" becomes null. `home_scrape.image_text(posts)` returns it per post and `text_with_image(text, desc)` renders `text + "\n[image: ...]"` for an encoder |
| `kind` | `repost` when the text starts with "RT @", else `original`. Replies do not occur; quotes cannot be told from originals (the scrape kept neither `is_quote_status` nor the quoted id) |
| author snapshot | followers, following, post count, media count, blue check, display name. No bio, no account creation date |
| not read | `is_viral`, `engagement_rate`, the hand-made text flags, `recency_days`, the three embedding files |

The reader also accepts the raw JSONL (nested `creator`), skipping the vectors.

In the model, `src_regime` (the expand track's sampling input) gets six more values for this source, one per capture
age: `home_0_1h`, `home_1_3h`, `home_3_6h`, `home_6_12h`, `home_12_24h`, `home_24h_plus`
(`model/features.py`; appended after the four existing values, so earlier tables and checkpoints keep their codes).
A sampling regime alone would not do: the count depends far more on the age at capture than on anything else.

## 3. The training rows

`scripts/scrape/build_rows.py` (the pipeline stages of `scripts/expand/build_rows.py`), then the expand track's
`embed_rows.py` (bge-small, cosine to stored vectors 0.99999 on the 2,000-row check) and `build_table.py`.

| Stage | Posts |
|---|---|
| Read (one row per id) | 271,226 |
| Already a training row (any split) | -4,410 |
| Accounts with a benchmark calibration pair | -1,435 |
| Validity | 265,374 |
| Kind (reposts out) | 259,328 |
| Recency (2023 on) | 259,316 |
| English | 256,407 |
| Hygiene (25 automated accounts) | 252,686 |
| Cap of 400 newest posts per account | **250,256 posts, 50,655 accounts** |

Decisions that differ from the pipeline, each because of what section 1 found:

- **Maturity filter not applied.** It would leave 6,351 posts. Rows keep the counts as read and carry the capture age
  through `src_regime`. This is the central compromise of the track and section 4 measures what it costs.
- **Every scrape account is a separate account**, also when its handle is in the corpus (2,275 training accounts).
  Otherwise one history would step from Community Archive rows (final likes, no views) into scrape rows (likes after
  4 hours, with views).
- **No minimum number of posts**: 28,943 accounts have one post, 122,445 rows come from accounts under 20 posts. The
  variants choose with `--train-exclude`.
- **Audience flag**: the pipeline's account rule for accounts with 5+ posts (2,439 accounts, 53,771 rows); smaller
  accounts are general accounts.
- **Split by account**: 80% train (198,331 rows, 40,440 accounts), 10% validation (25,623 rows), 10% held out
  (26,302 rows, 5,171 accounts; in the table's test split, never trained on and never used to pick a checkpoint).
- As in the expand track, the verified flag and video duration are blanked in the new rows; one more input
  (`audience_zone_known`, constant in the existing rows) was set to that constant.

Table (`table/` in the track folder is not uploaded, 1.2 GB; it is rebuilt by the commands at the end): 715,931 rows,
57,876 accounts; train 501,027, validation 57,754, test 157,150; views on 404,916 rows (154,660 before). Existing rows
are bit-identical to the old table (asserted). Capture-age values of the new rows: under 1 h 68,915; 1 to 3 h 36,522;
3 to 6 h 36,516; 6 to 12 h 47,065; 12 to 24 h 47,790; 24 h and over 13,448.

## 4. Views: does adding the scrape to training help?

All models: `scripts/train_one.py`, small preset, the 46 text-structure columns, 4,000 steps, batch 64, seeds 1 and 2,
one job at a time on 4 CPU cores (`scripts/scrape/queue.sh`). Nothing was tuned: one recipe, fixed before any result.

| Name | Trains on | Training rows | Sampling and capture-age input |
|---|---|---|---|
| Control (`ctl_s1`, `ctl_s2`) | the existing table | 302,696 | none. The expand track's controls: `ctl_s1` is bit-identical to the stored `runs/ts46_s1`, `ctl_s2` is the stored `runs/ts46_s2` |
| + accounts 20+, no input (`hist_*`) | existing + scrape accounts with 20 or more posts | 403,470 | none (`--drop-cat src_regime`): the plain "add the accounts with enough history" |
| + accounts 20+, sampling input (`histsrc_*`) | the same rows | 403,470 | `src_regime` with the capture-age values |
| + all posts, sampling input (`allsrc_*`) | existing + every scrape post; accounts down to one post are single-post accounts with an empty history | 501,027 | `src_regime` |
| + posts captured 12 h+, sampling input (`h12src_*`) | existing + scrape posts read 12 hours or more after posting (younger posts stay in the table as history only) | 351,241 | `src_regime`. Added after the first results, to test whether the young counts are what hurts |

Checkpoint selection uses each run's audience validation rows: 20,684 for the control, plus 3,625 (accounts 20+) or
5,133 (all) scrape validation rows for the others. Test numbers are always on identical rows.

### 4.1 Train, validation and test side by side (same-account pairs, views)

Mean of the two seeds [seed 1, seed 2]; differences against the control in points with a 95% interval from resampling
accounts (`eval/tables.md` has every slice, for views and likes; `scripts/scrape/eval_table.py`, `summarize.py`).

| Slice | Pairs (accounts) | Control | + accounts 20+, no input | + accounts 20+, sampling input | + all posts, sampling input |
|---|---|---|---|---|---|
| **Existing accounts** | | | | | |
| Train (600 audience accounts) | 46,568 (240) | 68.18% [68.1, 68.3] | 65.31%, -2.87 (-4.3 to -2.1) | 67.48%, -0.70 (-2.9 to +0.5) | 65.28%, -2.90 (-5.0 to -1.5) |
| Validation | 19,559 (170) | 59.31% [58.9, 59.7] | 59.71%, +0.40 (-0.4 to +1.0) | 58.91%, -0.41 (-1.0 to +0.3) | 59.01%, -0.30 (-1.1 to +0.6) |
| **Test: the existing view test pairs** | 24,481 (851) | **62.05%** [62.3, 61.8] | 61.53% [61.5, 61.6], **-0.52** (-1.3 to +0.7) | 60.38% [60.7, 60.1], **-1.67** (-2.6 to -0.1) | 60.27% [60.6, 59.9], **-1.78** (-2.9 to -0.1) |
| Test, company and founder accounts | 854 (55) | 68.15% | 68.56%, +0.41 (-1.5 to +1.7) | 68.74%, +0.59 (-2.1 to +2.8) | 67.56%, -0.59 (-3.6 to +1.9) |
| Test, accounts under 10k followers | 2,080 (178) | 63.75% | 62.98%, -0.77 (-2.0 to +0.5) | 62.88%, -0.87 (-2.7 to +0.9) | 62.50%, -1.25 (-3.2 to +0.7) |
| **Scrape accounts** | | | | | |
| Train (500 accounts) | 113,556 (500) | 52.18% | 58.58% | 71.21% | 71.01% |
| Validation | 255,270 (1,746) | 51.36% | 56.20% | 68.16% | 67.98% |
| **Held out (never trained or selected on)** | 288,055 (1,791) | **50.61%** | 55.46%, +4.84 (+3.6 to +6.4) | 70.62%, +20.01 (+16.9 to +23.3) | 70.77%, +20.16 (+17.1 to +23.4) |
| Held out, both posts captured 12 h+ | 19,567 (658) | 50.94% | 53.22%, +2.28 (-0.6 to +3.8) | 53.01%, +2.06 (+1.1 to +4.4) | 53.50%, +2.56 (+1.7 to +5.4) |
| Held out, accounts with 20+ posts | 273,032 (240) | 50.70% | 55.66% | 70.77% | 70.88% |
| Held out, accounts under 20 posts | 15,023 (1,551) | 48.93% | 51.70% | 67.91% | 68.88% |
| Held out, audience accounts | 53,519 (247) | 50.48% | 56.25% | 75.02% | 74.73% |
| Held out, accounts under 10k followers | 79,089 (926) | 48.49% | 55.10% | 74.48% | 74.55% |

Likes, the same rows: existing test pairs (683,152) control 59.97%; -0.10 (-0.3 to +0.1), -0.34 (-0.6 to -0.1) and
-0.07 (-0.3 to +0.2) for the three variants. Existing train 63.05% against 62.06%, 62.00%, 60.96%; validation 61.61%
against 61.78%, 61.59%, 61.71%.

Reading it:

1. **On the existing view test pairs the scrape does not help; with the sampling input it hurts.** -0.5 without the
   input (interval includes zero), -1.7 and -1.8 with it (both intervals exclude zero, both seeds below both control
   seeds). The existing train accuracy falls by up to 2.9 points while validation stays level: the models are not
   overfitting more, they have spent capacity on something else.
2. **What they spent it on is the capture age.** On held-out scrape accounts the control ranks an account's posts by
   views at chance (50.6%), because which of two posts has more views in this file is mostly which was read later.
   The sampling-input models reach 70.6% by reading the age from the input. Take the age away (pairs where both
   posts were captured 12 hours or later) and they are at 53.0% to 53.5%: +2.1 to +2.6 over the control, about the
   same as the model without the input (+2.3).
3. **So the scrape's own view pairs are not a test of what the product does.** The product ranks two drafts on their
   eventual reach. The scrape holds reach-so-far at a random moment. The 12 h+ slice is the nearest thing in the file
   to a like-with-like comparison and every model is near chance on it (51% to 54%), with counts that still differ
   from final by 1.5x at the median.
4. Overfitting check: train minus test for the control is 6.1 points on existing accounts (68.2 against 62.1); for
   the scrape models 3.8 to 7.1. Scrape accounts: train 71.2, validation 68.2, held out 70.6, so no gap there. The
   existing validation set ranks below the existing test set for every model (59% against 62%), as in earlier tracks:
   170 accounts with view pairs.

### 4.2 The benchmarks

Two-seed ensembles (`eval/results.md` has every member and gap size; `scripts/expand/summarize.py` with
`XVIRAL_TRACK_DIR=data/scrape_track`).

| | Control | + accounts 20+, no input | + accounts 20+, sampling input | + all posts, sampling input |
|---|---|---|---|---|
| Benchmark, 2,851 same-topic pairs 1.25x apart | 59.5% (57.3 to 61.6) | 59.2%, -0.2 (-1.7 to +0.9) | 60.1%, +0.6 (-1.1 to +2.2) | 60.0%, +0.6 (-0.7 to +1.8) |
| Benchmark, 2,071 pairs 2x apart | 60.5% | 61.1% | 62.1% | 62.0% |
| Confident calls on 2x pairs: share, correct | 25%, 69.6% | 26%, 71.7% | 27%, 70.6% | 27%, 69.9% |
| Benchmark, among confident calls, difference | | -0.2 (-2.2 to +1.9) | -1.2 (-3.7 to +0.7) | -1.1 (-3.4 to +0.9) |
| Launch pairs (850 pairs, 151 accounts) | 63.4% (59.4 to 67.7) | 63.5%, +0.1 (-2.6 to +2.7) | 61.1%, -2.4 (-5.0 to +0.2) | 62.6%, -0.8 (-3.4 to +1.5) |
| Launch pairs, confident and 2x apart: share, correct | 29%, 77.8% | 29%, 75.0% | 27%, 75.6% | 26%, 77.5% |
| Predicted level of a launch post over the account's usual (actual 2.19x, 576 posts) | 0.98x | 0.95x | 1.09x | 1.10x |

Per seed the benchmark differences are -0.4 and -0.2 (no input), -0.3 and +1.0 (20+ with input), 0.0 and +0.1 (all):
level with the control, inside the 1.5 points two seeds can resolve. Per seed, launch pairs lean negative for every variant
(-0.7 to -3.2); no interval excludes zero.

**No checkpoint beats the control**, so none is put forward. All run folders (`best.pt`, `metrics.json`) are uploaded
under `runs/` for the record.

### 4.3 Is it the young counts? The 12-hour variant

Run after the results above, as a diagnosis, with the same recipe: only scrape posts captured 12 hours or more after
posting are training rows (48,545 train rows of 61,238; the younger 189,018 stay in the table as history).

| | Control | + posts captured 12 h+, sampling input (`h12src`) |
|---|---|---|
| Existing view pairs: train / validation / **test** | 68.18% / 59.31% / **62.05%** | 67.62% / 59.27% / **62.13%** [62.2, 62.1], +0.08 (-0.6 to +1.2) |
| Existing like test pairs | 59.97% | 59.96%, -0.01 (-0.2 to +0.2) |
| Existing test, company and founder accounts, views / likes | 68.15% / 64.20% | 67.92% / 65.02% (+0.81, -0.0 to +1.4) |
| Held-out scrape accounts, both posts captured 12 h+, views | 50.94% | 53.70%, +2.76 (+1.3 to +3.6) |
| Forecast level, held-out scrape posts captured 12 h+ (predicted / actual views) | 0.26x / 0.29x | 1.05x / 1.05x |
| Forecast level, existing view test rows | 1.12x / 1.11x | 1.13x / 1.12x |
| Benchmark, 2,851 same-topic pairs | 59.5% | 59.8%, +0.3 (-1.0 to +1.5); per seed -1.1 and +1.1 |
| Benchmark, confident calls on 2x pairs: share, correct | 25%, 69.6% | 27%, 71.3% |
| Launch pairs (850) | 63.4% | 61.4%, -2.0 (-4.4 to +0.2) |
| Predicted level of a launch post over the account's usual (actual 2.19x) | 0.98x (0.88 to 1.09) | 1.30x (1.21 to 1.44); both seeds 1.32x |

- The loss on the existing view pairs is gone (62.13% against 62.05%; against the all-ages variants +1.8). So the
  damage in 4.1 comes from the 76% of scrape rows read under 12 hours, whose "views" are mostly age, not from the
  source as such.
- It still buys nothing: every pair measure is level with the control and launch pairs lean negative as in the other
  variants (-2.0; the interval reaches +0.2).
- One thing moves: the predicted level of launch posts rises from 0.98x to 1.30x of the account's usual (actual
  2.19x), with identical values in both seeds, while the level on ordinary existing rows does not move (likes
  0.92x / 0.94x against 0.93x / 0.94x). The three all-age variants show 0.95x, 1.09x and 1.10x. I have no tested
  explanation; a model that has seen 48,545 day-old posts the ranker chose to distribute may have learned to rate
  announcement-style posts higher. It is a level change without a ranking gain, on 576 posts, seen once: a lead for the
  post-type correction, not a result.
- The probe on these models gives a smaller selection estimate (the same post declared "captured 24 h+" against
  "search sample": 1.5x to 1.6x on scrape accounts, 1.25x to 1.35x on existing accounts), since they never saw young
  posts; the capture-age values under 12 hours are untrained in them.


## 5. The level bias

### 5.1 What the data says (`scripts/scrape/levels.py`, `levels.json`)

Two biases with opposite signs sit in every count of this file.

**Selection, upward.** 853 accounts have both posts the home timeline showed (in the scrape) and other posts of the
same weeks in the Community Archive, and the archive read all of them later, mature. Inside those accounts the shown
posts have **3.4x the final likes of the account's other archive posts** (2.9 to 4.0; 6,379 shown posts against 39,739;
shown is higher in 89% of accounts). The archive is itself a sample of posts that travelled (about 4x an account's full
timeline, expand track), so against a full timeline the home-timeline sample sits higher still; the product of the
two, about 13x, is an extrapolation, not a measurement, because no account here has a full timeline for these weeks.
The home timeline shows 14% of these accounts' archive posts.

**Immaturity, downward.** Section 1.5: likes at capture are 1/22 of final in the first hour, 1/5 at 1 to 3 hours, 1/3
at 3 to 6, under 1/2 at 6 to 12, 2/3 at 12 to 24.

**Net.** Views against the corpus's own view-labelled posts (the 2026 view corpus, a search sample of mature counts) at
the same follower tier, paid posts removed:

| Followers | Corpus median views | Scrape median, as read | As read / corpus | Captured under 1 h | 1 to 6 h | 6 to 12 h | 12 to 24 h | 24 to 72 h |
|---|---|---|---|---|---|---|---|---|
| 100 to 1k | 142 | 224 | 1.58x | 0.20x | 1.21x | 2.41x | 3.54x | 25x |
| 1k to 10k | 1,238 | 463 | 0.37x | 0.08x | 0.36x | 0.95x | 1.69x | 8.1x |
| 10k to 100k | 13,707 | 1,566 | 0.11x | 0.03x | 0.17x | 0.70x | 1.26x | 4.3x |
| 100k to 1M | 120,731 | 13,309 | 0.11x | 0.01x | 0.17x | 0.92x | 1.07x | 0.84x |
| 1M and over | 553,800 | 159,798 | 0.29x | 0.01x | 0.36x | 0.40x | 0.42x | 0.39x |

So the file as a whole reads **lower** than the corpus for accounts above 1,000 followers (0.11x to 0.37x), because
most posts were read young, and several times **higher** once the posts are a day old for accounts under 100,000
followers (4x to 25x). The few posts captured late are also the ones that were still travelling after a day, which is
a further selection. For small accounts the selection is strongest: a 500-follower account only reaches a stranger's
For You feed when the post is far above its usual. Any single level correction for "this source" would be wrong by an
order of magnitude in one direction or the other depending on the capture age and the account size.

### 5.2 What the models do with it

Level of the view forecast, median predicted over actual (`eval/tables.md`):

| Rows | Control | + accounts 20+, no input | + accounts 20+, sampling input | + all posts, sampling input |
|---|---|---|---|---|
| Existing view test rows (5,140) | 1.12x / 1.11x | 1.12x / 1.13x | 1.08x / 1.11x | 1.10x / 1.12x |
| Existing test rows, company and founder accounts (327) | 1.11x / 1.11x | 1.10x / 1.12x | 1.12x / 1.04x | 1.06x / 1.11x |
| Existing likes test rows (95,357) | 0.93x / 0.94x | 0.94x / 0.94x | 0.94x / 0.92x | 0.90x / 0.88x |
| Held-out scrape rows, all ages (26,302) | 0.84x / 0.96x | 1.02x / 1.03x | 1.01x / 1.04x | 1.09x / 1.09x |
| Held-out scrape rows captured 12 h+ (6,533) | **0.26x / 0.29x** | 0.44x / 0.45x | 0.83x / 0.86x | 1.08x / 1.08x |
| the same, share of outcomes above the model's 90th percentile | 28% / 26% | 20% / 18% | 8% / 8% | 7% / 8% |
| Held-out scrape rows, accounts under 20 posts (12,657) | 0.59x / 0.67x | 0.85x / 0.87x | 0.87x / 0.92x | 1.07x / 1.07x |

- **The scrape does not move the level of forecasts for existing accounts**, with or without the input (1.08x to
  1.13x against 1.11x to 1.12x on views; likes unchanged except "all posts", 0.88x to 0.90x against 0.93x). The
  forecast is anchored on the account's own history, and histories were kept separate by source.
- **On the scrape's own rows the input fixes the level.** A day-old home-timeline post gets about a quarter of its
  actual views from the control (and its outcome is above the control's 90th percentile 27% of the time instead of
  10%); the model trained on all posts with the input is at 1.08x with 7% to 8% above the 90th percentile. Without
  the input the model can only average over ages (0.44x).
- **What the input has learned** (`eval/regime_probe.json`, `scripts/scrape/regime_probe.py`: the last post of 806
  held-out scrape accounts and of 594 existing test accounts, scored under each value of the input):

| Input value | Predicted views over "captured 24 h+", scrape accounts (4 models) | Final over captured, measured on likes (5.1) |
|---|---|---|
| captured under 1 h | 0.04 to 0.05 | 1/22 |
| 1 to 3 h | 0.20 to 0.23 | 1/5 |
| 3 to 6 h | 0.41 to 0.46 | 1/3 |
| 6 to 12 h | 0.57 to 0.65 | 1/2.2 |
| 12 to 24 h | 0.75 to 0.80 | 1/1.5 |

  The learned curve for views follows the growth measured on likes of the same posts. And the same post declared
  "home timeline, captured 24 h+" instead of "search sample" is predicted **3.0x to 3.7x the views** on scrape
  accounts and 1.9x to 2.3x on existing accounts (likes: 2.1x to 2.2x and 1.45x to 1.53x); "stream sample" (the
  archive) against "search sample" is 1.2x to 1.6x. That is the models' own estimate of the selection, and it has the
  size of the measured one (3.4x on likes against the archive).
- **What it does not fix, and cannot be tested here:** the level for a full timeline, which is what a customer's own
  account is. This table has no full-timeline rows (the expand track's twitter100m and brand rows are not in it), so
  no model here was ever told what an unselected account looks like, and `full_timeline` is an untrained value of
  the input in these checkpoints. The input separates this source from the others; it does not turn this source into
  evidence about unselected posts.

Answer to the brief's question: the source implies forecasts would be too high by roughly 3x to 4x against the
archive (and more against a full timeline) if its day-old counts were taken as typical, and too low by 3x to 10x if
its counts were taken as read. A sampling-regime input alone does not fix that; a sampling input that also carries the
capture age does, for this source's own rows, and it keeps the bias out of forecasts for other accounts. It buys no
accuracy.

## 6. Images: does what the picture shows predict views?

`scripts/scrape/images.py`, `images.json`, `image_kinds.parquet`. English original posts: 262,016 posts, 50,990
accounts. The stored text, image and description vectors are used as they are (encoders unknown, section 1.2).

### 6.1 Across accounts, with the account controlled

Five folds by account; every number is out of fold. Stage 1 is gradient boosting on what is known without reading the
post (followers, following, post and media counts, blue check, **age at capture**, hour, weekday, capture day, media
type and count, text length), with or without the account's level (the mean stage-1 residual of the account's other
posts, accounts with 5+). Stage 2 is a ridge regression of the stage-1 residual on the vectors.

Stage 1 alone explains 72.8% of the variance of log views (79.4% on accounts with 5+ posts, **88.8% with the account
level**); residual spread 0.69 in log10 without the account level and 0.43 with it.

| Posts | Controls | Text | Image | Text + image | Description | Text + description | Text + image + description |
|---|---|---|---|---|---|---|---|
| Photos with a description: 53,338 posts, 19,063 accounts | followers, age | 7.4% | 5.8% | 9.1% | 4.7% | 8.5% | 9.3% |
| the same, accounts with 5+ posts: 38,591 posts, 6,724 accounts | + account level | 2.6% | 1.6% | 3.2% | 1.2% | 2.9% | 3.1% |
| the same, audience accounts: 11,720 posts, 1,955 accounts | + account level | 3.0% | 1.1% | 2.7% | 0.9% | 2.7% | 2.1% |
| All photos: 97,632 posts, 25,865 accounts | followers, age | 7.1% | 5.5% | 8.6% | | | |
| First frame of videos: 46,525 posts, 14,149 accounts | followers, age | 9.1% | 7.1% | 11.5% | | | |
| Photos with a description, **likes**, accounts 5+ | + account level | 3.1% | 2.8% | 4.4% | 1.8% | 3.8% | 4.3% |

(Share of the residual variance explained in held-out accounts.)

Same-account pairs in held-out accounts, the product's question (which of an account's two posts did better):

| Posts | Pairs (accounts) | Controls only | + text | + text + image | Image over text, points | Text + image + description over text |
|---|---|---|---|---|---|---|
| Photos with a description, views | 258,579 (6,682) | 79.3% | 78.7% | 78.7% | -0.02 (-0.3 to +0.3) | +0.08 (-0.3 to +0.4) |
| the same, account level controlled | 256,034 (4,968) | 79.8% | 80.2% | 80.1% | -0.09 (-0.3 to +0.1) | +0.01 (-0.3 to +0.3) |
| All photos, views | 684,738 (10,033) | 82.1% | 80.8% | 80.8% | -0.03 (-0.3 to +0.2) | |
| Video first frames, views | 250,199 (5,062) | 71.1% | 72.5% | 72.1% | -0.5 (-1.1 to +0.1) | |
| Photos with a description, audience accounts, views | 83,003 (1,467) | 84.9% | 84.8% | 84.5% | -0.4 (-0.9 to +0.3) | -0.6 (-0.9 to -0.2) |
| Photos with a description, **likes**, account level controlled | 245,178 (4,891) | 74.2% | 74.7% | 75.0% | +0.3 (-0.1 to +0.7) | +0.4 (+0.1 to +0.8) |

- The controls alone rank 79% to 85% of pairs: that is the capture age again.
- **Across accounts the picture carries a little information about views beyond the text and the account**: adding
  the image vector to the text vector raises the explained share of the residual from 7.4% to 9.1% (followers
  controlled) and from 2.6% to 3.2% (account level controlled). The description vector adds less than the image
  vector and nothing on top of it.
- **Inside an account it does not help rank posts on views**: 0.0 points, interval a third of a point wide, on
  258,579 pairs. For audience accounts it is slightly negative. On likes there is a small gain (+0.4, 0.1 to 0.8).
- The ceiling is low for everything here: with the account level in, text plus image explain 3% of what is left.
  Most of what is left is how the count would have grown, which the file does not hold.

### 6.2 Do the images track's kind findings hold on views here?

Kinds are read from X's description of the picture by rules on its first 160 characters (`KINDS` in `images.py`),
fixed before any outcome was looked at. Read by hand, 14 to 20 descriptions per kind, one reader: code or terminal
20 of 20, text card 18 of 20, chart or table 17 of 20 (most are trading and price charts, few are benchmark tables),
person 13 of 14, diagram or infographic 13 of 14, post or chat screenshot 12 of 14, article screenshot 12 of 14,
product screen 11 of 14, meme or cartoon 11 of 14; "other photo" is a mixed remainder (6 of 20 show a named person or
a graphic). Effects are inside accounts, on the stage-1 residual (so the capture age and the follower count are out):
the kind against the account's other described photos; ratio with a 95% interval from resampling accounts.

| Kind | Images track (likes, company and founder accounts) | Here, views, all accounts | Here, likes, all accounts | Here, views, startup-list accounts | Accounts / posts (all) |
|---|---|---|---|---|---|
| Any photo against no media | 1.23x (A); 1.25x on views | 0.97x (0.96 to 0.99) | 0.94x (0.92 to 0.95) | 1.03x (0.97 to 1.10) | 10,552 / 70,752 |
| Video against no media | (launch track: video is the lever) | 0.77x (0.74 to 0.80) | 0.84x (0.81 to 0.87) | 0.94x (0.85 to 1.05); likes 1.11x (1.01 to 1.22) | 5,275 / 26,063 |
| Chart or table | 1.23x (A) | 0.99x (0.94 to 1.04) | 0.95x (0.90 to 1.00) | 1.08x (0.95 to 1.25); likes 1.12x (0.95 to 1.35) | 1,369 / 2,409 |
| Text-heavy card | 0.73x (A) | 0.99x (0.93 to 1.05) | 0.93x (0.88 to 0.97) | 1.17x (0.98 to 1.44) | 1,254 / 1,905 |
| Screenshot of a post or chat | 0.80x (A) | 1.06x (1.01 to 1.10) | 0.99x (0.95 to 1.02) | 1.01x (0.86 to 1.19) | 1,594 / 2,728 |
| Product screen | indistinguishable | 1.03x (1.00 to 1.06) | 0.97x (0.94 to 1.00) | 0.96x (0.89 to 1.04) | 3,050 / 7,797 |
| Person | indistinguishable | 1.07x (1.04 to 1.10) | 1.15x (1.11 to 1.18) | 0.93x (0.83 to 1.02) | 3,162 / 7,857 |
| Meme, cartoon, illustration | indistinguishable | 0.95x (0.90 to 1.00) | 1.02x (0.97 to 1.07) | 1.09x (0.94 to 1.32) | 1,169 / 2,227 |
| Diagram or infographic | (not a class there) | 0.92x (0.86 to 0.99) | 0.92x (0.84 to 0.99) | 0.87x (0.68 to 1.12) | 702 / 1,060 |
| Article screenshot | (not a class there) | 0.87x (0.80 to 0.96) | 0.88x (0.80 to 0.96) | 0.77x (0.56 to 1.10) | 386 / 527 |
| Code or terminal | (not a class there) | 1.06x (0.98 to 1.14) | 1.05x (0.98 to 1.12) | 1.28x (1.06 to 1.58) | 551 / 1,063 |

**None of the images track's graded effects reappears here on views.** A photo against no media is 0.97x, a chart
0.99x, a text card 0.99x, a post or chat screenshot 1.06x. In the startup-list accounts (the population the images
track measured: 706 accounts with both kinds of post, 115 with a chart) the chart effect has the same sign as there
(1.08x views, 1.12x likes) and an interval that includes 1; the text card does not (1.17x). What does show here, in
all accounts: a person in the picture (1.07x views, 1.15x likes), article screenshots down (0.87x), diagrams down
(0.92x), video against no media down on views (0.77x).

This is not a refutation of the images track, for three reasons that all come from how the file was made:

1. Every post here was selected by the ranker. If a chart helps a post get distributed, the file contains the chart
   posts that were shown and the plain posts that were shown; comparing them compares winners with winners. An effect
   on "does it get shown" is invisible in a sample of shown posts.
2. The outcome is the count at capture. The stage-1 model removes the average effect of age, not the difference
   between a post read at its peak and one read on its way up.
3. The kinds are not the same classes: "chart" here is mostly price charts from trading accounts, and the labels come
   from descriptions, not from the images track's classifier.

So: across accounts the picture carries a small signal for views; inside accounts on this file it carries none for
views; and the file cannot confirm or overturn the kind effects.

## 7. Recommendation

### 7.1 Should this source enter the production training set?

**No, not as training rows.** The numbers:

| Reason | Number |
|---|---|
| Labels are not the outcome the model forecasts | 97.7% of counts read under 72 hours after posting, median 4.4 hours; final likes are 3.1x the captured ones at 3 to 6 hours (middle half 2.1x to 5.3x). The pipeline's maturity rule keeps 6,351 of 271,226 posts |
| It does not help the existing view test pairs | -0.5 (-1.3 to +0.7) without the input; -1.7 (-2.6 to -0.1) and -1.8 (-2.9 to -0.1) with it; +0.1 (-0.6 to +1.2) when only posts captured 12 h+ are used; 24,481 pairs, 851 accounts |
| It does not help the benchmarks | same-topic pairs -0.2 to +0.6 across five variants, every interval includes zero; launch pairs +0.1 to -2.4; launch level 0.95x to 1.30x against an actual 2.19x |
| What a model learns from it is the capture age | held-out scrape pairs 50.6% to 70.6% with the age as input, 51% to 54% when both posts are a day old |
| It is off audience and thin per account | 11% of posts match the strict tech pattern; median 1 post per account; 810 audience accounts with 20+ posts |
| The stored vectors cannot be used at serving | encoders unknown; 213,034 of the description vectors are one placeholder vector |

If it is merged anyway for coverage (small accounts, late-2025 posts), merge only the posts captured 12 hours or
more after posting (61,238 rows; section 4.3: level with the control on every measure, where the all-age variants lose
1.7 points on view pairs), and keep the conditions every variant here had: its own source name and the capture-age values of `src_regime`; scrape accounts
kept apart from corpus accounts of the same handle; reposts dropped; the paid posts dropped (captured 48 hours or
more after posting with under 0.3 likes per 100 views: 4,819 posts); never the plain add, which leaves forecasts for
day-old scrape posts at 0.44x.

### 7.2 What it is good for as it stands

| Use | What is there |
|---|---|
| Early counts against final counts | 10,609 posts with likes and views at a known age (minutes to a day) and the final likes from the Community Archive. This is the only data in the project on how a post's count at hour N relates to where it ends: the table in 1.5, and the basis for an "is this post on track after N hours" reading in the product. `same_post_two_readings.parquet` |
| Follower counts for accounts that have none | 594 of the 2,258 training accounts without any follower count are in the scrape (30,429 training rows), with followers, following, post and media counts read in late 2025. 2,275 training accounts get a late-2025 account snapshot. Not tested in a model here |
| A list of accounts to scrape | 2,621 audience accounts with 5+ posts, 1,922 of them not in the training rows and 1,819 not in the corpus at all; 1,532 handles of the startup list (692 under 10,000 followers), 388 of which have no post in any other source. These are accounts the ranker was distributing in late 2025. `source/accounts.parquet` |
| Launch posts of small accounts with views | 1,388 own launch posts, 689 from accounts under 10,000 followers and 245 under 1,000 (the data track found 10 with views under 1,000); but only 276 were read 12 hours or later (165 under 10,000 followers) and 81 a day or later. Evidence only after a second reading |
| Picture descriptions | 57,146 photos with X's own description, usable to build and check a text-based image-kind labeller (6.2 precision figures); not usable for outcomes |
| The selection itself | shown posts have 3.4x the final likes of the same account's other archive posts: a measured size for "what the For You feed picks", per account |

### 7.3 What a re-scrape should do differently, in order of value per request

1. **Read these 271,226 posts again, once, by id.** They are now eleven months old. One lookup per id returns the final
   views, likes, replies, reposts, quotes and bookmarks, the full text of long posts and the quote and reply fields.
   That turns the whole file into mature labels with an early reading attached (the first reading becomes an input,
   not a label): 250,000 view-labelled posts, 34,000 from accounts under 1,000 followers, 689 small-account launch
   posts, and the first view growth curve the project has. It is still a sample of shown posts, so it enters with its
   own sampling value, but the age problem is gone. Nothing else on this list is as cheap.
2. **Whole timelines of the startup list**, originals and replies, each post read at least 72 hours after it was
   made: first the 1,532 list handles seen here and the 1,922 audience accounts not in training (known to be active),
   then the 14,732 list handles with no post in any source (the 6,987 YC handles first). This is the only route to the
   `full_timeline` regime with views for 2025 to 2026, which is what a customer's account is; the expand track
   measured a 1.5x over-forecast on such accounts and this track could not test it for lack of any.
3. **Do not drop repeat sightings.** The scraper skips a post id it has seen (`seen_ids`). Saving every sighting with
   its time and counts gives repeated snapshots of the same post at no extra request: the feed serves a travelling
   post many times over a day. For a panel of a few thousand startup accounts, add fixed readings of every new post
   at about 1, 3, 6, 12, 24 and 72 hours and 7 days.
4. **Keep the fields that were thrown away**: the author's id; `note_tweet` text; `is_quote_status`, quoted id and
   quoted text; `conversation_id` and `in_reply_to`; reply, repost, quote and bookmark counts; `lang`; media type,
   size, duration and video variants; the author's bio and account creation date; which scraper account saw the post,
   its position in the feed, and the entry's promoted and social-context markers (so paid posts can be removed by
   rule instead of by guess). Simplest: store the raw response per entry.
5. **Collect conversation modules.** The parser keeps only entries whose result is a `Tweet`; a post shown with its
   replies is a module and is skipped. Those hold the author's self-replies (threads, "link in the reply").
6. **Record the encoder of any stored vector**, or store none. A vector whose model is unknown can only be used
   inside the file it came with.
7. Scraper accounts shape the sample: three accounts' For You feeds are 11% tech. If the home timeline is used again
   for discovery, follow the startup list from the scraper accounts so the feed fills with the audience.

## 8. Honest limits

- Two seeds per variant. On the benchmark nothing under about 1.5 points can be seen, on launch pairs about 3. The
  -1.7 on the existing view test pairs has both seeds below both control seeds and an interval that excludes zero by
  0.1; it is one test set of 851 accounts.
- `ctl_s2` is the stored `runs/ts46_s2`, not a retrain here (the expand track showed `ctl_s1` retrains bit for bit).
- The runs pick their checkpoint on different validation rows (the scrape runs add their scrape audience validation
  rows, whose labels are counts at capture). Scrape runs were still improving at step 4,000 (best step 3,996 to 4,000).
- The 12-hour variant was added after seeing the first results. It is a diagnosis of why the others lose, on the same
  test sets, and is not offered as a selected configuration.
- No training length other than 4,000 steps, no loss weighting by source, no variant that uses the capture age as a
  number instead of six classes, and no variant that corrects the label with the growth curve were tried. A model
  with the early reading as an input and a mature label needs the second reading (7.3, item 1).
- Views have no second reading: every statement about view growth is from likes of the same posts, or from the
  capture-age curve inside accounts, which also contains selection (posts served late are the ones still travelling).
- "Shown against not shown" compares posts in the scrape with the same account's other Community Archive posts. The
  archive is itself a sample, and three scraper accounts are not every For You feed: a post "not shown" here was shown
  to someone.
- The comparison with the corpus's view posts by follower tier (5.1) is between different accounts and months
  (late 2025 against early 2026) and uses the scrape's follower counts against the corpus's partial ones.
- The paid-post rule (48 hours, 0.3 likes per 100 views) is a guess from the accounts it catches; the file has no
  promoted flag.
- Image analysis: unknown encoders; ridge on stored vectors, not a fine-tuned model; kinds from description rules,
  one reader, 14 to 20 descriptions per kind; descriptions exist for photos only (55,987 of 101,861 photos, none of
  the videos), and which photos X described is not random.
- The image description is carried in the converted source (`media[0].alt`) and a renderer exists
  (`text_with_image`), but no model was trained with the description in its text: the 6.1 result (nothing inside
  accounts on views) did not justify a seventh training family in the time available.
- The eight-candidate check of open text encoders against the stored vectors did not run (refused by the session's
  permission check).
- Audience flags of scrape accounts come from the pattern rule on 5+ posts, not from the corpus's selection.
- The table and the row file are not uploaded with the scrape's text in them beyond the converted source; the table
  (1.2 GB) is rebuilt by the commands below.
- Tests: `tests/test_home_scrape.py` 4 pass. Full suite: 137 pass, 1 skipped, 4 fail: three need `fastapi`, which is not
  installed here, and `test_end_to_end.py::test_phase0_end_to_end` fails in the full run and passes alone, with and
  without these changes (the expand track saw the same).

## Exact commands

From `virality/`, after the setup lines of `scripts/worker_job.sh` (plus `pip install lightgbm scikit-learn pytest`),
with `HF_TOKEN=hf_proxyinjected`. One heavy job at a time.

```bash
# data: scrape/converted/*, work/model/table/*, work/emb/*, work/filtered/posts.parquet, work/dataset/rows.parquet, config.yaml,
#       tracks/pairs/benchmark_pairs.parquet, tracks/pairs/bench/cache_*.pkl, tracks/launch/*, tracks/data/startup_accounts.parquet,
#       tracks/expand/runs/ctl_s{1,2}/*, tracks/expand/bench/ctl_s{1,2}/*, tracks/expand/launch/ctl_s{1,2}.parquet  -> data/
mkdir -p data/launch data/pairs data/bench3 data/scrape_track/{launch,eval,runs}
cp data/tracks/launch/launch_posts.parquet data/launch/table.parquet; cp data/tracks/pairs/benchmark_pairs.parquet data/pairs/
for i in 0 1 2 3; do ln -sf $PWD/data/tracks/pairs/bench/cache_$i.pkl data/bench3/cache_$i.pkl; done
for s in 1 2; do cp -r data/tracks/expand/runs/ctl_s$s data/scrape_track/runs/; mkdir -p data/bench3/ctl_s$s; cp data/tracks/expand/bench/ctl_s$s/*.parquet data/bench3/ctl_s$s/; cp data/tracks/expand/launch/ctl_s$s.parquet data/scrape_track/launch/; done

# 1. read, validate, describe
python -c "from xviral.sources import home_scrape; import json; d,r=home_scrape.read('data/scrape/converted/posts.parquet'); d.write_parquet('data/scrape_track/posts.parquet'); json.dump(r,open('data/scrape_track/reader_report.json','w'))"
python scripts/scrape/describe.py; python scripts/scrape/levels.py; python scripts/scrape/probe_encoders.py
# 2. rows, embeddings, table
python scripts/scrape/build_rows.py --out data/scrape_track
python scripts/expand/embed_rows.py --rows data/scrape_track/new_rows.parquet --out data/scrape_track/emb --check-table data/work/model/table
python scripts/expand/build_table.py --new-rows data/scrape_track/new_rows.parquet --emb data/scrape_track/emb --out data/scrape_track/table
#    then: audience_zone_known of the new rows set to the existing rows' constant; under20_ids.npy, under12h_ids.npy, orig_test_ids.npy (see REPORT section 3)
# 3. models, benchmark, launch posts
bash scripts/scrape/queue.sh; bash scripts/scrape/post.sh; bash scripts/scrape/queue2.sh; bash scripts/scrape/post2.sh
# 4. tables
python scripts/scrape/eval_table.py --run <name>=data/scrape_track/runs/<name>/best.pt ... --out data/scrape_track/eval/slices.json
python scripts/scrape/summarize.py --families ctl,hist,histsrc,allsrc,h12src --out data/scrape_track/eval/tables
XVIRAL_TRACK_DIR=data/scrape_track python scripts/expand/summarize.py --families ctl,hist,histsrc,allsrc,h12src --slices none --out data/scrape_track/eval/results
python scripts/scrape/regime_probe.py --run ... --out data/scrape_track/eval/regime_probe.json
python scripts/scrape/images.py
```

## Code changes (`patch.diff`)

- New: `xviral/sources/home_scrape.py` (the reader), `tests/test_home_scrape.py`.
- `xviral/schema.py`: `home_scrape` in `SOURCES`. `xviral/model/features.py`: six capture-age values appended to
  `src_regime`, `with_src_regime` picks one from `snapshot_at_ms - created_at_ms` for this source.
- `scripts/expand/summarize.py`: the track folder can be set with `XVIRAL_TRACK_DIR`.
- New under `scripts/scrape/`: `describe.py`, `levels.py`, `probe_encoders.py`, `build_rows.py`, `eval_table.py`,
  `summarize.py`, `regime_probe.py`, `images.py`, `queue.sh`, `queue2.sh`, `post.sh`, `post2.sh`.
- Nothing committed, nothing pushed.

## Files (`tracks/scrape/`)

| Path | Content |
|---|---|
| `REPORT.md`, `patch.diff` | this report; code changes and new files |
| `source/posts.parquet` | the converted source: 271,226 posts in the post schema, picture description in `media[0].alt` |
| `source/reader_report.json`, `describe.json`, `post_meta.parquet`, `accounts.parquet` | guard results; every number of section 1; language, topic flags and post type per post; per-handle counts |
| `levels.json`, `same_post_two_readings.parquet`, `coverage.json`, `encoder_probe.json` | section 5.1; the 10,609 posts read twice; section 7.2 counts; section 1.2 |
| `rows/new_rows.parquet`, `build_counts.json`, `filter_log.csv` | the 250,256 training rows (columns of `work/dataset/rows.parquet` plus `acct_n`, `age_h`, `acct_kind`, `handle`), counts by stage |
| `under20_ids.npy`, `under12h_ids.npy`, `orig_test_ids.npy` | row lists for `--train-exclude` and `--eval-rows` |
| `runs/<name>/best.pt`, `metrics.json` | `hist_*`, `histsrc_*`, `allsrc_*`, `h12src_*` (none beats the control) |
| `bench/<name>/shard_*.parquet`, `launch/<name>.parquet` | per-pair benchmark scores; launch-post forecasts |
| `eval/tables.md`, `tables.json`, `slices.json`, `slices.npz` | pairs and level by slice for every model, with per-account tallies |
| `eval/results.md`, `results.json`, `results_bench.json`, `results_launch.json` | benchmark and launch tables |
| `eval/regime_probe.json`, `regime_probe_h12.json` | the same post under each value of the sampling input |
| `images.json`, `image_kinds.parquet` | section 6 |
