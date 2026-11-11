# Virality Engine: data pipeline and feature engineering

Companion to `ARCHITECTURE.md` (stages) and `DESIGN.md` (evidence, budget, gates). This document specifies how raw posts become training rows: sources, ingestion, filtering, deduplication, normalization, labels, splits, and every feature with its justification. A feature that cannot be tied to a line in this repo or to measured evidence is listed in section 10 as excluded, with the reason.

Rule used throughout: **Phoenix's inputs are reproduced exactly; everything we add either replaces a Phoenix input we cannot have (per-viewer history, the post-ID memory table) or feeds a stage that is X's own code (filters, value model, eligibility).**

---

## 1. Sources and what each contributes

| Source | Format | Fields that matter | Views | Role |
|---|---|---|---|---|
| **Your 2025 tech scrape** | JSONL, ~13 GB with embeddings; raw V1/V2/V4 | text, created_at, likes, impressions, media_url_https, screen_name, followers_count, following_count, posts_count, media_count, is_blue_verified | yes | Audience seed for the relevance filter; training rows; validation anchor |
| **USC X 2024 corpus** | 883 gzip CSV chunks of 50k rows in `part_n/` (part 36 removed). At least five header layouts; nested fields are Python reprs with `datetime.datetime(...)` calls; some layouts store `id` as a float, which destroys it, so the ID comes from the status URL | `url` (exact ID and handle), `user` (followers, following, posts, media count, created, verified, bio), `media` (type, URL, mp4 variants, duration), `links` (expanded URLs), `viewCount`, `likeCount`, `retweetCount`, `replyCount`, `quoteCount`, `quotedTweet` (boolean only), reply fields or `inReplyToTweetId`; the "octobergap" layout has the placeholder `PW` instead of views and reply data | yes, except "octobergap" | Largest view-labeled slice after audience filtering; research model only (CC BY-NC-SA). About 27% of rows are originals or quotes; the rest are replies |
| **Community Archive** | Public REST API (PostgREST), paginated by keyset into Parquet; the bulk export pointer was down on 2026-11-07. 474 donor uploads, 22.2M tweets, 22.7M donor likes, 897k account rows | `tweets` (text, created, likes, reposts, reply-to), `all_account` (followers, following, tweets, created, snapshot time), `profile` (bio, website), `tweet_media`, `tweet_urls`, `quote_tweets`, `followers` and `following` per donor, `likes` (the donor's own likes) | no | Long per-author histories, same-author pairs, **real follow-graph edges** (exact mutual-follow share for donors), and the only per-viewer engagement data available outside X: donors' likes, which feed the viewer-level research track |
| **Analytics exports** (yours and donors) | CSV from X analytics | per post: impressions, engagements, detail expands, profile clicks, link clicks, new follows, replies, reposts, likes, bookmarks, shares | yes, plus clicks | The only labels for click, profile-click, link-open, follow and share heads; calibration set |
| **Live media** | fetched from media URLs in the above | image bytes, video bytes | n/a | Embedded once, then discarded; only embeddings and sampled frames are stored |

Not used: Bluesky (held in reserve as content-tower pretraining if the X corpus is under ~150k view-labeled rows after filtering), Weibo and other cascade corpora (different platform mechanics), any live scraping of x.com.

---

## 2. Ingestion into one schema

Every source is parsed into the same Parquet schema. Fields absent at a source are null, never imputed at this layer.

```
post_id              int64      snowflake id
author_id            int64      (resolved from screen_name where only that exists)
author_handle        string
created_at_ms        int64      from snowflake: ((id >> 22) + 1288834974657); the source's own
                                timestamp is kept as created_at_src_ms and must agree within 1 s
snapshot_at_ms       int64      when the counts were observed (see 2.1)
text                 string     raw, untouched
text_note            string     long-post body if the post is a note tweet (Community Archive)
lang_src             string     source language field
kind                 enum       original | quote | reply | repost
in_reply_to_post_id  int64
quoted_post_id       int64
quoted_text          string
quoted_author_id     int64
reposted_post_id     int64
media[]              struct     {type: photo|video|gif, url, width, height, duration_ms, alt}
urls[]               struct     {raw, expanded, final_domain, is_media_tco}
mentions_n           int32
hashtags_n           int32
views                int64      null if the source has none
likes, replies, reposts, quotes, bookmarks   int64 (null where absent)
author_followers     int64      nearest known snapshot
author_following     int64
author_posts_count   int64
author_created_at_ms int64
author_verified      bool
source               enum       user_scrape | usc | community_archive | analytics | nitter_2026 | twitter100m
source_file          string
```

### 2.1 Snapshot time, per source

Counts are only meaningful relative to when they were read.

- **Your scrape:** the file's scrape timestamp per row if present, else the file date.
- **USC:** no per-row snapshot. Measured on real chunks: within a 2.7-hour chunk, the rank correlation between posting time and views per follower is about 0.0 (range −0.04 to +0.06 across eleven chunks from May to November). Counts read minutes after posting would show a strong negative trend, so USC counts are mature. Rows are marked `assumed_mature`, and the ingester re-runs this audit on every chunk; a chunk with |rho| > 0.1 is dropped by the maturity filter.
- **Community Archive:** the donor's archive generation date from the manifest.
- **Analytics:** the export date; impressions in exports are effectively final after a few days.

### 2.2 Snowflake creation time

Phoenix derives post creation time from the ID (`phoenix/xrex/data/recsys/recsys_batch.py:996-997`; serving `util.rs:345-352`). We do the same, millisecond-precise, and treat a disagreement with the source timestamp of more than 1 s as a corrupt row.

### 2.3 Author resolution

Your scrape keys authors by `screen_name`; USC by `user_id` plus a nested `user` object; Community Archive by `accountId`. A handle-to-ID table is built from every row that carries both. Handles can change, so when a handle maps to two IDs over time the row is assigned by date proximity, and unresolved handles get a synthetic negative ID and are excluded from author-history features.

---

## 3. Deduplication

Order matters; each step is logged with counts.

1. **Exact post ID** across all sources. Keep the row with the latest `snapshot_at_ms` (highest counts), carry the union of non-null fields from the others.
2. **Reposts.** Rows with `reposted_post_id` set are dropped as training rows (X never indexes them for out-of-network retrieval, `phoenixRankAllCandidateProcessor.strato:496-497`). Their existence is kept as a count on the original (a public repost count substitute where the source lacks it).
3. **Same-author exact text within 30 days.** Keep the first; mark the author with `copypasta_events += 1`, which is a suppression-risk feature (X's `COPYPASTA_SPAM` bots cluster duplicate text).
4. **Near-duplicate text across authors.** MinHash over 5-gram shingles, Jaccard ≥ 0.9, within a 7-day window. Keep all rows (they are real posts with real outcomes) but tag `near_dup_cluster_id` and `near_dup_rank` (order within the cluster by creation time). Late copies of a viral post behave differently from the original, and X's DPP penalizes near-duplicates of a higher-scoring post in the same feed.
5. **Near-duplicate media.** pHash on images and on the first video frame; SigLIP cosine ≥ 0.97 as a second check. Tag, do not drop.
6. **Your scrape versus the rest.** Your merged file is reconciled against V1, V2, V4 by post ID before anything else, so the merge is verified rather than assumed.

---

## 4. Filtering: what we keep and why

Filters run in this order so that the expensive ones (embedding-based relevance) see the fewest rows.

### 4.1 Hard validity

Drop rows that are: ad rows (USC rows where every field but `text` is NaN); missing `post_id` or `created_at_ms`; `text` empty and no media; `author_id` unresolved; views present but below 0; counts that exceed views by more than 2× (corrupt snapshot).

### 4.2 Post kind

Keep `original` and `quote`. Drop `reply` and `repost` as training rows. Justification: out-of-network replies and reposts are filtered before scoring (`home-mixer/filters/oon_retweet_reply_filter.rs`), replies are indexed only after a like and that source is off by default, and the product is for original posts and launches. Replies are retained in a side table for two features only: the author's reply volume (a suppression-risk signal) and the author's replies-per-original ratio.

### 4.3 Count maturity

Keep rows whose `snapshot_at_ms − created_at_ms ≥ 72 h`. Ninety-five percent of impressions arrive within 24 h (Pfeffer et al. 2023), and X stops serving at 48 h, so a 72 h floor gives counts that are final for practical purposes. Rows between 48 h and 72 h are kept in a separate "young" table for later use with an age-aware correction, not in the main training set.

### 4.4 Recency

Keep `created_at ≥ 2023-01-01`. Views are only public on posts after December 2022, and the ranker and its weights date from 2026. Within the kept range, training weight rises with recency: weight 1.0 for 2025 onward, 0.5 for 2024, 0.25 for 2023. The final calibration pass uses only the newest 6 months available.

### 4.5 Language

Keep `en` as detected by our own detector (not the source's field, which is unreliable on short posts), plus posts that are mostly code, numbers, or links where the detector is unsure. Everything else is dropped. The audience is SF and tech; Phoenix is multilingual but our label budget is not, and every non-English row costs embedding compute while teaching the model about an audience that will not use the product. Revisit when a non-English customer appears.

### 4.6 Audience relevance (the filter that matters most)

Goal: keep posts that look like what customers will submit, from accounts that look like customers' audiences, and drop the political, sports, celebrity and general-news bulk of the USC corpus without hand-written keyword lists.

Procedure:

1. **Seed set.** Every account in your 2025 scrape is a positive account, plus an optional handle list. Community Archive donors are scored, not assumed.
2. **Cheap pre-pass.** Qwen3-Embedding-0.6B over `text` for the candidate pool, about 100× cheaper per post than the 8B multimodal encoder. USC is first thinned by a political-term blocklist (any one hit drops the post). Measured: about 200 short posts per second on a 4-core CPU, so this stage does not need a GPU at all for a few million posts.
3. **Topic gate.** Each post is compared, as a retrieval document, against 20 short descriptions of target topics (launches, demo videos, fundraising, YC, ML research, LLMs and agents, developer tools, open source, SF tech) and 13 off-target ones (elections, partisan politics, war, sports, celebrity, religion, daily life, crime, memecoins). `topic_margin` = best target similarity minus best off-target similarity. Found on real data: even tech accounts post heavily about politics in election season, so a classifier trained on everything seed accounts post learns politics. Only on-topic seed posts become positives, and only on-topic posts are kept.
4. **Classifier.** Logistic regression on the 0.6B embedding: positives = on-topic posts from 80% of seed accounts; negatives = a random sample of non-seed posts, with 10% of the corpus held out. The default threshold keeps 90% of on-topic posts from the 20% of seed accounts held out of training, which measures how an unseen relevant account scores rather than how well the model memorises the seed accounts. After the manual audit, the threshold is reset to the lowest score that keeps ≥ 85% precision.
5. **Account-level decision.** An account is kept whole if ≥ 60% of its last 50 originals pass both the gate and the threshold; between 30% and 60% it contributes only its passing posts; below 30% it is dropped. Seed accounts are kept whole.
6. **Caps.** At most 2,000 rows per account and 50 per near-duplicate cluster.
7. **Audit.** 1,000 kept, 1,000 dropped and 1,000 boundary posts are written to `audit_to_label.csv` for a manual pass before any GPU time is spent on embeddings. The report records held-out AUC (unseen seed accounts against unseen corpus posts) and the held-out corpus pass rate.

### 4.7 Account hygiene

Drop accounts that look automated or that X would label: more than 200 originals in any 24 h window; follower/following ratio under 0.05 with over 5,000 following (follow-churn pattern); over 80% of posts containing the same external domain; copypasta events over 20. These are also the suppression-risk signals in section 7.6, so the thresholds for dropping are set far beyond the thresholds for flagging.

### 4.8 Outcome of filtering

Expected survivors are reported in phase 0 as a table of counts by source, year, kind, media type, follower tier and views-availability. The go/no-go gate reads this table before any embedding run.

---

## 5. Normalization

Two text forms are kept, because the encoder and the scalar features want different things.

- **`text_render`** (for the encoder and the teacher): exactly what X's renderer sees. t.co links removed, long-post body substituted where present, quoted post appended in X's format, image pads inserted at the positions `mm_encoder.py` uses, `Transcript:` appended for video with speech. No lowercasing, no emoji stripping, no normalization of any kind; the encoder is trained on real posts.
- **`text_feat`** (for scalar features): NFKC-normalized, URLs replaced by a token, used only to count things.

Counts: stored as int64, null preserved. Phoenix's transform is applied at feature time, not at storage: `log2(count + 1) / 32` (`recsys_feature_prep.py:38, 739`). Counts are zeroed and a stale flag set for posts older than 14 days *at impression time* in Phoenix; for us the analogue is the age grid in the surrogate, so no zeroing at storage.

Timestamps: UTC milliseconds. Audience-local hour is computed at feature time from the author's estimated audience time zone (section 7.4).

Follower counts: stored raw; log1p at feature time; follower tier buckets as X's own `TweetTypeMetrics` buckets (<100, 100–1k, 1k–10k, 10k–100k, 100k–1M, 1M+; `tweet_type_metrics_hydrator.rs`).

Media: images resized once to 448 px (VLM) and 384 px (SigLIP) and stored as JPEG at quality 90; videos decoded once to the three frame sets described in `ARCHITECTURE.md` S0 and the audio transcribed; original bytes deleted after embedding.

---

## 6. Labels

Everything is derived; nothing is copied from a source's own "viral" flag. Your `is_viral` is kept as a column for comparison only.

### 6.1 Per-impression action rates (the Phoenix-shaped targets)

For rows with views:

```
rate_fav    = likes    / views
rate_reply  = replies  / views
rate_repost = reposts  / views
rate_quote  = quotes   / views
rate_bmk    = bookmarks / views        (where present)
```

Clipped to [0, 1]. Rows with views < 100 keep their rates but receive low loss weight (section 6.6) because a rate from 30 impressions is noise.

Analytics rows additionally provide `rate_click` (detail expands / impressions), `rate_profile_click`, `rate_link_open`, `rate_follow`, `rate_share`. These are the only labels for Phoenix heads 29, 30, 39, 21 and 37, and they are precious: every analytics row is kept regardless of other filters, flagged by source.

### 6.2 Reach

```
reach_ratio      = views / author_followers_at_post        (section 7.3 for the "at post" estimate)
log_reach_ratio  = log1p(reach_ratio)
```

`reach_ratio` is the primary regression target. It is the quantity X's own exploration label is defined on (`recsys_batch.py:105-135`), and it is what makes a 300-follower account and a 3M-follower account comparable.

### 6.3 Exceedance tiers (the product's "odds")

Binary labels: `reach_ratio ≥ 1`, `≥ 10`, `≥ 100`. Plus author-relative labels: `views ≥ author's trailing p90 of views over the previous 50 originals` and `≥ trailing p99`. The author-relative labels are computed strictly from posts created before the target post, never from later ones.

### 6.4 X's exploration label, ported

`post_unexplored_24h = views_24h_est < 0.03 × followers × (1 − e^{−ln2·24/8}) / (1 − e^{−ln2·24/8})`, which at 24 h reduces to `views < 0.03 × followers`. Because our views are final rather than at 24 h, the label is computed on final views and documented as such. It is both a head (index 59) and a product output.

### 6.5 Pairwise labels

For every author, all pairs of originals created within 7 days of each other, both with views, both at least 72 h mature. Label = sign of the difference in `log_reach_ratio`; pairs whose difference is under 0.2 nats are dropped as ties. For rows without views (Community Archive), the comparison quantity is `log1p(likes / followers)`, and these pairs train only the pairwise term. Pairs are capped at 200 per author to stop prolific accounts dominating.

### 6.6 Loss weights per row

`w_row = recency_weight × min(1, log1p(views) / log1p(1000))`, so that a post with 1,000 or more impressions carries full weight and a 50-impression post carries about 0.57. Analytics rows get `w_row × 2` because their labels are exact and rare.

### 6.7 Known label noise, and what we do about it

- Views are a snapshot, not a 24 h reading: mitigated by the 72 h maturity floor.
- Views include the author's own and repeat views: ignored, consistent across rows.
- Follower count is at snapshot, not at post time: section 7.3 reconstructs an estimate; the residual error is largest for fast-growing accounts and is why author-relative labels exist alongside ratio labels.
- Deleted posts are absent, and they skew toward posts that did badly or got the author in trouble: unfixable, noted.

---

## 7. Feature engineering

Features are grouped by the role they play in the architecture. For each: definition, the line in this repo or the evidence that justifies it, and whether it is a Phoenix input reproduced exactly (**P**), a replacement for a Phoenix input we cannot have (**R**), an input to a ported deterministic stage (**D**), or an addition with measured evidence (**E**).

### 7.1 Phoenix candidate inputs, reproduced exactly (P)

| Feature | Definition | Source in repo |
|---|---|---|
| `sid[6]` | RQ-KMeans codes of `e_mm` against our codebook, 1-indexed, 0 = missing | `phoenix/reference/sid_codebook.py`, consumed at `recsys_feature_prep.py:783-875` |
| `sid_hash_level` | hash of the post ID mixed into a 257-way code, as Phoenix does | `recsys_feature_prep.py` (sid_hash_level) |
| `post_age_bucket` | `(age_minutes // 60) + 1`, clipped to 81, 0 if unknown; evaluated at each age on the grid | `recsys_feature_prep.py:51-72` |
| `local_hour` | hour in the viewer's time zone, 25 classes (24 + unknown); we use the audience time-zone estimate | `recsys_feature_prep.py:227-229, 694-703` |
| `count_fav/reply/repost/quote/view` | `log2(c + 1) / 32`, five values plus a constant 1 into Phoenix's 6→64→D MLP; zero at draft time, filled by the momentum loop at later ages | `recsys_feature_prep.py:32-39, 730-754` |
| `is_author_followed_by_viewer`, `is_author_following_viewer` | 2-row tables; set (1, m) in the in-network context where m is the mutual share, (0, 0) out of network | `recsys_feature_prep.py:711-728` |
| `is_stale_14d` | 0 for anything we score | `feature_config.py:87` |
| `product_surface` | constant: home ranking | `recsys.proto:1190-1212` |
| `e_mm` (1024) | the embedding behind the SID; Phoenix keeps the raw vector off in production but the SID is a lossy view of it, and we have no post-ID memory to compensate, so the raw vector is also a token | `mm_encoder.py`; off-switch at `xrecsys.py:364` |

Why the age grid: Phoenix scores a post at impression time with its age and counts at that moment. A draft has age 0 and counts 0. The momentum loop in `ARCHITECTURE.md` S6 re-evaluates at 0.25, 1, 2, 6 and 24 h with simulated counts, so these features are evaluated, not stored, per age step.

### 7.2 Home-mixer-hydrated fields the ranker ignores but we need (E)

Phoenix can ignore these because its 100M-row post-ID table memorizes how each post performed. We cannot memorize unseen posts, so content detail has to carry the signal. Each of these is computed by home-mixer in `models/content_features.rs` or hydrated in `candidate_hydrators/`, which is why they are the right set and not an arbitrary one.

| Feature | Definition | Why |
|---|---|---|
| `weighted_text_len` | X's 280-weighting: CJK and emoji count 2, URLs count 23 | `content_features.rs:44-80` |
| `newline_count` | line breaks in the text | `content_features.rs` |
| `has_url`, `external_url_count` | URLs excluding media t.co links | `content_features.rs:8-36`; link-open weight 0.2 (`param.rs:437`); 8× view penalty observed (arXiv 2410.17390) |
| `final_domain_class` | the expanded URL's final domain, bucketed: own site, GitHub, YouTube, app store, news, social, other; plus a learned domain-quality score | S2 URL verdict chain (`scarecrow/legacy/UrlVerdictRegistry.scala`) |
| `has_media`, `has_photo`, `has_video`, `media_count` | from the media array | `media_info_hydrator.rs:62-120` |
| `video_duration_ms`, `video_gt_10s` | duration and the > 10 s gate | `MinVideoDurationMs` 10,000 (`param.rs:707-712`); video index gate (`eventProcessing.strato:403-418`) |
| `video_duration_bucket` | ≤ 10 s, 10–60 s, > 60 s | `tweet_type_metrics_hydrator.rs:80-84` |
| `video_aspect`, `video_resolution_class` | vertical / square / horizontal; SD / HD / Full HD | `video_media_hydrator.rs`; X's embedding plan renders "quality SD/HD/Full HD" |
| `video_has_speech`, `speech_in_first_3s`, `transcript_len` | from Whisper | X's grox pipeline transcribes and appends; the first 3 s is what autoplay shows |
| `is_quote`, `quoted_has_media`, `quoted_video_duration_ms` | quoted post structure | `quote_hydrator.rs:165-266`; `is_quote` in `TweetBoolFeatures` |
| `is_long_post` | text over 280 weighted chars (note tweet) | "show more" action head 45 exists; long posts render truncated |
| `mentions_n`, `hashtags_n` | counts | X computes mention counts for mention-abuse detection; hashtag abuse is a spam sub-type; neither is a ranking signal, so these are low-priority scalars kept because they cost nothing |
| `lang` | detected language | sent to the ranker as `language_code`; used by the reply model |

### 7.3 Author features, replacing Phoenix's author-ID table and the viewer's follow relationship (R)

Phoenix learns an embedding per author from billions of impressions and sees whether each viewer follows the author. We replace this with explicit statistics plus a learned author embedding where the author has enough rows.

| Feature | Definition | Why |
|---|---|---|
| `log_followers_at_post` | followers reconstructed at post time: linear interpolation between known snapshots for the author; if only one snapshot, that value | Normalizer for reach; cold-start cap 50k (`param.rs:626-631`); popular-author cutoff; TweetTypeMetrics tiers |
| `follower_tier` | the six X buckets | `tweet_type_metrics_hydrator.rs` |
| `log_following`, `follow_ratio` | following and followers/following | follow-churn pattern (BDSM heads); input to mutual-share estimate |
| `mutual_share_est` | estimated fraction of the author's followers who are also followed back. Computed exactly for Community Archive donors from `follower.js` ∩ `following.js`; a small regression from (followers, following, account age, verified) is fit on donors and applied to everyone else | Reply weight is 5 for non-mutuals and 20 for mutuals (`param.rs:410-416`, `weights.rs:86-93`); this is the single largest per-viewer weight difference in the value model and it is a property of the author's graph |
| `account_age_days` | from `author_created_at_ms` | notifications filters use < 72 h; new-account viewer filters exist; credibility correlates |
| `verified` | blue check | no weight in the value model (report 02 §2.5), kept only as an author descriptor |
| `cold_start_eligible_author` | followers ≤ 50,000 | `param.rs:626-631` |
| `popular_author_est` | followers above the estimated cutoff for the top 0.005% of 7-day active posters (cutoff estimated from the follower distribution of active accounts in the corpus; expected to be in the low millions) | `popular_authors.rs:11, 25-43` |
| `cadence_24h`, `cadence_7d` | originals posted in the prior 24 h and 7 d, computed strictly before the target post | author-diversity decay 0.5 toward 0.25 within a feed (`scoring.rs:261-282`); Thunder per-author cap 50 originals |
| `hours_since_prev_original` | gap to the author's previous original | same |
| `trailing_rate_{fav,reply,repost,quote}_{p50,p90}` | over the author's previous 50 originals with views, strictly earlier | the author's audience's base rates; the closest observable analogue to Phoenix's learned author embedding; literature: author history is what moves R² from < 0.05 to 0.45 |
| `trailing_reach_ratio_{p50,p90,p99}` | same window | author-relative labels and features |
| `trailing_reply_to_fav` | replies / likes over the window | controversy proxy; replies carry weight 5 but controversial authors also attract the −234 report and −59 mute heads we cannot observe directly |
| `trailing_media_share`, `trailing_video_share`, `trailing_link_share` | fraction of recent originals with media / video / external links | describes the account's format habits and the audience trained on them |
| `author_centroid` (1024) | mean `e_mm` of the previous 50 originals | audience-topic prior; input to `AUDIENCE_IN` |
| `author_niche_sid2` | SID prefix-2 of the centroid | the learned audience embedding key |
| `author_id_hash` | hashed author embedding (learned), zero-initialized and only trained for authors with ≥ 20 rows | Phoenix's author table analogue, limited to where it can learn |
| `history_n` | how many prior originals the author statistics are based on | lets the model discount thin histories; Phoenix treats < 128 history events as "new user" |

All trailing features use only posts created before the target post. This is enforced in code by computing them in a time-ordered pass per author, and verified by a unit test that shuffles future rows and checks the features do not change.

### 7.4 Audience and time (R)

| Feature | Definition | Why |
|---|---|---|
| `audience_tz_est` | the author's audience time zone, estimated as the mode of the author's own posting hours mapped to the three US zones plus Europe and India, unless the account provides a follower time-zone histogram | Phoenix embeds the viewer's local hour (`recsys_feature_prep.py:694`) |
| `local_hour_audience` | planned hour in `audience_tz_est` | same |
| `dow_audience` | weekday; Phoenix has the table but disables it in production, so this is a low-priority scalar | `recsys_feature_prep.py:660-686` (off) |
| `audience_in_key` | (`author_niche_sid2`, `follower_tier`) | learned embedding standing in for "who the followers are" |
| `audience_oon_key` | the post's own SID prefix-2 | learned embedding standing in for "who Phoenix retrieval would show this to" |

### 7.5 Neighbors and trend (E)

Phoenix's viewer history tells it what the viewer has recently engaged with. We cannot have that per viewer, but we can know what the corpus recently did with similar content. Retrieval-augmented popularity prediction is the largest single gain in the 2024–2025 literature (RAGTrans, SKAPP).

| Feature | Definition | Leakage rule |
|---|---|---|
| `nbr_global[16]` | the 16 nearest posts by `e_mm` cosine among posts created in the 7 days *before* the target post, other authors, with their `log_reach_ratio`, `rate_fav`, `rate_reply`, follower tier, age gap and cosine | strictly earlier creation time; index built per week |
| `nbr_density_7d` | number of corpus posts with cosine ≥ 0.85 in the prior 7 days | topic saturation versus novelty |
| `nbr_best_reach_7d` | max `log_reach_ratio` among them | "is this topic currently moving" |
| `self_sim_48h` | max cosine to the author's own originals in the prior 48 h | DPP suppresses near-duplicates of a higher-scoring post in the same feed (`dpp.rs`, theta 0.65); video carousel dedupes at cosine 0.72 and SID prefix-3 |
| `dup_rank`, `dup_cluster_size` | from section 3 | late copies of a viral post |

### 7.6 Suppression and negative-feedback proxies (D, feeding S2 and the negative heads)

The value model's largest weights are negative and unobservable from public counts. These features are the best available proxies and they also feed the ported visibility rules.

| Feature | Definition | Repo basis |
|---|---|---|
| `p_nsfw_img`, `p_nsfw_video`, `p_nsfw_card` | open adult classifier on images, frames, card image | thresholds 0.95 / 0.999 / video recall bar (`IsHighPrecisionNsfwNewModelLogic.df`) |
| `p_gore` | open violence classifier | 0.6 image, 0.95 video (`ViolenceAndGore_ImageModel_MTT.bot`) |
| `p_bait` | classifier distilled from an LLM labeler over X's spam sub-types: engagement baiting, farming, card manipulation, hashtag abuse, scam links | `ptos/state.py:134-203`; fires regardless of credibility (`result_sink.py:233-248`) |
| `p_slop` | gibberish / LLM-slop classifier | `enforcement_post.yaml:45-50` |
| `url_verdict` | expanded chain against blocklists plus a learned domain score | `UrlVerdictRegistry.scala`; bad → drop everywhere |
| `copypasta_events_author`, `reply_burst_author` | from section 3 and the reply side table | `BBQDuplicateTextProd.bot`; BDSM reply-spam head |
| `author_nsfw_risk` | share of the author's recent media over the NSFW threshold | 3 flagged of last 5 → 7-day account label (`postToUserLabelRules.strato:395-426`) |
| `cred_proxy` | log followers, account age, verified, follow ratio, and the share of the author's replies that receive likes, combined by a small model fit to predict whether an account would clear the user-cred ≥ 54 bar in the one place we can observe it (donor accounts with full graphs) | `IsHighPageRankUser.df`: score ≥ 54, fallback followers > 25k; exempts authors from most spam rules |

### 7.7 Media content features (E)

| Feature | Definition | Why |
|---|---|---|
| `e_img[i]` (1152 × up to 4) | SigLIP-2 So400m/14 @384 per image and card image | visual detail the 1024-d post embedding compresses away; required because we lack post-ID memory |
| `e_frame[t]` (1152 × up to 18) + `t_norm` | per sampled frame, with normalized time | temporal structure of a launch video |
| `e_thumb` | first frame | the frame the feed shows before play |
| `frame_motion` | mean cosine distance between consecutive frames | static slideshow versus cut-heavy video; cheap and informative |
| `img_text_density` | fraction of the image area the VLM encoder's attention places on text-like regions is not available cheaply, so this is **excluded**; the Qwen3-VL embedding already reads text in images | section 10 |

### 7.8 Features kept only for the baseline and ablations

Your 433 columns and three 768-d embeddings are loaded as a separate feature group for the phase-0 LightGBM baseline and for one ablation ("old features versus new features on the same rows"). They are not inputs to the surrogate.

---

### 7.9 How the post is written: vetted before admission (E)

X's ranker sees a post through its embedding, and a frozen embedding encodes topic much better than layout. Layout is also the thing a writer can change. `features/text_structure.py` defines 46 candidate columns in six groups, and `scripts/vet_features.py` admits a group only if it improves same-author ranking for trees that already have every current model input including the embedding. The decision is made on the most recent 15% of training rows, so the test split is never used to pick features. A group passes when its gain is positive in at least 95% of author-level bootstrap resamples.

Result on 152k like-labelled posts from the archive's donor accounts (base 59.4% pairwise):

| Group | Columns | Gain in pairwise accuracy | 90% interval | Decision |
|---|---|---|---|---|
| Link placement | link in the first line, link at the end, share of text before the link | +0.55 | +0.34 to +0.79 | admitted |
| Opening line | length, words, share of the post, question, colon, number, first person, announcement phrasing, leading mention | +0.38 | +0.17 to +0.57 | admitted |
| Layout | words, lines, blank lines, bullet lines, mean and longest line, sentences, words per sentence | +0.22 | +0.02 to +0.43 | admitted |
| Voice | I, we, you, negation, superlatives, comparisons, how and why framing, thread marker, call to action | +0.20 | 0.00 to +0.38 | held: no gain on top of the three above |
| Typography | capitals, exclamation and question marks, emoji, numbers, money, percent, quotes, arrows, code | +0.18 | -0.06 to +0.38 | held |
| Relative to own history | length, media, link, gap and line count against the account's trailing median | +0.04 | -0.16 to +0.24 | dropped: the model already sees the history |

The three admitted groups together add 0.77 points; all six add 0.88. The 20 admitted columns are computed in the feature stage and in serving, and are in the model's scalar inputs. The numbers are in `research/10_text_structure_vetting.json`. The run has to be repeated on view labels and with X's own encoder, since a stronger encoder may absorb some of this.

## 8. Embedding computation plan

| Step | Model | Rows | Hardware | Hours | Credits |
|---|---|---|---|---|---|
| Relevance pre-pass | Qwen3-Embedding-0.6B | up to 46M texts (USC thinned first) | L4 spot | ~20 | ~$10 |
| Faithful post embedding | Qwen3-VL-Embedding-8B, X's rendering | all kept rows, ~0.5–1M, with media where fetched | L40S spot | ~15–30 | ~$20–40 |
| Codebook | RQ-KMeans 6×256 via `sid_codebook.py` | all `e_mm` | CPU | ~2 | $0 |
| Image embeddings | SigLIP-2 So400m @384 | ~300k images + cards | RTX 2060 | ~3 | $0 |
| Frame embeddings | SigLIP-2 | ~30k videos × ~18 frames | RTX 2060 | ~5 | $0 |
| Transcription | Whisper large-v3-turbo | ~30k videos | RTX 2060 | ~8 | $0 |
| Safety classifiers | open NSFW, gore, bait, slop | all media and texts | RTX 2060 | ~6 | $0 |
| Neighbor index | FAISS over `e_mm`, weekly shards | all rows | CPU | ~1 | $0 |

Embeddings are stored as fp16 memmaps keyed by `post_id`; media bytes are deleted once embedded.

---

## 9. Splits and leakage controls

- **Test:** the newest 10% of rows by creation time, all authors. This is the number reported to anyone.
- **Validation:** 10% of authors, chosen at random among the remaining rows, held out entirely; used for model selection and calibration.
- **Train:** the rest.
- **Neighbor features** are built only from rows created before the target, so test rows can see train-period neighbors but never other test rows from the future.
- **Author trailing features** are strictly time-ordered.
- **Pairwise labels** never cross the split boundary.
- **Analytics rows** are split by account, never by row, so a connected account's posts are all on one side.
- A **shuffled-label control** is trained once per phase; if it scores above chance on test, a leak exists and nothing else is reported until it is found.

---

## 10. Excluded, with reasons

| Candidate feature | Reason for exclusion |
|---|---|
| Sentiment scores, emotion categories | No role in the algorithm; weak and unstable in the literature; the embedding carries tone already |
| Emoji counts, capitalization ratio, exclamation counts as primary features | Not algorithm signals; kept only in the baseline group for comparison |
| Hashtags as a boost | No ranking weight references hashtags; hashtag abuse is a spam sub-type. Count kept as a low-priority scalar, nothing more |
| "Best time to post" tables | The algorithm has no time-of-day weight; what exists is the viewer's local hour as a Phoenix input, which is why we model audience local hour instead |
| PCA-compressed embeddings | Loses the detail we are paying the encoder for; the transformer consumes full vectors |
| OCR of images as a separate feature | The Qwen3-VL embedding reads text in images; a second pass adds cost and no new information |
| Raw follower count as a target or as the dominant input | It is the normalizer, not the outcome; using it as an outcome is how last year's model became a follower-count detector |
| Readability scores, keyword lists for "virality words" | Hand-built heuristics with no basis in the ranker; the pairwise-trained model learns phrasing effects from data |
| Absolute like, repost counts as targets | Per-impression rates are what Phoenix predicts; absolute counts conflate reach with rate |
| Viewer-level demographics | Phoenix has them (country, language, age, gender) but we score for an audience, not a viewer; subsumed by the audience keys |
| Synthetic engagement labels from the shipped generator | Random affinities with hand-set base rates; teaches the generator, not X |

---

## 11. Phase-0 deliverables from this pipeline

1. The unified Parquet with the schema in section 2, and a count table by source × year × kind × media × follower tier × views-availability, before and after each filter.
2. The relevance classifier, its threshold, and the 2,000-row manual audit.
3. The dedup report.
4. Label distributions: reach ratio histograms by follower tier, exceedance base rates, pairwise pair counts per author.
5. The feature store with sections 7.2, 7.3, 7.4, 7.6 and 7.8 populated (7.1, 7.5 and 7.7 need the embedding runs in phase 1).
6. The LightGBM baseline with the pairwise objective on the available features, reported on the time-forward test split, plus the shuffled-label control.

---

## 12. Implementation status (code in `xviral/`)

Every stage above is implemented and tested; `README.md` has the commands.

| Stage | Module | Verified on |
|---|---|---|
| USC ingest, all layouts | `sources/usc.py` | 12 real chunks across May to November: zero missing authors or follower counts, zero ID-time mismatches, maturity rho within ±0.06 |
| Community Archive download and join | `sources/community_archive.py` | live REST schema, synthetic tables in that schema |
| Tech scrape | `sources/user_scrape.py` | nested and flat layouts, rows with and without IDs |
| Analytics exports | `sources/analytics.py` | both known X export layouts; unknown layouts fail loudly |
| Merge and dedup | `unify.py`, `dedup.py` | priority rules, synthetic-ID reconciliation, MinHash clusters |
| Filters | `filters.py` | each rule, analytics exemptions, per-source filter log |
| Author features | `authors.py` | leakage test: changing every later post leaves earlier features unchanged |
| Relevance | `relevance.py` | real USC data with Qwen3-Embedding-0.6B on CPU |
| Content, audience, suppression features | `features/` | the X repository's own weighted-length test cases |
| Labels, pairs, splits | `labels.py`, `splits.py` | pairs equal to a brute-force enumeration |
| Baseline and leak control | `baseline.py` | shuffled-label control at chance |
| Media, ASR, encoders, SIDs, neighbours | `media/`, `embed/`, `features/neighbors.py` | ffmpeg sampling on a real clip; encoder inputs identical to `mm_encoder.py`; neighbour leakage test. Model weights run only on the GPU box |

---

## 13. Measured on real data (2026-11-07)

Topic gate with Qwen3-Embedding-0.6B on CPU (`topic_margin` = best target-topic similarity minus best off-target similarity).

| Corpus slice | Originals scored | margin > 0 | margin > 0.05 | margin > 0.1 | What passes at 0.1 |
|---|---|---|---|---|---|
| USC, 12 chunks May to November 2024, after the political pre-thin | 135,630 (18,443 after pre-thin) | 1.64% | 0.39% | 0.09% | spam bots, memecoin "launches", political posts; essentially no target content |
| Community Archive, 4,996 posts from 1 June 2025 | 2,281 | 24.4% | 10.8% | 5.0% | ML infrastructure, benchmark repos, startup news, AI tooling |

Consequences:

1. **USC is not a source of target-audience data.** It is an election crawl. Extrapolated to the full corpus, it would yield a few thousand relevant posts mixed with noise. It stays supported by the code, and useful only as a scale test and for generic author statistics; it should not receive GPU time.
2. **The Community Archive is the audience.** At a 0.05 margin it implies on the order of 500k relevant originals from 2023 on (22.2M tweets, roughly 46% originals, roughly 11% passing), from accounts kept whole by the account rule. It has likes and reposts but no views, so it trains the pairwise and per-author objectives.
3. **Views for the target audience come from the tech scrape and analytics exports.** No free public corpus found has both views and this audience. The scrape's 250k to 450k rows are therefore the primary view-labeled set after all; the free corpora add depth (author histories, same-author pairs, donor graphs and likes), not view labels.
4. **Stand-in seeds do not work in an off-topic corpus.** Accounts with tech-sounding bios in USC posted 24 on-topic posts among them. Seeds must come from the scrape or from a handle list of known accounts.
5. **The default margin is 0.05**, between the permissive 0 (many off-topic posts) and the strict 0.1 (high precision on the CA sample). The manual audit replaces it.

---

## 14. Community Archive: two kinds of row, and donor-account download (2026-11-10)

Checked against the live API:

- **Archive rows** (`archive_upload_id` set) come from a donor's uploaded archive. Counts are as of `archive_upload.archive_at`. 404 donor accounts have completed uploads.
- **Stream rows** (`archive_upload_id` null) are captured by the archive's browser extension as posts appear in timelines, often seconds after posting. Counts are as of `updated_at`. They are about 80% of recent rows, spread over roughly 750k accounts with a few rows each.

The pipeline now takes the snapshot time per row (archive date or capture time), and the 72-hour maturity filter removes posts whose counts were read too early. Before this fix a stream post captured five seconds after posting would have been trained on with zero likes.

This revises section 13's estimate. The archive's usable, mature, deep-history data is the donor accounts: `RestDownloader.download_account_tweets` pulls a list of accounts' posts through the account index (about 0.8 s per 1,000 rows). All 404 donors since 2023: 1.78M posts, of which 417k are originals, 372k mature, 358k English, from 392 accounts with a median of 735 posts each. The same call serves any account list, including your seed accounts if they are in the archive.

Added for like-only sources: `trailing_likes_p50/p90` and `trailing_like_ratio_p50` (strictly-earlier author history on likes), labels `log_like_ratio` and `y_likes_beat_own_p90`, and `relevance.enabled: false` to pass an already on-audience corpus through.

---

## 15. Two more public sources, the full archive, and account selection (2026-11-11)

### 15.1 Sources added

| Source | Code | What it carries | Use |
|---|---|---|---|
| `SinclairSchneider/tweets_sample_2026` on Hugging Face, 73 shards, 75.8M posts from early 2026 | `sources/nitter_sample.py` | Text, likes, replies, reposts, quotes, **views**, handle. No follower counts | The only public corpus found with view counts from the current ranking system. Primary view-labeled set until the private scrape arrives |
| `enryu43/twitter100m_tweets` and `twitter100m_users`, timelines up to April 2023 | `sources/twitter100m.py` | Text, likes, replies, reposts, user bio and follower count. No views | Deep per-account history for accounts whose bio is tech. Pre-dates the current ranker, so it has its own recency floor (2020) and lowest merge priority |

Two defects in `tweets_sample_2026` are corrected at ingest, both verified on the data:

1. **The count columns are shifted by one.** The column named `quotes` holds likes and the column named `likes` holds views. After the correction likes are at most views in 99.99% of rows and the median like-to-view ratio is 1.3%, the usual rate. The reader checks this on every shard and refuses a shard where fewer than 99% of rows satisfy it, so a corrected re-upload of the dataset cannot be silently mis-read.
2. **Later posts were captured young.** The file has no capture time. Median views fall steeply for posts created from May 2026 on, which means those were read soon after posting. Posts created before 1 May 2026 are treated as mature; later ones are dropped by the maturity filter. About half of the rows that pass the other filters survive.

The corpus is a random sample of X, not of the target audience. Ingest is therefore streaming and selective: each shard is downloaded, reduced to posts whose author handle is in the Community Archive account index or whose text matches the tech pattern, written, and deleted. That keeps roughly 87k rows per shard, 43k of them mature.

### 15.2 The full Community Archive, measured

Downloading every table since 2023 by creation-time keyset (`ca-download --mode full`):

| Step | Rows |
|---|---|
| Raw tweet rows since 2023 | 17.1M |
| Originals and quotes, mature at capture | 3.67M |
| English | 3.48M |
| Accounts | 216,930 |
| Accounts with at least 30 posts | 9,840 |
| Accounts with at least 30 posts and a tech share of at least 0.15 | 2,412 accounts, 366,639 posts |

Stream rows (section 14) are biased: the browser extension captures what appears in its users' timelines, which over-represents posts that were already being distributed. Two consequences are handled in code. Maturity is decided per row from the capture time. And accounts are kept only when the archive holds enough of their posts to form a history, which removes the long tail of accounts seen once because one post travelled.

### 15.3 Account selection (`filters.select_accounts`, config `select`)

Runs after hygiene and before deduplication. It replaces the embedding topic gate as the first-line audience filter on large corpora, because it costs nothing and works at the account level, which is what the model needs: a post is only useful with its author's history.

- `min_posts_per_account`: fewer posts than this and the account cannot supply a history plus a candidate.
- `min_tech_share`: share of the account's posts matching `text/tech.py`'s pattern (AI, ML, startup, developer and product vocabulary). An account is kept or dropped whole, so its off-topic posts stay in as history and as negatives.
- `max_posts_per_account`: keeps the most recent N posts, so a few prolific accounts do not dominate the batches.
- `always_keep_sources`: analytics exports and the private scrape bypass the rule.

The embedding topic gate (4.6) remains for the final audit and for corpora with no account structure.

### 15.4 Memory

Unifying 17M rows in one pass ran out of memory. Two changes fixed it. The archive reader filters to mature originals and quotes while reading, before the frames are concatenated. And the cross-source merge groups only the post IDs that appear more than once, which is a small fraction, instead of grouping every row.
