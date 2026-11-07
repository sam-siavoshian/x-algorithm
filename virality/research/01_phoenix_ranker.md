# Phoenix RANKING model — feature inventory, architecture, heads, losses, data pipeline

All paths are relative to `/home/user/x-algorithm/phoenix/`. Line numbers are from the checked-out tree.
"Prod" = `home_direct_packed` / `home_direct_packed_gb300` (`xrex/configs/xrecsys.py:577-644`); "nano" = `home_direct_packed_nano` (`xrex/configs/xrecsys.py:645-668`). Both use the **feature-prep input path** (`feature_prep_enabled=True`, `xrecsys.py:375`), which is the path documented below. A legacy input path (`feature_prep_enabled=False`, `block_history_reduce` / `block_candidate_reduce`, `xrex/models/recsys_model.py:1947-2152`) still exists for `xrecsys_seqpack`/`xrecsys_search`; it consumes the same raw tensors with a concat-then-project combine instead of project-then-sum.

---

## 0. One-paragraph answer

The ranker never sees raw text, images, or video. A candidate post is represented to the transformer as the **sum of D-dim projections** of: (a) two hashed post-ID embeddings, (b) two hashed author-ID embeddings, (c) a 6-level×256 **semantic-ID** (residual-quantized code of the post's 1024-d multimodal embedding — this is the only content signal that reaches prod), (d) categorical context: product surface, post-age bucket (hour granularity, 82 buckets), timezone, local hour-of-day (dithered), (e) viewer↔author relationship bits (followed-by-viewer, following-viewer), (f) **log2 engagement counts** (fav/reply/repost/quote/view) through a 2-layer MLP, (g) a stale-post (>14 d) flag, (h) a "time since latest history impression" bucket (prod only). The raw 1024-d multimodal embedding input exists in code (`cand_multimodal_proj`) but is **off in every registered ranking config** (`multimodal_embedding_type=None`, `xrecsys.py:364,659`). Author follower count enters only as a **label gate** (post-unexplored head), never as an input. Language, media-type flags, bookmark count, text length, etc. are in the wire proto but are **not read** by the ranker.

---

## 1. Candidate-post (post-side) inputs — exact tensors

### 1.1 Data-layer tensors (`PostSeq`, `xrex/data/recsys/recsys_batch.py:330-370`)

Per candidate slot `[B, C]` (C = `candidate_seq_len`=64, doubled to 128 by in-batch negatives):

| PostSeq key | dtype/shape | Source (training parquet / serving proto) | Citation |
|---|---|---|---|
| `post_hashes` | int32 `[B,C,2]` | `tweetIdSeq` → 2 multiplicative hashes into a 100M (prod)/100k (nano) item table | `recsys_batch.py:1725`, `recsys_embedding.py:310-320`; serving `util.rs:682-686` |
| `auth_hashes` | int32 `[B,C,2]` | `authorIdSeq` → 2 hashes into a 30M/30k author table | `recsys_batch.py:1726`; `util.rs:676-680` |
| `ip_hashes` | int32 `[B,C,2]` | `ipAddressSeq` (Int64Feature slot 10) — used only to derive the *user* IP token, not a per-post feature | `recsys_batch.py:1476-1478,1797-1805` |
| `impr_ts` | int32 sec `[B,C]` | `impressedTimeMsSeq/1000`; at serving = `CandidateSet.requestTimestampMs` or wall-clock now | `recsys_batch.py:1262-1266`; `util.rs:383-391,690` |
| `post_creation_ts_sec` | int32 `[B,C]` | decoded from the **snowflake tweet id** `((id>>22)+1288834974657)/1000` | `recsys_batch.py:996-997`; `util.rs:345-352` |
| `product_surface` | int32 `[B,C]` | `productSurfaceSeq` / `CandidateSet.productSurface` (enum `ProductSurface`, `recsys.proto:1190-1212`) | `recsys_batch.py:1267`; `util.rs:688` |
| `client_app_id` | int32 `[B,C]` | `clientAppIdSeq` (metrics slicing only) | `recsys_batch.py:1268` |
| `actions` | bool `[B,C,64]` | `actionNameMultiHotSeqSeq` — **labels** for candidates | `recsys_batch.py:907-912` |
| `continuous_actions` | f32 `[B,C,8]` | `continuousActionValuesSeqSeq` — **continuous labels** (slot 1 = dwell seconds) | `recsys_batch.py:914-926` |
| `post_sids` | uint16 `[B,C,6]` | `semanticIdSeq` (+1, 0 = missing) / `TweetInfo.semanticIds` hydrated by the SID service | `recsys_batch.py:1259-1261`; `util.rs:130,699-705` |
| `embedding` | f32 `[B,C,1024]` | `embeddingV5Seq` (only if `multimodal_embedding_type` set) / fetched by post id from the MM-embedding cache at serving | `recsys_batch.py:138-146,1052-1066,1178-1186`; `util.rs:881` |
| `categorical_features` | int16 `[B,C,18]` | see §1.2 | `feature_config.py:8-22` |
| `bool_features` | bool `[B,C,3]` | `isStalePost14d`, `isAuthorFollowedByViewerSeq`, `isAuthorFollowingViewerSeq` | `feature_config.py:45-48` |
| `float_features` | f32 `[B,C,2]` | `matchedWordFractionSeq` (search only) | `feature_config.py:54-55` |
| `int64_features` | int64 `[B,C,17]` | see §1.2 | `feature_config.py:58-72` |
| `safety_label_mask` | int64 `[B,C]` | `safetyLabelMaskSeq`; bit 2 = author NSFW | `feature_config.py:42`, `recsys_batch.py:1110-1112` |
| `promoted_ids`, `line_item_objective` | — | ads; zero for organic | `recsys_batch.py:1269-1272` |
| `search_query_embeddings` | f32 `[B,C,d]` | search surface only | `recsys_batch.py:1038-1050` |

**Categorical slots** (`CategoricalFeature`, `feature_config.py:8-22`): 0 `productSurfaceSeq`, 5 `timezoneSeq`, 6 `localHourOfDaySeq` (computed), 7 `localDayOfWeekSeq` (computed), 8 `postAgeBucketSeq` (computed), 9-13 `fav/reply/repost/quote/viewCountBucketSeq` (computed, legacy path only), 14 `authorIsNsfwSeq` (computed from safety mask), 15 `webConvTrackingIntegrationSeq` (ads), 16 `exactPhraseSeq`, 17 `matchedWordFractionBucketSeq` (search).
**Int64 slots** (`feature_config.py:58-72`): 0 `clientAppIdSeq`, 1 `promotedIdSeq`, 5 `favCountSeq`, 6 `replyCountSeq`, 7 `repostCountSeq`, 8 `quoteCountSeq`, 9 `viewCountSeq`, 10 `ipAddressSeq`, 11-12 DPA product key hashes (ads), 13-14 web-conv time-on-site (ads), 15 `requestTimeSecSeq` (computed), 16 `latestHistoryImpressionTimeSecSeq` (computed).

Local hour / day-of-week are computed from `impr_ts` + a 32-entry timezone-enum→UTC-offset table (`recsys_batch.py:2381-2438`; serving `util.rs:323-345,420-424`). Engagement counts are clipped to int32 (`recsys_batch.py:1480-1497`) and **zeroed for posts older than 14 d** (`STALE_POST_14D_TTL_SEC=1_213_200`, `feature_config.py:87`; `recsys_batch.py:1515-1535`; serving `util.rs:778-805`) with `isStalePost14d=True`.

### 1.2 What the model actually embeds for a candidate token (feature-prep path)

`build_feature_prep_streams` (`xrex/models/recsys_feature_prep.py:1270-1360`): `candidate_tokens = _build_seq_tokens + _add_context_features + _add_sid_features + _add_candidate_features`, then `* input_scale (= sqrt(D))`. Everything is **summed** into one D-dim token (D=2560 prod, 512 nano). Each component, with the prod flags from `_home_direct_packed_base()["feature_prep"]` (`xrecsys.py:375-400`) and the `home_direct_packed` override (`xrecsys.py:599-603`):

| # | Component | Param / math | On in prod? | Code |
|---|---|---|---|---|
| 1 | Post-ID hash embeddings | 2 rows of the shared table `[input_vocab, 1024]` → `cand_post_hash_{0,1}_proj` (1024→D), summed | yes | `recsys_feature_prep.py:1163-1192, 367-372` |
| 2 | Author-ID hash embeddings | 2 rows → `cand_author_hash_{0,1}_proj` (1024→D) | yes | same |
| 3 | Product surface | one-hot(16) · `cand_product_surface_emb [16,D]` | yes | `:635-640` |
| 4 | Post age | bucket = `(impr_ts−creation)//60//60 + 1`, clipped to 81 (4800 min = 80 h), 0 = unknown → `cand_post_age_emb [82,D]` | yes | `:55-72, 642-650, 51` |
| 5 | Timezone | `cand_timezone_emb [32,D]` | yes | `:652-658` |
| 6 | Local hour of day | `cand_hour_of_day_emb [25,D]`, training-time dither +1 h w.p. 0.1 | yes (`hour_of_day_dither_fraction=0.1`) | `:688-700, 376-380` |
| 7 | Local day of week | `[8,D]` | **no** (`enable_day_of_week=False` default) | `:702-709` |
| 8 | Time-of-day / time-of-week cyclic kernels | 48/14 centers, cosine kernel | no | `:660-686, 428-466` |
| 9 | `isAuthorFollowedByViewer`, `isAuthorFollowingViewer` | two `[2,D]` tables | yes | `:711-728` |
| 10 | **Engagement counts** | `v = log2(count+1)/32` for fav,reply,repost,quote,view (+ constant 1) → `cand_count_lograw_mlp_in [6,64]` → GELU → `cand_count_lograw_mlp_out [64,D]` | yes (`enable_engagement_counts=True`) | `:32-39, 730-754` |
| 11 | Stale-post flag (candidates only) | `cand_is_stale_post_14d_emb [2,D]` | yes (`enable_stale_post=True`) | `:756-765` |
| 12 | **Semantic ID** | 6 codes (1..256) + 1 "hash level" code derived from `post_hashes[...,0]` (`murmur-style mix % 256 + 1`), looked up in `feat_prep_post_sid_combined [(6+1)·257, 1024]`, summed over levels, projected `feat_prep_post_sid_proj [1024,D]`. Missing SID ⇒ all zeros | yes (`use_post_sid=True, sid_hash_level=True, sid_cross_attn=False`) | `:783-875, 878-932`; `xrecsys.py:365-370` |
| 13 | Multimodal post embedding | `cand_multimodal_proj [1024,D]` on `candidate_seq["embedding"]` | **no** (`multimodal_embedding_type=None` in every ranking cfg; README table `README.md:412`) | `:1100-1118` |
| 14 | Search-query embedding | `cand_search_query_proj` | no (search cfg only) | `:1120-1133` |
| 15 | Time since latest history impression | `gap = requestTime − latestHistoryImpressionTime`, log2 buckets `60·2^k`, 19 classes, zero-init table, 4× LR | yes in `home_direct_packed` (`enable_time_gap=True`), no in base/nano | `:75-81, 1135-1150`; `xrecsys.py:599-603` |

Padding/validity of a candidate slot = `post_hashes[...,0] != 0` (`:1189`).

**Not present anywhere in the ranker input graph** (present in `TweetInfo`, `recsys.proto:1421-1473`, but never stamped by `util.rs` nor read by `from_record_batch`): `languageCode`, `hasVideo`, `hasImage`, `tweetBoolFeatures` (is_reply/is_retweet/is_quote/has_media/is_promoted), `contentFeatures` (video duration, photo, media count, weighted text length, newline count, hasUrl), `bookmarkCount`, `notInterestedCount`, `firstVideoDurationMs`, `authorInfo.followers`, `postEngagers`, `retweeting/quoting/inReplyTo*`, `scoreInfo`, `authorEngagementCounts`, `embedding` (field 26 — the serving engine fetches MM vectors itself by post id, `util.rs:646,881`). grep confirms the only consumers of `languageCode`/`followers`/`firstVideoDuration` are the separate reply two-tower model (`xrex/models/recsys_reply_two_tower_v2_model.py:605-801`, `crates/common/xai-recsys/src/reply.rs`).

### 1.3 Serving-side stamping of a candidate (`crates/common/xai-recsys/src/util.rs:642-895`)
`tweet_id`→item hashes, `author_id`→author hashes, `CandidateSet.productSurface`, `impr_ts = requestTimestampMs|now`, creation from snowflake, `ad_info` ids, `semantic_ids`, set-level `search_query_embedding`, `device_feature.timezone` → local hour/dow, `fav/reply/retweet/quote/view_count` (zeroed when stale), `is_author_followed_by_user`, `author_info.is_following_user`, `search_lexical_match`, author-NSFW from `safety_label_mask>>2 &1`, `isStalePost14d`, MM embedding (optional `Vec<f16>` passed in). That is the complete candidate-side contract.

---

## 2. Viewer-side inputs

### 2.1 User prefix tokens (2 tokens in prod: `num_user_prefix_tokens = enable_user_embedding + has_user_features`, `xrecsys.py:862-866`)

**Token 1 — hashed identity** (`_build_user_token`, `recsys_feature_prep.py:488-519`): Σ of 2 user-ID hash rows → `user_hash_{i}_proj` + 2 **IP-address** hash rows (`ipAddressSeq` of the latest event, 10M-row table) → `user_ip_hash_{i}_proj`. Mask = `user_hashes[:,0]!=0`.

**Token 2 — profile features** (`_build_user_features_token`, `:522-623`), prod flags `xrecsys.py:377-383`:
- country (`userCountryCode`, 250-way table; serving = `crc32(country_code) % 256`, `util.rs:501-506`) — on
- language (`userLanguageCode`, 200) — on
- state (100), DMA code (300) — off
- location: lat/lon → 8-frequency sin/cos Fourier features (32-d) → `user_feat_location_proj` — on
- gender: profile gender (3) + inferred gender (2) × `userInferredGenderScore` — on
- age: `userAgeBracket` (10) + `log1p(ageInYears)` × learned vector — on
- installed apps: 64-way multi-hot → projection — **off** in `home_direct_packed` (on in `xrecsys_seqpack`)
Source columns: `UserCategoricalFeature`/`UserFloatFeature` (`feature_config.py:118-135`); serving `UserContext`/`ClientContext` (`recsys.proto:53-99`, `util.rs:905-990`).

### 2.2 History sequence (`history_seq`, up to `history_seq_len=1022` events)
Each history event is one **aggregated (viewer, post) pair**: `AggregatedUserAction{tweetInfo, actions[], impressedTimeMs}` (`recsys.proto:1554-1560`). Training takes the **last** `num_history` non-candidate rows (`recsys_batch.py:1170-1174`, `user_history_indices[-num_history:]`); serving takes the last `history_seq_len` aggregated actions (`util.rs:1069-1070`).

Per-event fields consumed:
- post hashes, author hashes (same tables as candidates; prefix `hist_`)
- `actions` multi-hot `[64]` — **all action names** observed on that post (fav, reply, repost, quote, bookmark, click, VQV, dwelled/not-dwelled, report, block, …), embedded by `hist_action_emb [64,D]` as a multi-hot dot (`recsys_feature_prep.py:943-949`); serving sets bit `action_name` for every `ActionInfo` (`util.rs:1105-1109`)
- `continuous_actions[...,1]` **dwell time (s)** — serving sums `UserActionMeta.dwellTime` ms over the actions (`util.rs:1104-1118`). Prod embeds it via a **piecewise-linear bucket table** `hist_dwell_time_table [34,D]` with knots = `(0, cread thresholds 2s…240s)` + two sentinel rows (dwell==2.0 s sentinel → "negative-feedback" vs "other" row), zero-init (`:950-985`, `enable_dwell_bucket_table=True, dwell_bucket_cap=240`). Fallback: `clip(dwell,0,30)/30 × vector` (`:986-993`)
- `continuous_actions[...,2]` click-dwell (`log1p(clip(x,0,60))/log1p(60)` × vector) — on (`enable_click_dwell_time=True`, `:1006-1015`)
- slots 3 / 0: home video continuation secs / profile visit secs (`log1p(clip(x,0,300))`, zero-init vectors) — `home_direct_packed` only (`enable_home_excursion_history`, `:1017-1037`)
- product surface, post age at impression, timezone, local hour, followed/following bits, engagement counts (same `_add_context_features` with prefix `hist`; stale flag is skipped for history, `:756`)
- semantic IDs (same table/projection — `feat_prep_post_*` params are shared between history and candidates, `:910-930`)
- time gap `request_ts − impr_ts` bucket `hist_time_gap_emb [19,D]` (prod, `:1076-1095`)
- author NSFW categorical (slot 14) is stamped but only used on the legacy path (`recsys_model.py:3115-3170`); on the feature-prep path there is no `author_is_nsfw` embedding.
Mask = `post_hashes[...,0]!=0`.

History length distribution in prod training is forced by sequence packing: Beta(1,1) over `[126, 1022]`, mean 414 (`xrecsys.py:401-409, 583-585`; `sequence_packing.py:33-63`). Metrics tag users with `<128` history events as "new users", `==0` as "no history" (`recsys_model.py:4784-4795, 5026-5031`).

### 2.3 Not used by ranking (exists for other models)
`id_list_features` (hydrator top-k engaged post/author lists, `xrex/data/recsys/id_list_presets.py:33-80`) — no ranking config sets `id_list_features`. `UserContext.followed_grok_topics`, `user_followers_count`, `is_nsfw_user` etc. are not read by `util.rs` for ranking.

---

## 3. Architecture

### 3.1 Sequence layout and attention
`tokens = [user_id_token, user_features_token | history (1022) | candidates (64 own + 64 in-batch negatives)]`, padded candidates to multiple of 128 (`pad_candidates_to_128`, `recsys_model.py:4830-4875`). Prod trains **packed** (`use_seqpack=True`): several variable-length users per row, varlen attention kernel (`cutedsl_ranker_varlen_attn`), candidate packing with 128 candidate tokens per row (`xrecsys.py:579-581`).

Attention mask: `segment_ids` = 1 for user/history, **−1 for candidates**, 0 for padding (`recsys_model.py:2277-2288`). User+history attend bidirectionally among themselves; each candidate attends to user+history **and only itself** (README diagram `README.md:196-240`; kernels `xrex/pallas/ranker_attention_*`, bounds `recsys_attention.py:38-43`). Non-causal, RoPE positions: **right-anchored** so the most recent history event sits at position `history_end−1` and all candidates share position `history_end` (`right_anchored_rope_positions`, `recsys_model.py:260-281`).

### 3.2 Trunk (`TransformerConfig`, `xrecsys.py:1041-1077`; `xrex/models/transformer.py:62-111`)

| | prod | nano |
|---|---|---|
| `emb_size` D | 2560 | 512 |
| layers | 8 (prod `home_direct_packed`: last 2 are **candidate-only layers**, `num_candidate_only_layers=2`, `xrecsys.py:586-587`; `transformer.py:789-800,909-940`) | 4 |
| Q / KV heads (GQA), key size | 20 / 4, 128 | 4 / 2, 128 |
| FFN | non-gated, widening 2 → 5120 | 1024 |
| norms | pre-norm RMSNorm, `qk_norm=True`, `primer_norm=False`, final `final_layer_norm` RMS (`recsys_model.py:4360-4365`) | same |
| attn logit cap | −1 (off) in home; 80 soft-sign in immersive | |
| embedding table width | 1024 | 128 |
| per-device batch | 1024 (H100) / 512 (GB300) | 64 |
| μP scaling | `ScaleConfig` (base width 512): `input_scale=sqrt(D)`, `output_scale=2·sqrt(512/D)`, emb LR mult `0.5·sqrt(512/D)` (`xrex/models/scaling.py:9-47`) | |

### 3.3 Hash-embedding scheme (`xrex/models/recsys_embedding.py`)
One shared table `emb_table[input_vocab_size, emb_table_width]` (bf16, `recsys_model.py:1451-1469`), `input_vocab_size = user + item + author + ip + 60 + 1` rounded up to 512 (`xrecsys.py:146-156`): prod 100M+100M+30M+10M → **240,000,512 rows × 1024** (≈2.46e11 params, ~492 GB bf16, sharded `ep=64`); nano 240,128 × 128. Row layout: `[0]=pad, 1..64 reserved, offset_user=65, offset_item=65+U, offset_author=…, offset_ip=…` (`recsys_embedding.py:234-248`). Hash `h_j(id) = ((id·scale_j + bias_j) mod modulus) mod (table−1) + 1`, 2 independent (scale,bias) pairs per entity type, `id==0 → 0` (`:154-172`; constants `xrecsys.py:160-171`). The identical hash is implemented in Rust for serving (`crates/common/xai-recsys/src/model_config.rs:154-260`). Lookups are gathered with `shard_map` all-gather/all-to-all over the expert axis (`trainer_recsys.py:873-972`); optimizer = rowwise AdaGrad with half-life decay (`TRAINING.md:27-30`, `xrecsys.py:452-459`).

### 3.4 Heads
- `unembeddings [D, 64]` → 64 action logits per candidate, scaled by `output_scale` (`recsys_model.py:3288-3306, 4395-4398`)
- `continuous_unembeddings [D, 8]` → 8 continuous slots (`:3326-3343`)
- per CREAD head: `{metric}_cread_unembeddings [D, 30]` (`:3452-3478`)
- (ads only) value-bin heads `[D,32]` (`:3345-3375`)

### 3.5 Parameter counts implied
Prod dense: per layer attn `Wq 2560²+Wk 2560·512+Wv 2560·512+Wo 2560² = 15.73M`, FFN `2·2560·5120 = 26.21M` → 41.9M/layer × 8 = **335M**; feature-prep projections 12 × (1024×2560) = 31.5M; SID table 7·257·1024 = 1.84M + proj 2.6M; categorical/action/dwell tables ≈ 1–2M; heads ≈ 0.5M → **≈ 375M dense** + **≈ 246B** embedding-table parameters. Nano: 4 × 1.84M = 7.3M trunk + ~3.2M feature-prep + 30.7M table.

---

## 4. Prediction heads, labels, losses

### 4.1 Discrete heads (64 BCE heads = `ActionName` enum indices 0..63)
`output_vocab_size = roundup(ACTION_TYPE_MAP_LEN=60, 64) = 64` (`xrecsys.py:47-48,157-159`); labels are `actionNameMultiHotSeqSeq[...,64]` with index = `ActionName` value (`recsys.proto:456-697`). Meaningful organic heads inside 0..63: 1 fav, 4 reply, 5 quote, 6 retweet, 10 server report, 11 recap dwelled, 12 recap not-dwelled, 13 video quality view, 14 photo expand, 15 click mention, 16 client report, 17 not interested, 19 see fewer, 21 follow author, 23 block, 25 mute, 27 mute conversation, 29 click, 30 click profile, 31 click share, 32 share copy link, 33 send DM, 35 click hashtag, 36 bookmark, 37 share, 39 open link, 40 screenshot, 41 translate, 42 not relevant, 44 post video, 45 show more, 46 grok analyze, 47 video open, 49 quoted photo expand, 59 **post unexplored** (computed label), 60 external link long dwell. The "primary engagement" metric group (`constants.py:25-122`) names the heads production reports on: Favorited, Replied, Retweeted, Quoted, Bookmarked, Shared(+copy link, DM), NotInterested, MuteAuthor, BlockAuthor, Reported, VideoQualityViewed, VideoOpened, ProfileClicked, RecapDwelled/NotDwelled, OpenLink, Clicked, PostUnexplored, PostedVideo, ExternalLinkLongDwelled, ExternalLinkSession<3/5/10 >3/5/15/20/25/30/60 s.

Loss: sigmoid BCE on all 64 heads, masked by `loss_mask`, summed over heads, **normalized by the number of valid candidate slots** (not heads) (`loss_recsys.py:21-55`; `recsys_model.py:5063-5071`). Head-level weighting:
- `view_loss_weight` (0.5 in `home_direct_packed`, `xrecsys.py:596`) multiplies heads 11/12/13 (recap dwelled, not dwelled, VQV) (`constants.py:566-570`; `recsys_model.py:4990-4995`)
- negative-feedback heads (16,17,19,25,23,42,27,52,104) are masked out on in-batch negatives (`mask_neg_feedback_on_negatives=True`, `recsys_model.py:1298,4969-4974`; `constants.py:587-597`)
- optional `mask_candidate_positive_when_negative_action_present` (off in home) zeroes positive labels when a negative-feedback action is present (`:4930-4962`)
- per-row `sampleWeight` × **log-Q weight** (see §4.4)
- regularizer `act_l2_weight=5e-7` on final hidden activations (`xrecsys.py:1086`; `recsys_model.py:5235`)

### 4.2 Continuous slots (`ContinuousActionName`, `recsys.proto:699-708`)
0 HOME_PROFILE_VISIT_SECS, 1 DWELL_TIME, 2 CLICK_DWELL_TIME, 3 HOME_VIDEO_CONTINUATION_SECS, 4 BRIDGE_PROBABILITY, 5 ADS_WEB_CT_PURCHASE_VALUE, 6 ADS_VIDEO_WATCH_TIME_SECS, 7 ADS_VIDEO_LENGTH_SECS. Configured losses (`xrecsys.py:869-948, 282-330`), weights for `home_direct_packed`:

| metric | slot | loss | surfaces | weight (prod / nano) |
|---|---|---|---|---|
| dwell-time-tweedie | 1 | Tweedie p=1.5, softplus, cap 300 | gallery(immersive) only | 0 / 0 (0.1 in `immersive_page_rl`) |
| dwell-time-mae | 1 | MAE on `clip/30`, sigmoid | non-gallery | 0 / 0 |
| click-dwell-binary | 2 | BCE on `click_dwell > 10 s` | all | 1.0 / 1.0 |
| dwell-time-cread | 1 | CREAD, 30 log-spaced thresholds 2 s→120 s + Huber restoration (δ=5 s, w=0.5) | non-gallery | 0.1 / 0.1 |
| dwell-time-cread-immersive | 1 | CREAD 2 s→240 s | gallery | 0.1 / 0.1 |
| home-video-continuation-cread | 3 | CREAD 2→300 s, **conditioned on head 47 (video open)** | home + ranked-following | 0.013 / 0 |
| home-profile-visit-secs-cread | 0 | CREAD 2→300 s, conditioned on head 30 (click profile) | home surfaces | 0.0035 / 0 |

CREAD (`loss_recsys.py:110-187`; decode `recsys_model.py:3452-3478`): thresholds `t_k = 2·(S/2)^{k/29}`, `k<29`, plus `S`; logits `[D,30]`; classification BCE on `y > t_k`; predicted value `Σ_k sigmoid(l_k)·(t_k − t_{k−1})`; plus Huber on the restored value. Sentinel handling: dwell == 2.0 s is the "default sentinel"; losses drop labels equal to 0 or 2.0 (`cread_mask_sentinel_labels`, `recsys_model.py:4658-4668`); valid dwell range 0..86 400 s (`recsys_feature_prep.py:401-403`; `recsys_model.py:4641-4650`). Continuous losses are **never trained on in-batch negatives** (`mask_negatives=True`, `loss_recsys.py:58-65`). At inference sigmoid heads are rescaled back to seconds by `norm_scale` (`recsys_model.py:550-575`).

### 4.3 Computed label: post-unexplored (head 59)
`compute_post_unexplored_labels` (`recsys_batch.py:105-135,999-1023`): original post (`inReplyToPostIdSeq==0`), age ≤ 24 h, and `viewCount < 0.03 × authorFollowerCount × (1−e^{−ln2·age/8h})/(1−e^{−ln2·24/8})`. Only enabled with `compute_post_unexplored_label=True` (prod yes, nano no, `xrecsys.py:356,667`). **This is the sole use of author follower count in the ranker.**

### 4.4 Negative sampling and log-Q
`num_negatives_per_example=1` (`xrecsys.py:197`): for each user, the 64 candidate slots of **one random other user in the same batch** are appended as slots 64..127 (`apply_negative_sampling`, `recsys_batch.py:1770-2120`; chosen at `:1945-1953`). Negatives keep all post features but their label vector is reset to **only `ClientTweetRecapNotDwelled=1`** (`:1990-1993`). In the loss, slots ≥ `candidate_seq_len` are `negative_sample_mask` (`recsys_model.py:4468-4484`). With `log_q_correction=True`, each negative slot gets weight `1/count(hashed_post_id among negative slots in the batch)` (`get_candidate_tweet_counts`, `:1523-1537,4486-4495`), i.e. popular posts that recur as negatives are down-weighted; positives keep weight 1. Global (corpus) negatives exist (`apply_global_negative_sampling`, `:2121+`) but `num_global_negatives_per_example=0` for ranking.

### 4.5 Ads / search / page-RL heads (present in code, off for home organic)
Purchase-value binned heads (`value_heads.py`, `value_bin_tables.py`: 31 log-spaced edges 0.02…50, softmax readout), ads head masking, conversion-delay slices, search relevance heads 105/106 conditioned on prompt, `page_rl` slate-RL decoder (`immersive_page_rl` cfg). Ignore for virality.

---

## 5. Training data pipeline

### 5.1 What one training row is (parquet = Kafka replacement; `reference/dump_gen.py:208-279`, required columns `feature_config.py:148-208`)
One row = one **user session snapshot** of length `S = 1022 + 64`: a time-ordered sequence of `(tweetIdSeq, authorIdSeq, impressedTimeMsSeq, productSurfaceSeq, timezoneSeq, clientAppIdSeq, ipAddressSeq, safetyLabelMaskSeq, fav/reply/repost/quote/viewCountSeq, actionNameMultiHotSeqSeq[64], continuousActionValuesSeqSeq[8], semanticIdSeq[6])` plus `paddingMask` and `newEventMask`. `newEventMask=False` ⇒ **history** event; `newEventMask=True` ⇒ **candidate** (post served in the request that generated this row). User-level columns: `userId`, `userCountryCode`, `userLanguageCode`, `userState`, `userGender`, `userAgeBracket`, `userDmaCode`, `userLongitude/Latitude`, `userAgeInYears`, `userInferredGender(+Score)`, `installedAppsMultiHot[64]`, `sampleWeight`, `kafka_timestamp_ms`. Optional: `embeddingV5Seq[64×1024]` (candidates only), `authorFollowerCountSeq`, `inReplyToPostIdSeq`, `searchQueryEmbeddingSeq`, ads columns.

### 5.2 Label collection (impression → actions)
Upstream (not in repo) aggregates every `UserAction` on a `(user, tweet)` into `AggregatedUserAction{actions[], impressedTimeMs}` (`recsys.proto:1544-1560`); `UserActionMeta` carries `dwellTime`, `productSurface`, `timeZone`, `safetyLabelMask`, `servedTimeMs`, `percent_screen_height_100k`, `fair_view_time_ms` (`:1389-1419`). The multi-hot is simply "did action X ever occur on this impression"; the label window is a property of the upstream aggregator and **is not specified in the repo**. Hints: `impressed_time_ms` is the sort key; `ServedFeed` proto logs what was served per request (`crates/serving/xai-recsys-proto/proto/served_feed.proto`); ads use explicit `conversionDelayMsSeq` columns and "late window" streams (`xrex/data/recsys/ads_late_window.py`), implying organic labels are closed within hours. Production also **downsamples rows with no rare action** (report/not-interested/block/mute/open-link/ext-link>30 s, indices 16,17,23,25,39,89) and reweights survivors by `10000/rate` (`dump_keep_sample_weights`, `recsys_batch.py:75-102`).

### 5.3 Batch assembly (`from_record_batch`, `recsys_batch.py:790-1770`)
history = last ≤1022 non-candidate rows; candidates = first ≤64 `newEvent` rows; creation ts from snowflake; local hour/dow; author NSFW bit; stale-post zeroing; `requestTimeSecSeq = min(candidate impr_ts)`, `latestHistoryImpressionTimeSecSeq = max(hist impr_ts ≤ request)` (`:2441-2459`); then in-batch negatives; hashing; `user_ip_hashes` from the last non-zero IP (`:1797-1805`). Data reader: partition-interleaved parquet with time-window/block shuffle (`xrex/data/parquet_recsys.py:795-1700`), `num_kafka_partitions=1024` (`xrecsys.py:198`). Serving builds the same tensors from the gRPC `PredictNextActionsRequest{sequences, candidateSets, clientContext, userContext}` (`recsys.proto:139-191`; `util.rs:996-1350`).

### 5.4 Synthetic generator semantics (`reference/world.py`) — useful as a sanity model, not real data
Engagement probability = base rate per action (fav .14, click .10, VQV .08, repost .045, reply .035, bookmark .02, quote .015, share .012; `world.py:53-62`) × sigmoid gate of `affinity = user_topic_pref × author_quality × post_quality` (`:292-307`); dwell ~ lognormal in affinity, `dwelled` ⇔ engaged & dwell ≥ 10 s, `not_dwelled` ⇔ no engagement (`:308-330`). Counts: `views ∝ popularity^(Zipf 1.05) × 3e6`, `fav = views×0.03×quality_gate`, repost 0.008, reply 0.005, quote 0.002 (`:574-598`). Posts 1–36 h old at serve time.

---

## 6. How popularity / engagement counts / author size feed the model
1. **Per-post engagement counts (fav, reply, repost, quote, view)** as they stood at impression time are a direct input on both history and candidate tokens: `log2(c+1)/32` → 2-layer MLP (64 hidden) → D (`recsys_feature_prep.py:730-754`). Zeroed and flagged for posts >14 d old. Legacy path uses log2 **buckets** (32 classes, per-feature caps fav 18, reply 13, repost 15, quote 13, view 25; `recsys_model.py:230-247`) with 16-d tables (`xrecsys.py:993-1017`).
2. **Author identity** via 2 hashed author-ID embeddings (30M rows) — author "size" is learned implicitly from how often the author appears/gets engaged; no explicit follower count input.
3. **Viewer–author graph**: `isAuthorFollowedByViewer`, `isAuthorFollowingViewer` 2-way embeddings.
4. **Author follower count** only gates the post-unexplored label (§4.3) — i.e., the model learns to *predict* "this post is under-exposed relative to its author's reach", which downstream can use for exploration.
5. **Post ID embedding** itself (100M-row hashed) memorizes per-post popularity for posts seen in training; semantic IDs give generalization to unseen posts.
6. **Post age** bucket (hourly to 80 h) and time-since-last-history gap.
7. Retrieval/eval sidecar metadata (`post_author_pairs_metadata/metadata_1day.parquet`: `fav_count, view_count, author_followers_count, has_video`) is used for cold-pool filtering (`xrex/data/cold_pool_filter.py:26-63`: fav<8 & view<500) and eval slicing (`xrex/eval/recsys_eval.py:118-128`), not as ranker input.

---

## 7. What is NOT in the repo
- **Weights/checkpoints**: none (`TRAINING.md:99-100`, `reference/README.md:176`).
- **Real data**: no Kafka feed, no dumps; `reference/dump_gen.py` writes a deterministic synthetic world (`reference/README.md:11-16`). Dataset path is a placeholder `"/path/to/offline_kafka_dump"` (`xrecsys.py:188`).
- **Upstream feature/label services**: the user-action aggregator (impression→actions, dwell computation, label windows), the hydrator that fills `TweetInfo` counts / follow bits / safety masks, the id-list hydrator, the SID assignment service (reference `sid_index_server.py` + mirrored `sid_codebook.py`/`sid_assign.py` ship), the MM-embedding service (reference `mm_encoder.py` = Qwen3-VL-Embedding-8B over rendered post text+images+quoted post, truncated to 1024-d and L2-normalized; `reference/mm_encoder.py:9-20`, `reference/README.md:110-121`; **weights not shipped**).
- **Production orchestration**: multi-host scheduling, telemetry, Kafka creds (`xrex/configs/data_feeds.py`), internal "tuned RMS-Adam" (AdamW substituted for non-Muon configs, `TRAINING.md:21-25`).
- **The 64-wide action taxonomy used in prod**: `ACTION_TYPE_MAP_LEN=60` is hard-coded (`xrecsys.py:47`) even though the proto defines 256 names; which of 0..63 carry real positives in production depends on the unshipped aggregator.
- **Downstream scoring**: how the 64 probabilities + dwell predictions are combined into a final rank is in other repo dirs (`home-mixer/`, `xai-value-model/`, `phoenix-rankall*`), not in `phoenix/`.

---

## 8. Mirror checklist for a virality model (what the ranker's candidate token actually contains)
`post_id (hashed ×2)`, `author_id (hashed ×2)`, `semantic_id[6] (RQ codes of a 1024-d text+image+quote embedding)`, `fav/reply/repost/quote/view counts at impression`, `post_age_minutes` (hourly bucket ≤80 h), `product_surface`, `viewer timezone`, `local hour`, `author_followed_by_viewer`, `author_following_viewer`, `is_stale_14d`, `time_since_viewer's_last_history_event`; viewer context = 2 prefix tokens (id/IP hash; country, language, lat/lon, gender, age) + up to 1022 past `(post, author, actions[64], dwell, surface, age, SID, counts)` events. Targets = 64 binary action heads (+ post-unexplored) and CREAD dwell / excursion-time distributions, trained with in-batch impression negatives labelled "not dwelled" and log-Q down-weighting of popular negatives.
