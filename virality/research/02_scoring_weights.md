# 02 — Scoring weights, value model, and post-scoring adjustments (x-algorithm)

Repo: /home/user/x-algorithm. All paths below are relative to that root. Line numbers are from the checked-out files.
Scope: how Phoenix per-action predictions become a final score and rank (home-mixer value model + vm-ranker + selectors), plus popular-posts retrieval.

Two copies of the weight defaults exist and resolve from the SAME feature-switch keys (`rust_home_mixer_*`):
- `home-mixer/params/param.rs` — header says "mirrored from config feature-switch defaults; last sync 2026-11-05T16:00:35Z" (param.rs:1). Used by home-mixer for the local weighted score / cold-start / fallback.
- `vm-ranker/params.rs` — the vm-ranker binary's compiled fallbacks for the same keys; in prod it resolves them from the synced config git repo (vm-ranker/main.rs:69-98, ranking_config.rs:55-67). Where the two files differ I list both; treat the home-mixer value (newer sync) as the production value.

---------------------------------------------------------------------------------------------------

## 1. The exact formula

### 1.1 Weighted sum (`xai-value-model/scoring.rs:79-218`)

```
weighted(c) = Σ_h  w_h · x_h(c)              # 30 terms, weighted_terms()
apply(score, w) = score.unwrap_or(0.0) * w   # missing head => contributes 0   (scoring.rs:57-59)
product(p, v)   = Some(p? * v?)              # both must be present            (scoring.rs:61-63)
```

Term list, in code order (scoring.rs:102-217), with the input each weight multiplies:

| # | term name | input x_h | weight symbol | home-mixer default (param.rs) | vm-ranker default (vm-ranker/params.rs) |
|---|---|---|---|---|---|
| 1 | favorite | P(like) | FavoriteWeight | **0.708** (404-409) | 0.5 (3) |
| 2 | reply | P(reply) | ReplyWeight (+ bidirectional boost, see 2.4) | **5.0** (410) | 5.0 (4) |
| 3 | retweet | P(repost) | RetweetWeight | **1.0** (423) | 1.0 (5) |
| 4 | photo_expand | P(photo expand) | PhotoExpandWeight | **0.05** (424-429) | 0.05 (6-11) |
| 5 | video_open | P(video open) | VideoOpenWeight | **0.0** (430-435) | 0.07 (12-17) |
| 6 | click | P(click) | ClickWeight | **0.274** (436) | 0.3 (18) |
| 7 | open_link | P(open link) | OpenLinkWeight | **0.2** (437) | 0.2 (19) |
| 8 | profile_click | P(profile click) | ProfileClickWeight | **0.0** (438-443) | 0.0 (20-25) |
| 9 | vqv | P(video quality view), gated by vqv_eligible | VqvWeight | **0.0** (444) | 0.0 (26) |
| 10 | share | P(share) | ShareWeight | **2.118** (445) | 2.0 (27) |
| 11 | share_via_dm | P(share via DM) | ShareViaDmWeight | **5.0** (446-451) | 5.0 (28-33) |
| 12 | share_via_copy_link | P(share via copy link) | ShareViaCopyLinkWeight | **20.0** (452-457) | 20.0 (34-39) |
| 13 | dwell | P(dwell) | DwellWeight (+ bidirectional dwell boost) | **0.05** (458) | 0.05 (40) |
| 14 | quote | P(quote) | QuoteWeight | **5.0** (459) | 5.0 (41) |
| 15 | quoted_click | P(click on quoted post) | QuotedClickWeight | **0.05** (460-465) | 0.05 (42-47) |
| 16 | quoted_vqv | P(VQV on quoted video), gated | QuotedVqvWeight | **0.0** (466-471) | 0.0 (48-53) |
| 17 | dwell_time | E[dwell seconds] (continuous) | ContDwellTimeWeight | **0.0075** (496-501) | 0.004 (54-59) |
| 18 | click_dwell_time | E[post-click dwell seconds] (continuous) | ContClickDwellTimeWeight | **0.385** (502-507) | 0.4 (60-65) |
| 19 | video_continuation | P(video_open) × E[home_video_continuation_secs] | VideoContinuationWeight | n/a — hard-coded 0.0 in home-mixer (scorers/value_model.rs:39) | 0.0 (66-71) |
| 20 | user_video_continuation | P(video_open) × user_secs (see 1.3) | UserVideoContinuationWeight | hard-coded 0.0 (value_model.rs:40) | 0.0 (72-77) |
| 21 | profile_visit_secs | P(profile_click) × E[home_profile_visit_secs] | ProfileVisitSecsWeight | hard-coded 0.0 (value_model.rs:41) | 0.0 (78-83) |
| 22 | follow_author | P(follow author) | FollowAuthorWeight | **4.0** (472-477) | 4.0 (84-89) |
| 23 | not_interested | P(not interested) | NotInterestedWeight | **-53.07** (509-514) | -47.52 (114-119) |
| 24 | block_author | P(block) | BlockAuthorWeight | **-31.2** (515-520) | -31.2 (120-125) |
| 25 | mute_author | P(mute) | MuteAuthorWeight | **-58.8** (521-526) | -58.8 (126-131) |
| 26 | report | P(report) | ReportWeight | **-234.0** (527) | -234.0 (132) |
| 27 | not_dwelled | P(not dwelled) | NotDwelledWeight | **-0.02** (528-533) | -0.02 (133-138) |
| 28 | post_unexplored | P(post unexplored / "pdwell"), in-network only unless flag | PostUnexploredWeight | **0.015** (478-483) | 0.02 (90-95) |
| 29 | post_video | P(post video) | PostVideoWeight | **0.0** (484-489) | 0.0 (102-107) |
| 30 | grok_analyze | P(Grok analyze) | GrokAnalyzeWeight | **0.0** (490-495) | 0.0 (108-113) |

Boosts / gates that modify the weights above:

| name | default | effect | cite |
|---|---|---|---|
| BidirectionalFollowReplyWeightBoost | **15.0** | reply weight becomes 5 + 15 = **20.0** for eligible candidates | param.rs:411-416; weights.rs:86-93 |
| BidirectionalFollowDwellWeightBoost | **0.0** | would add to dwell weight for eligible candidates (shipped off) | param.rs:417-422; weights.rs:95-102 |
| bidirectional eligibility | — | `!is_reply && !is_retweet && is_mutual_follow_author` | xai-value-model/inputs.rs:20-22 |
| MinVideoDurationMs | **10_000** ms | vqv eligible only if candidate.min_video_duration_ms > 10 000 ms | param.rs:707-712; util/candidates_util.rs:19-34 |
| MAX_FOLLOWERS_THRESHOLD | **10_000** | vqv weight forced to 0 if the VIEWER has ≥ 10 000 followers | util/candidates_util.rs:4, 24-30 |
| EnableQuotedVqvDurationCheck | false | quoted_vqv eligible without a duration check | param.rs:714-719; candidates_util.rs:36-45 |
| PostUnexploredIncludeOutOfNetwork | false | post_unexplored weight only applies when in_network == Some(true) | vm-ranker/params.rs:96-101; scoring.rs:95-100 (home-mixer hard-codes false: value_model.rs:29) |
| WeightPerturbationSigma / Salt | 0.0 / "" | per-user ±σ multiplicative log-perturbation of every weight (md5 sign); off | vm-ranker/params.rs:163-174; weights.rs:104-112, 202-209 |

Observations on transforms:
- No probability clipping, log, binning, or squashing anywhere in the fusion. Each weight multiplies the raw Phoenix output. The only clamp is inside the (disabled) user-debias path: probabilities recentred in logit space, then `clamp(1e-7, 1-1e-7)` (vm-ranker/scoring/debias.rs:33, 221-235).
- Continuous heads (`dwell_time`, `click_dwell_time`) enter as seconds × weight, linearly, uncapped (scoring.rs:152-159).
- `home_video_continuation_secs` and `home_profile_visit_secs` are "home excursion" continuous heads fetched from Phoenix (`ContinuousActionName::HomeVideoContinuationSecs/HomeProfileVisitSecs`, phoenix_scorer.rs:171-183, enabled by EnableHomeExcursionScores=true param.rs:308-313) but their weights are 0, so they do not affect ranking today.
- `set_user_video_continuation` (scoring.rs:220-237): user_secs = Σ_c P(open_c)·cont_c / Σ_c P(open_c) over the request's candidates (an expected watch-time-per-open for this viewer); only run when UserVideoContinuationWeight != 0 (vm-ranker/scoring/value_model.rs:169-171).

### 1.2 Offset (makes net-negative posts sort below every net-positive post) — `scoring.rs:239-248`

```
NEGATIVE_SCORES_OFFSET = 0.001                        (xai-value-model/weights.rs:4; home-mixer/params/config.rs:42)
positive_sum = Σ positive-side weights (reply counted at base 5.0, NOT the +15 boost)   (weights.rs:51-72)
negative_sum = -(not_interested + block + mute + report + not_dwelled)                  (weights.rs:74-80)
total_sum    = positive_sum + negative_sum                                              (weights.rs:82-84)

offset_score(x) = x + 0.001                               if x >= 0
                = (x + negative_sum) / total_sum * 0.001  if x <  0     # maps [-negative_sum, 0) into [0, 0.001)
                = max(x, 0)                               if total_sum == 0
```
With home-mixer defaults: negative_sum = 377.09; positive_sum = 43.465; total_sum = 420.555. A post whose weighted sum is negative therefore lands in (0, 0.001) and sorts below any non-negative post, with ordering among negatives preserved.
`fuse_heads = offset_score(compute_weighted_score)` (scoring.rs:65-67) is what home-mixer stores as `weighted_score` (scorers/value_model.rs:83-89; vm_ranker.rs:59-75).

### 1.3 Post-fusion multipliers — `scoring.rs:261-406`

Default path (`multiplier_pre_offset = false`, vm-ranker/params.rs:199-204; scoring.rs:393-405):
```
base_i     = offset(weighted_i)  (+ author_exploration_bonus_i, applied in un-offset space; bonus default 0.0)
adjusted   = apply_cold_start_decisions(base)              # lifts the single cold-start candidate (see 2.3)
m_i        = author_diversity_i × oon_i                     # computed on the ordering of `adjusted`
score_i    = adjusted_i × m_i                               # PostFusionMultiplier::apply  (scoring.rs:23-25)
```
- `author_diversity_i = (1 - floor) · decay^k + floor`, k = number of higher-`adjusted`-scoring candidates in the SAME request from the same author (scoring.rs:261-282). Defaults decay **0.5**, floor **0.25**, enabled (vm-ranker/params.rs:175-192). Sequence per author: 1.0, 0.625, 0.4375, 0.34375, 0.297, … → 0.25.
- `oon_i = ctx.effective_oon_weight` when `in_network == Some(false)`, OR when `in_network == Some(true) && (is_reply || is_retweet)` (EnableOonRescoreForInNetworkRepliesRetweets = true, vm-ranker/params.rs:193-198; scoring.rs:284-293). `in_network == None` → 1.0.
- `effective_oon_weight` (vm-ranker/scoring/value_model.rs:123-147): **0.75** (OonWeightFactor, params.rs:217-222); **0.5** for topic requests (TopicOonWeightFactor, 211-216); **0.00001** for "new users" = account age < NewUserAgeThresholdSecs (**0**, i.e. disabled) AND following ≥ NEW_USER_MIN_FOLLOWING = 5 (params.rs:223-234; value_model.rs:21).
- Because multipliers apply to the offset score, even net-negative posts (in (0,0.001)) are multiplied; ordering is unchanged.
- Home-mixer's own local copy of the weights disables diversity (decay 1.0, floor 1.0, enable false) and OON rescoring (scorers/value_model.rs:46-50): those adjustments happen only in the vm-ranker service. Home-mixer uses the local score only as the DPP input fallback when the RPC fails (vm_ranker.rs:107-114).

### 1.4 Where the pieces run

- `PhoenixScorer` (home-mixer/scorers/phoenix_scorer.rs:84-157) calls the Phoenix ranker (cluster "Experiment1Fou", param.rs:55-60) and fills `phoenix_scores` per candidate.
- `VMRanker` scorer (home-mixer/scorers/vm_ranker.rs:84-141): computes local `weighted_score` + cold-start lift, then sends everything to vm-ranker with `value_model_id = "dpp"`, `compute_value_model = true` (vm_ranker_request.rs:8, 38-47). On RPC error it serves the local scores.
- vm-ranker `rank()` (vm-ranker/scoring/mod.rs:40-94): resolves per-viewer params, computes the value model (above), then DPP (section 3). Returned `score` overwrites the candidate's score; `weighted_score` and `top_heads` are also returned (vm_ranker.rs:123-136).

---------------------------------------------------------------------------------------------------

## 2. Post-scoring adjustments

### 2.1 Author diversity decay
Factor **0.5**, floor **0.25**, enabled by default (`rust_home_mixer_author_diversity_decay/_floor/enable_author_diversity`, vm-ranker/params.rs:175-192). Formula and k-counting: scoring.rs:261-282. Applied multiplicatively after cold-start, before DPP.

### 2.2 Out-of-network discount
- Value **0.75** (OonWeightFactor, vm-ranker/params.rs:217-222); topic requests 0.5; (disabled) new-user path 0.00001.
- Applies to all OON posts and ALSO to in-network replies and retweets (flag default true, params.rs:193-198; scoring.rs:284-293).
- Separately, OON retweets are dropped outright and OON replies are dropped unless `EnablePhoenixOonReplies` (false, param.rs:67-72) — `filters/oon_retweet_reply_filter.rs:17-28`. Any reply with no hydrated ancestors is also dropped (line 27).

### 2.3 New-author / cold-start boost (`home-mixer/scorers/author_cold_start.rs`)
Enabled by EnableViewerColdStart = true (param.rs:646-651). Runs in home-mixer on the local weighted scores (vm_ranker.rs:68-70) and the decision (`cold_start_lift_to_rank`) is forwarded so vm-ranker re-applies it on the value-model scores (scoring.rs:341-358; vm_ranker_request.rs:65). Not applied on topic requests (author_cold_start.rs:384-386).

Eligibility (author_cold_start.rs:188-193, 296-308):
- original post: no `in_reply_to_tweet_id`, no `retweeted_tweet_id`;
- `author_followers_count` ≤ **50 000** (ColdStartFollowerCap, param.rs:626-631);
- post age ≤ **7200 s** (ColdStartMaxPostAgeSecs, 632-637);
- `view_count_on_home` < **200** (ColdStartImpressionThreshold, 608-613);
- current position among non-zero scores < 0.97 × (#non-zero) (LowImpressionsMaxPositionRatio **0.97**, 639-644);
- experiment-arm corpus gating (holdout viewers: every author eligible; author_cold_start.rs:228-234, 400-405).

Pick (one post per request): Thompson sampling (EnableColdStartThompsonSampling = true, param.rs:652-657): sample r ~ Beta(0.75 + x, 49.25 + (n − x)) with n = ColdStartImpressionScale(1.0) × view_count_on_home, x = min(fav_count, n) (ColdStartBetaAlpha0/Beta0, param.rs:658-669; author_cold_start.rs:243-255); take the top ColdStartTsTopK = **2** by sample, then the highest value-model score among them (670-675; 257-283). A brand-new post (n = 0) samples from Beta(0.75, 49.25) — the prior — so it competes on variance (test "prefers uncertain over peaked zero", 987-1007).

Target position: a uniform rank in [ColdStartSlotMin, ColdStartSlotMax) = [15, 16) → rank index **15**, i.e. the 16th slot (param.rs:614-625; author_cold_start.rs:216-226). Score becomes `max(own, score_at_rank_15)` (330-331; scoring.rs:355). The author-diversity/OON multipliers are applied AFTER the lift (scoring.rs:398-404), so an OON cold-start post lands slightly below slot 16 unless its own score was higher.

### 2.4 Bidirectional ("mutual follow") boost
- Hydrator: `is_mutual_follow_author = author follows viewer && viewer follows author` (candidate_hydrators/bidirectional_follow_hydrator.rs:54-66), EnableBidirectionalFollowHydration = true (param.rs:839-844), skipped for cached-posts requests (line 17).
- Effect: reply weight 5 → **20** on original posts from mutuals (replies/retweets excluded) (weights.rs:86-93; inputs.rs:20-22). Dwell boost 0.0.
- History (docs/BIDIRECTIONAL_BOOST_CHANGE.md:5-11, 478-495): A/B tested 5/10/15/20 from 2026-07-10, launched at 20 on 2026-07-13, lowered to 15 on 2026-07-24.
- Separate `author_exploration_bonus` (vm-ranker/params.rs:205-210, default 0.0) is an additive bump in un-offset score space for authors in an FS bucket (vm-ranker/scoring/value_model.rs:243-262; scoring.rs:384, 396, 408-413).

### 2.5 Verified / premium / subscriber / language / media multipliers
- **None in the value model.** Grep of scorers, selectors, vm-ranker and xai-value-model finds no verified/blue/premium/subscriber multiplier. Subscriptions only appear as `IneligibleSubscriptionFilter` (a filter in phoenix_candidate_pipeline.rs:441).
- Language: only used as a feature-switch recipient attribute (vm-ranker/ranking_config.rs:89; vm_ranker_request.rs:83) — it can select parameter overrides via config rules, but there is no language term in code.
- Media: no media multiplier. `has_video/has_photo/media_count/max_video_duration_ms/weighted_text_len/newline_count/has_url` are computed (models/content_features.rs:8-36) and sent to vm-ranker only as the user-debias payload (scorers/vm_ranker_debias_payload.rs:33-40, 59-69); user debias is off (UserDebiasEnabled false, vm-ranker/params.rs:260-265).

### 2.6 Score caps
- No upper cap. Lower side: negative nets compressed into (0, 0.001) (section 1.2).
- DPP sets non-selected candidates' score to exactly 0.0 (vm-ranker/scoring/dpp_model.rs:196-206).
- `TopKScoreSelector` treats a missing score as −∞ (selectors/top_k_score_selector.rs:32-34).

---------------------------------------------------------------------------------------------------

## 3. Final ranking, DPP, top-K selection, candidate budgets

### 3.1 Pipeline stage order (`home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs`)
1. Query hydrators (285-348): scoring/retrieval sequences, blocked/muted/followed/subscribed ids, cached posts, mutual follow, demographics, Grok topics, starter packs, installed apps, carousel CTR model, recent searches, engagement signals, impression bloom filter, IP, carousel seen, inferred gender.
2. Sources (380-390), in order: `ThunderSource` (in-network), `PopularPostsSource`, `TweetMixerSource` (off), `SimclustersSource`, `PhoenixSource`, `PhoenixTopicsSource`, `PhoenixMOESource` (0 results), `SidSource` (off), `CachedPostsSource`.
3. Hydrators (392-423): InNetwork → BidirectionalFollow → CoreData → PhoenixReplyAncestors → Quote → MediaInfo → Subscription → Gizmoduck → BlockedBy → FilteredTopics → LanguageCode → EngagementCounts → PostEngagers → SemanticId → NewUserSoftNsfwLabel.
4. Filters (429-453): DropDuplicates, ZeroFavCodivert, CoreDataHydration, NewUserSoftNsfw, Age (MAX_POST_AGE = 48 h, params/config.rs:39), SelfTweet, OONRetweetReply, OONNsfwSimclusters, RetweetDedup, IneligibleSubscription, PreviouslySeen, PreviouslySeenBackup, PreviouslyServed, ViewerMutedKeyword, AuthorSocialgraph, Video, TopicIds, NewUserMinEngagement (off), InventoryHoldout (off), FavHoldout (off), **FairShareRanking**.
5. Scorers (475-481): `PhoenixScorer` → `ImmersiveVideoScorer` (carousel only) → `VMRanker` → `PageDecodeRerankScorer` (off, param.rs:326-331) → `PageRlExplorationScorer` (off, 356-373).
6. Selector (483): `TopKScoreSelector` — sort by `score` desc, keep **TOP_K_CANDIDATES_TO_SELECT = 50** (params/config.rs:17; top_k_score_selector.rs:15-17).
7. Post-selection hydrators (485-509): VF, AdsBrandSafetyVf, TweetTypeMetrics, FollowingRepliedUsers, MutualFollowJaccard, TopicFeedbackContext, AiTrendFeedbackContext, VideoMedia, VideoEmbedding, AncestorVideo.
8. Post-selection filters (511-517): VF, AncillaryVF, AncestorVideo, DedupConversation, VideoCarousel.
9. `result_size = RESULT_SIZE = 35` (config.rs:18; pipeline 1149-1151).
10. `ForYouCandidatePipeline` (for_you_candidate_pipeline.rs:219-239) blends those 35 scored posts with ads (AdsBlenderType "multi_risk", param.rs:957-962), WhoToFollow at position 6, prompts at 0, feed survey at 12, video carousel at 6 (`BlenderSelector`, selectors/blender_selector.rs:39-106; positions config.rs:29-32, param.rs:1229-1234); FOR_YOU_MAX_RESULT_SIZE = 35 + 5 + 8 = 48 (config.rs:19-22).

### 3.2 FairShareRankingFilter (pre-scoring budget)
Caps the Phoenix-scoring pool at PhoenixScoresResultSize = **2800** (param.rs:1357-1362; EnableFairShareRanking true 1363-1368). Each served_type keeps `count × 2800 / total` (largest-remainder rounding), taking the first N of each group in source order (filters/fair_share_ranking_filter.rs:68-110). So no source is favoured; the OON/in-network mix entering scoring is proportional to what retrieval returned.

### 3.3 Per-source candidate caps
| source | cap | served_type | cite |
|---|---|---|---|
| Thunder (in-network, followed authors) | ThunderMaxResults **1200** (×quality_factor) | ForYouInNetwork | param.rs:23-28; sources/thunder_source.rs:35 |
| Popular posts | PopularPostsMaxResults **500** | ForYouPopularPosts | param.rs:1125-1130; sources/popular_posts_source.rs:36-37 |
| Phoenix retrieval | PhoenixMaxResults **1000** + PhoenixColdStartMaxResults **200** | ForYouPhoenixRetrieval / …Cold / …ZeroFav | param.rs:4-21; sources/phoenix_source.rs:129-130, 165-171 |
| Phoenix topics | PhoenixMaxResults 1000 | ForYouPhoenixRetrieval | phoenix_topics_source.rs:54 |
| SimClusters | MAX_RESULTS **800**, candidate age ≤ 48 h | ForYouSimclusters | sources/simclusters_source.rs:26, param.rs:753-758 |
| Phoenix MoE | PhoenixMOEMaxResults **0** (off) | ForYouPhoenixRetrievalMoe | param.rs:205-210 |
| TweetMixer | 800 but EnableTweetMixerSource **false** | ForYouTweetMixer | param.rs:35-46 |
| SID | 800 but EnableSidSource **false** | — | param.rs:759-770 |
`quality_factor::apply` is an external load-shedding scaler from `xai_candidate_pipeline` (not in this repo).

### 3.4 OON fraction of the feed
There is **no explicit OON quota or target fraction** anywhere in the scoring/selection code. The mix is emergent: retrieval budgets above (roughly 1200 in-network vs ~2300 OON candidates before filtering), FairShare proportional trimming to 2800, the ×0.75 OON multiplier, cold-start (one slot), and DPP. Only `in_network_only` (Ranked Following surface) forces 100% in-network by disabling OON sources (e.g. phoenix_source.rs:106, popular_posts_source.rs:29).

### 3.5 DPP re-ranker (vm-ranker)
- Enabled: DppEnabled **true**; disabled for topic requests (TopicDppEnabled false) (vm-ranker/params.rs:241-252; scoring/mod.rs:96-98). value_model_id must be "dpp" (mod.rs:47-52).
- Config: theta **0.65**, max_selected_rank **150** from feature switches (params.rs:253-259; mod.rs:55-59); top_k from CLI `--dpp-top-k` default **50**, `--embedding-dim` default **1024** (vm-ranker/args.rs:18-28; main.rs:31-48).
- Embeddings: post embeddings from the "MM" embedding store (`xai_recsys_mm_server::mm_embedding_client`, O2 preload; vm-ranker/embedding_store.rs:8-22) as f16 vectors; looked up by `retweeted_tweet_id` if a retweet else `tweet_id` (scoring/dpp_model.rs:80-84). Missing embedding → random unit vector (≈ orthogonal, so no diversity penalty) and flagged (dpp_model.rs:23-37, 43-51).
- Algorithm (vm-ranker/dpp.rs:35-215, 223-314): take the top `max_selected_rank` (150) by value-model score; `alpha = theta / (2(1-theta))` = 0.9286 at theta 0.65; quality `q_i = exp(alpha · score_i / max_score)`; kernel `L_ij = q_i q_j cos(e_i, e_j)`; greedy MAP-DPP via incremental Cholesky picks up to top_k = 50 items, stopping when the marginal conditional volume ≤ 1e-6 ("rank_exhausted"). Optional `seed_tweet_id` is pinned first and suppresses near-duplicates (dpp_model.rs:102-125; dpp.rs:135-142).
- Output: selected items keep their value-model score; everything else gets score **0.0** (dpp_model.rs:196-206). Diversity trade-off: theta→0 ⇒ alpha→0 ⇒ all q=1 ⇒ pure embedding diversity; theta→1 ⇒ q differences explode ⇒ pure relevance. Scores are not re-scaled, only filtered ("filter_only_preserves_original_scores" test, dpp.rs:434-464).
- Then home-mixer's `TopKScoreSelector` takes the top 50 by score, so in practice the DPP's ≤50 picks ARE the selected set, in value-score order.

### 3.6 Other (inactive) rerankers
- Page decode (sequence decoding of a page by the Phoenix model; PageDecodePoolN 200, 64 steps, temperature 0.0, reward head 0) — off (param.rs:326-355; scorers/page_decode_rerank_scorer.rs).
- Page RL exploration (Plackett–Luce sampling over top-64 with temperature 0.2) — off (param.rs:356-373; page_rl_exploration_scorer.rs:81-109).
- User debias (per-viewer logit recentering of 14 heads, alpha 0, clip 2.0) — off (vm-ranker/params.rs:260-301; scoring/debias.rs:17-32).

---------------------------------------------------------------------------------------------------

## 4. Popular posts source

Two offline jobs feed one retrieval source; nothing here changes a post's score, it only adds candidates.

**Popular authors** (`home-mixer/popular_authors_job.rs`, `util/popular_authors.rs`):
- Read a BigQuery snapshot `twttr-bq-timelines-prod.pulse.popular_posting_authors` (main.rs:37-41) of accounts that posted in the last 7 days, ranked by follower count (popular_authors_job.rs:217-239).
- Keep the top `ceil(active_posters_7d × TOP_POSTING_AUTHORS_FRACTION)` with fraction **0.00005 = 0.005 %** (popular_authors.rs:11, 25-43; job 259-263, log at 287). Snapshot must be < 30 h old (main.rs:46-47). Refreshed hourly (main.rs:33-34).

**Popular posts** (`home-mixer/popular_posts_job.rs`, `util/popular_posts.rs`), every **300 s** (main.rs:52-53):
- For each popular author, fetch from Thunder up to 50 posts/author, `max_replies_reposts_per_author = 0`, and additionally drop anything with `is_reply || is_retweet` → originals only (popular_posts_job.rs:22-24, 67-69, 83-86, 104).
- Fetch `view_count` per post; keep posts with age in [0, 24 h] (window_hours **24**) (job 114-131; popular_posts.rs:65).
- Quality = projected 24-hour views under an exponential-decay accrual model:
  `λ = ln2 / half_life_hours(8)`; `saturation = 1 − e^{−λ·24}`; `elapsed = 1 − e^{−λ·max(age_h, 0.5)}`; `quality = views × saturation / elapsed` (popular_posts.rs:48-53; SelectionConfig defaults 36-46: per_author 5, budget 500, half_life 8 h, floor 0.5 h, window 24 h; CLI defaults main.rs:54-59).
- Keep top **5 per author**, then global top **500** by quality (popular_posts.rs:61-86). Stored in Manhattan; home-mixer caches it and refreshes every 60 s (popular_posts.rs:7, 190-206).

**Serving** (`sources/popular_posts_source.rs`): enabled when not in-network-only, not a cached-posts request, EnablePopularPostsSource true (27-31; param.rs:1119-1124); takes the first PopularPostsMaxResults = **500**, drops already-seen ids, tags `ServedType::ForYouPopularPosts` (36-61). Candidates then flow through all normal hydrators/filters/scoring — being "popular" confers no weight.

So "popular" means: an original post, ≤ 24 h old, by an author in the top 0.005 % of recent posters by follower count, with high view velocity relative to an 8-hour half-life curve.

---------------------------------------------------------------------------------------------------

## 5. "What drives rank" for a NEW post with zero history

Assumptions: the post has just been created (views/favs ≈ 0), the viewer is a typical established account, the post is retrievable (in-network via Thunder if the viewer follows the author; otherwise via Phoenix retrieval / SimClusters / popular posts). All engagement predictions are personalized per viewer by Phoenix; the parameters below say how each prediction or attribute moves the final score.

| rank driver | parameter(s) / mechanism | direction & magnitude | cite |
|---|---|---|---|
| Predicted report | ReportWeight −234 | strongest single lever; any non-trivial P(report) makes the net negative → score < 0.001 → bottom of pool | param.rs:527; scoring.rs:239-248 |
| Predicted mute / not-interested / block | −58.8 / −53.07 / −31.2 | same, large negatives | param.rs:509-526 |
| Predicted share via copy link | 20.0 | largest positive weight | param.rs:452-457 |
| Predicted reply, from a mutual-follow author | 5.0 + 15.0 = 20.0 | original posts from mutuals get 4× reply weight | param.rs:410-416; weights.rs:86-93 |
| Predicted reply (non-mutual), quote, share via DM | 5.0 each | | param.rs:410, 446-451, 459 |
| Predicted follow-author | 4.0 | | param.rs:472-477 |
| Predicted share | 2.118 | | param.rs:445 |
| Predicted repost | 1.0 | | param.rs:423 |
| Predicted like | 0.708 | | param.rs:404-409 |
| Predicted post-click dwell seconds | 0.385 /s | continuous, uncapped | param.rs:502-507 |
| Predicted click / open link | 0.274 / 0.2 | | param.rs:436-437 |
| Predicted dwell probability / dwell seconds / photo expand / quoted click / post_unexplored (in-network only) | 0.05 / 0.0075 / 0.05 / 0.05 / 0.015 | small | param.rs:424-429, 458-465, 478-483, 496-501 |
| Predicted not-dwelled | −0.02 | small | param.rs:528-533 |
| Video open, VQV, profile click, post_video, grok_analyze, video continuation, profile visit secs | 0.0 | no effect today | param.rs:430-444, 484-495; value_model.rs:39-41 |
| In-network vs out-of-network | OonWeightFactor 0.75 | OON post (viewer does not follow author) scores ×0.75; in-network reply/retweet also ×0.75 | vm-ranker/params.rs:217-222; scoring.rs:284-293 |
| Being a reply or retweet | OONRetweetReplyFilter; bidirectional/cold-start ineligibility; ×0.75 | OON replies/retweets removed entirely; in-network ones lose the mutual boost and cold-start and get ×0.75 | oon_retweet_reply_filter.rs:17-28; inputs.rs:20-22; author_cold_start.rs:188-193 |
| Another post by the same author ranked above it | AuthorDiversityDecay 0.5, floor 0.25 | ×0.625 for the 2nd, ×0.4375 for the 3rd, … | vm-ranker/params.rs:181-192; scoring.rs:261-282 |
| Cold-start slot (zero-history posts are exactly the target) | follower cap 50k, < 200 home impressions, ≤ 2 h old, original, EnableViewerColdStart | one such post per request is lifted to ≥ the 16th-ranked score (then multiplied by diversity/OON); Thompson prior Beta(0.75, 49.25) decides which | param.rs:608-669; author_cold_start.rs:216-331 |
| Post age | MAX_POST_AGE 48 h; cold-start ≤ 2 h; popular-posts window 24 h | hard eligibility, not a decay term — no freshness decay in the value model | config.rs:39; param.rs:632-637; popular_posts.rs:65 |
| Embedding similarity to higher-scored posts in the same request | DPP theta 0.65, top_k 50, pool 150 | near-duplicate content of a better-scoring post is dropped to score 0 | vm-ranker/params.rs:253-259; dpp.rs:92-136 |
| Author in the top 0.005 % of recent posters by followers | PopularPostsSource | extra retrieval channel (500 slots) for the author's originals with high view velocity; no score change | popular_authors.rs:11; popular_posts.rs:48-53 |
| Viewer attributes | TopicOonWeightFactor 0.5; vqv zeroed for viewers with ≥10k followers; FS recipient rules (country, language, client, account age) | can change effective params per viewer | vm-ranker/params.rs:211-216; candidates_util.rs:4; ranking_config.rs:70-103 |

Net: for a zero-history post, rank is almost entirely the Phoenix predictions weighted as above; the deterministic adjustments that matter most are (1) follow relationship (OON ×0.75; mutual follow → reply weight 20), (2) author repetition decay, (3) the single cold-start slot at position 16, (4) DPP content de-duplication, (5) reply/retweet status.

---------------------------------------------------------------------------------------------------

## 6. Parameters that reference media type, text length, links, hashtags, replies, quotes

| topic | parameter / code | role | cite |
|---|---|---|---|
| Video | MinVideoDurationMs 10_000 | VQV eligibility threshold (vqv weight is 0 anyway) | param.rs:707-712; candidates_util.rs:19-34 |
| Video | VideoOpenWeight 0.0, VqvWeight 0.0, PostVideoWeight 0.0, QuotedVqvWeight 0.0, VideoContinuationWeight 0.0 | all zero → no video term in the value model | param.rs:430-435, 444, 466-471, 484-489; vm-ranker/params.rs:66-77 |
| Video | VideoFilter | drops posts with video when `query.exclude_videos` | filters/video_filter.rs |
| Video carousel | VideoCarouselVideoOpenWeight 1.0, ImmersiveVqv 0.0, OpenTimesImmersiveDwell 0.0, VideoContinuation 0.0; EnableVideoCarousel false; position 6; extra candidates 30; embedding dedupe threshold 0.72 | separate carousel scoring, off by default | param.rs:1217-1348; top_k_score_selector.rs:18-25 |
| Photo / image | PhotoExpandWeight 0.05 | P(photo expand) | param.rs:424-429 |
| Media / text / links (features only) | has_video, max_video_duration_ms, has_photo, media_count, weighted_text_len (twitter-text weighting, URLs count 23), newline_count, has_url (external URLs excluding the media t.co) | sent to vm-ranker as user-debias ContentFeatures; debias disabled → no ranking effect today | models/content_features.rs:4-36, 44-84; vm_ranker_debias_payload.rs:33-40, 59-69 |
| Links | OpenLinkWeight 0.2 | P(open link) | param.rs:437 |
| Hashtags | — | no parameter or code path references hashtags in scoring | (grep) |
| Replies | ReplyWeight 5.0; BidirectionalFollowReplyWeightBoost 15.0 | P(viewer replies) | param.rs:410-416 |
| Replies (candidate is a reply) | OONRetweetReplyFilter; EnablePhoenixOonReplies false; PhoenixOonRepliesRootOnly false; oon_rescore_in_network_replies_retweets true; bidirectional/cold-start ineligible; DedupConversationFilter | OON replies removed; in-network replies ×0.75 and no boosts | param.rs:67-78; oon_retweet_reply_filter.rs; scoring.rs:284-293 |
| Retweets | RetweetWeight 1.0 (P(viewer reposts)); candidate retweets: OON dropped, in-network ×0.75, RetweetDeduplicationFilter; DPP embeds the original tweet | | param.rs:423; dpp_model.rs:80-84 |
| Quotes | QuoteWeight 5.0 (P(viewer quotes)); QuotedClickWeight 0.05; QuotedVqvWeight 0.0; EnableQuotedVqvDurationCheck false; quoted content features | | param.rs:459-471, 714-719; content_features.rs:22-36 |
| Text length | weighted_text_len only (feature, debias payload) | no weight | content_features.rs:44-80 |
| Engagement counts | fav_count, view_count_on_home (cold-start Thompson); view_count/author_followers_count (pacing payload, VMRankerSendPacingInputs false) | not in the weighted sum | author_cold_start.rs:243-255; vm_ranker_debias_payload.rs:42-47 |
| New-user min engagement | EnableNewUserMinEngagementFilter false (metric "fav", threshold 0.0, account age < 1800 s) | would drop OON posts under an engagement floor for brand-new viewers | param.rs:851-880; filters/new_user_min_engagement_filter.rs |

---------------------------------------------------------------------------------------------------

## Appendix A — complete numeric parameter inventory from `home-mixer/params/param.rs` (1368 lines)

Format: Name = default (line) — meaning.

Retrieval / sources
- PhoenixMaxResults = 1000 (4-9) — Phoenix retrieval cap.
- EnablePhoenixSource = true (10-15).
- PhoenixColdStartMaxResults = 200 (16-21) — extra "HomeCold" dataset results.
- ThunderMaxResults = 1200 (23-28) — in-network cap.
- ThunderAlgorithm = "default" (29-34).
- TweetMixerMaxResults = 800 (35-40); EnableTweetMixerSource = false (41-46).
- EnableSimclustersSource = true (48-53).
- PhoenixInferenceClusterId = "Experiment1Fou" (55-60) — ranker cluster; PhoenixRetrievalInferenceClusterId = "Experiment1Fou" (61-66).
- EnablePhoenixOonReplies = false (67-72); PhoenixOonRepliesRootOnly = false (73-78).
- ShadowTrafficPhoenixClustersList = [] (79-84); ShadowTrafficPhoenixRetrievalClustersList = [] (85-90); ShadowTrafficPhoenixClusterRates = Experiment18/19/22/23Fou at 0.3 (91-101); ShadowTrafficPhoenixClusterAggregationTypes = Experiment16Fou:DENSE_WITH_LONG_DWELL_HOME_EXCURSION (102-107); ShadowTrafficDefaultPercent = 0.3 (108-113).
- EnablePhoenixRetrievalStatsExperimentBucket = false (114-119); EnablePhoenixScoreStatsExperimentBucket = false (120-125).
- RetrievalCandidatesKafkaSamplePercent = 5.0 (126-131); RetrievalCandidatesKafkaMaxCandidates = 100000 (132-137).
- EnableResponseDiversityStatsExperimentBucket = false (138-143).
- PhoenixRetrievalTopicInferenceClusterId = "Experiment2Fou" (144-149); TopicFilteringId = "PostBased50Pct" (150-155); TopicFilteringOverrides = "" (156-161).
- UserActionsClusterId = "Fou" (162-167); PhoenixAggregationType = "DENSE_WITH_LONG_DWELL_HOME_EXCURSION" (168-173); PhoenixRetrievalAggregationType = "DENSE_WITH_LONG_DWELL" (174-179).
- EnablePhoenixMOESource = false (181-186); PhoenixMoeCodivertViewerIsControl/IsTreatment = false (187-198); PhoenixRetrievalMOEInferenceClusterId = "Experiment2Memy04" (199-204); PhoenixMOEMaxResults = 0 (205-210); PhoenixMoeColdStartMaxResults = 200 (211-216); AuthorPhoenixMoeEnabled = false (217-222); AuthorIsControl/IsTreatment = false (223-234); ViewerZeroFavCodivertIsTreatment = false (235-240); AuthorZeroFavCodivertIsTreatment/IsControl = false (241-252).
- PhoenixRetrievalNewUserInferenceClusterId = "Experiment1Lap7" (254-259); PhoenixRetrievalNewUserHistoryThreshold = 0 (260-265); PhoenixXdsRetrievalMaxRetries = 0 (266-271); PhoenixRetrievalExcludeSeenPosts = false (273-278); RetrievalValueModelWeights = "" (280-285, JSON of head→weight forwarded to Phoenix retrieval).
- PhoenixRankerNewUserInferenceClusterId = "Experiment1Fou" (287-292); PhoenixRankerNewUserHistoryThreshold = 0 (293-298).

Ranking switches
- EnableRanking = true (300); LogSlateContext = false (301-306); RerankerHeadTag = 0 (307); EnableHomeExcursionScores = true (308-313); EnablePageLogging = true (314-319); PageLogTtlSeconds = 3600 (320-325).
- EnablePageDecodeRerank = false (326-331); PageDecodePoolN = 200 (332-337); PageDecodeSteps = 64 (338-343); PageDecodeTemperature = 0.0 (344-349); PageDecodeRewardHead = 0 (350-355).
- EnablePageRlExploration = false (356-361); PageRlExplorationTemperature = 0.2 (362-367); PageRlExplorationTopK = 64 (368-373).

Value-model weights (375-533) — see section 1 table. Verbatim: FavoriteWeight 0.708; ReplyWeight 5.0; BidirectionalFollowReplyWeightBoost 15.0; BidirectionalFollowDwellWeightBoost 0.0; RetweetWeight 1.0; PhotoExpandWeight 0.05; VideoOpenWeight 0.0; ClickWeight 0.274; OpenLinkWeight 0.2; ProfileClickWeight 0.0; VqvWeight 0.0; ShareWeight 2.118; ShareViaDmWeight 5.0; ShareViaCopyLinkWeight 20.0; DwellWeight 0.05; QuoteWeight 5.0; QuotedClickWeight 0.05; QuotedVqvWeight 0.0; FollowAuthorWeight 4.0; PostUnexploredWeight 0.015; PostVideoWeight 0.0; GrokAnalyzeWeight 0.0; ContDwellTimeWeight 0.0075; ContClickDwellTimeWeight 0.385; NotInterestedWeight −53.07; BlockAuthorWeight −31.2; MuteAuthorWeight −58.8; ReportWeight −234.0; NotDwelledWeight −0.02.

Facepile / VM ranker transport
- EnableFollowingRepliedUsersFacepile = false (535-540); FollowingRepliedUsersFacepileMinUsers = 0 (541-546); FollowingRepliedUsersFacepileMaxPosts = 0 (547-552).
- VMRankerEnableFallback = false (554-559); VMRankerClusterId = "Experiment1" (560-565); PhoenixExperimentOverrides = "" (566-571); VMRankerSendDebiasInputs = true (572-577); VMRankerSendPacingInputs = false (578-583); VMRankerSendHistoryTopics = false (584-589); HistoryTopicsEnabled = false (590-595); HistoryTopicsWindowDays = 28 (596-601); EnableSerendipityStatsExperimentBucket = false (602-607).

Cold start
- ColdStartImpressionThreshold = 200 (608-613); ColdStartSlotMin = 15 (614-619); ColdStartSlotMax = 16 (620-625); ColdStartFollowerCap = 50000 (626-631); ColdStartMaxPostAgeSecs = 7200 (632-637); LowImpressionsMaxPositionRatio = 0.97 (639-644); EnableViewerColdStart = true (646-651); EnableColdStartThompsonSampling = true (652-657); ColdStartBetaAlpha0 = 0.75 (658-663); ColdStartBetaBeta0 = 49.25 (664-669); ColdStartTsTopK = 2 (670-675); ColdStartImpressionScale = 1.0 (676-681); ColdStartTrackedIds = "" (682-687).

Caching / features / hydration
- EnableCachedPosts = true (689-694); EnableUnderTheHood = false (695-700); MaxPostsToCache = 750 (701-706); MinVideoDurationMs = 10000 (707-712); EnableQuotedVqvDurationCheck = false (714-719); EnableContextFeatures = true (721-726); EnableIpFeature = true (728-733); EnableExplicitEngagementSignals = true (735-740); EnableImplicitEngagementSignals = true (741-746); EngagementSignalsMaxPerType = 15 (747-752); SimclustersMaxCandidateAgeHours = 48 (753-758); EnableSidSource = false (759-764); SidSourceMaxResults = 800 (765-770); SidSourceMaxSeeds = 50 (771-776); SidSourceMaxPerSeed = 100 (777-782); SidSourceMinPrefixDepth = 3 (783-788).
- EnableUserInstalledAppsHydration = true (790-795); EnableRecentSearchQueriesHydration = false (797-802); EnableUserInstalledAppsV2 = false (804-809); EnableInferredGenderHydration = true (811-816); EnableRetrievalSequenceHydration = true (818-823); EnableScoringSequenceHydration = true (825-830); EnableMutualFollowJaccardHydration = false (832-837); EnableBidirectionalFollowHydration = true (839-844); EnableAllAuthorFollowHydration = true (845-850).
- EnableNewUserMinEngagementFilter = false (851-856); NewUserMinEngagementFilterMetric = "fav" (857-862); NewUserMinEngagementFilterUseRatio = false (863-868); NewUserMinEngagementThreshold = 0.0 (869-874); NewUserMinEngagementFilterMaxAccountAgeSecs = 1800 (875-880); EnableNewUserSoftNsfwFilter = false (881-886); NewUserSoftNsfwFilterMaxAccountAgeSecs = 604800 (887-892); NewUserSoftNsfwFilterMaxScoringSeqLength = 100 (893-898); NewUserMinEngagementFilterMaxResurrectionAgeSecs = 1800 (899-904).
- EnableEngagementCountsHydration = true (905-910); EnablePostEngagers = false (911-916); EnableSemanticIdHydration = true (917-922); SidServerEndpoint = xds://xai-recsys-sid-v8-root.prod.recsys-infra:grpc (923-928).

Ads
- EnableAdsSource = true (929-934); EnableAdsInjectionLogging = true (936-941); EnableAdsHistoryXdcCacheWrite = true (943-948); EnableAdsFunnelTrace = true (950-955); AdsBlenderType = "multi_risk" (957-962); EnableAdsBrandSafetyVerdictV2 = false (964-969); EnableAdsAuthorBrandSafetyFallback = true (971-976); AdsTimeGapTSec = 4.0 (977-982); AdsTimeGapClampLo = 0.5 (983-988); AdsTimeGapClampHi = 2.0 (989-994); AdsTimeGapMinOrganicGap = 3 (995-1000).

Sequences / holdouts / misc
- UseXdsForUas = true (1001); UasSourceDataType = "NEW_UUA_ARROW" (1002-1007); MaxSeqLengthScoring = 1024 (1008-1013); MaxSeqLengthRetrieval = 1024 (1014-1019).
- EnableAdAdjacentServedFilter = false (1021-1026); EnableServedFilterAllRequests = true (1027-1032); EnableInventoryHoldout = false (1034-1039); InventoryHoldoutOriginals/Replies/RetweetsPercent = 0 (1040-1057); EnableFavHoldout = false (1059-1064); EnableEntryIdRandomHash = false (1066-1071); EnableTopicFeedbackContext = false (1073-1078); EnableAiTrendFeedbackContext = false (1080-1085); EnablePhoenixRequestCacheSideEffect = true (1087-1092); PhoenixRequestCacheSideEffectTtlSeconds = 10800 (1093-1098); EnableUrtMigrationComponents = true (1100-1105); EnableAuthorServedMetricsExperimentBucket = false (1107-1112); AuthorServedMetricsAuthorIds = [] (1113-1118).
- EnablePopularPostsSource = true (1119-1124); PopularPostsMaxResults = 500 (1125-1130); EnablePublishSeenIdsToKafka = true (1132-1137); EnableWhoToFollowModule = true (1139-1144); WhoToFollowFatigueHours = 30 (1145-1150); EnablePrompts = true (1152); ExcludeServedTweetIdsDuration = 10 min (1154-1159); ExcludeServedTweetIdsNumber = 100 (1160-1165).
- Jetfuel frames all false (1167-1202); EnableFeedSurvey = false (1204-1209); FeedSurveyFatigueMinutes = 1440 (1210-1215).
- Video carousel: EnableVideoCarousel = false (1217-1222); VideoCarouselFatigueMinutes = 60 (1223-1228); VideoCarouselPosition = 6 (1229-1234); EnableVideoCarouselAuthorDedupe = true (1235-1240); VideoCarouselSemanticIdDedupeLevels = 3 (1241-1246); EnableVideoCarouselEmbeddingDedupe = true (1247-1252); VideoCarouselEmbeddingDedupeThreshold = 0.72 (1253-1258); EnableVideoCarouselCtrModel = false (1259-1264); VideoCarouselCtrModelName = "" (1265-1270); VideoCarouselEmbeddingVersion = "v8_search_1" (1271-1276); EnableVideoCarouselHighRiskFilter = true (1277-1282); EnableVideoCarouselMediumRiskFilter = false (1283-1288); EnableVideoCarouselSeenFilter = false (1289-1294); EnableVideoCarouselCandidatesLogging = true (1295-1300); VideoCarouselVideoOpenWeight = 1.0 (1301-1306); VideoCarouselImmersiveVqvWeight = 0.0 (1307-1312); VideoCarouselOpenTimesImmersiveDwellWeight = 0.0 (1313-1318); VideoCarouselWeightedScoreWeight = 0.0 (1319-1324); VideoCarouselVideoContinuationWeight = 0.0 (1325-1330); VideoCarouselImmersiveClusterId = "Experiment3Fou" (1331-1336); VideoCarouselImmersiveAggregationType = "IMMERSIVE_LONG_DWELL" (1337-1342); VideoCarouselExtraCandidates = 30 (1343-1348).
- EnableXaiVfClient = true (1350-1355); PhoenixScoresResultSize = 2800 (1357-1362); EnableFairShareRanking = true (1363-1368).

Compile-time constants (`home-mixer/params/config.rs`): TOP_K_CANDIDATES_TO_SELECT 50 (17); RESULT_SIZE 35 (18); FEED_MODULE_SLOTS 5 (19); MAX_JETFUEL_FRAMES_PER_RESPONSE 8 (20); FOR_YOU_MAX_RESULT_SIZE 48 (21-22); RANKED_FOLLOWING_MAX_RESULT_SIZE 38 (23); FOLLOWING_MAX_RESULT_SIZE 110 (24); WHO_TO_FOLLOW_POSITION 6 (29); PROMPTS_POSITION 0 (31); FEED_SURVEY_POSITION 12 (32); VIDEO_CAROUSEL_MAX/MIN_VIDEOS 5/3 (34-35); MAX_POST_AGE 48 h (39); NEW_USER_MIN_FOLLOWING 5 (41); NEGATIVE_SCORES_OFFSET 0.001 (42).

## Appendix B — worked example (home-mixer weights, non-mutual OON original post)
P(fav)=0.10, P(reply)=0.01, P(retweet)=0.02, P(share_copy_link)=0.002, P(report)=0.0005, click_dwell=1.0 s, everything else 0:
weighted = 0.708·0.10 + 5·0.01 + 1·0.02 + 20·0.002 + 0.385·1.0 − 234·0.0005 = 0.0708+0.05+0.02+0.04+0.385−0.117 = 0.4488
offset   = 0.4498; OON ×0.75 → 0.3374 (first post from this author). If the same author already had a higher post in the pool: ×0.625 → 0.2109.
