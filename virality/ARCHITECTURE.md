# Virality Engine: end-to-end architecture specification

Companion to `DESIGN.md` (which holds the evidence, data plan, budget and gates). This document specifies the system: what goes in, what each stage does, what comes out, and why it should work. Repo paths refer to this repository; the ported stages cite the file they are ported from.

---

## 0. Input contract

A `Submission` is one draft, or a set of draft variants, from one account.

```
Submission
  account
    handle                      string
    followers, following        int          (snapshot)
    account_age_days            int
    verified                    bool
    history[]                   optional     last N posts with text, media, outcomes
                                             (from a connected account or analytics export)
    audience_timezone_hist      optional     distribution of follower local time zones
  post
    text                        string       (plain text; URLs allowed)
    media                       one of:
                                  images[1..4]  (jpg/png/webp)
                                  video         (mp4/mov, any length; "launch video" is the main case)
                                  gif
                                  none
    link_preview_url            optional     (a URL whose card would render)
    quoted_post                 optional     {text, media, author}
    kind                        original | quote | reply   (reply is accepted but flagged)
    planned_post_time           optional     ISO time; if absent the system recommends one
  options
    variants[]                  optional     alternative texts / media to compare
    rewrite                     bool         ask the optimizer for rewrites
```

Audio is not a separate input. A launch video's audio track is processed as part of the video (transcription), which matches how X's content pipeline treats it.

---

## 1. Stage map

```
 Submission
     │
 S0  Normalize ─────────────── URL expansion, card fetch, language, text metrics,
     │                          media probing, frame sampling, transcription
     ▼
 S1  Content representation ── X's encoder + our codebook (ported),
     │                          SigLIP-2 image/frame embeddings, scalar features
     ▼
 S2  Pre-flight ────────────── ported visibility rules + open safety classifiers
     │                          → ALLOW / INTERSTITIAL / DROP_OON / DROP_ALL
     ▼
 S3  Eligibility timeline ──── ported retrieval, cold-pool, cold-start, popular,
     │                          age gates → where and when the post can be served
     ▼
 S4  Phoenix surrogate ─────── THE TRAINED MODEL: Phoenix's candidate features in,
     │                          Phoenix's action heads out, for in-network and OON
     ▼
 S5  Value model + placement ─ ported weighted sum, offset, OON ×0.75, author
     │                          diversity, cold-start lift, DPP; rank vs real pool
     ▼
 S6  Momentum loop + reach ─── counts feed back into S4 at later ages (X's own
     │                          feedback path); calibrated views distribution
     ▼
 S7  Optimizer ─────────────── solver over deterministic levers, best-of-N
     │                          rewrites, video hook/thumbnail/trim analysis
     ▼
 Report (section 9)
```

Trained components: S4 (the surrogate and its phase-2 teacher), the open safety classifiers in S2, the calibrators in S6, and the generator used in S7. Everything else is ported code.

---

## 2. S0: Normalize

- **Text.** Expand t.co and short links, follow redirects to the final domain. Compute `weighted_text_len` with X's weighting (URL counts 23), `newline_count`, `has_url`, external URL count, mention count, hashtag count, emoji count, question/exclamation marks, language via a fastText-class detector. Strip t.co from the rendered text, as X's encoder does.
- **Card.** If a link would render a card, fetch title, description and card image. The card image is embedded like a post image and is checked by the sexual-card classifier in S2.
- **Images.** Decode, record resolution and aspect. Produce a 448 px version for the VLM encoder and a 384 px version for SigLIP-2.
- **Video.** Probe with ffmpeg: duration, resolution class (SD / HD / Full HD), aspect, fps, audio present. Sample three frame sets: (a) 4 uniformly spaced frames for the X-faithful embedding view (X's post-embedding plan uses up to 4 frames per video), (b) up to 16 frames at about 1 fps capped, plus first and last, for SigLIP-2 temporal features, (c) the first frame alone, because it is the thumbnail the feed shows. Record whether all media are video and longer than 10 s (X's video-index and video-quality-view gate).
- **Audio.** If a video has speech, transcribe with Whisper large-v3-turbo and keep up to 2,000 characters (X's `MAX_ASR`). The transcript is appended to the rendered document as `Transcript:` exactly as X's plan does. Speech ratio and whether the first 3 s contain speech are kept as scalars.
- **Account.** If history is present, compute trailing statistics over the last 50 posts: median and p90 of likes/views, replies/views, reposts/views, quotes/views, bookmarks/views where available; posting cadence; media ratio; mean content embedding (the author centroid). Otherwise mark `history_unknown`.

---

## 3. S1: Content representation

| Output | How | Dim |
|---|---|---|
| `e_mm` | Render as `phoenix/reference/mm_encoder.py` does: author line, text, image pads in position, quoted post, video frames, transcript; ChatML-wrap with X's instruction; encode with `Qwen/Qwen3-VL-Embedding-8B`; truncate 4096 → 1024; L2-normalize | 1024 |
| `sid` | Assign against our RQ-KMeans codebook trained with `phoenix/reference/sid_codebook.py` on the corpus | 6 codes in [1, 256] |
| `e_img[i]` | SigLIP-2 So400m/14 @384 per image and per card image | 1152 each, up to 4 |
| `e_frame[t]` | SigLIP-2 per sampled frame, with timestamp | 1152 each, up to 18 |
| `e_thumb` | SigLIP-2 of the first frame | 1152 |
| `scalars` | The S0 text, media and video scalars, plus `kind` one-hot and language | ~40 |

`e_mm` and `sid` are what Phoenix sees. The rest is what we add because we lack Phoenix's 100M-row post-ID memory.

---

## 4. S2: Pre-flight (ported visibility filtering)

Rule engine ported from `visibility-filtering/rules/levels.rs`, `tweet_rules.rs`, `author_rules.rs`, evaluated in X's order for two surfaces: `TimelineHome` (followers) and `TimelineHomeRecommendations` (non-followers). Labels that X's upstream systems would attach are estimated by:

| Label family | Estimator | Threshold source |
|---|---|---|
| NSFW high precision / near perfect / high recall | Open adult-content classifier on images, frames and card image | image precision ≥ 0.95, near-perfect ≥ 0.999, video recall bar lower (report 05) |
| Gore and violence | Open violence classifier | image precision > 0.6, video raw ≥ 0.95 |
| Spam high recall: engagement bait, farming, card manipulation, scam links, hashtag abuse | A small text classifier distilled from an LLM labeler over the policy sub-types named in `grox/` | fires regardless of author credibility, as in X's sink |
| Malicious / low-quality URL | Redirect-chain expansion, public blocklists, a learned domain-quality score | any hop bad → drop everywhere |
| Sexual card image | Adult classifier on the card image | near-perfect |
| Duplicate / copypasta | Near-duplicate match against the author's history and a global index | cluster rule |
| Gibberish / slop | Perplexity and a slop classifier | post label 30 d |
| Author-level NSFW | 3 flagged of last 5 posts | 7 d account label |

Output: `verdict ∈ {ALLOW, INTERSTITIAL, DROP_OON, DROP_ALL}`, the rule that fired with its repo citation, and a concrete fix ("move this link to a reply", "this phrasing matches engagement-bait patterns").

---

## 5. S3: Eligibility timeline (ported distribution gates)

A deterministic function of post kind, media, author size and the S2 verdict:

| Gate | Ported from | What it yields |
|---|---|---|
| In-network availability | `thunder/` | Immediate to followers; per-author caps (50 originals per 48 h); replies shown only to the root or followed users |
| Out-of-network indexing | `phoenix-rankall-strato/.../phoenixRankAllCandidateProcessor.strato` | `post_creation` index at +2 min; `1fav` index after first like; `32fav` after 32 likes; video index only if all media video > 10 s; replies only after a like and off by default; reposts never |
| Cold pool | `phoenix/xrex/data/cold_pool_filter.py`, two-tower config | Under 8 likes and under 500 views and under 2 h: own 200-slot retrieval budget |
| Cold-start slot | `home-mixer/scorers/author_cold_start.rs`, `param.rs` | Eligible if original, author ≤ 50k followers, age ≤ 2 h, home impressions < 200; Thompson prior Beta(0.75, 49.25); lifted to position 16 when selected |
| Popular posts | `home-mixer/util/popular_authors.rs`, `popular_posts.rs` | Only authors in the top 0.005% of 7-day posters by followers; projected-24h-views formula with 8 h half-life |
| SimClusters | `simclusters/` thresholds | Needs a like from a user with interest clusters; served when cosine > 0.5; ≤ 48 h |
| Exploration target | `phoenix/xrex/data/recsys/recsys_batch.py:105-135` | Views needed by age t to stop being "underexposed": 3% of followers on the 8 h half-life curve |
| Expiry | `home-mixer/params/config.rs` | 48 h |

Output: a timeline of windows and the list of gates with pass / fail / conditional and the condition.

---

## 6. S4: the Phoenix surrogate (trained)

### 6.1 Role

Phoenix answers: for this viewer and this post, what is the probability of each action. We cannot condition on one viewer's 1022-event history, so the surrogate answers the audience-averaged version: for this audience and this post, what fraction will take each action. That is exactly what per-impression rates measure, so the target is observable.

The surrogate is evaluated in **two contexts**, because X scores a post differently for followers and non-followers: `ctx = IN` (follow bit on, no OON multiplier) and `ctx = OON` (follow bit off, OON multiplier downstream). It is also evaluated at several **post ages and count states**, because Phoenix's inputs include age bucket and engagement counts and the product must forecast a draft with zero counts and then the same post with momentum.

### 6.2 Inputs, as tokens

Candidate token, built with Phoenix's own feature prep (`phoenix/xrex/models/recsys_feature_prep.py`), one summed vector of width D = 1024:

| Feature | Source in Phoenix | Draft-time value |
|---|---|---|
| Semantic ID, 6 levels + hash level | shared SID table → projection | from S1 |
| Post age bucket (hourly, 82 classes) | `cand_post_age_emb` | evaluated at 0.25 h, 1 h, 2 h, 6 h, 24 h |
| Local hour (25) | `cand_local_hour_emb` | over the audience time-zone histogram; default US-centric prior for tech accounts |
| Follow bits (viewer follows author, author follows viewer) | two 2-row tables | IN: (1, m) with m the mutual share estimate; OON: (0, 0) |
| log2 counts of like, reply, repost, quote, view through MLP | `cand_count_lograw_mlp` | zeros at draft time; set by S6 at later ages |
| Stale flag | `[2, D]` | 0 |
| Product surface | `[16, D]` | home ranking |

Extension tokens (not in Phoenix; each projected to D with a learned type embedding):

| Token | Content |
|---|---|
| `MM` | `e_mm` projected |
| `IMG_i` | `e_img[i]` projected, up to 4 |
| `FRAME_t` | `e_frame[t]` projected plus a sinusoidal time position, up to 18 |
| `THUMB` | `e_thumb` projected |
| `SCALAR` | MLP over the ~40 scalars and the S2 classifier scores |
| `AUTHOR` | MLP over log followers, log following, ratio, account age, verified, cadence, media ratio |
| `AUTHOR_HIST` | MLP over trailing per-action rate medians and p90s, or a learned `unknown` vector |
| `AUTHOR_CENTROID` | mean `e_mm` of the author's history, projected |
| `AUTHOR_ID` | hashed author embedding table (learned; zero for unseen authors) |
| `AUDIENCE_IN` | learned embedding of the author's niche (SID prefix-2 of the author centroid) crossed with follower tier |
| `AUDIENCE_OON` | learned embedding of the post's own SID prefix-2 neighborhood, representing who X would retrieve it for |
| `TIME` | planned hour and weekday in audience-local time |
| `NEIGHBOR_j` | for the 16 nearest recent corpus posts by `e_mm`: projected embedding plus their observed outcome and recency; this carries trend saturation and "what this kind of post did last week" |

Sequence length is about 50 tokens per candidate. Multiple drafts are scored in one pass using Phoenix's candidate-isolation mask (`segment_ids`: candidates attend to shared context and themselves, never to each other), so variant scores are independent and cacheable, as in X.

### 6.3 Trunk

The `home_direct_packed` block design at reduced width: 8 pre-norm layers, D = 1024, 16 query heads, 4 key-value heads, head dim 128, non-gated FFN with widening 2, RMSNorm, qk-norm, final RMSNorm. About 110M trainable parameters in the trunk, 30 to 60M in projections, tables and heads. Frozen in front of it: Qwen3-VL-Embedding-8B, SigLIP-2 So400m, Whisper turbo.

### 6.4 Heads

Sixty-four sigmoid heads indexed as Phoenix's `ActionName` enum, so the ported value model reads them by index:

| Index | Head | Label at launch |
|---|---|---|
| 1 | favorite | likes / views |
| 4 | reply | replies / views |
| 5 | quote | quotes / views |
| 6 | repost | reposts / views |
| 36 | bookmark | bookmarks / views where present |
| 29 | click (detail expand) | analytics exports only; masked otherwise |
| 30 | profile click | analytics only |
| 39 | open link | analytics only |
| 21 | follow author | analytics only |
| 14 | photo expand | masked |
| 47 / 13 | video open / video quality view | masked; weights are 0 in current `param.rs` anyway |
| 37 / 32 / 33 | share / copy link / DM | masked; these carry weights 2.1 / 20 / 5, see 6.6 |
| 10, 16, 17, 23, 25 | report, client report, not interested, block, mute | proxy label from S2 classifier scores at launch; real labels from connected accounts later |
| 59 | post unexplored | ported formula applied to observed views |
| others | masked | |

Auxiliary heads outside Phoenix's set: `log_reach_ratio` (Huber), exceedance of 1x / 10x / 100x followers and of the author's trailing p90 (BCE), used for calibration and for the pairwise loss.

### 6.5 Losses

For each labeled head: BCE between the predicted rate and the observed rate, weighted by log(1 + views) so that a post seen by 5 people does not count like one seen by 5 million. Masked heads contribute nothing. Huber on log reach ratio. Pairwise logistic on the ported value score between posts from the same author within 7 days (the literature's confound control). In phase 2, a distillation term toward the teacher's head outputs. Rows without views (Community Archive) enter only through the pairwise term with likes per follower as the comparison.

### 6.6 The share problem, stated

The three largest positive weights (copy link 20, DM share 5, share 2.1) have no public counts. At launch they are masked, and the value score the product reports is computed over the heads we can estimate, with a disclosed "unobserved share of weight." Connected-account analytics expose shares, and those heads unmask as that data arrives. This is the single biggest known gap between the surrogate and Phoenix, and it is a data gap, not a design gap.

### 6.7 Teacher

Phase 2 trains Qwen3-VL-8B with QLoRA to read the rendered post, the images or frames and an author card, and emit the same heads. It is slower and better on content nuance. Its scores over the corpus are distilled into the surrogate; it can also rerank the top candidates in S7 when latency allows.

---

## 7. S5: value model and placement (ported)

1. **Weighted sum.** `score_ctx = Σ_h w_h · P_h(ctx)` with `w` from a pinned `home-mixer/params/param.rs`, arithmetic from `xai-value-model/scoring.rs`. The mutual-follow reply boost (+15) applies to the mutual share `m` of the in-network audience. The report compares `score_IN` and `score_OON` and shows each term's contribution.
2. **Offset.** Ported: non-negative scores get +0.001; net-negative scores are compressed into (0, 0.001).
3. **Multipliers.** OON ×0.75. Author diversity: given the account's planned posting cadence, the expected number of its higher-scoring posts already in a feed, applying 0.75·0.5^k + 0.25.
4. **Cold-start lift.** If S3 says eligible, simulate the Thompson draw against the pool's eligible candidates and lift to the score at rank 15 when selected.
5. **Placement.** A reference pool of 2,800 real recent posts from the corpus, matched on hour and sampled with X's source proportions, each with precomputed surrogate scores. Insert the draft, run the ported DPP (`vm-ranker/scoring/dpp.rs`, theta 0.65, pool 150, select 50) on `e_mm` cosine, take the top 35. Repeat over 200 sampled pools and the age grid.

Output per context and age: probability the post is shown in a feed, expected rank quantiles, and which competitor types beat it.

---

## 8. S6: momentum loop and reach

X's feedback loop is explicit in its features: engagement counts are inputs to Phoenix, so early engagement raises later scores, which raises later engagement. We do not invent a cascade model; we run X's loop:

```
t = 0.25 h, 1 h, 2 h, 6 h, 24 h
  for each t:
    views_IN(t)  ≈ p_shown_IN(t) · active_followers(t)
    views_OON(t) ≈ p_shown_OON(t) · OON_exposure(t | eligibility timeline, cold pool, cold-start)
    counts(t)    = Σ_h P_h · views(t)        (likes, replies, reposts, quotes)
    feed counts(t), age(t) back into S4 → S5 for the next step
```

`active_followers(t)` and `OON_exposure(t)` are the two quantities the corpus cannot give us directly; they are fit as monotone functions of follower tier and time from observed views, jointly with the final calibrator. The output of the loop is a simulated 24 h view count. A per-follower-tier isotonic calibrator maps (simulated views, value score, eligibility flags) to observed views, with split-conformal intervals on held-out authors.

Reported: `views_24h` as p10 / p50 / p90, probability of exceeding 1x, 10x and 100x followers, probability of exceeding the author's own trailing p90, and the "underexposed at 24 h" probability using X's own formula. When a post is live and counts are supplied at any checkpoint, the same loop restarts from observed counts, which is the live forecast.

---

## 9. S7: optimizer

**Deterministic levers (a solver, not a model).** Each lever is a discrete choice that changes only scalars, context or eligibility, so re-running S3 to S6 is cheap because embeddings are cached:

| Lever | Why it exists in the algorithm |
|---|---|
| Post as original, not quote or reply | Replies and reposts are filtered or discounted out-of-network; quotes lose the cold-start and mutual boosts |
| Link in the post vs. in the first reply vs. none | Link-open weight 0.2; external links observed to cut views up to 8x; bad-domain links drop the post |
| Native video vs. a link to a video host | A hosted link is a link; a native video is content with its own retrieval index |
| Video length ≥ 10 s and the first frame as thumbnail | Video index and quality-view gates require > 10 s; the first frame is the shown image |
| Posting hour and weekday in audience-local time | Local hour is a Phoenix input; the 48 h window and the 2 h cold window make the first hours decisive |
| Spacing from the account's previous post | Author-diversity decay 0.5 toward 0.25 within one feed |
| One post vs. a thread | Only the root is an original; each reply is a reply |
| Text length, line breaks, mention and hashtag count | Scalar inputs the surrogate has learned from |

Output: a ranked list of lever changes with the predicted change in the 10x probability and in the value score, each with its repo citation.

**Content rewrites (best-of-N with a guard).** A generator (Qwen3-32B class on an L4, or the teacher) produces N = 32 variants under constraints: preserve named entities, product claims and calls to action; stay within a length target; avoid phrasings the S2 bait classifier fires on. Each variant is re-rendered and re-embedded (S1), scored by the surrogate, and independently scored by a second model with a different inductive bias (the phase-0 LightGBM and, when available, the teacher). Variants are ranked by the surrogate's 10x probability, discounted by disagreement with the second model, and must keep cosine ≥ 0.85 to the original `e_mm`. The top three are returned with the heads that moved and why.

**Launch-video analysis.** Gradient-times-input attribution over `FRAME_t` and `THUMB` tokens shows which seconds drive the score; the report gives a hook score for the first 3 s, the best thumbnail frame among the first few seconds, a trim suggestion, and caption text drawn from the transcript that scores best as post text.

**Later.** A posting-time bandit per account once it posts through the product, and an RL-trained editor (DPO on observed same-author pairs, then GRPO against the surrogate with a KL penalty to the base model and the agreement filter as a hard constraint).

---

## 10. Output contract

```
Report
  preflight        {verdict, rules_fired[{rule, repo_ref, reason}], fixes[]}
  eligibility      {in_network: "immediate", oon_entry: "after first like (~2 min)",
                    cold_pool_window_h, cold_start_slot: {eligible, reason},
                    popular_posts_eligible, video_index_eligible, expires_h: 48}
  heads            {IN: {favorite, reply, repost, quote, bookmark, ...},
                    OON: {...}, masked: [...], age_grid: [...]}
  value_score      {IN, OON, breakdown[{action, p, weight, contribution}],
                    negative_risk, unobserved_weight_share}
  placement        {p_shown_IN(t), p_shown_OON(t), rank_quantiles(t)}
  reach            {views_24h: {p10, p50, p90}, p_exceed_1x, p_exceed_10x,
                    p_exceed_100x, p_beat_own_p90, p_underexposed_24h,
                    interval_method, calibration_n}
  levers[]         {lever, change, delta_p_10x, delta_score, repo_ref, rationale}
  rewrites[]       {text, p_exceed_10x, delta_vs_original, agreement, heads_moved}
  video            {hook_score_3s, thumbnail_frame_t, trim_suggestion, segment_attribution[]}
  live             {present only when counts were supplied: updated reach, next checkpoint}
  caveats[]
```

The "probability distribution" you asked for is `reach`: a full distribution over 24 h views with tier probabilities, not a single flag. The "how to make it go viral" is `preflight.fixes`, `levers`, `rewrites` and `video`, each with a predicted effect and a pointer into X's code.

---

## 11. Why this will work

1. **The error surface is small and measurable.** Everything after prediction is X's code. The system can be wrong only in the surrogate, the safety classifiers and the reach calibrator, and each of those has a held-out metric and a literature anchor. There is no hand-built "virality heuristic" anywhere in the chain.
2. **The target is the quantity X optimizes, in the units X uses.** Per-impression action rates are the audience average of Phoenix's outputs; the weighted sum is X's; the reach normalization is X's exploration formula. The product does not predict a proxy of a proxy.
3. **The content representation is X's.** The encoder, prompt, rendering and codebook recipe come from this repo. Whatever Phoenix can and cannot distinguish about a post, the surrogate sees the same distinctions, and SigLIP-2 fills in the visual detail Phoenix compensates for with memory.
4. **The product asks questions the data can answer.** Calibrated odds, the ranking of drafts for the same account, the deterministic levers, and the live update are all within the literature's demonstrated range (pairwise 65 to 77%, author plus content AUROC 0.72 to 0.82, 0.9 plus with early counts). The one question nobody can answer, "will this specific post be the top 0.1%", is reported as a probability with an interval rather than promised.
5. **Early dynamics come from X's own feedback path.** Counts are Phoenix inputs; the momentum loop just runs them forward. The cold pool, the cold-start slot and the 48 h expiry are the actual mechanics, not assumptions.
6. **Capacity is where the data supports it.** About 9B frozen parameters that already understand images, video and language; a trainable part of 150 to 300M sized to a corpus in the high hundreds of thousands; a teacher that adds nuance and is distilled rather than served. Fine-tuned encoders beat zero-shot frontier models on this task in every 2024 to 2026 study; from-scratch giants overfit it.
7. **Optimization is guarded.** Best-of-N with an independent second scorer, a meaning constraint, and later a KL-bounded editor. This is the configuration the reward-hacking literature says degrades slowest, and it is the opposite of a single model grading its own rewrites.
8. **It is gated.** Phase 0 is free and returns the corpus size and the first pairwise and AUROC numbers in two weeks. If the pairwise number is under 60% on a clean author-and-time split, we stop and fix data. Every later phase has its own threshold before it spends credits.
9. **It tracks X.** The weights, filters and constants are read from pinned repo files and refreshed from upstream. When X moves a weight, the product moves the same day with no retraining. When X adds a feature the surrogate can learn, it is one more token.
10. **It is cheap to run.** Under a cent per thousand predictions for the surrogate; the expensive encoders run once per draft; the teacher is optional. A marketing team can score every draft and every variant.

Where it will be weakest, so nobody is surprised: share-type heads until connected-account data arrives; negative-feedback heads until the same; the extreme tail; and calibration for accounts far outside the tech niche the corpus covers.

---

## 12. Simulating an audience, and the viewer-level track

**Question considered:** instead of training a predictor, show the post to 10,000 Phoenix agents and simulate.

Phoenix is per viewer: its input is one viewer's last 1,022 engagements and profile, its output is that viewer's action probabilities. Running it across many viewers is exactly how X works. It is not available to us for two reasons: the repository ships no trained weights, and no public dataset contains the viewer histories (what each person saw and did) that a viewer-level model needs. An untrained Phoenix shown to 10,000 synthetic agents returns noise; the shipped synthetic world draws outcomes from random affinities.

What the design does instead, in two parts:

1. **Audience segments (in S4, uses aggregate data).** The surrogate predicts per-action probabilities for K latent audience segments (for example ML researchers, founders, investors, general tech), and an account's audience is a mixture over them. Training constrains the mixture average to reproduce observed per-post rates, which is all public data can identify. At inference the post is "shown" to N viewers sampled from the account's segment mixture plus the out-of-network audience its semantic-ID neighbourhood implies, and the ported value model and placement (S5) run per sampled viewer. Output: the reach distribution plus which segments engage and which do not.
2. **A real viewer-level model (research track, uses donor data).** The Community Archive holds 474 donor archives with 22.7M of those donors' own likes and their follow lists, mostly tech accounts. That is per-viewer engagement data from the target audience: positives only, with likely-seen negatives derived from posts by accounts each donor follows. It supports training a small viewer-conditional Phoenix (same feature prep, real histories) and validating the segment model against real individuals. It does not support a population of 10,000 distinct simulated people; it does support checking that simulated segments behave like real ones.

LLM personas are not used as agents: zero-shot LLMs score at chance on engagement prediction (research/06_literature.md, section C), and they do not reproduce X's ranking.
