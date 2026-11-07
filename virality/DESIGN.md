# Virality Engine: architecture proposal

Status: proposal for review, 2026-11-07. Nothing here is built yet.
Evidence: seven research reports in `virality/research/` (five on this repo's code with file:line citations, one literature review, one compute and data costing). This document is the synthesis and the recommendation.

---

## 1. The decision in one page

**Build a three-model system, not one RL agent.**

| Component | What it answers | Method | Why this method |
|---|---|---|---|
| **A. Pre-flight suppression check** | "Will X refuse to distribute this at all?" | Rules + small classifiers that mirror `visibility-filtering/` | X's visibility filtering is binary (show or drop, never rank-penalized). Rules are known from code. No learning needed for most of it. |
| **B. Phoenix surrogate inside a ported X pipeline** (the core) | "Given this draft and this account, what reach will it get, and what are the odds it goes viral?" | X's own filters, value model, post-fusion and reranker ported as code; one supervised transformer stands in for Phoenix and emits Phoenix's action heads | The only thing X learned is Phoenix. Everything else is code in this repo. We port the code and train only the surrogate, with its interface fixed by the algorithm. |
| **C. Live reach forecaster** | "It has been up 30 minutes. Where will it land at 24 hours?" | Small sequence model over early engagement counts conditioned on B's prior | With one hour of signal, final reach is predictable to within roughly 25%. This is the piece that makes the product provably accurate to customers. |
| **D. Optimizer** | "How do I change it to do better?" | (1) Deterministic levers from the algorithm, (2) best-of-N rewriting scored by B with an independent check, (3) later: an RL-trained editor (DPO then GRPO) with B as reward | This is where RL belongs: the policy that *edits* content, not the model that *predicts* reach. |

**Where your instinct lands.** You proposed a transformer trained through RL. The system has that: component D's editor is a transformer trained by reinforcement learning against a learned reward. But the reward model (B) must be supervised, because the ground truth is what real audiences did, and no amount of self-play produces that. Training B with RL would be training a thermometer by reinforcement.

**On "guaranteed to go viral."** Not achievable by anyone, and I will not pitch it. The literature is unambiguous: content alone explains under 5% of variance in absolute reach on Twitter (Martin et al. 2016, 852M tweets). Account size, audience, timing and luck dominate. What *is* achievable, and what customers will pay for:

- A calibrated probability ("18% chance this exceeds 10x your follower count in 24h, versus 4% for your typical post").
- A reliable *ranking of drafts* for the same account (literature ceiling 65 to 77% pairwise accuracy; humans score 61%).
- A near-certain forecast once the post is 30 to 60 minutes old (AUROC above 0.9 for top-1% posts).
- Concrete, algorithm-backed changes with predicted lift, and a hard list of things that get a post silently dropped.

That is a product. "Guaranteed" is a refund request.

---

## 2. What the repo actually tells us (the ground truth we build on)

These are the findings that shape the design. Each is cited to code in the research reports.

### 2.1 The ranker never sees text or pixels. It sees one embedding.

The Phoenix ranker (`phoenix/xrex/models/recsys_feature_prep.py`) builds each candidate post from: hashed post ID, hashed author ID, a 6-level **semantic ID** (residual-quantized code of a 1024-d multimodal embedding), post age bucket, local hour, engagement counts (log2 of like/reply/repost/quote/view through a small MLP), and two bits for whether the viewer follows or is followed by the author. That is the whole content contract. Media flags, video duration, language, text length and URL presence are sent on the wire but **not consumed** by the home ranker (report 01, section 1.2; report 04, section 1).

The embedding is produced by the public `Qwen/Qwen3-VL-Embedding-8B` model over a rendered document (author line, text, images, quoted post, video frames, speech transcript), with X's own instruction prompt, truncated to 1024 dims and L2-normalized (`phoenix/reference/mm_encoder.py:18-25`, `phoenix/reference/README.md:110-121`). The semantic ID codebook trainer (RQ-KMeans) ships too (`phoenix/reference/sid_codebook.py`).

**Consequence:** we do not need to guess a CNN. We run the same open-weight encoder X runs, on the same rendering, and train our own codebook. Our model then sees content the way Phoenix sees it. This is the single strongest argument for this design over a generic "VLM looks at the post" approach.

### 2.2 The objective is a public weighted sum, and it is lopsided.

`xai-value-model/scoring.rs` computes `score = Σ weight_i × P(action_i)` with weights in `home-mixer/params/param.rs` (synced from production on 2026-11-05):

| Action | Weight | | Action | Weight |
|---|---|---|---|---|
| share via copy link | 20.0 | | like | 0.708 |
| reply, from a mutual follower | 5 + 15 = 20.0 | | click | 0.274 |
| reply, quote, share via DM | 5.0 each | | open link | 0.2 |
| follow author | 4.0 | | dwell, photo expand | 0.05 |
| share | 2.118 | | every video-specific weight | **0.0** |
| repost | 1.0 | | | |
| **report** | **-234** | | **mute** | **-58.8** |
| **not interested** | **-53.07** | | **block** | **-31.2** |

No clipping, no freshness decay, no verified or premium multiplier, no language multiplier. A net-negative post is compressed into (0, 0.001) and ranks below every clean post. Out-of-network posts are multiplied by 0.75. Each additional post from the same author in one feed decays by 0.5 toward a floor of 0.25 (report 02).

**Consequence:** the product's score is not "likes." It is X's value score. We predict per-impression rates of the actions we can observe and combine them with X's weights. When X changes a weight (the repo syncs daily by cron), we change a config file and the product updates without retraining.

### 2.3 X defines "underexposed" with a formula we can reuse as a label.

`phoenix/xrex/data/recsys/recsys_batch.py:105-135`: an original post is "unexplored" if, within 24 hours, its views are below `0.03 × followers × (1 − e^{−ln2·age/8h}) / (1 − e^{−ln2·24/8})`. The popular-posts job uses the same 8-hour half-life curve to project 24-hour views (`home-mixer/util/popular_posts.rs:48-53`).

**Consequence:** reach normalized by follower count, on an 8-hour half-life curve, is X's own yardstick. Our virality label is defined on that scale, not on raw counts.

### 2.4 The first two hours decide out-of-network reach.

A clean original post reaches followers immediately via Thunder. It enters out-of-network retrieval about two minutes after its **first like** and lives in a dedicated cold pool (under 8 likes and under 500 views, age under 2 hours) with its own 200-slot budget. A separate cold-start slot lifts one post per feed request to position 16 for authors under 50k followers, under 2 hours old, under 200 home impressions, chosen by Thompson sampling (report 02 section 2.3, report 03 section 1). Reposts are never indexed for out-of-network retrieval. Replies reach non-followers only after a like, and that path is disabled by default. Everything is gone from the feed at 48 hours.

**Consequence:** component C (live forecaster) must sample at 5, 15, 30, 60 and 120 minutes, and the product should tell a customer to post originals, not reposts, and to secure early engagement from mutuals (reply weight is 4x from a mutual follower).

### 2.5 Suppression is binary and mostly about links, bait and media safety.

Visibility filtering returns show or drop for For You; no rank penalty exists (`home-mixer/filters/vf_filter.rs:22-27`). The drops that matter for a marketer, ranked by consequence (report 05, section 6): links to bad-reputation domains (dropped for everyone, retroactively), pinning a low-quality link, engagement bait or farming (dropped even for high-credibility authors), adult or gory media, sexual preview cards, copy-pasted text across posts, mass replies, follow churn, gibberish or LLM slop. External links are not suppressed by rule, but an observational study of 40M posts found they cut views up to 8x, and the link-click weight is only 0.2.

**Consequence:** component A is a cheap deterministic gate, and it is the first thing the product shows.

### 2.6 What is not in the repo

No trained weights. No real data. No user-action aggregator. The shipped synthetic world (`phoenix/reference/world.py`) draws engagement from hand-set base rates and a sigmoid of random affinities. **It cannot generate labels for us.** This is the answer to the synthetic-data idea: we can mirror X's *structure* exactly, but the *outcomes* must come from real posts.

---

## 3. Architecture: X's pipeline as code, with one learned function

### 3.1 The governing principle

X's For You pipeline is a chain of deterministic stages wrapped around one learned function. The deterministic stages are in this repo as code: filters, visibility rules, the value model, the post-fusion multipliers, the cold-start slot, the DPP reranker, retrieval eligibility, the popular-posts projection, the content encoder and the codebook. The one learned function is Phoenix: given a viewer and a post, the probability of each action.

So the system is built the same way. **Everything that is code in X stays code here, ported from this repo. The only thing we train is a surrogate for Phoenix, and it is trained to produce exactly the outputs Phoenix produces, so that X's own downstream code can run on them unchanged.** The learned part is as small as the problem allows, and its interface is dictated by the algorithm, not by us.

```
                 draft post + account
                         │
   ┌─────────────────────▼───────────────────────────────┐
   │ LAYER 0: PORTED X CODE (no training)                 │
   │  content token   phoenix/reference/mm_encoder.py     │
   │                  + sid_codebook.py / sid_assign.py   │
   │  pre-flight      visibility-filtering rules          │
   │  eligibility     thunder, phoenix-rankall gates,     │
   │                  cold pool, cold-start slot,         │
   │                  popular-posts formula, 48h age      │
   │  post-type       OON reply/repost filters            │
   └─────────────────────┬───────────────────────────────┘
                         │ candidate features in Phoenix's own layout
   ┌─────────────────────▼───────────────────────────────┐
   │ LAYER 1: PHOENIX SURROGATE (the only trained model)  │
   │  candidate side: Phoenix's feature prep, verbatim    │
   │  viewer side:    audience tokens instead of one      │
   │                  viewer's 1022-event history         │
   │  outputs:        Phoenix's action heads              │
   │                  P(like) P(reply) P(repost) P(quote) │
   │                  P(bookmark) P(click) P(follow) ...  │
   │                  masked where no label exists        │
   └─────────────────────┬───────────────────────────────┘
                         │ per-action probabilities
   ┌─────────────────────▼───────────────────────────────┐
   │ LAYER 2: PORTED X CODE AGAIN (no training)           │
   │  value model     xai-value-model/scoring.rs with     │
   │                  home-mixer/params/param.rs weights  │
   │  post-fusion     offset, OON ×0.75, author-diversity │
   │                  decay, cold-start lift              │
   │  reranker        vm-ranker DPP, theta 0.65, top 50   │
   │  placement       rank against a reference pool of    │
   │                  real competing posts                │
   └─────────────────────┬───────────────────────────────┘
                         │ rank distribution
   ┌─────────────────────▼───────────────────────────────┐
   │ LAYER 3: CALIBRATION + PRODUCT                       │
   │  rank → reach mapping fit on observed views          │
   │  odds per tier, score breakdown by action × weight   │
   │  solver over deterministic levers + surrogate        │
   │  live forecaster once trajectories exist             │
   └─────────────────────────────────────────────────────┘
```

### 3.2 Layer 0 and Layer 2: what gets ported, from where

| X stage | Repo location | What we do with it |
|---|---|---|
| Content token | `phoenix/reference/mm_encoder.py`, `sid_codebook.py`, `sid_assign.py` | Run as-is: Qwen3-VL-Embedding-8B, X's prompt and rendering, truncate to 1024, our RQ-KMeans codebook trained with their trainer. |
| Candidate feature prep | `phoenix/xrex/models/recsys_feature_prep.py`, `data/recsys/feature_config.py` | Reuse the age buckets (hourly to 80h), log2 count MLP input, SID tables, stale flag, follow-bit tables. Our surrogate consumes the same tensors Phoenix consumes. |
| Pre-scoring filters | `home-mixer/filters/*.rs` | Port the ones a submitted post can trip: age, reply/repost out-of-network, video exclusion, NSFW SimClusters, muted-keyword exposure estimate. |
| Visibility filtering | `visibility-filtering/rules/levels.rs`, `tweet_rules.rs`, `author_rules.rs`, `botmaker-rules/`, `scarecrow/` | Port the rule logic for For You and For You recommendations. Classifiers that produce labels (NSFW, gore, bait, link reputation) are replaced by open models at X's documented thresholds. |
| Retrieval eligibility | `phoenix-rankall-strato/.../phoenixRankAllCandidateProcessor.strato`, `phoenix/xrex/data/cold_pool_filter.py`, `home-mixer/util/popular_posts.rs`, `simclusters/` thresholds | Port as a timeline simulator: when the post enters each index, the cold pool, the popular list; which gates it must clear. |
| Exploration label | `phoenix/xrex/data/recsys/recsys_batch.py:105-135` | Port verbatim; it defines our reach normalization. |
| Value model | `xai-value-model/scoring.rs`, `home-mixer/params/param.rs` | Port verbatim, weights read from a pinned copy of `param.rs` and refreshed from upstream weekly. |
| Post-fusion | `xai-value-model/scoring.rs:239-406`, `home-mixer/scorers/author_cold_start.rs` | Port verbatim: offset, OON multiplier, author diversity, cold-start Thompson lift. |
| DPP reranker | `vm-ranker/scoring/dpp.rs`, `dpp_model.rs` | Port verbatim; run over our 1024-d embeddings, which are the same embeddings X feeds it. |
| Selection | `home-mixer/selectors/top_k_score_selector.rs`, pool size 2800, top 50, result 35 | Port the constants; use a sampled reference pool of real posts with their own surrogate scores as competitors. |

Rust can be compiled and called through FFI where the file is self-contained (`scoring.rs`, `dpp.rs`); the rest is a line-by-line Python port with the repo file and line cited in a comment, and a test that reproduces the repo's own unit tests where they exist.

### 3.3 Layer 1: the Phoenix surrogate

**What Phoenix is.** A transformer over `[viewer tokens | viewer's last 1022 engagements | candidates]`, candidate-isolated, outputting 64 sigmoid action heads and continuous dwell heads. Candidate inputs: hashed post ID, hashed author ID, semantic ID, age bucket, surface, timezone, local hour, follow bits, log2 counts, stale flag. Trained on billions of impressions with in-batch negatives and log-Q correction (report 01).

**What we can and cannot copy.**

| Phoenix element | Surrogate | Why |
|---|---|---|
| Candidate-side feature prep | Identical, reusing their code | It is in the repo and needs no data we lack. |
| Semantic ID + content token | Identical encoder, our codebook | Encoder is public; codebook is trained with their trainer on our corpus. |
| Hashed post-ID table (100M rows memorizing each post) | Dropped | We score unseen posts, which is exactly the case where Phoenix itself falls back to the semantic ID. |
| Hashed author-ID table | Author embedding learned on our corpus plus explicit author statistics | Same role, learnable from our data. |
| One viewer with 1022-event history | Audience tokens: the author's follower population described by aggregate statistics, plus an out-of-network audience token from the semantic-ID neighborhood | Public counts are averages over the real audience, so the target is audience-average probability, and the input must describe the audience, not one person. |
| 64 action heads | Same head names and indices; heads without labels are masked | Keeps the output contract identical so Layer 2 code runs unchanged. As connected-account data arrives, more heads unmask (clicks, profile clicks, follows, link opens). |
| Continuous dwell heads (CREAD) | Masked until trajectory or analytics data exists | No public label. |
| Negative-feedback heads (report, mute, block, not interested) | Predicted from a proxy trained on the pre-flight classifiers and later on connected-account data | These carry the largest weights, so even a coarse estimate matters. |
| In-batch negatives, log-Q | Replaced by same-author pairwise loss and impression-weighted BCE | Our rows are posts with rates, not impressions with binary outcomes. |
| Trunk | Same block design as `home_direct_packed` at reduced width: 8 layers, width 1024, candidate isolation kept | Same shape, sized to our label budget. |

**Training target.** For each post, per-impression rates: likes / views, replies / views, and so on, which are the audience-average of the probabilities Phoenix predicts. Losses: impression-weighted BCE per head, Huber on log reach ratio as an auxiliary, same-author pairwise logistic on the X value score. The teacher in phase 2 (Qwen3-VL-8B QLoRA) is trained on the same heads and distilled in.

**What the surrogate adds beyond Phoenix's inputs.** SigLIP-2 image and frame embeddings and the content scalars (text length, links, newlines, media count, video duration) that home-mixer computes and Phoenix ignores. Phoenix can ignore them because its post-ID table memorizes outcomes; we cannot, so they are inputs. They are additive and can be ablated to measure how far "pure Phoenix inputs" get.

### 3.4 Layer 3: from X's rank to reach and odds

X's code gives a rank in a feed, not a view count. The mapping from "expected rank and eligibility timeline" to "views at 24h" is fit on observed data: a monotone calibration per follower tier, with conformal intervals, on held-out authors. This is a small, well-posed regression, and it is the only place the product converts X's internal quantities into the numbers a customer reads.

Product outputs, all traceable to a repo file:

- Pre-flight verdict and the rule that fired.
- Eligibility timeline: when the post can enter out-of-network retrieval, cold-pool window, cold-start slot eligibility, popular-posts eligibility.
- X value score with its breakdown: each predicted probability times its weight from `param.rs`.
- Expected rank distribution against real competitors, after OON multiplier, author diversity and DPP.
- Reach odds per tier with intervals.
- Levers: a solver enumerates the deterministic choices (original versus quote, link placement, post spacing to avoid author-diversity decay, posting hour, media type) and re-runs Layers 0 to 3; content rewrites go through best-of-N scored by the surrogate with a second model as the agreement check.

### 3.5 The live forecaster and the RL editor

Unchanged from the earlier draft and deferred until the data for them exists: the live forecaster needs trajectory samples from connected accounts; the RL editor (DPO on observed pairs, then GRPO against the surrogate with KL and agreement constraints) needs the surrogate first and live outcomes to stay honest.

---

## 4. Why this beats the alternatives

| Alternative | Why not as the core |
|---|---|
| Pure RL agent predicting virality | No environment to act in. Prediction is supervised by definition. RL is used in D only. |
| Generic VLM prompted "will this go viral?" | Zero-shot frontier LLMs score at chance (AUROC 0.53 on top-5% prediction, Vastu 2026). Fine-tuned small encoders beat them. |
| Gradient-boosted trees only | Right as the phase-1 baseline and will likely stay in the ensemble, but it cannot read images or video and cannot rank rewrites. |
| Our own copy of Phoenix trained on synthetic data | The synthetic world is random affinities with hand-set base rates. A model trained on it learns the generator, not X. |
| Scrape tens of millions of posts | X's terms set liquidated damages of $15,000 per million posts per 24 hours. The API and donated archives are the compliant path. |
| A generic CNN for media | Phoenix sees media only through the Qwen3-VL embedding. Using the same encoder is both more faithful and stronger than a CNN trained from scratch on our label budget. SigLIP-2 covers what the embedding smooths over. |

What makes this hard to copy: the exact content encoder and codebook recipe from the repo, the X value score with live-synced weights, the X-native reach normalization, and (once customers connect accounts) first-party impression and click data that no public dataset has.

---

## 5. Data plan: one unified, audience-filtered corpus

### 5.1 Principle

Every source is poured into one schema, then filtered hard for **audience relevance** and **recency** before a single GPU-hour is spent on it. Old posts, posts outside the tech, startup, ML, YC and SF world, and post types the current algorithm distributes differently (replies, reposts) are dropped at ingestion. Compute goes only to posts that look like what customers will submit.

Unified post schema: post id, author id, created_at, text, media URLs and types, like, reply, repost, quote, bookmark and view counts where present (null otherwise), author followers and following at the nearest known time, verified flag, source tag, snapshot time. Labels are derived, never copied, from the source.

### 5.2 The filter, and your dataset's real job

Your 2025 scrape is the cleanest available sample of the target audience. Its account list and its posts become the **positive seed** for a relevance classifier: embed everything with the same encoder, train a small classifier on "from your account set" versus a random sample of the public corpora, and keep public posts above a threshold, with a manual check of 500 borderline cases to set it. Author-level rules on top: accounts whose recent posts are majority-relevant stay as whole accounts, so author histories remain intact for the trailing-rate features.

Recency: views exist on X only for posts after December 2022, and the current ranker and weights date from 2026. Keep 2023 onward, weight 2025 onward higher in training, and run a final calibration pass on the newest slice available.

Post type: originals and quotes only. Replies and reposts are filtered or crippled by the current pipeline (section 2.4), so they are the wrong training signal for a product about original posts and launches.

Video and launch content are rare in every corpus. They are never dropped by the filter, get oversampled in training, and get their own validation slice so a weak video result cannot hide behind a strong text result.

### 5.3 Sources after filtering

| Source | Raw size | Expected yield after the audience filter | Carries views | Role |
|---|---|---|---|---|
| USC X 2024 corpus (CC BY-NC-SA, research only) | ~46M posts | Unknown until audited; politics-collected, so plausibly a few hundred thousand relevant posts, mostly from accounts that post about both | Yes | Largest view-labeled slice for the research model |
| Community Archive | Millions of posts, tech-skewed donors | High fraction relevant; filtered to 2023+ originals | No | Author histories, same-author pairs, media via archive URLs |
| Your 2025 scrape | 250k to 450k | All relevant by construction | Yes | Audience seed, validation anchor, and training rows once deduped against the rest |
| Your analytics exports and donors | Small | All relevant | Yes, plus clicks and follows | Calibration and the click and follow heads |
| Bluesky open API | Very large; tech community is sizable | Filtered the same way | No (likes, reposts) | Optional pretraining of the content tower; skipped unless the X corpus is too small |
| Customer accounts (post-launch) | Grows | All relevant | Yes, rich | Flywheel and the compliant commercial retrain |

The honest expectation, revised after measuring real data (DATA_PIPELINE.md section 13): the USC corpus contains essentially no target-audience posts, so the view-labeled corpus for this audience is the tech scrape (250k to 450k rows) plus analytics exports, and the Community Archive adds roughly 500k relevant originals with likes but no views, used for pairwise and per-author training. That is enough for the design in section 3 because the heavy capacity is frozen and the trainable part is sized to the data, but it makes the scrape and connected-account analytics the critical inputs.

### 5.4 Why this will not repeat last year's result

Your XGBoost ensemble had the right instinct and four structural problems the new design removes:

1. **Target.** An absolute `is_viral` label is mostly a follower-count detector. The new target is reach relative to followers on X's own exploration curve, plus author-relative percentiles.
2. **Confounding.** Rows were independent. The new training pairs posts from the same author within a week, so the model is forced to learn what differed in the content, not who posted.
3. **Content representation.** Unknown 768-d embeddings compressed to 128 dims through PCA. The new representation is the exact Qwen3-VL embedding X's ranker sees, plus our codebook, plus SigLIP-2 for images and video frames, read by a transformer rather than a tree over PCA components.
4. **Evaluation.** Random splits leak authors and time. The new splits hold out authors and a later time window, and the product reports calibrated odds rather than a binary label.

Trees stay in the system as the phase-0 baseline and as a member of the final ensemble. They are not the model.

---

## 6. Compute, time and budget, pinned to $300 to $500 of AWS credits

Hardware: AWS credits for the GPU steps, the RTX 2060 for overnight work (SigLIP embeddings, Whisper, LightGBM, data packing). Everything on spot instances with checkpoints every 15 minutes.

| Phase | Weeks | What ships | Compute | Credits |
|---|---|---|---|---|
| 0. Unified corpus + baseline | 2 | Ingest all sources into one schema, dedupe, audience relevance classifier seeded from your account set, recency and post-type filters, label definitions, author and time splits, LightGBM baseline with pairwise objective. Outputs the real corpus counts and the first AUROC and pairwise numbers. **This is the go/no-go gate.** | CPU and 2060; one L4 spot day for embedding the candidate pool for the relevance filter | $10 |
| 1. Faithful embeddings + predictor v1 | 3 | Qwen3-VL-Embedding-8B over the filtered corpus with X's rendering, our 6×256 codebook, SigLIP-2 image embeddings, fusion transformer (150 to 300M) with all heads, calibration, pre-flight checker | g6e.xlarge spot (L40S) ~15h for embeddings; g6.xlarge spot (L4) ~40h for sweeps | $20 + $20 |
| 2. Teacher + distillation | 2 | Qwen3-VL-8B QLoRA on ~200k examples, same heads; score corpus; distill into v1 | g6e.xlarge spot ~30h training + ~10h scoring | $55 |
| 3. Optimizer v1 | 2 | Algorithm levers, best-of-N rewriter (Qwen3-8B generation) with dual-model agreement, timing recommendation, API | g6.xlarge spot ~40h | $20 |
| Storage, S3, EBS | | | | $25 |
| Reserve for reruns and spot interruptions | | | | $100 |
| **Total** | **9 weeks** | | | **$240 to $260, ceiling $360 with reserve** |

That leaves $140 or more unspent against a $500 ceiling for the first RL editor cycle (DPO on observed pairs, GRPO against the predictor with KL and agreement constraints) once live outcomes exist.

Model size at the end of phase 2: ~9B frozen encoder parameters, an 8B LoRA teacher used offline, a 150 to 300M fusion model that serves at under $0.01 per 1,000 predictions.

---

## 7. Success criteria (what "working" means, measured before any customer sees it)

| Metric | Target after phase 1 | Target after phase 2 | Basis |
|---|---|---|---|
| Cold-start AUROC, "exceeds 10x followers in 24h" | 0.74 | 0.80 | Literature range 0.72 to 0.82 for content + author |
| Cold-start precision at top 1% | 5x lift over base rate | 10x lift | Literature range 5 to 15x |
| Same-author pairwise accuracy (which draft did better) | 66% | 72% | Literature 62 to 77%, humans 61% |
| Spearman on log reach ratio, held-out authors | 0.55 | 0.65 | SMP challenge winners 0.70 to 0.78 with richer author data |
| Live forecast at 30 min, AUROC for top 1% | 0.88 | 0.92 | Cascade literature with 1h signal |
| Live forecast at 60 min, error on 24h views | 35% | 25% | SEISMIC-class 15 to 25% for moving posts |
| Calibration error (ECE) on held-out authors | under 0.05 | under 0.03 | Required for stating odds to customers |
| Pre-flight drop detection on a labeled audit set | recall 0.9 on link and bait rules | plus media safety | Rules are known from code |

If phase 1 misses the pairwise target by more than 5 points, the honest conclusion is that our data is too thin and the next dollar goes to data, not to the teacher.

---

## 8. Risks, stated plainly

- **Your corpus is a 2025 snapshot without action breakdown or trajectories.** It is the right corpus for the cold-start predictor and the optimizer. The live forecaster and the per-action value-score heads wait for connected-account data. The 2026 algorithm change is handled by taking mechanics from current code and recalibrating on new data.
- **The repo shows the structure, not the weights.** Our predictor learns from audience outcomes, not from Phoenix. It will disagree with Phoenix on individual posts. The value score and the normalization are X's; the probabilities are ours.
- **Weights drift.** `param.rs` is re-synced by cron and the reply boost changed twice in July 2026. We read weights from a config pinned to the repo commit and re-pull weekly.
- **Goodhart in the optimizer.** Mitigated by the dual-model agreement constraint, meaning-preservation constraint, KL penalty and, above all, measuring real outcomes through customer analytics.
- **Licensing.** All recommended encoders are Apache-2.0 or MIT. VideoMAE v2 (CC BY-NC) and DINOv3 (custom license) are excluded. The USC dataset is non-commercial and stays out of the product model.
- **The tail is luck.** Top-0.1% virality is dominated by exogenous events (a large account quote-tweeting, a news cycle). We will report precision honestly at that tier rather than claim it.

---

## 9. What I need to start phase 0

1. **Your dataset files** (the 13 GB merge and the V1, V2, V4 raws) and the account list behind them. They seed the audience filter.
2. **Media stance.** Bounded, rate-limited image and video-frame fetching from stored URLs, or embeddings-only for media.
3. **Your X analytics export**, and anyone else's you can get.
4. **A go.** Phase 0 touches about $10 of credits and ends with real corpus counts and the first accuracy numbers.

## Appendix: research reports

- `research/01_phoenix_ranker.md`: full input contract of the ranking model, heads, losses, parameter counts, training data schema.
- `research/02_scoring_weights.md`: every weight and threshold in `param.rs` and `scoring.rs`, the DPP reranker, cold-start slot, popular-posts source.
- `research/03_distribution.md`: Thunder, retrieval index eligibility, SimClusters, every pre-scoring filter, lifecycle of a new post.
- `research/04_media_content.md`: media and text understanding, Grox classifiers, the multimodal embedding recipe, audio transcription.
- `research/05_visibility_suppression.md`: label table, suppression rules, user credibility, the 15 things not to do.
- `research/06_literature.md`: cascade prediction, cold-start multimodal popularity, LLM predictors, industry recsys, counterfactual optimization, RL versus supervised, datasets and legal.
- `research/07_compute_data.md`: GPU prices, feasibility matrix per hardware option, memory arithmetic, storage, encoder licenses, serving cost, phased budget.
