# Virality Engine: architecture proposal

Status: proposal for review, 2026-11-07. Nothing here is built yet.
Evidence: seven research reports in `virality/research/` (five on this repo's code with file:line citations, one literature review, one compute and data costing). This document is the synthesis and the recommendation.

---

## 1. The decision in one page

**Build a three-model system, not one RL agent.**

| Component | What it answers | Method | Why this method |
|---|---|---|---|
| **A. Pre-flight suppression check** | "Will X refuse to distribute this at all?" | Rules + small classifiers that mirror `visibility-filtering/` | X's visibility filtering is binary (show or drop, never rank-penalized). Rules are known from code. No learning needed for most of it. |
| **B. Phoenix-mirror predictor** (the core) | "Given this draft and this account, what reach will it get, and what are the odds it goes viral?" | Supervised multi-task transformer over the *exact* content representation X uses, plus author and context | Every state-of-the-art engagement predictor is supervised. The repo tells us the inputs and the objective X optimizes, so we copy both. |
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

## 3. Architecture in detail

```
                       ┌──────────────────────────────────────────────┐
  draft post ───────►  │ A. PRE-FLIGHT  (rules + classifiers)          │──► drop risks, fixes
  (text, media,        └──────────────────────────────────────────────┘
   account)                               │ passes
                                          ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │ B. PHOENIX-MIRROR PREDICTOR                                        │
  │                                                                    │
  │  content tokens                 author tokens        context       │
  │  ├ Qwen3-VL-Embedding-8B (1024) ├ followers, age     ├ local hour  │
  │  │  + our 6×256 semantic ID     ├ trailing per-post  ├ day         │
  │  ├ SigLIP-2 image emb (per img) │  rate distribution ├ original /  │
  │  ├ SigLIP-2 ×8 video frames     ├ posting cadence    │  reply /    │
  │  ├ Whisper transcript (in text) ├ author centroid    │  quote      │
  │  ├ scalars: has_url, text len,  │  (SimClusters      ├ k nearest   │
  │  │  newlines, media count,      │   proxy)           │  recent     │
  │  │  video secs, language        └ credibility proxy  │  posts +    │
  │  └ safety scores from A                              │  outcomes   │
  │                                                                    │
  │  ──► fusion transformer (150–300M trainable) ──► heads:            │
  │        per-impression rate: like, reply, repost, quote, bookmark   │
  │        log(views / followers)   [Huber]                            │
  │        exceed 1x / 10x / 100x followers  [BCE, calibrated]         │
  │        same-author pairwise "B beats A"  [logistic]                │
  │        X value score = Σ w_i · rate_i  (weights from param.rs)     │
  │                                                                    │
  │  teacher (phase 3): Qwen3-VL-8B QLoRA reading the rendered post,   │
  │  same heads; distilled into the fusion model for serving           │
  └────────────────────────────────────────────────────────────────────┘
                 │ prior                                  ▲
                 ▼                                        │
  ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │ C. LIVE FORECASTER            │        │ D. OPTIMIZER                 │
  │ counts at 5/15/30/60/120 min  │        │ 1. deterministic levers      │
  │ + prior ──► 24h reach, CI     │        │ 2. best-of-N rewrite, scored │
  └──────────────────────────────┘        │    by B + independent model  │
                                          │ 3. DPO/GRPO editor (later)   │
                                          │ 4. posting-time bandit       │
                                          └──────────────────────────────┘
```

### 3.1 Component B: the predictor

**Content representation (mirrors Phoenix exactly, then adds what Phoenix throws away).**

1. Render the post the way `mm_encoder.py` does: author line, text with t.co links stripped, card and article titles, quoted post, image pads, up to 4 uniformly sampled video frames, `Transcript:` from Whisper (X's grox pipeline transcribes audio and appends it; report 04 section 5). Encode with `Qwen3-VL-Embedding-8B`, X's instruction, truncate to 1024, normalize. This is the token Phoenix sees.
2. Train a 6×256 RQ-KMeans codebook on our corpus with the shipped `sid_codebook.py`. Feed prefix codes as learned embeddings, as Phoenix does. This gives the model the same "topic neighborhood" generalization that lets Phoenix score a brand-new post.
3. Add what Phoenix cannot use but which we know matters for cold start: SigLIP-2 So400m embeddings per image and per video frame (aesthetics, text-in-image, faces), and the scalar features home-mixer computes but the ranker ignores (`weighted_text_len`, `has_url`, `newline_count`, `media_count`, video duration, language). Phoenix doesn't need these because it has a 100M-row post-ID table memorizing each post's popularity. We don't, so content detail has to work harder.

**Author representation.** Followers, following, account age, verification, posting cadence, the trailing distribution (median, p90) of per-impression like/reply/repost/quote rates over the author's last 50 posts, the author's content centroid in embedding space, and the author's trailing reach ratio distribution. In the literature, author history moves R² from under 0.05 to 0.42 to 0.48. In Phoenix, author identity is a hashed embedding and the viewer-author follow bits; this is our stand-in.

**Context.** Local hour and weekday of the author's audience (Phoenix uses the viewer's local hour), original/reply/quote flag (replies and reposts have crippled reach, section 2.4), and retrieved neighbors: the k most similar recent posts in embedding space and how they performed. Retrieval-augmented popularity prediction (RAGTrans, SKAPP) is the strongest recent gain in the literature and models trend saturation.

**Fusion.** A transformer over the token set, 8 to 12 layers, width 1024, roughly 150 to 300M trainable parameters, with ~9B frozen encoder parameters in front of it. Multi-task heads as drawn. Losses: Huber on log reach ratio, BCE on exceedance thresholds, pairwise logistic on same-author pairs posted within 7 days of each other (the standard confound control from Tan 2014, Hessel 2017, RePALM), and a distillation term from the teacher in phase 3.

**Why not simply "go big" on the trainable part.** We will have 0.5 to 2M labeled posts. Phoenix trains on billions of impressions. With our label budget a from-scratch multi-billion-parameter model overfits; the capacity that helps is *frozen* capacity that already understands images, video and language. The 8B embedding encoder plus an 8B fine-tuned teacher is the right place for size. The trainable fusion model is sized to the data, and it is the model that serves at under a cent per thousand predictions.

**Targets and the definition of viral.** Primary scale: reach ratio r = views at 24h / followers at post time. Tiers: r ≥ 1 ("broke out of followers"), r ≥ 10, r ≥ 100, plus "top 1% of this author's trailing year." Secondary: absolute view tiers for brand accounts whose follower count is not the right denominator. All heads calibrated by isotonic regression on held-out authors, with conformal intervals.

### 3.2 Component C: the live forecaster

Inputs: B's prior, plus views, likes, replies, reposts, quotes, bookmarks at each checkpoint, plus the derived cold-pool state (under 8 likes and under 500 views, under 2h). A small GRU or temporal transformer with a Hawkes-style intensity head, trained on the same corpus with trajectory snapshots. Output: 24h and 48h reach with a confidence interval, and "cold-start slot still eligible" status. Trajectory data requires polling posts after publication, which the data plan in section 5 covers.

### 3.3 Component D: the optimizer

Ordered by evidence and cost:

1. **Deterministic levers from the algorithm**, each scored by B as a counterfactual: move the external link to a reply; post as original instead of quote or reply; cut to one post per hour so author-diversity decay does not apply; switch video to a photo plus text if the video is under 10 seconds (video weights are zero today anyway); adjust posting hour to audience local time; remove engagement-bait phrasing (which risks a drop, not just a penalty).
2. **Best-of-N rewriting.** An open LLM (Qwen3-32B class) generates 32 to 64 variants under a meaning-preservation constraint (embedding cosine to the original above a threshold). B scores them. A second predictor with a different backbone must agree, and we discount variants where the two disagree. This is the Goodhart guard; without it, optimizing against a single learned predictor produces posts the predictor loves and humans do not (Gao et al. 2023; Meta's CharacterFlywheel, where reward-model win rate rose while real engagement fell).
3. **The RL editor (your transformer trained through RL).** Once we have observed same-author pairs and customer A/B outcomes, train the rewriter with DPO on real pairs, then GRPO against B with a KL penalty to the base model and the agreement filter as a hard constraint. This is the correct, bounded use of RL here. It is phase 4 because it needs B to exist first and needs live outcomes to stay honest.
4. **Posting-time bandit.** Thompson sampling over hour-of-day per account, exactly the family X uses for its cold-start slot. Cheap and deployed in industry since 2015.

### 3.4 Component A: the pre-flight check

URL expansion and domain reputation (several open blocklists plus a learned domain-quality score), open NSFW and gore classifiers run with X's documented thresholds (image precision ≥ 0.95, gore precision > 0.6, video raw ≥ 0.95), an engagement-bait and farming classifier (LLM-labeled training set over the policy categories named in `grox/`), copypasta detection against the author's own history, and the behavioral flags (reply bursts, posting cadence). Output: probability of an out-of-network drop and a plain-language fix.

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

## 5. Data plan (zero cash; AWS credits only)

Constraint: no money for X API reads. Everything below is free to obtain. The compute to process it comes out of AWS credits.

| Source | What it gives | License / terms | Use |
|---|---|---|---|
| **USC X 2024 election dataset** (~46M posts, May to Nov 2024) | Full post objects with `viewCount`, like, reply, repost, quote counts, media fields, author metrics | CC BY-NC-SA: **research and prototyping only** | The only free corpus with views at scale. Trains and validates every component in phases 0 to 2. Topically skewed to politics, so we sample 1 to 2M posts weighted away from pure political content and toward accounts that also post product, tech, business and creator material. |
| **Community Archive** (donated X archives) | Per-author longitudinal histories with like and repost counts and media URLs; no views | Donated under the project's open terms | Same-author pairwise training (the strongest content signal in the literature) and author-history features. Donors skew toward tech and startup accounts, which matches the first customers. |
| **Your own accounts and anyone who will donate** | X analytics export: per-post impressions, engagements, profile clicks, link clicks, detail expands, new follows | First-party | The richest labels that exist, and they map onto Phoenix heads directly. Even 10 accounts with full histories is a valuable calibration set. Ask friends, founders, and the first design partners. |
| **Bluesky firehose / open API** (235M+ posts) | Text, images, like, repost, reply counts; fully open | Open | Auxiliary pretraining of the content tower on "engagement given content and author" in general. Different platform, so it is a pretraining signal and a regularizer, never a label for X. |
| **Media for the above** | Images and video frames via media URLs in the datasets | X's terms restrict automated collection from its services; fetching referenced media is a gray area | Sample, do not sweep: pull media for a bounded subset (about 300k posts with images, 30k with video), rate-limited, and store SigLIP and Qwen3-VL embeddings plus sampled frames rather than raw files. If you want zero exposure here, media coverage comes from Community Archive donors and your own accounts only. |
| **Customers' analytics** (post-launch) | Impressions, clicks, follows per post | Consented | The flywheel. This is also the only compliant path to a commercial model, because the USC corpus cannot ship in a product. |

**What changes without views in part of the data.** Community Archive posts have no impression counts. For those, the reach proxy is likes relative to followers and the training signal is pairwise (which of two posts from the same account within 7 days did better). On X, likes per impression run roughly 1 to 3%, so the proxy is noisy but consistent, and the literature's best content-only results come from exactly this pairwise setup. The USC set carries the absolute view targets and the live-trajectory data.

**Licensing consequence, stated once.** The USC data is non-commercial. The research model (phases 0 to 2) uses it. The model that ships to paying customers is retrained on Community Archive, donated analytics, Bluesky pretraining and customer data once those exist. The architecture, code and codebook are identical, so the retrain is a job, not a project.

## 6. Compute, time, size and budget

Hardware ranking for this project: **AWS credits first, the RTX 2060 second (free, useful), Mac mini last.** A 2060 handles SigLIP embeddings, tabular baselines and dataloading overnight. The 8B embedding encoder and the 8B teacher need 24 to 80 GB cards. A Mac mini can run the fusion model and a 2 to 4B VLM pilot but is 10 to 30x slower than an A100 for the big steps. If the credits can only be spent on AWS, the single-GPU `g6e.xlarge` (L40S 48 GB, about $1.32/h spot) does everything; otherwise an A100 80 GB on a marketplace at about $1.05/h is 2 to 3x cheaper for the teacher.

| Phase | Weeks | What ships | Compute | Dollars |
|---|---|---|---|---|
| 0. Data + baselines | 2 | API pull, media pipeline, Qwen3-VL embeddings for the corpus, SigLIP and Whisper features, train/val/test by time and author, LightGBM baseline with pairwise objective, metrics dashboard | 2060 for SigLIP/Whisper; one 24 to 48 GB GPU for ~30 h of Qwen3-VL-Embedding-8B inference | GPU $20 to $50 |
| 1. Predictor v1 + pre-flight | 3 | Fusion transformer (150 to 300M) over cached embeddings, all heads, calibration, component A, live forecaster v1 from trajectory polls | A10G or L4 spot, 6 to 10 sweep runs of 5 to 10 h | $50 to $80 |
| 2. Teacher + distillation | 3 | Qwen3-VL-8B QLoRA on 300 to 500k examples with the same heads; score the corpus; distill into v1 to get v2 | A100/H100 marketplace ~35 h, or L40S spot ~50 h | $100 to $150 |
| 3. Optimizer v1 | 3 | Deterministic levers, best-of-N rewriter with dual-model agreement, timing bandit, product API | A10G/L4 spot for generation and scoring | $50 to $100 |
| 4. RL editor + flywheel | ongoing | DPO on observed pairs, GRPO against B with KL and agreement constraints, customer analytics ingestion, weekly retrain | A100 marketplace, periodic | $100 to $200 per cycle |

Totals for phases 0 to 3: about **11 weeks** and **$250 to $450 of AWS credits** including storage. Everything runs inside AWS. The teacher phase uses `g6e.xlarge` (L40S 48 GB, about $1.32/h spot) rather than a marketplace A100, which costs about $60 more than the cheapest route but needs no cash.

How the plan scales with the credit balance:

| Credits available | What we do |
|---|---|
| Under $150 | Phases 0 and 1 in full (embeddings for 1M posts, fusion predictor, pre-flight, live forecaster). Skip the 8B teacher; use a 4B Qwen3-VL teacher on 150k examples if $40 remains. |
| $150 to $300 | Phases 0 to 2 with the 8B teacher on 300k examples on L40S spot. Optimizer v1 (deterministic levers and best-of-N) uses the same predictor so it costs little extra. |
| $300 to $500 | Full plan including sweeps, a second predictor backbone for the agreement guard, and corpus-wide teacher scoring for distillation. |
| Above $500 | Teacher on the full labeled corpus, the Bluesky pretraining pass, and the first DPO cycle of the RL editor. |

The 2060 and a Mac mini, if available, take the overnight jobs (SigLIP embeddings, Whisper, LightGBM, data packing) and stretch the credits by roughly a third.

Model size at the end of phase 2: about 9B frozen encoder parameters (Qwen3-VL-Embedding-8B, SigLIP-2 So400m, Whisper turbo), an 8B LoRA teacher used offline, and a 150 to 300M fusion model that serves. Serving cost: under $0.01 per 1,000 predictions on an L4 for the fusion model; about $0.03 to $0.10 per 1,000 if the teacher is used as a reranker on the top candidates.

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

- **The free corpus with views is non-commercial and topically skewed.** The USC set builds and validates the system but cannot ship in the product, and its political skew means the research metrics will be optimistic for political content and conservative for product launches. Community Archive, donated analytics and customer data are what the shipping model trains on.
- **The repo shows the structure, not the weights.** Our predictor learns from audience outcomes, not from Phoenix. It will disagree with Phoenix on individual posts. The value score and the normalization are X's; the probabilities are ours.
- **Weights drift.** `param.rs` is re-synced by cron and the reply boost changed twice in July 2026. We read weights from a config pinned to the repo commit and re-pull weekly.
- **Goodhart in the optimizer.** Mitigated by the dual-model agreement constraint, meaning-preservation constraint, KL penalty and, above all, measuring real outcomes through customer analytics.
- **Licensing.** All recommended encoders are Apache-2.0 or MIT. VideoMAE v2 (CC BY-NC) and DINOv3 (custom license) are excluded. The USC dataset is non-commercial and stays out of the product model.
- **The tail is luck.** Top-0.1% virality is dominated by exogenous events (a large account quote-tweeting, a news cycle). We will report precision honestly at that tier rather than claim it.

---

## 9. Decisions needed from you

1. **Credit balance.** Tell me the approximate AWS credit amount so I pin the plan to one row of the table in section 6.
2. **Media collection stance.** Bounded, rate-limited fetching of media referenced by the datasets, or donor and first-party media only.
3. **Donated analytics.** Export your own X analytics and ask anyone you can for theirs. Each account is a free, high-quality calibration set.
4. **Go for phase 0.** On approval I lay out the code: `virality/` package with ingestion for the USC and Community Archive formats, the faithful post renderer and embedding pipeline reusing `phoenix/reference/mm_encoder.py` and `sid_codebook.py`, the feature store, the baseline trainer and the evaluation harness.

## Appendix: research reports

- `research/01_phoenix_ranker.md`: full input contract of the ranking model, heads, losses, parameter counts, training data schema.
- `research/02_scoring_weights.md`: every weight and threshold in `param.rs` and `scoring.rs`, the DPP reranker, cold-start slot, popular-posts source.
- `research/03_distribution.md`: Thunder, retrieval index eligibility, SimClusters, every pre-scoring filter, lifecycle of a new post.
- `research/04_media_content.md`: media and text understanding, Grox classifiers, the multimodal embedding recipe, audio transcription.
- `research/05_visibility_suppression.md`: label table, suppression rules, user credibility, the 15 things not to do.
- `research/06_literature.md`: cascade prediction, cold-start multimodal popularity, LLM predictors, industry recsys, counterfactual optimization, RL versus supervised, datasets and legal.
- `research/07_compute_data.md`: GPU prices, feasibility matrix per hardware option, memory arithmetic, storage, encoder licenses, serving cost, phased budget.
