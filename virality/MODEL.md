# The model: architecture, training, and evidence

Companion to `ARCHITECTURE.md` (system stages) and `DATA_PIPELINE.md` (data and features). This document specifies the one trained component, the Phoenix surrogate (stage S4), as implemented in `xviral/model/`, and the ported X code it feeds (`xviral/xalgo/`). Section 9 reports what it scores on real posts.

---

## 1. What the model has to do

Given a draft post and an account, output a probability distribution over reach, the probability of each engagement action per impression, and a score that ranks drafts for the same account. Three facts about the problem decide the design:

1. **The account explains most of the variance; content explains the part a customer can change.** A model judged on raw accuracy learns the account and stops. So the model is built and judged on *within-account* skill: which of two posts by the same account did better.
2. **Each account's audience has its own taste.** What works for one account does not work for another. The model must condition on what this account's audience has rewarded before.
3. **The product reports odds.** A point estimate is not enough; the output has to be a calibrated distribution.

## 2. The central idea: Phoenix's layout, with the author's history as the audience's history

Phoenix scores a post for a viewer by reading the viewer's last 1,022 engagements: `[viewer tokens | history | candidates]`, with candidates isolated from each other (`phoenix/README.md`). We cannot have per-viewer histories. What we can have is the aggregate of them: **the outcomes of an account's earlier posts are its audience's collective history with that account.**

So an input is an *episode*, in Phoenix's layout:

```
[ context | history: the account's previous posts, each with its content and its outcome | candidates ]
              oldest ................................................. newest            drafts, isolated
```

- **Context (1 token):** the account's baseline, computed only from visible history: medians and 90th percentiles of likes and views, how much history exists, follower count.
- **History (up to 64 to 128 tokens):** each earlier post as one token: its content, its features, and what happened to it.
- **Candidates (1 to 8 tokens):** the drafts. Each attends to context, history and itself, never to another candidate, so a draft's score does not depend on which other drafts are scored with it. This is Phoenix's candidate isolation, kept for the same reason X keeps it.

The transformer then does in-context what a per-account model would do: it compares the draft to the account's own past posts and reads off how similar posts performed. A new customer needs no retraining; their history is the conditioning. This is also why there is no account-ID embedding: Phoenix has one because it serves known accounts, we score accounts the model has never seen.

## 3. Tokens

Every post, in history or as a candidate, becomes one token built the way Phoenix builds a candidate token: each feature group is projected to the model width and the projections are summed (`recsys_feature_prep.py::build_feature_prep_streams`).

| Group | Content | Source |
|---|---|---|
| Post embedding | 1024-d vector from `Qwen3-VL-Embedding-8B` over X's rendering of the post | X's own encoder (`phoenix/reference/mm_encoder.py`) |
| Semantic ID | 6 codes × 256, summed embeddings | X's RQ-KMeans trainer (`sid_codebook.py`); Phoenix's only content input |
| Numeric scalars | ~50 standardized features: text length in X's weighting, media and video scalars, follower count at post time, cadence, duplicate and suppression signals; missing-value indicators added automatically | `DATA_PIPELINE.md` 7.2 to 7.6 |
| Categoricals | link domain class, media aspect, video duration bucket, post kind, audience-local hour (25 classes as in Phoenix), weekday, follower tier | same |
| Image and frame embeddings | SigLIP-2 vectors, when present | 7.7 |
| Outcome (history and neighbours only) | Phoenix's count encoding, `log2(count+1)/32` for like, reply, repost, quote, view through a small MLP, plus presence bits (sources lack some counts) and the two follower-normalized ratios | `recsys_feature_prep.py:730-754` |
| Age | Phoenix's hourly post-age bucket (82 classes) | `recsys_feature_prep.py:55-72` |
| Time gap | Phoenix's log-spaced gap bucket (19 classes, edges 60·2^k s) between the post and the episode cutoff | `recsys_feature_prep.py:75-81` |
| Role | history, candidate or neighbour | |

**Neighbours.** For each candidate, up to 16 similar recent posts by *other* accounts (retrieved by embedding from earlier days only) are encoded as post tokens with outcomes and pooled into the candidate token by attention. This is the "what did posts like this do last week" signal. They are pooled rather than placed in the sequence because they differ per candidate, and candidates must not share them.

**A candidate token carries no outcome.** In cold-start scoring its count features are absent. The same slot accepts early counts and a post age, which is Phoenix's own candidate contract (counts and age at impression time) and is how the live forecast will work (section 6).

## 4. Trunk

Phoenix's block as configured in production (`phoenix/xrex/configs/xrecsys.py:1041-1077`), reimplemented in PyTorch at reduced width:

- pre-norm RMSNorm, eps 1e-5; final RMSNorm
- grouped-query attention (more query heads than key/value heads), RMS norm on queries and keys, rotary positions with base 10000
- non-gated feed-forward with widening 2, tanh-approximate GELU, no biases
- right-anchored positions: the newest history post sits next to the candidates, and all candidates share one position
- candidate isolation mask

| Preset | Width | Layers | History | Trainable parameters | Use |
|---|---|---|---|---|---|
| nano | 64 | 2 | 16 | ~0.2M | tests |
| small | 192 | 3 | 48 | ~2M | CPU experiments |
| base | 512 | 8 | 96 | ~25M | first GPU runs |
| large | 1024 | 12 | 128 | ~150M | full corpus |

Phoenix's production trunk is width 2560 with 8 layers, trained on billions of impressions. Ours is sized to hundreds of thousands of posts; the capacity that matters most sits in the frozen 8B encoder, and section 8 covers unfreezing it.

## 5. Outputs and losses

**Phoenix's action heads.** 64 sigmoid outputs indexed by Phoenix's `ActionName` enum, so the ported value model reads them by index. Trained on per-impression rates (likes/views and so on) where views exist, with soft-target cross-entropy weighted by impressions. Heads with no label in the data are untouched and unmask as analytics data arrives.

**Distributional heads.** Four categorical distributions over fixed bins in log10 units:

| Head | Target | Range | Trained on |
|---|---|---|---|
| `reach` | (views+1)/(followers+1) | 10^-4 to 10^4 | rows with views |
| `like` | (likes+1)/(followers+1) | 10^-6 to 10^2 | every row; the bridge between sources |
| `rel_view` | views over the account's own median views in visible history | 10^-3 to 10^3 | rows with views and ≥ 5 visible history posts |
| `rel_like` | likes over the account's own median likes | 10^-3 to 10^3 | rows with ≥ 5 visible history posts |

Targets are smoothed over neighbouring bins (HL-Gauss). Classification over bins is more robust than regression on noisy heavy-tailed outcomes and yields the whole distribution. X models its own heavy-tailed quantity, dwell time, with the cumulative form of the same idea (CREAD, `loss_recsys.py:110-187`).

Everything the product reports is read off these distributions: quantiles (p10, p50, p90), the probability of exceeding 1×, 10×, 100× followers (tail mass of `reach`), and the probability that the post lands in the account's own top 10% (tail mass of `rel_*` above the account's own p90). The `rel_*` heads do not depend on the follower count, which matters because some sources know it only at one point in time.

**Pairwise loss.** Among the candidates of one episode (same account, within 7 days), a logistic loss on the difference of predicted means against which post actually did better, ignoring near-ties. This is the literature's confound control for content effects and the direct training signal for "rank my drafts".

**Total loss** = distribution cross-entropies + action cross-entropy + pairwise loss, each row weighted by recency and by how many impressions back its label.

## 6. Leakage rules

Enforced in `episodes.build` and covered by tests that alter the data and check the outputs do not move:

1. History is strictly earlier in the account's timeline than the first candidate.
2. A history post's outcome is hidden if the post is under 24 hours old at the cutoff. At serving time that post's counts are still growing, so training must not see them final.
3. A neighbour's outcome is hidden under the same rule; neighbours not strictly earlier than the candidate are removed.
4. A candidate's own counts and labels never enter its token. The outcome encoder computes its ratios from the counts it is handed and never reads a label column.
5. The account baseline in the context token and in the `rel_*` targets comes from visible history only.
6. Test rows are later in time than all training rows; validation accounts are disjoint from training accounts.

**Live forecast.** The candidate slot accepts observed counts and an age. Training with early counts needs trajectories, which no current source has. `episodes.thin_counts` can simulate them with the saturation curve X itself uses to project 24-hour views (`home-mixer/util/popular_posts.rs::quality`), with the half-life drawn from a wide range. That is an assumption standing in for data, so it is off by default and is never used to report accuracy. It switches on when connected accounts supply real trajectories.

## 7. From the model to X's score

`xviral/xalgo/value_model.py` is a port of `xai-value-model/{weights,scoring}.rs`: the weighted sum over heads, the mutual-follow reply boost, the offset that pushes net-negative posts below every clean post, author-diversity decay and the out-of-network factor. Weights are parsed from `home-mixer/params/param.rs` and `vm-ranker/params.rs` at load time, so an upstream weight change arrives with a `git pull`. The model's 64 outputs times `head_weight_vector` is X's weighted score for the audience average, with the reply weight blended by the account's mutual-follow share.

## 8. Training plan with the larger compute budget

| Stage | What | Why |
|---|---|---|
| A (implemented) | Frozen embeddings, train the trunk and heads | Cheap, cacheable; establishes the numbers in section 9 with the real encoder |
| B | LoRA on `Qwen3-VL-Embedding-8B`, trained through the same episode loss with candidates encoded live and history from cache, refreshed per epoch | Fine-tuned encoders beat frozen ones on engagement prediction in every 2024 to 2026 study; the pairwise loss gives the encoder a direct "which post wins" signal |
| C | Ensemble of 3 to 5 seeds and sizes of stage B, distribution-averaged; a gradient-boosted model on tabular features as an independent second scorer | Calibration, and the agreement check the optimizer needs |
| D | Viewer-level model on donors' own likes (22.7M) with Phoenix's real layout | Validates audience behaviour against real individuals |

Stage A at `large` on 1M posts is a few GPU-hours. Stage B on an 8B encoder is the compute-heavy step: tens of GPU-hours on 80 GB cards per run.

## 9. Evidence

See the results section appended below by the experiment run.
