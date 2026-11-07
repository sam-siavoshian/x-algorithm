# Literature review: predicting (and editing for) virality of X/Twitter posts

Date: 2026-11-07. Scope: cascade/popularity prediction, cold-start multimodal popularity, LLM/VLM predictors, X's own ranker (including the open-sourced `xai-org/x-algorithm` repo checked out in this workspace), counterfactual content optimization, RL vs supervised, and data/legal realities.

Conventions: numbers quoted from abstracts/tables where I could fetch them; items marked **[mem]** are from memory of the paper and should be re-verified before being quoted in a design doc. MSLE = mean squared log error on log(1+count); SRC = Spearman rank correlation.

---

## A. Cascade / popularity prediction (early-observation models)

All of these need to *observe* part of the cascade first; none is cold-start. Typical protocol (Weibo): observe 0.5–1 h, predict the 24 h retweet count, keep only cascades with >= 10 observed retweets. That filter alone removes ~all ordinary posts, so reported MSLEs are for already-moving cascades.

| Paper | Venue/yr | Core method | Early window | Reported result | Data |
|---|---|---|---|---|---|
| Cheng, Adamic, Dow, Kleinberg, Leskovec, *Can cascades be predicted?* https://arxiv.org/abs/1403.4608 | WWW 2014 | Framed prediction as "given k reshares, will the cascade double?" (balanced, avoids heavy tail). Random forest / logistic regression on temporal, structural, content, root-user features. | k = 5 reshares observed | ~0.795 accuracy at k=5 (chance 0.5); temporal (speed of first k reshares) features dominate, content features weakest; accuracy rises with k. **[mem]** | Facebook photo reshares (~150K photos, ~9M reshares) |
| Zhao, Erdogdu, He, Rajaraman, Leskovec, *SEISMIC* https://arxiv.org/abs/1506.02594 | KDD 2015 | Self-exciting point process; final size = closed-form function of observed events weighted by follower counts; no feature engineering. | 10 min – 1 h | ~15% relative error on final retweet count after 1 h, ~25% after 10 min, "for reasonably popular tweets"; 30% better than prior SOTA. | 1 month of Twitter, cascades >= 50 retweets |
| Mishra, Rizoiu, Xie, *Feature driven and point process approaches for popularity prediction* https://arxiv.org/abs/1608.04862 | CIKM 2016 | Compares feature regression vs Hawkes; hybrid (Hawkes params as features) wins. **[mem]** | 10 min – 1 h | Hybrid beats either alone; features matter most at very short windows. | Twitter (news/links) |
| Rizoiu et al., *Hawkes processes for events in social media* (tutorial) https://arxiv.org/abs/1708.06401 ; Rizoiu et al. *Expecting to be HIP* https://arxiv.org/abs/1602.06033 | 2017; WWW 2017 | Hawkes/HIP: exogenous promotion + endogenous self-excitation; derives "virality" and "promotion sensitivity" per item. | days | HIP fits YouTube view series; interpretable but needs a series. | YouTube |
| Li, Ma, Guo, Mei, *DeepCas* https://arxiv.org/abs/1611.05373 | WWW 2017 | Random walks on cascade graph -> GRU + attention -> size increment. | 1–3 h (Twitter hashtags), years (AMiner) | First end-to-end cascade model; MSLE reproduced on Weibo 1 h: 2.74 (HierCas table). | Twitter hashtag cascades, AMiner citations |
| Cao, Shen, Shi, Zhou, Cheng, *DeepHawkes* (CIKM 2017) https://dl.acm.org/doi/10.1145/3132847.3132973 | CIKM 2017 | Hawkes ingredients (user influence, self-excitation, time decay) as differentiable modules. | 1 / 2 / 3 h | Weibo 1 h MSLE ~2.6 (repro); original paper ~2.44 **[mem]** | Weibo (119,313 posts, 1 June 2016), APS |
| Chen, Zhou, Zhang, Zhong, *CasCN* https://arxiv.org/abs/1810.05184 | ICDE 2019 | Dynamic GCN over cascade snapshots + LSTM. | 0.5–1 h | Weibo 1 h MSLE 2.60 (repro). | Weibo, HEP-PH |
| Xu, Zhou, Zhong, Trajcevski, *CasFlow* https://github.com/Xovee/casflow (TKDE 2021) | TKDE 2021 | Hierarchical (local cascade + global social graph) embeddings + VAE + normalizing flows; models uncertainty. | 0.5–1 h | Weibo 1 h MSLE 1.98 (repro); Twitter (MMCas data) 1 h MSLE ~1.51. | Weibo, Twitter, APS |
| Tang et al., *TempCas* (AAAI 2021) | AAAI 2021 | BiGRU + CNN over cascade sequence and temporal counts. | 1 h | Among best baselines in several follow-ups. | Weibo, APS |
| Chen et al., *MUCas* (2021) | 2021 | Multi-scale GCN on sub-cascade graphs + dynamic routing + attention. | 0.5–1 h | Weibo 0.5 h / 1 h MSLE 2.14 / 1.98 (repro). | Weibo, APS |
| Wang et al., *CasSeqGCN* https://arxiv.org/abs/2110.06836 | Inf. Sci. 2022 | GCN per snapshot + LSTM over snapshots. | 0.5–1 h | Weibo 0.5 h / 1 h MSLE 2.43 / 2.34 (repro). | Weibo, DBLP, APS |
| *HierCas* https://arxiv.org/abs/2310.13219 | 2023 | Temporal graph attention over the whole evolving graph (no snapshots). | 0.5 / 1 h | Weibo 0.5 h / 1 h MSLE **1.82 / 1.70**; APS 3y/5y 0.94/0.77. Table reproduces all baselines above. | Weibo (6.7M nodes), APS |
| *CasDO* (TKDE 2024) https://www.xoveexu.com/file/paper/24-09-TKDE-CasDO.pdf ; *ConCat* (TKDE 2024) https://arxiv.org/abs/2409.16623 | TKDE 2024 | Probabilistic diffusion + neural ODE (CasDO); neural ODE + temporal point process in continuous time (ConCat). | 0.5–1 h | Explicit predictive uncertainty; small MSLE gains over CasFlow. | Weibo, Twitter, APS |
| *AutoCas* https://arxiv.org/abs/2502.18040 | 2025 | Tokenizes cascades; autoregressive LLM-style modeling + prompt learning; claims scaling behaviour. | 0.5–1 h | "Significantly outperforms" baselines (no numbers in abstract). | Weibo, Twitter, APS |
| Jing et al., *MMCas: Modeling Multimodal Information Cascade with Interpretable MoE* (WWW 2026, DOI 10.1145/3774904.3793030) | WWW 2026 | Adds user profile + text + image experts to cascade structure. | 1 h | Twitter 1 h MSLE ~1.26 vs CasFlow 1.51, CasCN 1.76 -> content/profile modalities cut error ~17% *on top of* 1 h of cascade. | New Twitter multimodal cascade dataset (check release) |
| *Tabular foundation models for multi-view cascade popularity* https://arxiv.org/abs/2608.25048 | 2026 | TabPFN-style in-context learning over cascade/user/content views. | 1 h | Competitive with MMCas on the MMCas Twitter set. | Weibo, Twitter |

Takeaways for our design: (i) once you have ~1 h of engagement, final size is predictable to within a factor of ~e^sqrt(1.7) ~ 4x (MSLE 1.7) in the general case and ~15% for already-popular tweets; (ii) every published gain since 2017 is in *how* early signals are encoded, not in cold-start; (iii) content/author modalities still help even when cascade data is available (MMCas), which argues for a two-stage model: cold-start prior + early-engagement update.

---

## B. Content-only / cold-start popularity prediction

### B1. How much do content vs author/network explain? (Twitter-specific)

| Paper | Finding |
|---|---|
| Suh, Hong, Pirolli, Chi, *Want to be retweeted?* (SocialCom 2010) https://ieeexplore.ieee.org/document/5590452 | 74M tweets. URLs and hashtags are the strongest content correlates of retweeting; follower/followee counts and account age matter; number of past tweets does not. **[mem]** |
| Bakshy, Hofman, Mason, Watts, *Everyone's an influencer* (WSDM 2011) https://dl.acm.org/doi/10.1145/1935826.1935845 | 1.6M seed users, 74M URL cascades. Regression trees on seed features: past local influence + followers are the only reliable predictors; content ratings (interestingness, positivity) add almost nothing; most cascades are tiny; targeting "ordinary influencers" is most cost-effective. R^2 ~0.34 **[mem]**. |
| Petrovic, Osborne, Lavrenko, *RT to win!* (ICWSM 2011) https://ojs.aaai.org/index.php/ICWSM/article/view/14176 | 21M tweets, passive-aggressive classifier; social (author) features dominate tweet features; F1 ~47% on whether a tweet will be retweeted; humans about as good. **[mem]** |
| Tan, Lee, Pang, *The effect of wording on message propagation* (ACL 2014) https://arxiv.org/abs/1405.1438 | Same-author, same-URL tweet pairs (controls topic + author + follower count, pairs within 12 h, authors > 5K followers). Humans 61.3%; model with wording features + BoW 66.5% CV / 65.6% held-out. The canonical "wording matters, but only ~15 points over chance" result. |
| Martin, Hofman, Sharma, Anderson, Watts, *Exploring limits to prediction in complex social systems* (WWW 2016) https://arxiv.org/abs/1602.01013 | 852M tweets with URLs (Feb 2015), ex-ante prediction of retweet count. Content-only R^2 "negligibly low" (< 0.05 even with past URL success); basic user features R^2 ~0.2; adding the user's *past success* R^2 0.42–0.48; everything else adds nothing. Argues a hard ceiling well below R^2 = 1 due to intrinsic randomness. Also Hofman, Sharma, Watts, Science 2017 "Prediction and explanation in social systems". |
| Pfeffer et al., *The Half-Life of a Tweet* https://arxiv.org/abs/2302.09654 (2023) | Impression counts from the API (post-Dec-2022). Impressions/sec peak ~72 s after posting; median half-life ~80 min; ~95% of tweets receive essentially no impressions after 24 h. Implication: "viral" is decided in the first 1–3 h; early-signal update must be fast. |
| *Revealing the secret power: how algorithms influence content visibility on X* https://arxiv.org/abs/2410.17390 (2024, v4 2026) | >40M posts, >9M accounts with view counts (Ukraine war + 2024 US election). Posts with external links get up to ~8x fewer views regardless of stance/source reliability; visibility differs strongly by account. |
| *The significance of user characteristics for reposting prediction on X under distribution shift* https://arxiv.org/abs/2505.15370 (2025/26) | Individual-level repost prediction; DT/MLP/BERT/Qwen. Content features: F1 ~0.12 on unseen topics; user features: F1 > 0.70. "Reposting decisions are largely content-agnostic" at the individual level. |
| Belli et al., *Privacy-aware recommender systems challenge on Twitter's home timeline* (RecSys Challenge 2020) https://arxiv.org/abs/2004.13715 ; NVIDIA winning write-up https://medium.com/rapids-ai/winning-solution-of-recsys2020-challenge-gpu-accelerated-feature-engineering-and-training-for-cd67c5a87b1f | 160M tweets (200M user–tweet pairs), 4 engagement targets within the home timeline, BERT token IDs, media type, hashtags, author/engager follower counts. Winners (GBDT) relied on target-encoded user/author engagement history; text added modestly; no cold-start setting (engager known). Dataset no longer distributed. |

### B2. Pairwise / content-controlled studies (the trick that isolates content)

| Paper | Setup | Accuracy |
|---|---|---|
| Lakkaraju, McAuley, Leskovec, *What's in a name?* (ICWSM 2013) https://ojs.aaai.org/index.php/ICWSM/article/view/14421 | Same image re-submitted to Reddit with different titles/communities/times; separates content, title, community, timing. Title and community/timing effects large once content is held fixed. **[mem]** | — |
| Hessel, Lee, Mimno, *Cats and captions vs. creators and the clock* (WWW 2017) https://arxiv.org/abs/1703.01725 | Pairs of image posts to the same subreddit within seconds–minutes; linear ranker on feature differences. | Content (image+text) 62.7–67.7% vs creator features 54.6–68.4% vs timing-only ~50%. Content beats creator in 5/6 communities *when timing/community are controlled*. |
| Ding, Ma, Wang, *Intrinsic image popularity assessment* (ACM MM 2019) https://arxiv.org/abs/1907.01985 | Same-user Instagram image pairs; learning-to-rank CNN. | 76.65% pairwise accuracy, slightly above humans on a 1,000-pair psychophysical test; raw-likes regression baseline 66.5%. |
| Tan et al. 2014 (above) | Same author, same URL. | 66.5% vs humans 61.3%. |
| *Judging a book by its cover: marginal impact of title on Reddit popularity* https://arxiv.org/abs/2112.12191 (2021) | Attention model on titles controlling for posting time; title effect varies by community. | SOTA claimed, numbers in paper. |
| Upworthy headline A/B archive (32,487 randomized tests; Matias et al. 2021) — see C for LLM results | Randomized pairs, same article, same time. | Fine-tuned 8B LLM 82–84% on pairs with significant CTR differences; near chance on all pairs with prompting. |

Reading across these: once author + timing are controlled, content-only pairwise accuracy sits at **62–77%** (chance 50%), humans ~61%. That is the realistic ceiling for "which of these two drafts will do better" from text/image alone.

### B3. Multimodal post popularity (SMP Challenge, Instagram, micro-video)

| Paper | Venue | Method / data | Result |
|---|---|---|---|
| SMP Challenge (Wu et al.; summary ACM MM 2024 https://dl.acm.org/doi/10.1145/3664647.3688996) | ACM MM 2019–2025 | SMPD: ~486K Flickr posts, ~69K users, image + title/tags + rich user metadata; target log-popularity; metrics SRC, MAE. | Winners every year are GBDT ensembles (LightGBM/CatBoost) over user-level features + pretrained image/text embeddings; user/metadata features dominate. |
| *MMF: Winning solution SMP 2024* https://dl.acm.org/doi/10.1145/3664647.3688997 | ACM MM 2024 | Multimodal fusion + GBDT. | Best SRC/MAE in 2024. |
| *HyperFusion* https://arxiv.org/abs/2507.00926 (3rd, 2025 image track) | 2025 | Hierarchical multimodal ensemble. | SRC 0.7324, MAE 1.2402. (1st place numbers not located; same ballpark.) |
| *MVP: Winning solution SMP 2025 video track* https://arxiv.org/abs/2507.00950 | 2025 | Pretrained video features + user metadata -> GBDT. | MAPE 0.1754. |
| *Anchoring trends* https://arxiv.org/abs/2507.19863 | 2025 | Temporal anchoring + feature clustering to fight drift. | MAPE 17.47% public / 19.58% private. |
| Cheng et al., *RAGTrans: retrieval-augmented hypergraph for multimodal SMPP* https://dl.acm.org/doi/10.1145/3637528.3672041 | KDD 2024 | Retrieve similar historical posts into a multimodal hypergraph transformer. | Beats non-retrieval baselines; later work says gains are coarse. |
| Xu et al., *SKAPP: selective retrieval knowledge augmentation* https://ojs.aaai.org/index.php/AAAI/article/view/32078 ; follow-up *Learning to curate context* (AAAI 2026) | AAAI 2025/26 | Meta retriever + selective refiner (drop retrieved posts that don't help). | 26.7–48.2% improvement across metrics on three datasets (code github.com/Xovee/skapp). |
| Xie/Zhu/Chen, *MMVED* https://arxiv.org/abs/2003.12724 ; *HMMVED* (TKDE) | WWW 2020 / TKDE | Variational multimodal encoder–decoder; stochastic embeddings; product-of-experts fusion. | Standard micro-video baselines; nMSE ~0.75–0.85, SRC ~0.45–0.55 on NUS (Vine) / Xigua sets **[mem]**. |
| Zhong et al., *MMRA: micro-video popularity via multimodal retrieval augmentation* https://github.com/ICDM-UESTC/MMRA | SIGIR 2024 | Retrieval of similar videos + attention. | Third-party re-eval: nMSE 0.807, SRC 0.480, PCC 0.458. |
| *SHORTS-CAST / WEBSHORTS: Will it go viral? grounding micro-video popularity on the open web* https://arxiv.org/abs/2605.18653 | 2026 | "Evidence card" of open-web trend context at upload time + LLM rationales + online adaptation on delayed labels. 14K videos, 7-day view curves. | Beats content-only and corpus-retrieval baselines; trend context matters more than content. |
| *OmniTrend* https://arxiv.org/abs/2604.26252 | 2026 | Factorizes popularity = content attractiveness x contextual exposure (posting time, author activity, trends, retrieval neighbourhood). | Transfer across platforms claimed. |
| *SMTPD* https://arxiv.org/abs/2503.04446 | 2025 | YouTube multilingual temporal popularity benchmark. | "Early popularity and temporal alignment are crucial". |
| *MMG-Pop / MMG-PopNet benchmark* https://arxiv.org/abs/2606.27539 | 2026 | Unified benchmark (Bluesky + Reddit), multimodal + reply graph; zero-shot / RAG / fine-tuned LLM vs graph model. | Graph model best; documents "LLM prediction limitations". |

### B4. Typical ceilings (what to expect)

| Information available | Metric | Typical range (sources) |
|---|---|---|
| Content only, absolute counts | R^2 on log count | < 0.05 (Martin 2016); SRC ~0.2–0.4 on SMPD when user features ablated **[mem]** |
| Content only, author/time-controlled pairwise | pairwise acc. | 62–77% (Tan 2014, Hessel 2017, Ding 2019, Upworthy-significant 82–84%) |
| + author features (followers, past success) | R^2 / SRC | R^2 0.42–0.48 (Martin 2016); SRC 0.70–0.78 (SMPD leaderboards) |
| + 10 min early engagement | rel. error | ~25% for popular tweets (SEISMIC) |
| + 1 h early engagement | MSLE / rel. error | MSLE 1.7–2.6 Weibo, ~1.26 Twitter multimodal (MMCas); ~15% rel. error on popular tweets (SEISMIC) |
| Zero-shot LLM, top-5% upvote classification | AUROC | ~0.53 (Vastu 2026) |

---

## C. LLM / VLM-based engagement prediction (2023–2026)

| Paper | Venue | Method | Data | Result / verdict |
|---|---|---|---|---|
| El-amrany et al., *Reddit-V* https://aclanthology.org/2025.ranlp-1.41/ | RANLP 2025 | 27K Reddit image posts; "viral" = top 20% upvotes & comments; zero-shot instruction LLMs vs fine-tuned CLIP / IDEFICS. | Reddit | Zero-shot LLMs poor; fine-tuned CLIP +3 pts, IDEFICS +7 pts over best zero-shot LLM; up to +10 F1 over text-only prompting. Fine-tuning > prompting; images help. |
| *Vastu: LMs struggle to recognize online community values* https://arxiv.org/abs/2601.12491 | 2026 | Predict top-5% comments; zero-shot vs LoRA SLMs. | Reddit | GPT-5-mini zero-shot AUROC 0.53 / F1 0.20; fine-tuned small models clearly better. |
| Ye, Yoganarasimhan, Zheng, *LOLA* https://arxiv.org/abs/2406.02611 | 2024 (Mgmt Sci. R&R) | Upworthy A/B archive; prompting vs embeddings+regression vs LoRA fine-tune; then LLM prior + Thompson sampling. | 32K headline tests | Pure LLM picks best headline only 46.9% (multi-arm; chance lower); prompting ~chance; LLM+bandit beats both pure LLM and pure bandit on clicks. |
| MSI working paper 24-150 (Upworthy pairwise) https://thearf-org-unified-admin.s3.amazonaws.com/MSI_Report_24-150.pdf | 2024 | 17,681 headline pairs with significant CTR gaps; fine-tuned Llama-3-8B. | Upworthy | 82–84% pairwise; prompting near chance. |
| Synerise 2025 (title rewrites) — cited in search | 2025 | Fine-tuned BERT pairwise 74% vs GPT-4o near random. | news titles | Small fine-tuned encoder > frontier zero-shot. |
| Chen et al., *Predicting social media popularity with LLMs: transforming metadata into semantic-enriched text* (IEEE Access 2024) | 2024 | Serialize SMPD metadata + captions into text; LoRA fine-tune. | Flickr SMPD | "Satisfactory"; does not beat GBDT winners by a wide margin (survey). |
| LLM instruction tuning + DPO human alignment for SMPP (2024; cited in Springer 2026 survey https://link.springer.com/article/10.1007/s44443-026-00909-x) | 2024 | LoRA + improved DPO on popularity preferences. | SMPD | Competitive, not dominant. |
| *BuzzProphet: enriching hashtag popularity prediction with LLM reasoning* https://arxiv.org/abs/2510.08481 | CIKM 2025 | LLM writes rationales (virality, reach, timing) -> features for a regressor; HashView benchmark (7,532 hashtags). | Twitter hashtags | RMSE −2.8%, correlation +~30% vs baselines. LLM-as-feature-extractor, modest. |
| *Can LLMs simulate social media engagement?* https://arxiv.org/abs/2502.12073 | 2025 | Predict retweet/quote/rewrite action; frozen vs fine-tuned BERT vs GPT-4o-mini/o1-mini/DeepSeek-R1 prompting. | Twitter | Fine-tuned BERT ~57% acc, prompting lower. |
| *PopALM* (LREC-COLING 2024) https://lrec.elra.info/lrec2024-main-1127 ; *RePALM* (ACL Findings 2024) https://aclanthology.org/2024.findings-acl.570/ | 2024 | GPT-2-based popularity *reward model* trained on tweet pairs (>= 10 interaction gap, < 12 h apart, same thread) then PPO. | Twitter quote tweets | Pairwise reward model + curriculum PPO; evaluates on predicted popularity (not live). |
| *AutoCas* (above) | 2025 | LLM backbone over tokenized cascades. | Weibo/Twitter | Needs early cascade; not cold-start. |
| *HotComment* https://arxiv.org/abs/2604.25614 | 2026 | 1.4M content–comment pairs (NetEase/Tencent/Bilibili); popularity-prediction track for LLMs/MLLMs. | Chinese news/video | Benchmark; LLMs mediocre at popularity, fine with quality. |
| *MMG-Pop* (above) | 2026 | Zero-shot / RAG few-shot / SFT LLM vs graph model on Bluesky+Reddit. | — | Graph model wins; SFT LLM with early cascade is competitive but not best. |
| *Assessing capabilities of LLMs in social media* https://arxiv.org/pdf/2604.18955 | 2026 | Survey/eval incl. engagement prediction. | — | Same pattern. |

Verdict: (1) **Zero-shot frontier LLMs are near chance** at absolute virality and only modestly above chance at pairwise (Upworthy, Reddit-V, Vastu). (2) **Fine-tuned models — even 100M-parameter encoders — beat prompting by a wide margin**; fine-tuned 8B LLMs reach 82–84% only on *controlled pairs with decisive outcomes*. (3) Against strong tabular/GBDT or graph baselines with author features, LLM/VLM content encoders add a few points of SRC, not a regime change. (4) The useful LLM roles are: content encoder (embeddings / LoRA head), rationale generator for features (BuzzProphet), retrieval of similar past posts (SKAPP/RAGTrans), and *pairwise* reward model (PopALM/RePALM). (5) Same-author pairing is the established trick to cancel follower-count confounding: Tan 2014 (same URL, < 12 h), RePALM (same thread, < 12 h, >= 10 gap), Ding 2019 (same user), Hessel 2017 (same subreddit, seconds apart).

---

## D. Industry recommenders relevant to X

### D1. Twitter 2023 open-source stack
- *Twitter's Recommendation Algorithm* blog (31 Mar 2023) https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm ; code github.com/twitter/the-algorithm and the-algorithm-ml (AGPL-3.0). Pipeline: ~1,500 candidates per request, ~50% in-network (RealGraph-ranked) / 50% out-of-network (SimClusters, TwHIN, GraphJet); Light Ranker (logistic) -> Heavy Ranker, a ~48M-parameter parallel **MaskNet** over ~6,000 features predicting ~10 engagement probabilities, combined with fixed weights (reply ~13.5, reply-engaged-by-author ~75, like ~0.5, retweet ~1, profile click ~12, negative feedback/report large negatives **[mem]**); heuristics: author diversity, OON penalty, social proof, visibility filtering.
- Wang, She, Zhang, *MaskNet* https://arxiv.org/abs/2102.07619 (DLP-KDD 2021): instance-guided masks for feature-wise multiplication in CTR models.
- El-Kishky et al., *TwHIN* https://arxiv.org/abs/2202.05387 (KDD 2022): heterogeneous information-network embeddings for users/tweets/ads; used for candidate generation and as ranking features.
- Satuluri et al., *SimClusters* (KDD 2020) https://dl.acm.org/doi/10.1145/3394486.3403370 : ~145K communities from the follow bipartite graph; tweets embedded by which communities engage; still a candidate source in the 2026 repo.
- Kamath et al., *RealGraph* (WWW 2014) https://dl.acm.org/doi/10.1145/2566486.2568043 : predicts user–user interaction probability from the follow graph + interactions; drove in-network ranking.
- Sharma et al., *GraphJet* (VLDB 2016) — real-time bipartite engagement graph random walks.

### D2. X 2026: Phoenix (from `xai-org/x-algorithm`, checked out at /home/user/x-algorithm; Apache-2.0; open-sourced 20 Jan 2026, production model code shipped 13 Aug 2026, last README update 26 Oct 2026)
Facts read directly from the repo (README.md, phoenix/README.md, home-mixer/params/param.rs, xai-value-model/scoring.rs, phoenix/xrex/models/recsys_feature_prep.py, phoenix/xrex/data/recsys/feature_config.py):
- **Architecture.** Two-tower retrieval (user tower = transformer over the viewer's engagement history + one profile token; *no learned per-user ID*; candidate tower = **semantic IDs** = 6-level x 256-code residual quantization of each post's multimodal embedding + hashed author ID). Ranking = Grok-architecture transformer over [user tokens | history (posts + authors + actions + SIDs + context + dwell) | candidates], with **candidate isolation** (candidates attend only to the viewer context) and **hash-based embeddings** (new posts representable immediately). Outputs logits for every action + dwell regression (losses: BCE multi-hot, MSE/MAE/Huber, "cread" classification+restoration, Tweedie).
- **Candidate-side features.** Hashed post ID, hashed author ID, SIDs (content/topic), post-age bucket, product surface, viewer timezone / local hour, engagement-count buckets (fav/reply/repost/quote/**view**), author-is-NSFW, is-author-followed-by-viewer / following-viewer (off by default in config), search match features. **No hand-crafted text features** (hashtags, length, sentiment): content reaches the ranker only via the multimodal embedding -> SIDs, i.e., *topic/semantic similarity to what the viewer engaged with*. Viewer-side: country, language, location, gender, age bracket, installed apps, IP hash.
- **Score = sum_i w_i * P(action_i)**, weights (param.rs, Oct 2026): favorite 0.708, reply 5.0 (+15.0 when the viewer and author mutually follow — the July 2026 "bidirectional follow boost", A/B tested at 5/10/15/20), retweet 1.0, quote 5.0, share 2.118, share-via-DM 5.0, share-via-copy-link 20.0, follow-author 4.0, click 0.274, open-link 0.2, photo-expand 0.05, dwell 0.05, continuous dwell time 0.0075, click-dwell-time 0.385, post-unexplored 0.015; video-open / VQV / profile-click / grok-analyze 0.0; negatives: not-interested −53.07, block −31.2, mute −58.8, **report −234**, not-dwelled −0.02. The repo stresses weights multiply *predicted probabilities*, not counts.
- **Post-score adjustments.** Author diversity: 2nd+ post from the same author x (0.75 * 0.5^k + 0.25); out-of-network multiplier **0.75** (also applied to in-network replies/retweets); **New-author/cold-start boost**: posts < 2 h old with < 200 impressions from authors with <= 50,000 followers are lifted toward slot 15–16, using **Thompson sampling with a Beta(0.75, 49.25) prior** (i.e., a ~1.5% prior engagement rate) — a production bandit for exploration of new content.
- **Filters.** Posts older than 48 h are never ranked; OON retweets/replies dropped; NSFW SimClusters OON dropped; `popular_posts_source` injects a shared list of popular recent posts "initially from the most-followed active accounts"; `NewUserMinEngagementFilter` hides low-engagement OON posts from new users.
- **Content understanding (Grox).** Runs at publish time: spam/adult/violence classifiers and text+image numeric representations (CLIP-trained). Grox prompts are withheld. `user-cred-v2` = PageRank over follow + engagement edges -> per-account credibility; `agatha` = account labels from block/report/spam ratios relative to favorites.
- **Retrieval/ranking objective changes (Oct 2026):** ranking now predicts time spent better; retrieval trained to surface posts people read all the way through.

How to approximate Phoenix externally (for a creator-side model): the ranker's inputs reduce to (a) *who the viewer is and what they engaged with*, (b) *the post's topic embedding*, (c) *the author identity*, (d) *post age and current engagement counts*. A creator cannot see (a) per viewer, but the expectation over the author's reachable audience is approximable by: follower count and follower quality (proxy for user-cred PageRank), the author's historical per-post engagement rates per action type (proxy for the learned author-hash embedding), the cosine similarity between the post's embedding and the author's past high-performing posts / the audience's interests (proxy for SID match), presence of external link (open-link weight 0.2 and observational 8x view penalty), predicted negative-feedback propensity (controversy -> report/not-interested terms with huge negative weights), media type (video continuation / photo-expand heads), and reply-bait structure (reply weight 5, x4 for mutuals). Early impressions/engagements within the first 30–120 min are directly consumed (count buckets) and gate the cold-start boost.

### D3. Generative / scaling-law recommenders
- Zhai et al., *Actions speak louder than words: trillion-parameter sequential transducers (HSTU)* https://arxiv.org/abs/2402.17152 (ICML 2024): generative recommender over user action sequences, up to 1.5T params, +12.4% online metrics, 285x faster than FlashAttention baselines on long sequences; scaling-law behaviour. Phoenix's history-token design is in this family.
- Zhang et al., *Wukong: towards a scaling law for large-scale recommendation* https://arxiv.org/abs/2403.02545 (ICML 2024): stacked factorization machines give power-law scaling across two orders of magnitude.
- Kuaishou, *OneRec technical report* https://arxiv.org/abs/2506.13695 (2025): single encoder–decoder replaces retrieval/pre-rank/rank, items as semantic-ID token sequences, RL preference alignment with a reward model; serves 25% of QPS, +0.54%/+1.24% app stay time, OPEX 10.6% of the cascade. Semantic IDs (also used by Phoenix) make *topic* the unit of content representation.

---

## E. Counterfactual content optimization, Goodhart, calibration

| Paper | Approach | Evidence |
|---|---|---|
| RePALM (ACL Findings 2024), PopALM (LREC-COLING 2024) | PPO against a pairwise popularity reward model (GPT-2) + consistency reward with simulated reader responses; curriculum to handle noisy like labels. | Improves reward-model popularity; no live deployment. |
| *RL-TweetGen* https://doi.org/10.3390/jtaer20030218 (2025) | LLM tweet generation with an XGBoost engagement regressor as reward. | Gains on predicted engagement only. |
| *Amplifying your social media presence: personalized influential content generation with LLMs* https://arxiv.org/abs/2505.01698 (2025) | Content-centric and structure-aware prompts (network info in prompt) to rewrite Weibo posts. | +21.79% *estimated* influence (repost predictor + independent-cascade simulation), not live. |
| *The impact of LLMs on content quality in social media* https://www.mdpi.com/2079-9292/15/13/2820 (2026) | Rewrite-to-maximize-engagement prompts; explicit quality penalty lambda = 0.5. | Documents engagement/quality trade-off. |
| *CHASE: how content ecosystems are reshaped when ranking is the only target* https://arxiv.org/abs/2608.30466 (2026) | 20 rounds of creators rewriting documents to climb an LLM ranker, six domains. | Quality–rank alignment fell in all six domains (mean Delta Spearman −0.068); drift caused by ranking incentives, not rewriting per se. |
| Meta, *CharacterFlywheel* https://arxiv.org/pdf/2603.01973 (2026) | Production RLHF loop with engagement goals. | Reward-model win rate rose to 70.7% while real engagement fell; they now cap win-rate at < 65% as a Goodhart guard. |
| Gao, Schulman, Hilton, *Scaling laws for reward model overoptimization* https://arxiv.org/abs/2210.10760 (ICML 2023) | Gold vs proxy RM; BoN and RL vs KL. | Proxy reward keeps rising while gold reward peaks then falls; peak occurs at larger KL for bigger RMs/more data; BoN reaches the peak at smaller KL than RL. |
| Wen et al., *Rethinking reward model evaluation* (ICLR 2025) https://mlanthology.org/iclr/2025/wen2025iclr-rethinking/ | RM accuracy does not predict overoptimization (regressional Goodhart). | — |
| LOLA (above) | Predictor as prior, bandit for truth. | Combining beats either. |
| Calibration/uncertainty: Martin 2016 (irreducible noise), CasFlow (VAE/flows), CasDO (diffusion), ConCat (TPP intensities), Phoenix (Tweedie / threshold-classification heads) | — | No conformal-prediction paper specific to cascades found; practice is log-transform + heavy-tail-aware losses (Tweedie/Huber/quantile) and ordinal "will exceed threshold k" heads (Cheng 2014; Phoenix `cread`). |

Design implications: use the predictor as a *ranker of candidate edits* under a tight edit budget (best-of-N with semantic-similarity / edit-distance constraint, N small), never as an RL reward for unconstrained generation; hold out an independent "gold" check (a second predictor trained on disjoint data/architecture, or live A/B) and watch for divergence; report calibrated exceedance probabilities (P(top 1%), P(top 0.1%)) with abstention when the author has too little history.

---

## F. Where RL belongs (candid)

- **Supervised learning is the right tool for the predictor.** Labels are abundant and offline; the target is observational. Every SOTA system above (SMP winners, Phoenix, HSTU) is supervised multi-task. Pairwise/preference losses on same-author pairs are still supervised.
- **Offline preference learning (DPO-style) on observed same-author pairs** is the cheap, stable way to get an "editor" that prefers higher-engagement phrasing — it is RL in name only and avoids reward hacking because the "reward" is real outcomes, not a proxy model.
- **RL with a learned engagement reward (PPO/GRPO) is where the evidence is weakest**: RePALM/PopALM/RL-TweetGen only show gains on the proxy; CharacterFlywheel and CHASE show proxy gains decoupling from real engagement. If used, constrain with KL + edit budget and stop early (Gao 2023), and validate live.
- **Bandits are appropriate where real feedback is cheap and repeated**: posting time (Spasojevic et al., *When-to-post on social networks*, KDD 2015 https://arxiv.org/abs/1506.02089 — personalized schedules from audience reaction times, deployed to millions; LSE blog of it notes no single best time), headline/variant selection (LOLA: LLM prior + Thompson sampling), and X itself runs Thompson sampling for cold-start posts. A contextual bandit over {post now vs. in k hours, variant A/B} with the supervised predictor as prior is sound; a full MDP is not needed.
- **Not RL**: cascade size prediction, feature attribution, "what makes posts viral" analysis.

---

## G. Data, API, legal

### G1. X API v2 (official pricing page https://docs.x.com/x-api/getting-started/pricing , read 2026-11-07)
- **Pay-per-usage is the default** (announced 6 Feb 2026 per vendor blogs): prepaid credits; **$0.005 per post read** (also lists/spaces/communities), **$0.010 per user / follower / following record**, $0.001 per like/mute/block; writes $0.015 per post create (**$0.200 if it contains a URL**), $0.010 for counts-all/trends. Resources deduplicated within a 24 h UTC window. **Capped at 3 million post reads per monthly billing cycle**; beyond that Enterprise (custom; vendors report ~$42K+/month).
- Legacy tiers (vendor sources; verify in the developer console): Free (write-only, 500 posts/mo) retired with a one-time $10 voucher; **Basic $200/mo (15K post reads, 50K writes)** migrated to pay-per-use from 1 Jun 2026; **Pro $5,000/mo (1M reads, filtered stream)** reported deprecated Aug/Sep 2026 but still active for some grandfathered accounts. Cost arithmetic: 1M posts = $5,000; 3M = $15,000/month; hydrating 100K posts + 100K authors = $1,500.
- Impression counts: `public_metrics.impression_count` exists for posts created after **15 Dec 2022** (views public on the web since then). Views include any load of the post, so views/follower is the natural "reach" normaliser.

### G2. Terms / legality
- X ToS (effective 15 Nov 2024, still present in current terms): scraping without written consent prohibited; **liquidated damages of $15,000 per 1,000,000 posts** accessed in any 24 h period in breach of the terms; Knight Institute criticised it as chilling research. Developer agreement limits redistribution of content to IDs (historically <= 1.5M tweet IDs / 30 days **[mem]**), which is why academic X datasets are ID lists that now cost $0.005/post to hydrate.
- *X Corp v. Bright Data* (N.D. Cal., No. 3:23-cv-03698): May 2024 dismissal (contract/tort claims pre-empted by copyright; no harm shown); Nov 2024 partial revival on server-access and UCL fraud theories; tweet-misappropriation claims stayed dead. *X Corp v. CCDH* dismissed Mar 2024 (anti-SLAPP). *hiQ v. LinkedIn* (9th Cir. 2022): CFAA does not reach public pages, but hiQ later lost on breach of contract. Net: logged-out scraping of public posts is not a crime in the US, but it is a contract/ToS exposure with an explicit damages clause; the compliant path is the API or user-donated archives.

### G3. Datasets with engagement counts (and whether media/views are available)

| Dataset | Size / dates | Engagement | Media | Views | Access / licence |
|---|---|---|---|---|---|
| **Community Archive** https://github.com/TheExGenesis/community-archive , https://community-archive.org | User-donated full X archives (accounts: a few thousand, mostly tech/"tpot" cluster; tweet count in the tens of millions — verify in the manifest) | `favorite_count`, `retweet_count`, text, entities, reply fields, followers/following lists, likes | No media files (only avatar/header URLs); media URLs recoverable from entities | **No** (archives lack impression counts) | Bulk Parquet on GitHub releases; keyed API v1 + semantic-search API (legacy Supabase REST retires 2026-11-19); stated goal "public-domain dataset", code MIT. Best source for *same-author longitudinal* pairs. |
| **USC X 2024 US election** (Balasubramanian et al.) https://arxiv.org/abs/2411.00376 , github.com/sinking8/usc-x-24-us-election | ~46M posts (46 parts x ~1M), 1 May–30 Nov 2024, custom scraper | likeCount, replyCount, retweetCount, quoteCount | `media` field (images/videos) | **Yes (viewCount)** | csv.gz on GitHub; README says CC BY-NC-SA 4.0 (sidebar shows CC0 — check). Largest open X set with views; topic-biased. |
| *Revealing the secret power* dataset https://arxiv.org/abs/2410.17390 | >40M posts, 9M accounts (Ukraine war + election) | yes | — | Yes | Via authors. |
| OSINT Russo-Ukrainian war tweets https://arxiv.org/abs/2409.01052 | Jan 2022–Jul 2023 | yes | links | Yes (post-Dec-2022 part) | Check paper. |
| MMCas Twitter multimodal cascades (WWW 2026) | cascades with user profile + text + image | cascade timestamps | images | no | Check authors' release. |
| MiDe22 https://arxiv.org/abs/2210.05401 | EN+TR misinformation tweets, 2020–22 | 4 engagement types | image links | no | Academic. |
| TwiBot-22 (NeurIPS 2022 D&B) https://twibot22.github.io | 1M users, 88M tweets (2022) | public_metrics | media URLs | no (pre-views) | By request, research only. |
| Kaggle scrapes (Feb/Mar 2024, "fastcurious") | 3–4K tweets | yes | some | Yes | Unlicensed scrape; toy only. |
| RecSys Challenge 2020/2021 (Twitter) | 160M / ~1B pairs | 4 targets within 24 h | media *type* only | no | No longer distributed. |
| Weibo (DeepHawkes) / Twitter15-16 / APS | cascades | retweet graphs | no | no | Public; cascade models only. |
| SMPD (Flickr) https://smp-challenge.com | 486K posts, 69K users | views-based popularity | image URLs (downloadable) | n/a | Challenge licence. |
| Reddit-V | 27K image posts | upvotes/comments | yes | no | Research. |
| Upworthy Research Archive https://osf.io/jd64p/ | 32,487 headline A/B tests, 150K headlines | clicks/impressions (randomized!) | thumbnails partly | n/a | CC-BY-ish, open. The only *causal* text-variant dataset. |
| Bluesky (Failla & Rossetti 2024) https://arxiv.org/abs/2404.18984 , Zenodo | 235M posts, 4M users; likes/reposts/replies/follows | yes | fetchable via AT Protocol | no | Open; ideal substitute for free large-scale text+image+engagement with an open API. |
| MicroLens (2023) https://arxiv.org/abs/2309.15379 ; NUS micro-video (Vine); WEBSHORTS (14K, 2026) | 1B interactions / 1M videos (MicroLens) | views/likes | raw video | yes | Research. |

Practical note: because every X source except user-donated archives is either paid ($0.005/post) or ToS-exposed, the cheapest *compliant* path to a multi-million-post X training set with views is: Community Archive for same-author history (no views) + a purchased pay-per-use slice (<= 3M posts/month) for recent posts with `impression_count` + the USC election set for scale (views, media, NC licence).

---

## H. Synthesis

### H1. What accuracy can a content+author model realistically reach for "viral vs not"?
Define viral_1 = top 1% and viral_01 = top 0.1% of impressions / follower count (author-normalised), measured at 24–48 h (Phoenix drops posts after 48 h; 95% of impressions arrive in 24 h).

- **Cold-start (text + media + author history, no engagement yet).** The literature puts content-only pairwise discrimination at 62–77% and R^2 on absolute counts < 0.5 even with full author history (Martin 2016). Translating to a 1%-prevalence classifier: expect **AUROC ~0.72–0.82** and **precision@1% of roughly 5–15% (5–15x lift over base rate)** with a well-tuned GBDT/transformer that uses the author's per-post engagement distribution (the single most predictive feature family), topic/semantic embedding, media type, link presence, reply-bait structure, and posting time. Author-normalisation removes the follower-count shortcut, so content carries *more* of the signal than in absolute-count tasks, but the irreducible randomness documented by Martin/Watts means most variance stays unexplained. For **top 0.1%**, expect AUROC to stay similar but **precision@0.1% in the 1–3% range**; the tail is dominated by exogenous events (who quote-tweets it, trend timing) that no content model sees. Zero-shot LLM judgement adds nothing measurable (AUROC ~0.5–0.6).
- **With early engagement (10–60 min of impressions/likes/replies).** SEISMIC/CasFlow-class evidence says the problem becomes easy: **AUROC 0.90+ for viral_1 and ~15–25% relative error on final size for posts already moving**. The Pfeffer half-life (80 min) means a 30-minute checkpoint already captures ~25–30% of lifetime impressions.
- **Pairwise edit evaluation ("is draft B better than draft A for this author?").** Realistic ceiling **65–75% accuracy** on same-author pairs with decisive outcomes (Tan 2014, Hessel 2017, Ding 2019), up to ~80% if pairs are filtered to large outcome gaps (Upworthy 82–84%). Humans ~61%.

### H2. The 10 most load-bearing design decisions (evidence-backed)
1. **Normalise the target by author reach** (views / followers, or residual vs the author's own trailing distribution). Martin 2016, Bakshy 2011, 2505.15370: raw counts are 80–90% author; without normalisation the model learns follower count and nothing about content.
2. **Train the content component on same-author pairs** (same author, within 7–30 days, decisive gap) with a pairwise/ranking loss, in addition to the pointwise head. Tan 2014, Ding 2019, Hessel 2017, RePALM pairs, Upworthy 82–84%. This is what makes an "edit suggester" possible at all.
3. **Two-stage model: cold-start prior + early-engagement update** (10/30/60-min checkpoints, Hawkes-style or count buckets). Cheng 2014, SEISMIC, CasFlow/HierCas, MMCas, Phoenix count-bucket features; prediction quality jumps from AUROC ~0.75 to 0.9+.
4. **Author features = per-action engagement *rates* history + follower count + follower quality/credibility proxy**, not just follower count. Martin 2016 (past success R^2 0.42–0.48 vs 0.2); Phoenix user-cred PageRank and author-hash embeddings.
5. **Represent content with a frozen multimodal embedding + small fine-tuned head (or LoRA) — not zero-shot prompting.** Reddit-V, Vastu, LOLA/MSI, Synerise; Phoenix itself consumes only a multimodal embedding via semantic IDs, so topic-level similarity to the author's audience is the operative content signal.
6. **Add retrieval of similar recent posts (by the same author and globally) as features** (kNN popularity stats, trend saturation). RAGTrans, SKAPP (+27–48%), SHORTS-CAST/WEBSHORTS, OmniTrend: exposure context explains more than content polish.
7. **Model negative-feedback propensity explicitly.** Phoenix weights: report −234, mute −58.8, not-interested −53.07 vs like 0.708; a post that raises P(report) by 0.3% loses more score than a 100% lift in P(like). Train auxiliary heads for controversy/NSFW/spam-likeness.
8. **Encode platform mechanics as features/constraints**: external link (open-link weight 0.2; observational 8x view penalty), reply/quote-eliciting structure (reply 5.0, quote 5.0, x4 for mutuals), media type (video continuation, photo-expand), post age (48 h cutoff), cold-start boost eligibility (<= 50K followers, first 2 h, < 200 impressions), local-hour of the audience.
9. **Use heavy-tail-aware objectives and calibrated exceedance heads**: predict log(views/followers) with Huber/Tweedie *and* ordinal "exceeds top-10% / 1% / 0.1%" heads (Cheng 2014's doubling framing; Phoenix `cread` head); calibrate with isotonic/conformal on held-out *authors* and report intervals.
10. **Edit suggestion = constrained best-of-N re-ranking with an independent check, not RL against the predictor.** Gao 2023 (BoN overoptimises slower than RL), CharacterFlywheel (proxy up, engagement down), CHASE (ranker-chasing degrades quality). Keep N small, bound semantic drift, prefer DPO on *observed* same-author pairs if a generator is trained, and close the loop with live posting-time/variant bandits (LOLA, Spasojevic 2015).

### H3. What the literature says will NOT work
- **Predicting absolute virality from content alone**: R^2 < 0.05 (Martin 2016); any such demo is leaking author/topic popularity.
- **Zero-shot or few-shot frontier-LLM "virality judges"**: near chance on Reddit-V, Vastu (AUROC 0.53), Upworthy prompting; fine-tuned 100M encoders beat them.
- **Hand-crafted "viral features" (hashtag count, emoji, sentiment, length) as the main signal**: consistently weak since Suh 2010/Bakshy 2011; Phoenix doesn't even ingest them. They are at best second-order modifiers after topic/audience fit.
- **Treating the 2023 heavy-ranker weights or the 2026 Phoenix weights as "N likes = 1 reply" arithmetic**: the repo explicitly says weights scale *predicted per-viewer probabilities*; reach is a product of per-viewer probabilities across the reachable audience, so optimisation must target probability lifts, not counts.
- **RL fine-tuning a generator against a learned engagement reward without external grounding**: proxy reward rises, true engagement does not (CharacterFlywheel), ecosystem quality drifts (CHASE), and RL overoptimises at lower KL than best-of-N (Gao 2023).
- **Expecting a cascade-model MSLE (~1.7–2.6) to transfer to cold-start**: those models require >= 10 observed retweets and 0.5–1 h; they describe a different, easier problem.
- **Predicting the top 0.1% tail with any precision from pre-publication signals**: dominated by exogenous amplification (quote-tweets by large accounts, trend timing); plan for probabilistic ranking and early-signal re-scoring instead of point predictions.
- **Hoping an individual-level "will user u repost" model generalises across topics from content**: F1 collapses to ~0.12 out-of-topic (2505.15370); audience/user features are what transfer.
- **Relying on scraping for training data**: $15K per 1M posts liquidated-damages clause; ID-only redistribution; use donated archives, the paid API (<= 3M reads/mo), the USC election set (NC), or Bluesky as a structurally similar open platform for pre-training.

---

## Source index (URLs)
Cascade: https://arxiv.org/abs/1403.4608 · https://arxiv.org/abs/1506.02594 · https://arxiv.org/abs/1608.04862 · https://arxiv.org/abs/1708.06401 · https://arxiv.org/abs/1611.05373 · https://arxiv.org/abs/1810.05184 · https://github.com/Xovee/casflow · https://arxiv.org/abs/2110.06836 · https://arxiv.org/abs/2310.13219 · https://arxiv.org/abs/2409.16623 · https://arxiv.org/abs/2502.18040 · https://arxiv.org/abs/2608.25048 · MMCas DOI 10.1145/3774904.3793030
Content/cold-start: https://arxiv.org/abs/1602.01013 · https://arxiv.org/abs/1405.1438 · https://arxiv.org/abs/1703.01725 · https://arxiv.org/abs/1907.01985 · https://arxiv.org/abs/2112.12191 · https://arxiv.org/abs/2302.09654 · https://arxiv.org/abs/2410.17390 · https://arxiv.org/abs/2505.15370 · https://arxiv.org/abs/2004.13715 · https://dl.acm.org/doi/10.1145/3664647.3688996 · https://dl.acm.org/doi/10.1145/3664647.3688997 · https://arxiv.org/abs/2507.00926 · https://arxiv.org/abs/2507.00950 · https://arxiv.org/abs/2507.19863 · https://dl.acm.org/doi/10.1145/3637528.3672041 · https://ojs.aaai.org/index.php/AAAI/article/view/32078 · https://arxiv.org/abs/2003.12724 · https://github.com/ICDM-UESTC/MMRA · https://arxiv.org/abs/2605.18653 · https://arxiv.org/abs/2604.26252 · https://arxiv.org/abs/2503.04446 · https://arxiv.org/abs/2606.27539
LLM: https://aclanthology.org/2025.ranlp-1.41/ · https://arxiv.org/abs/2601.12491 · https://arxiv.org/abs/2406.02611 · https://thearf-org-unified-admin.s3.amazonaws.com/MSI_Report_24-150.pdf · https://arxiv.org/abs/2510.08481 · https://arxiv.org/abs/2502.12073 · https://aclanthology.org/2024.findings-acl.570/ · https://lrec.elra.info/lrec2024-main-1127 · https://arxiv.org/abs/2604.25614 · https://arxiv.org/pdf/2604.18955 · https://link.springer.com/article/10.1007/s44443-026-00909-x
Industry: https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm · https://github.com/twitter/the-algorithm · https://arxiv.org/abs/2102.07619 · https://arxiv.org/abs/2202.05387 · https://dl.acm.org/doi/10.1145/3394486.3403370 · https://dl.acm.org/doi/10.1145/2566486.2568043 · https://github.com/xai-org/x-algorithm (local checkout) · https://arxiv.org/abs/2402.17152 · https://arxiv.org/abs/2403.02545 · https://arxiv.org/abs/2506.13695
Optimization/Goodhart: https://doi.org/10.3390/jtaer20030218 · https://arxiv.org/abs/2505.01698 · https://www.mdpi.com/2079-9292/15/13/2820 · https://arxiv.org/abs/2608.30466 · https://arxiv.org/pdf/2603.01973 · https://arxiv.org/abs/2210.10760 · https://arxiv.org/abs/1506.02089
Data/legal: https://docs.x.com/x-api/getting-started/pricing · https://github.com/TheExGenesis/community-archive · https://arxiv.org/abs/2411.00376 · https://github.com/sinking8/usc-x-24-us-election · https://arxiv.org/abs/2409.01052 · https://arxiv.org/abs/2210.05401 · https://arxiv.org/abs/2404.18984 · https://osf.io/jd64p/ · https://knightcolumbia.org/content/knight-institute-says-xs-new-terms-of-service-will-stifle-independent-research · https://natlawreview.com/article/x-corp-loses-battle-over-public-data-access · https://news.bloomberglaw.com/ip-law/x-partially-revives-lawsuit-against-israeli-data-scraping-firm
