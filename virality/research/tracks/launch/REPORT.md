# Track "launch": startup-specific evidence for launch posts and other company posts

2026-11-16. Deliverables in this folder: `LAUNCH_PLAYBOOK.md` (for a marketing lead), `launch_evidence.json` (same
shape as `xviral/data/edit_evidence.json`, plus fields), result files, `patch.diff` (new scripts under
`scripts/launch/`, no existing file changed), `launch_posts.parquet` and `accounts_labeled.parquet` (the labels).

## Headline

- 5,179 own launch posts and 3,130 other own announcements labelled in the 4.48M-post corpus; 270 company and 528
  founder accounts. View counts exist for only 205 of the company/founder launch posts, so every effect is on likes.
- A launch post gets 2.2x the account's own usual likes (2.06 to 2.44; 2,156 posts, 563 accounts); funding 6.9x;
  feature releases 1.2x; hiring is not distinguishable from an ordinary post.
- Of about 30 traits tested inside accounts, eight pass a rule fixed in advance. Only one (the first line names the
  product: 64.0%, 59.0 to 69.0, 1,672 pairs) is also confirmed separately before and after the test cutoff; layout on
  three or more lines is confirmed in both periods but weaker in the strict subset. Video against an image or against
  nothing, opening with the announcement and not the problem, an original post and not a quote, no link in the post,
  and a number in the first line pass the rule but are not separately provable after the cutoff.
- The benchmark has 3 held-out pairs with a launch post on either side. On a new held-out set of 850 same-account
  launch pairs from the test period the baseline ensemble is 63.7% correct (59.5 to 68.0); 73.3% (65.3 to 81.0) on
  the 26% of pairs where it states 60% or more; 78.3% (69.4 to 87.2) on confident pairs with a 2x real gap, which is
  28% of the 2x pairs. The model predicts launch posts at 1.03x the account's median where the truth is 2.19x.

## What the data allows and what it does not

| Asked for | Status |
|---|---|
| Bio signals | `author_bio` is null for every row of both files. Accounts are identified from display name and posting patterns. |
| View counts | Only in the `nitter_2026` source: 850 of 5,179 launch posts. That source stores text without line breaks and records almost no video (44 of 1,349 typed posts), so first-line, layout and video traits cannot be read there. All likes analyses use Community Archive posts only. |
| Link in a reply, thread body | Replies are not in the corpus (`is_reply` is false everywhere, `conversation_id` null). Only the head post's wording ("link in reply", 🧵, 1/) is visible. |
| Same-topic pairs | 44 pairs with a launch post on one side, 84 with any announcement: too thin for any trait (`same_topic.json`). The within-account pair design below replaces it. |
| Founder against company for the same launch | Team membership is not observable. A proxy (people who quote one company's announcements in the first person at least twice) gives 152 pairs from 30 people and 19 companies. |
| Day and hour | UTC only; the audience's zone is unknown. |

## 1. Accounts and post types

Scripts: `labels.py` (rules), `label_run.py`, `accounts.py`, `build_table.py`, `counts.py`.

**Post types.** Rules on the first 300 characters (links removed), first match in the order funding, hiring, launch,
open-source release, feature release, milestone, demo. A typed post counts as the account's *own* announcement only
if the opening is in the first person (we / our / I / my) or, on a company account, opens with an announcement word;
reprinted articles and posts starting with a handle are dropped. Three rounds of reading samples changed the rules
(matching anywhere in long posts labelled essays; "new:" openers labelled news; install commands labelled promotion
as open-source releases).

Precision of the final rules, read by hand inside company and founder accounts:

| Type | Correct | Note |
|---|---|---|
| funding | 17 of 20 | |
| hiring | 15 of 20 | misses are job-board lists and "X is hiring" about others |
| launch | 19 of 25 | misses are investments, events, workshop announcements |
| open-source release | about 14 of 17 | after removing install-command matches |
| feature release | 16 of 25 | |
| milestone | 17 of 25 | other people's milestones leak in |
| demo | about 8 of 20 | not used for effects |

**Accounts.** company: under 10% of posts use I/my, display name is not a person's name (first-name dictionary), at
least 3 own typed tech posts making up 6% or more of posts, 30% or more of posts tech-worded. founder: at least 2
own first-person typed tech posts, 3% or more of posts. News, digest and fan names, breaking-news aggregators and
token-promotion accounts excluded. Read by hand on 100 each: about three in four "company" accounts are product
companies or labs; about six in ten "founder" accounts are people announcing their own products, the rest tech
commentators. The post-level own-announcement test carries most of the precision.

Counts (`counts.json`), own announcements in `work/filtered/posts.parquet`; "hygiene" = also in
`work/hygiene/posts.parquet`:

| Type | Company posts / accounts | Founder posts / accounts | Other accounts | All with views | All without views | In hygiene |
|---|---|---|---|---|---|---|
| launch | 1,636 / 236 | 1,055 / 413 | 2,488 / 1,806 | 850 | 4,329 | 3,059 |
| feature release | 252 / 102 | 540 / 239 | 1,277 / 884 | 354 | 1,715 | 1,132 |
| hiring | 64 / 34 | 152 / 104 | 375 / 317 | 72 | 519 | 336 |
| demo | 66 / 44 | 199 / 116 | 328 / 269 | 29 | 564 | 399 |
| funding | 45 / 32 | 77 / 61 | 110 / 102 | 14 | 218 | 163 |
| open-source release | 32 / 23 | 48 / 43 | 75 / 67 | 20 | 135 | 117 |
| milestone | 7 / 5 | 31 / 27 | 45 / 42 | 10 | 73 | 51 |

Accounts: 270 company (207 in hygiene, 61 with any view counts), 528 founder (431 in hygiene, 120 with views).

Launch posts of company and founder accounts by follower tier (posts / accounts / with views): 100k+ 803 / 96 / 12;
10k to 100k 696 / 192 / 93; 1k to 10k 158 / 67 / 15; under 1k 18 / 10 / 7; unknown 1,016 / 284 / 135. Full tables for
every type are in `counts.json`.

## 2. What separates launch posts that beat the account's usual

Scripts: `effects.py`, `build_launch_evidence.py`, `same_topic.py`, `second_account.py`, `founder_vs_company.py`.

**Outcome.** lift = log((likes+1) / (median likes of the account's previous 30 posts + 1)), at least 8 previous
posts; posts captured under 3 days after posting dropped. Ordinary posts of these accounts have a median lift of
0.99x, so the baseline is unbiased.

**Three estimates per trait**, each with a 95% interval from resampling accounts (2,000 draws): (a) pairs: all pairs
of launch posts by one account, at most 365 days apart, exactly one with the trait, lifts differing by 1.25x or
more; win rate computed separately for "with it posted first" and "second" and averaged; at most 50 pairs per
account. (b) ratio: accounts with both; mean lift with minus without, precision-weighted. (c) adjusted: one
regression with account fixed effects and 19 traits.

**Rule, fixed before reading results by set.** MAIN = launch and open-source posts of company and founder accounts
(1,980 Community Archive posts, 530 accounts). STRICT = the subset whose opening has an announcement verb directly
followed by a product name (1,066 posts), which removes label noise: a mislabelled post gets about 1x where a real
launch gets 2.2x, so any trait correlated with label quality would look like an effect. *survives* = every test
passes in MAIN or STRICT, at least one passes in the other, same direction. *main_only*, *wider_only* (only when all
accounts are pooled, 3,654 posts), *leaning*, *none*, *too_thin* (under 200 pairs or 25 accounts). Then each trait
was estimated separately before and from 2026-04-23.

| Trait (stronger version) | Grade | MAIN win rate | Likes ratio | Adjusted | Pairs / accounts | STRICT win rate | Before cutoff | From cutoff |
|---|---|---|---|---|---|---|---|---|
| First line names the product | A | 64.0% (59.0-69.0) | 1.51 (1.27-1.78) | 1.36 (1.16-1.60) | 1,672 / 180 | 65.0% (51.6-75.4) | 58.8% (52.3-64.7) | 72.8% (64.1-80.3) |
| Three or more lines | A- | 59.2% (54.4-63.8) | 1.37 (1.18-1.59) | 1.34 (1.18-1.56) | 1,631 / 144 | 53.1% (45.9-59.9) | 60.9% (55.9-65.8) | 61.1% (53.1-69.6) |
| Video or GIF, against none | B | 65.4% (60.2-70.4) | 1.64 (1.33-2.02) | 1.58 (1.35-1.90) | 991 / 133 | 61.8% (49.7-70.5) | 73.6% (66.5-79.0) | 59.4% (41.0-78.2) |
| Video or GIF, against an image | B | 62.8% (56.0-69.3) | 1.61 (1.32-1.96) | n/a | 1,021 / 113 | 67.1% (58.2-76.0) | 63.1% (54.9-70.5) | 61.1% (41.3-75.0) |
| Announcement first, not a problem or question | B | 72.5% (59.5-81.8) | 1.91 (1.33-2.65) | 1.73 (1.22-2.38) | 379 / 48 | 77.3% (61.9-87.3) | 69.3% (56.8-79.7) | 56.8% (34.3-82.3) |
| Original post, not a quote | B | 72.1% (60.7-81.1) | 1.90 (1.33-2.62) | 1.60 (1.11-2.23) | 363 / 65 | 74.1% (51.4-88.5) | 68.1% (58.7-76.8) | 57.6% (36.7-72.4) |
| No external link in the post | B | 58.2% (50.4-65.7) | 1.29 (1.05-1.65) | 1.21 (0.99-1.53) | 927 / 95 | 63.3% (56.2-70.3) | 57.7% (49.1-65.6) | 59.6% (42.4-76.6) |
| Number in the first line | B | 55.8% (50.5-60.9) | 1.28 (1.06-1.52) | 1.18 (1.00-1.39) | 1,684 / 151 | 57.5% (51.1-63.8) | 54.0% (48.5-59.5) | 57.0% (42.7-70.0) |
| Image or video, against none | B- | 62.9% (58.8-66.6) | 1.44 (1.23-1.67) | n/a | 1,700 / 186 | 56.4% (47.3-64.0) | 67.5% (62.8-71.6) | 54.5% (42.4-66.2) |
| 2 hours or more after the previous post | C | 55.6% (44.8-65.0) | 1.40 (1.07-1.83) | 1.36 (1.04-1.73) | 786 / 57 | 55.4% (42.4-66.7) | 59.9% (48.8-69.7) | 58.6% (47.9-68.8) |

Before and from cutoff are on launch posts of every account. The spacing row is *wider_only*: pooled over all
accounts it is 57.9% (50.5-65.0), ratio 1.41 (1.14-1.72), adjusted 1.33 (1.09-1.58), 1,089 pairs, 122 accounts.

By follower tier (all accounts with a known count): video against none is 74.9% (67.3-81.4), ratio 2.58 (1.81-3.69)
under 100k followers (252 pairs, 59 accounts) and 65.2% (55.5-72.6), 1.50 (1.10-1.92) at 100k and above.

No effect or not established: image against none (ratio 1.17, 0.95-1.45; STRICT 48.0%), thread marker (51.2%),
result in the first line (49.4%), 280+ characters (adjusted 0.95), 600+ characters, emoji, tagging three or more
accounts, weekend, day of week, hour blocks (13-17 UTC against 18-23 UTC passes in MAIN and reverses in STRICT).
Leaning: no tags (59.1%, ratio interval includes 1), announcement verb in the first line (nothing once product
naming is held fixed). Too thin: link in a reply (17 pairs), hashtags (41 posts), question in the first line as a
separate trait (192 pairs, 31 accounts, 26.7% for the version with it). Not a lever: the account posting again
within 2 hours of the launch (69.1%), which is follow-up to launches that are already doing well.

Views: for launch posts no trait has more than 49 accounts with both versions, and the intervals all include no
effect; nothing is statable (`effects.json`, `views` blocks).

**Types** (company and founder accounts, likes against own trailing median; geometric mean, share beating the
median): funding 6.87x (5.00-9.48), 96%, 74 posts; open source 2.60x (1.68-4.10), 68%, 62; launch 2.23x
(2.06-2.44), 71% (68-74), 29% in the account's own top tenth, 2,156 posts; milestone 1.99x (1.21-3.20), 30 posts;
hiring 1.28x (0.99-1.66), 176; feature 1.19x (1.07-1.32), 55%, 691. Launch on views: 1.53x (1.25-1.96), 57%, 202
posts, 60 accounts. Launch on company accounts 2.20x (1.98-2.45), on founder accounts 2.31x (2.05-2.62).

**Second account.** 3,481 quote posts of launch posts by another corpus account. A first-person quote does 1.70x
(1.47-2.01) the quoter's own median (560 posts, 401 accounts) and 1.44x (1.27-1.67) the quoter's other quote posts;
it gets more likes than the original in 5% of 493 cases. With the team proxy (152 pairs): 2.4x (1.5-3.8) the
person's usual, more likes than the company post in 10%. Launches quoted within 48 hours do 2.30x the same
account's unquoted launches; that is circular and is not reported as an effect.

## 3. How well the current model ranks launch posts

Scripts: `bench_launch.py`, `score_launch.py`, `model_check.py`.

**Benchmark pairs (bench2).** Of 1,660 held-out pairs with a 1.25x gap, 3 have a launch-type post on either side and
11 have any own announcement; 14 pairs belong to company or founder accounts. Accuracy there is noise for every one
of the ten models (`bench_launch.json`). The benchmark does not cover launch posts.

**New held-out set.** Every launch-type post created on or after 2026-04-23 by an account with at least two
(582 posts, 167 accounts), scored through the draft path with the baseline ensemble (ts46_s1, ts46_s2,
compare_v2_full), each at its own time against the account's previous 56 posts. Pairs: same account, likes
differing by 1.25x or more, at most 30 per account: 850 pairs, 151 accounts.

| Slice | Pairs | Accounts | Correct |
|---|---|---|---|
| All | 850 | 151 | 63.7% (59.5-68.0) |
| Real gap 2x or more | 568 | 127 | 66.0% (60.9-71.6) |
| Real gap 3x or more | 401 | 105 | 67.8% (61.9-74.1) |
| Within 30 days of each other | 278 | 92 | 63.0% (56.2-69.1) |
| Company accounts | 601 | 69 | 63.7% (58.8-68.7) |
| Founder accounts | 119 | 43 | 63.0% (52.9-71.7) |
| Stated confidence 60%+ (26% of pairs) | 217 | 75 | 73.3% (65.3-81.0) |
| Confidence 60%+ and real gap 2x+ (28% of the 2x pairs) | 157 | 70 | 78.3% (69.4-87.2) |
| All three members agree (77% of pairs) | 657 | 131 | 66.5% (61.7-71.6) |

Confidence uses the scale fitted on the benchmark's calibration pairs (1.5156), not refitted here.

Other findings for the model tracks:
- The model does not recognise a launch: predicted median ratio to the account's trailing median 1.03x, actual
  2.19x (576 posts); it predicts 51% to beat the median, 76% do. Rank correlation of predicted and actual lift 0.48.
- Test-period pairs differing in "first line names the product": the version with it won 73.9% of 264; the model
  preferred it 61.4%. Pairs differing in video: the video version won 50.5% of 220; the model preferred it 67.7%.
- A rule score fitted only on launch posts before the cutoff (account fixed effects, 16 traits, 1,768 posts) ranks
  the same test pairs 55.7% (51.3-59.6): the pre-cutoff weights (video 2.4x) did not carry over in full, which is
  why video is graded B and not A.

## What did not work

- Same-topic pairs for launches: 44 pairs.
- Views as an outcome: too few posts, and the view source lacks line breaks and video.
- Bio-based account identification: no bios.
- First post-type rules (matching anywhere in the text): about half of "feature", "milestone" and "open source"
  labels were wrong; fixed by matching the opening only and requiring the first person.
- The pre-cutoff rule score as a predictor: barely above chance on the test period.

## Honest limits

- Observational, likes only. Associations inside accounts; not randomized.
- About 30 traits and three tests each; the rule (three tests, two sets, then two periods) limits false positives
  but "number in the first line" (55.8%) could be one.
- Labels are rules: about a quarter of "launch" labels are wrong, and the "founder" account list is loose. The
  STRICT subset controls for label noise in the effects; the type table in section 2 is diluted by it (true launch
  lifts are probably somewhat above 2.2x).
- The sample is technology and AI accounts with large followings; 18 launch posts from accounts under 1,000
  followers. Capped posts per account and missing replies and reposts mean "hours since the previous post" is an
  upper bound on the true gap.
- The traits "names the product" and "problem first" are regular expressions on the first line; they were checked
  by reading samples, not by a second labeller.

## Suggested next steps

1. Give the model a post-type input (or a launch prior on the output): it is off by 2x on launch posts.
2. Add the 850 launch pairs to the benchmark as a launch slice.
3. Wire `launch_evidence.json` into the advisor for drafts the rules type as a launch: operators "name the product
   in the first line", "move the problem statement below the announcement", "make it an original post" have a
   measured effect; "attach an image" does not for launch posts.
4. Collect replies for launch posts (link in reply, threads) and bios (founder and company pairs); these are the
   two questions the corpus cannot answer.

## Files

| File | Content |
|---|---|
| `LAUNCH_PLAYBOOK.md` | the playbook |
| `launch_evidence.json` | evidence table: one entry per trait (`description`, `win_rate`, `lo`, `hi`, `pairs`, `accounts`, `supported`, `outcome`, `design`) plus `status`, `grade`, ratios, strict subset, tiers, periods; `spacing`; `_types`, `_model_check`, `_bench_launch`, `_second_account`, `_founder_vs_company`, `_same_topic`, `_counts`, `_meta` |
| `effects.json` | every estimate for 12 sets of posts, likes and views |
| `counts.json`, `same_topic.json`, `second_account.json`, `founder_vs_company.json`, `bench_launch.json`, `model_check.json` | as named |
| `launch_scores.parquet` | model predictions for the 582 test-period launch posts |
| `launch_posts.parquet` | the 8,902 typed own posts with type, lift and traits |
| `accounts_labeled.parquet` | the 798 company and founder accounts |
| `patch.diff` | `scripts/launch/*.py` and this folder's documents |

Reproduce, from `virality/` with the data in `data/`, in this order (all under `scripts/launch/`):
`accounts_explore.py` (per-account language shares), `label_run.py`, `accounts.py` (needs `gender-guesser`),
`build_table.py`, `effects.py`, `same_topic.py`, `second_account.py`, `founder_vs_company.py`, `bench_launch.py`,
`score_launch.py --checkpoint ... --out data/launch/launch_scores.parquet`, `model_check.py`, `counts.py`,
`build_launch_evidence.py`. `sample_types.py <seed> <n> [types]` prints labelled samples for reading.
