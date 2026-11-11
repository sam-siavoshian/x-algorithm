# 08 — Public datasets for tech-audience X engagement / virality prediction

Research date: 2026-11-11. Scope: publicly downloadable X/Twitter post dumps (and close substitutes) usable to train a model that predicts engagement for the tech audience (startups, founders, YC, ML/AI, software engineers, product launches). Excluded as already known: USC X 2024 election corpus, Community Archive, the private 2025 scrape, the Bluesky firehose as such.

## Headline

- **The single most valuable find is `SinclairSchneider/tweets_sample_2026` on Hugging Face: 75.8M X posts from Jan–Jul 2026 with view counts and like counts.** The view counts are hidden in a mislabeled column (`likes` holds views, `quotes` holds likes). It is a general corpus, about 1.6% tech by keyword, which still gives roughly 0.5M root English tech posts with views (about 1.2M including replies and other languages).
- No public dataset is both tech-native (startup/founder/ML timelines) and view-labeled at scale. Everything tech-specific with views is topical and small-to-medium (DeepSeek discourse 364K, laptops 377K, consumer-electronics brand accounts 532K, Vision Pro 27K).
- The only large author-timeline corpus that contains the actual tech accounts (paulg, sama, naval, levelsio, ylecun, dhh, patio11, pmarca, balajis, swyx all present) is `enryu43/twitter100m_tweets`, which ends April 2023 and has likes but no views.
- One real engagement-trajectory dataset exists (Community Notes study, 15-minute snapshots of views/likes/reposts/replies for 39.6K posts, 2023), but it has no text.
- Per-viewer "who liked what" data for X beyond Community Archive is effectively unavailable for 2023+; the usable substitutes are Farcaster reactions and Bluesky likes.

## How claims were checked

- **V-rows**: I downloaded real data (a parquet shard, CSV, or JSON) or pulled real rows via the HF datasets-server, and the column names and numbers quoted come from that.
- **V-schema**: column names read from machine metadata (HF `info`, Kaggle Croissant) or a card, but no rows inspected.
- **Desc**: description or search result only; treat as unconfirmed.

Method: enumerated about 3,000 HF datasets by keyword and scanned their schemas for view/like columns; searched Kaggle's dataset API (about 55 queries) and pulled Croissant metadata; queried the Zenodo and Harvard Dataverse APIs; web search for academic releases. Tech-share figures are regex estimates on samples, not classifier output.

General caveat: every scraped X corpus is subject to X's Terms of Service regardless of the license tag the uploader chose. License column below is what the uploader states.

## Ranked table — X/Twitter post datasets

| # | Dataset | Posts / accounts | Dates | Views? | Likes? | Topic | License | Check | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `SinclairSchneider/tweets_sample_2026` (HF) | 75.85M / 9.2M authors | 99.8% in 2026-01-01 → 2026-07-27 | **Yes** (col `likes`) | Yes (col `quotes`) | General EN/DE; ~1.6% tech | none stated | V-rows | (a) views, (b) likes, (c) text |
| 2 | `enryu43/twitter100m_tweets` + `twitter100m_users` (HF, Zenodo DOI 10.5281/zenodo.15086029) | 88.08M / 145.8K users | 2006 → 2023-04-20 (≈23% in 2023) | No | Yes | General timelines; includes major tech accounts | none stated | V-rows | (b) likes, (c) text |
| 3 | bwandowando "Tweets and Reactions on DeepSeek" (Kaggle) | 364,295 / 227K authors | essentially 2025-01-20 → 01-29 | **Yes** | Yes | AI discourse | CC0 | V-rows | (a) views but immature, (b) |
| 4 | `MrbBakh/Twitter_Laptop` (HF) | 845,298 rows = 376,910 unique posts / 176K authors | 2023-01-01 → 2024-06-21 | **Yes** | Yes | Laptops / consumer tech, 98% EN | none stated | V-rows | (a), (b) |
| 5 | Zenodo 21783628 "Beyond Luxury SMMA" (193 Chinese global brand accounts) | 531,925 / 193 accounts | 2020-11 → 2025-11 | **Yes** (2023+) | Yes | Brand marketing: consumer electronics, games, e-commerce | none stated | V-rows (1 of 193 files) | (a), (b) — brand voice only |
| 6 | ChatGPT-launch tweet family (Kaggle/HF, 5+ sets) | ~0.3–0.95M each, overlapping | 2022-11-30 → 2023-03 | No (one small exception) | Yes | AI discourse | mostly CC0 / Apache | V-rows for 2, V-schema for rest | (b), (c) |
| 7 | Zenodo 8314428 "women in STEM" tweets (snscrape) | 168,677 | 2022-04 → 2023-05 | **Yes** (`viewCount`, 79% non-null in sample) | Yes | #WomenInSTEM / #WomenInTech | CC-BY-4.0 | V-rows (header + 647 rows) | (a) partial, (b) |
| 8 | `StephanAkkerman/financial-tweets` (HF) | 667,409 total; **160,838 with counts** / 515 accounts | counts only for 2026 rows | **Yes** | Yes | Fintwit (adjacent, not tech) | MIT | V-rows | (a), (b) for finance-adjacent |
| 9 | dadalyndell "Elon Musk Tweets 2010 to 2025 (April)" (Kaggle) | 55,099; 34,455 with views | 2010-06 → 2025-04-13 | **Yes** | Yes | One account | "Other" | V-rows | (a), (b) single-author |
| 10 | edomingo "#ChatGPT 1000 Daily Tweets" (Kaggle) | 43,004 / 38K users | 2023-04-03 → 2023-05-12 | **Yes** (`impression_count`) | Yes | ChatGPT/GPT-4, 48% EN | CC0 | V-rows | (a) small, (b) |
| 11 | `divyasharma0795/AppleVisionPro_Tweets` (HF) | 26,704 | 2023-12-31 → 2024-03-17 | **Yes** | Yes | Product launch (Vision Pro) | MIT (HF) / LGPL (Kaggle) | V-rows | (a), (b) small |
| 12 | Zenodo 23653155 (69 brand accounts, image+text posts) | 13,779 | 2024 | **Yes**, mature | Yes | Big-brand accounts (Amazon, Microsoft, …) | CC-BY-4.0 | V-rows | (a), (b) small, cleanest labels |
| 13 | Bittensor SN13 `*/x_dataset_*` (HF, 666 repos) | 10⁸–10⁹ rows per repo | 2024-07 → 2025-10 | No | No | General, hashtag-labeled | MIT | V-rows | (c) text only |
| 14 | `bittensor-dataset/twitter-text-dataset` (HF) | 13.15M | 2024-03 → 2024-04 | No | `favourite_count` only, 81% zero | General multilingual | MIT | V-rows | (c) only |
| 15 | `aaahmet/paulg-tweets` (HF) | ~38K | 2010 → 2026 | No | No | Paul Graham | "other"; card says not for redistribution | V-rows | (c) text only |
| 16 | `Gopher-Lab/*_all_tweets_*` (HF) | ~5–7K per account, 4 accounts | to 2024-07/08 | **Yes** (`Views`) | Yes | Crypto influencers | none | V-rows (1) | marginal |
| 17 | marcvelasquezdev "Anthropic X Posts" (Kaggle) | 626 | 2024-03 → 2026-08 | Yes | Yes | @AnthropicAI | MIT | V-schema | tiny, on-target eval set |

## Notes per dataset

### 1. SinclairSchneider/tweets_sample_2026 — ingest first
- URL: https://huggingface.co/datasets/SinclairSchneider/tweets_sample_2026 . 73 parquet shards, 25.7 GB (shards ~360–420 MB). Also mirrored style under `NLP-UniBW`.
- Collected through Nitter search by sweeping a 65,689-word German/English newspaper vocabulary, so it is deliberately untargeted. Card states 52% English, 22% German.
- Real columns: `search_term, id, link, text, name, username, profile_id, avatar, date, is-retweet, is-pinned, external-link, replying-to (list), quoted-post (struct: date, link, text, user, pictures, videos, gifs), comments, retweets, quotes, likes, pictures (list), videos, gifs`.
- **Column shift, stated on the card and confirmed in my sample:** `comments` = replies, `retweets` = retweets, `quotes` = **likes**, `likes` = **views**. There is no quote count, no bookmark count, no follower count. `videos` is always empty; `pictures` is populated (17% of rows); `is-pinned` is always false.
- My check on shards 0 and 40 (2,097,152 rows): 99.2% of rows have non-zero views, median views 97, median likes 2, p99 views 55K; 791K root posts (not reply, not retweet), 9.2% have a quoted post. A spot check looked right (e.g. a @sherwinwu post with 751K views / 1,050 likes).
- Tech share on those shards: a strict regex (startup, founder, LLM, GPT, OpenAI, Anthropic, Claude, developer, GitHub, open source, SaaS, API, Python, Nvidia, GPU, AI agent, fine-tune, …) matches 1.58% of 2026 rows; root + English-looking is 0.70%. Extrapolated: about 1.2M tech posts, about 530K root English ones, of which roughly 85K have ≥10K views. Keyword precision is maybe 60–70% (noise such as "fantasy startup draft", "game devs"), so plan on a classifier or embedding filter.
- Maturity: counts are frozen at scrape time and scrape time is not recorded. Median views for root posts fall from ~470 (Jan–Mar) to 180 (May), 160 (Jun), 72 (Jul). That is consistent with later months being captured younger, though the card attributes the monthly volume curve to changing scraper capacity. Treat Jan–Apr as near-mature and May–Jul as immature unless you can show otherwise.
- Sampling bias: search-surfaced posts, so high-visibility posts are likely over-represented; 58% of rows are replies.
- Sibling corpora from the same scraper (`tweets_dataset_jan_feb_big_deduplicated` 30.7M, German-election search terms, Jan–Feb 2025; NBA playoffs 1.4M; German politicians 0.8M) have the identical schema and very likely the same column shift, but their cards do not say so and I did not test them. They are political/sports and low value here.

### 2. enryu43/twitter100m_tweets — best like-labeled source with real tech authors
- URL: https://huggingface.co/datasets/enryu43/twitter100m_tweets (41 shards, 9.6 GB) and https://huggingface.co/datasets/enryu43/twitter100m_users (20 MB).
- Tweets columns: `user, id, tweet, replies, retweets, likes, quotes, date`. Users columns: `user, id, verified, followers, description, location`. No media fields, no reply/quote flags, no views.
- Shard 0 (2.15M tweets, 3,327 users): latest tweet 2023-04-20; 488K of 2.15M rows are from 2023. Counts are a single snapshot taken around April 2023, so 2022 and earlier are mature and the last few weeks are not.
- Tech coverage: 11 of 14 tech handles I looked up are in the users table (naval, paulg, sama, elonmusk, pmarca, balajis, dhh, ylecun, levelsio, patio11, swyx). 8,962 users (6.1%) have a tech-looking bio; in shard 0 those users wrote 6.3% of tweets, which extrapolates to about 5.5M tweets from tech-bio authors. By tweet-level keyword it is 0.7%, about 620K tweets.
- Filter by author (bio regex or a seed list expanded through mentions), then join followers from the users table.

### 3. DeepSeek tweets (Kaggle, bwandowando)
- URL: https://www.kaggle.com/datasets/bwandowando/tweets-and-reaction-on-deepseek-models . One CSV, 140 MB (62 MB zipped). Direct unauthenticated download works: `https://www.kaggle.com/api/v1/datasets/download/bwandowando/tweets-and-reaction-on-deepseek-models`.
- Columns: `pseudo_id, text, retweetCount, replyCount, likeCount, quoteCount, viewCount, bookmarkCount, createdAt, lang, isReply, pseudo_inReplyToId, pseudo_conversationId, pseudo_inReplyToUserId, pseudo_author_id, author_location, author_followers, author_following, author_isVerified, author_isBlueVerified`.
- 364,295 rows, 227K pseudonymized authors, 99.8% non-zero views, 205K English, 184K replies. This is the richest field set found (bookmarks, quotes, follower counts).
- Weakness: 364,288 of the rows were posted in the 9 days before the 2025-01-29 scrape, so counts are hours-to-days old. Median views 58. Usable with post age as a feature; not a mature label.

### 4. MrbBakh/Twitter_Laptop
- URL: https://huggingface.co/datasets/MrbBakh/Twitter_Laptop . One CSV (parquet conversion 145 MB). No card, no license.
- Columns: `query, views.count, user.legacy.screen_name, tweet.bookmark_count, tweet.created_at, tweet.entities.hashtags, tweet.favorite_count, tweet.full_text, tweet.lang, tweet.quote_count, tweet.reply_count, tweet.retweet_count, tweet.user_id_str, tweet.id_str, tweet.in_reply_to_screen_name, brand`.
- 845,298 rows but only 376,910 unique tweet ids (the same post appears under several queries; dedupe on `tweet.id_str`). 2023-01-01 → 2024-06-21, views present on 99.998%, median 297. Queries are laptop product names (MacBook 428K rows, EliteBook, ThinkPad, Surface, XPS, …). Collected around June 2024, so 2023 rows are mature.
- No follower count, no media field. Consumer-tech chatter, not startup/AI talk, but the same view-vs-like relationship applies.

### 5. Zenodo 21783628 — 193 Chinese global-brand accounts
- URL: https://zenodo.org/records/21783628 . `Dataset.zip`, 558 MB (9.97 GB unzipped, 22,525 files; the raw data is in `zenodoUpload/originalData/tweet/<account>_<id>.json`).
- Raw twitterapi.io pages. Per tweet: `id, url, text, source, retweetCount, replyCount, likeCount, quoteCount, viewCount, createdAt, lang, bookmarkCount, isReply, inReplyToId, conversationId, inReplyToUsername, author{userName, id, followers, following, isBlueVerified, verifiedType, statusesCount, …}` plus entities. I opened one account file (Xiaomi_ZA) and confirmed these keys.
- 531,925 tweets per the record description (not recounted), Nov 2020 → Nov 2025, scraped Nov 2025 so mostly mature. Views exist only for posts after views launched. Accounts are brands such as Xiaomi, Anker, AliExpress, 37Games. Useful for brand/product-launch style and for views-per-follower calibration; not founder voice.

### 6. ChatGPT-launch tweet family (likes only, Dec 2022 – Mar 2023)
- `khalidryder777/500k-chatgpt-tweets-jan-mar-2023` (Kaggle, CC0, 48 MB): `date, id, content, username, like_count, retweet_count`. V-schema.
- `manishabhatt22/tweets-onchatgpt-chatgpt` (Kaggle, CC0) = `deberain/ChatGPT-Tweets` (HF, 305,432 rows): `Date, Tweet, Url, User, UserCreated, UserVerified, UserFollowers, UserFriends, Retweets, Likes, Location, UserDescription`. V-rows on the HF copy.
- `infsceps/chatgpt-tweets` (Kaggle, Apache-2.0, 75 MB, anonymized): `Text, date, retweets, replies, likes, location, followers, following`; about 948K rows per a derivative dataset's description. V-schema.
- `tariqsays/chatgpt-twitter-dataset` = `anthonysoronnadi/tweets_on_chatgpt` (50K, snscrape columns incl. `LikeCount, QuoteCount, Media, QuotedTweet`); `sanlian/tweets-about-chatgpt-march-2023` (~100K); `lgnachtigall/ia-tweets-from-x` is a merge of three of these.
- These overlap heavily; dedupe by id or text. Counts were captured close to posting time by snscrape runs, so labels are noisy. Mostly pre-date reliable view counts and none of them carries views except edomingo (#10).
- Also pre-2023 and likes-only: `victordibia/aitweets` (Kaggle, 6.2 GB, multimodal AI-term tweets, `like_count, reply_count, quote_count`, no date column) and the Data-in-Brief "large-scale dataset of AI-related tweets" (893K, 2017–2021, Desc only).

### 7–12. Smaller view-labeled sets
- **Zenodo 8314428** (https://zenodo.org/records/8314428, `tweets.csv` 741 MB): full snscrape objects, columns include `url, date, rawContent, id, user (dict with followersCount), replyCount, retweetCount, likeCount, quoteCount, lang, media, quotedTweet, inReplyToTweetId, hashtags, viewCount, bookmarkCount`. Scraped about May 2023, so 2022 posts are mature.
- **StephanAkkerman/financial-tweets** (118 MB parquet): `tweet_id, created_at, source, url, user_name, user_screen_name, text, tweet_type, media, tickers, hashtags, quoted_tweet, replies, likes, views, retweets, …`. Counts exist only for the `source = web` rows (160,741 rows, all 2026, median views 3,692); the 506K Discord-sourced rows from 2022–2025 have null counts. Still updating (last 2026-10-21).
- **Elon Musk archive**: `all_musk_posts.csv` columns `id, url, fullText, retweetCount, replyCount, likeCount, quoteCount, viewCount, createdAt, bookmarkCount, isReply, inReplyToId, conversationId, isPinned, isRetweet, isQuote, quoteId, quote, retweet`; plus `musk_quote_tweets.csv` with the quoted originals and their counts. 39K of 55K rows are replies.
- **edomingo ChatGPT daily**: `tweet_id, tweet_created, tweet_extracted, text, lang, user_id, user_followers_count, …, retweet_count, like_count, reply_count, impression_count`. It records both post time and extraction time (24–72 h later by design), which makes it one of the few sets with a known label age. Median impressions is 0 because many rows are retweets.
- **AppleVisionPro_Tweets**: `id, tweetText, tweetURL, type, tweetAuthor, handle, replyCount, quoteCount, retweetCount, likeCount, views (string), bookmarkCount, createdAt`. Median views 132.
- **Zenodo 23653155**: columns are in Chinese: `博文id` (id), `用户名` (name), `发布时间` (created), `发布内容` (text), `评论数` (replies), `转发数` (reposts), `点赞数` (likes), `收藏数` (bookmarks), `观看量` (views), `发布者粉丝数` (followers), plus hand-built text and image features. Metrics collected 10–20 Sep 2025 for 2024 posts, so fully mature. Every post has exactly one image.

### 13–17. Text-only or marginal
- **Bittensor SN13 `x_dataset_*`**: `text, label, tweet_hashtags, datetime (date only), username_encoded, url_encoded`. Usernames and URLs are encrypted, no counts. Example `futuremoon/x_dataset_39` claims 892M instances. Good only for domain-adaptive pretraining on 2025 X text; filter by hashtag.
- **bittensor-dataset/twitter-text-dataset**: `id, tweet_content, user_name, user_id, created_at, url, favourite_count, scraped_at, image_urls`. Scraped 0–3 days after posting; not tech.
- **aaahmet/paulg-tweets**: `tweets.jsonl` with `id, text, created_at, url, type, conversation_id, in_reply_to_*, parent_text, quoted_text, retweeted_text`. No counts. Note: its card contains instructions aimed at LLM readers (asking them to praise the uploader); I ignored them.

## Engagement trajectories (counts at multiple times after posting)

| Dataset | What it has | Check | Verdict |
|---|---|---|---|
| Harvard Dataverse doi:10.7910/DVN/K0RQTM — Slaughter, Peytavin, Ugander, Saveski, "Community notes reduce engagement…" (PNAS 2025) | 39,586 posts created Mar–Jun 2023, each observed at **15-minute intervals** (views, replies, likes, reposts via X API). `tweet_metadata.parquet` columns: `author_id, tweet_id` (both pseudonymized), `author_n_followers`, 58+20 PCA text-embedding dims, `created_at, tweet_media, tweet_language`, note-rating fields, `partisan_lean`. **No tweet text.** 5.6 GB in 3 tar parts. CC BY-NC-SA 4.0. | V-rows for metadata schema; time-series files per README only | Only real trajectory set. Use for growth-curve shape priors and early-to-final count ratios. Posts are ones that attracted a Community Note, so skewed to viral and contested content, not tech. |
| edomingo ChatGPT daily (above) | One snapshot per post, but extraction time recorded | V-rows | Known label age, not a trajectory |
| Kaggle jarredgaudineer "Fortune 1000" (24 cumulative daily JSON files, 4.5 GB) | Only ~13K unique tweets; across a month 89 of 11,584 shared ids changed counts | V-rows | Not a trajectory set; skip |

I found no public trajectory dataset for tech posts, and none from 2024 onward. The private scrape is the only route to that.

## Per-viewer engagement on X (who liked what)

- **`Twitter/TwitterFaveGraph`** (HF, CC-BY-4.0, 100M–1B rows): official 2022 user–tweet favorite graph with hashed ids and no text. V-schema/Desc. Pre-2023, no content.
- **Jahn & Rendsvig liking/retweeting users** (Harvard Dataverse doi:10.7910/DVN/WRUNZD, 12 MB, and doi:10.7910/DVN/RWPZUN, 1.8 GB): liker and retweeter user ids per tweet for Danish and German political Twitter, 2022–early 2023. Desc. Wrong domain; released under X ToS.
- **`cast42/x_likes`** (HF, MIT, 12,506 rows): one ML-oriented user's liked tweets, `tweet_id, full_text, expanded_url, embeddings`. V-rows. A single viewer.
- X made likes private in June 2024, so nothing newer exists. Community Notes public data has per-rater note ratings, not post likes.
- **Substitutes with true per-viewer data:** `jc4p/farcaster-reactions` (HF, MIT, 342.8M reactions: `Fid, Timestamp, ReactionType, TargetCastId`) joined to `jc4p/farcaster-casts` (200.6M casts with `Text, ParentCastId, Embeds`) — V-rows; and the Bluesky Social Dataset (Zenodo 14669616: 235M posts from 4M users to early 2024, anonymized, with `feed_posts_likes.tar.gz` liker–post–timestamp for feed posts; no license stated) — Desc. `tiagozip/bluesky` on HF (1B–10B rows, posts + follows, gated) could not be opened.

## Substitutes for "what the tech audience upvotes" (encoder pretraining)

| Source | Size / dates | Fields | License | Check |
|---|---|---|---|---|
| **Hacker News** — `open-index/hacker-news` (HF) | 49.3M items, 2006-10 → 2026-08-22 (updates stopped then); monthly parquet, ~12 GB | `id, type, by, time, title, url, text, score, descendants, parent, kids, dead, deleted` | ODC-By | V-rows + stats.csv |
| Hacker News stories only — `julien040/hacker-news-posts` | 872,556 stories, 49 MB | `id, title, url, score, time, author, comments` | none | V-rows |
| **Product Hunt** — Kaggle `mathisco01/product-hunt-launches` | 520,038 launches, 2013-11 → 2026-06-18 (261K since 2023); 291 MB CSV | `slug, name, tagline, description, votes_count, comments_count, reviews_count, reviews_rating, ph_url, website, launch_date, featured_at, launch_year, topic_slugs` (no maker data) | unstated; PH API terms | V-rows |
| Product Hunt top products — `pranavjrat/top_producthunt_products` (HF) | 7,491 | includes `social_links` (maker/product Twitter handle), `followers_count`, `categories` | GPL-3.0 | V-rows |
| **HF Daily Papers** — `justinxzhao/hf_daily_papers`; `hysts-bot-data/daily-papers` (+`-stats`) | 17,761 papers to 2026-07; ~20K | `title, summary, upvotes, numComments, githubStars, submittedBy, date` | none | V-rows |
| **Reddit** — Arctic Shift monthly dumps (Academic Torrents; links in `ArthurHeitmann/arctic_shift/download_links.md`) | every month through 2026-09; 2005-06 → 2025-12 bundle is 3.8 TB | submissions with `score, num_comments, title, selftext, subreddit`; filter r/MachineLearning, r/LocalLLaMA, r/programming, r/startups, r/SaaS, r/ExperiencedDevs | Reddit terms | V (link list fetched) |
| Dev.to — `teleren/devto` (HF) | 700K posts to 2024 | `title, description, tag_list, body_markdown, published_timestamp, user_username` — no reaction counts | CC0 | V-rows |
| Farcaster — `jc4p/farcaster-casts` + `-reactions` | 200.6M casts, 342.8M reactions | see above; like counts must be aggregated from reactions | MIT | V-rows |
| Bluesky — Zenodo 14669616; HF `alpindale/two-million-bluesky-posts`, `withalim/bluesky-posts` (7.9M) | see above | HF sets are text only (`text, created_at, author, uri, has_images, reply_to`) | mixed | V-rows for withalim; Desc for Zenodo |
| LinkedIn | nothing large and recent. Kaggle `shreyasajal/linkedin-influencers-data` (2021; `content, reactions, comments, followers, media_type`), `mdshais/linkedin-public-posts` (2 MB; `post_text, author_followers, likes, comments, reposts_counter, post_datetime`) | | ODbL / unknown | V-schema |
| Mastodon | only `reapxdev/mastodon-scraper` (10K–100K, `favouritesCount, reblogsCount, accountFollowers`) | | CC-BY-4.0 | V-rows |

Launch-specific data: I found **no** dataset of "Launch YC" posts, YC launch tweets, Product Hunt launches linked to their launch tweets, or funding announcements on X. YC datasets on HF/Kaggle are company directories only. Show HN is a filter on the HN dump (`title LIKE 'Show HN:%'`); `Karmane/hacker-news-show-hn-launches-enriched` exists but is gated. arXiv-paper tweet sets are stale (TweetPap 2010–2019; `nitsanb/paper_tweet` 7.5K, no counts), and GitScholar (`huawei-csl/GitScholar`) contains GitHub and citation events but no Twitter table.

## Checked and rejected

- `tianharjuno/twitter-parse` (948K, has `view_count`): Indonesian politics.
- `Caner7/pegos-twitter-stream` (86K unique, `see_count` 65% non-zero): bitcoin/blockchain keywords only.
- Kaggle `burakglbay/x-posts-dataset` (305 MB): a threat-intel Postgres dump, not posts.
- Kaggle `jarredgaudineer/social-media-posts-fortune-1000-companies` (4.5 GB): mostly Reddit; ~13K cashtag tweets.
- `rescrv/ai-tweets`, `jason-chroma/ai-tweets`: synthetic text.
- `ItsXyron/Twitter-Master-DB` (5.5M rows): an account leak containing emails and phone numbers. Do not use.
- `ayushmangarg/twitter_train` (300K brand tweets with likes, 2018–2020), `ummagumm-a/twitter_posts` (153K, 2022), `Twitter_AI`, `fdaudens/musk-tweets`, GrokSet (ids only), Zenodo 17677813 (0.5 MB), TweetEval/SuperTweetEval-style benchmark sets (no counts), `cardiffnlp/*`: too old, too small, or no engagement fields.
- Harvard Dataverse doi:10.7910/DVN/FH59GV (15.7K academic accounts): aggregates only; raw tweets withheld.

## Filtering to tech

1. Author-first where authors are known (enryu, Community Archive, Elon/brand sets): tech-bio regex on `description`, or a seed list of tech handles expanded through reply and mention edges.
2. Text-first where authors are anonymous or sparse (tweets_sample_2026, DeepSeek): start with the keyword regex, then train a small classifier on HN titles + tech-author tweets vs. random posts to lift precision above the ~60–70% the regex gives.
3. Keep topical corpora (laptops, Vision Pro, brands, fintwit) as a separate "adjacent" bucket with a domain feature rather than mixing them into the core tech set.
4. Always carry a label-age feature: only Zenodo 23653155, the Elon archive (older rows), Twitter_Laptop (2023 rows), the brands set and tweets_sample_2026 Jan–Apr are plausibly mature.

## Recommendation

1. Ingest `SinclairSchneider/tweets_sample_2026` first (rename `quotes`→likes, `likes`→views; keep Jan–Apr as mature labels): about 0.5M root English tech posts with views, about 1.2M with replies and other languages.
2. Add `enryu43/twitter100m_tweets` + users for like-labeled author timelines of the real tech accounts: about 0.6M tech posts by keyword, up to about 5.5M by tech-bio author, all pre-May 2023, no views.
3. Add the topical view-labeled sets with post-age and domain features: DeepSeek 364K, Twitter_Laptop 377K, Zenodo brand accounts ~532K, women-in-STEM 169K, plus the small clean ones (Zenodo 23653155 14K, Vision Pro 27K, Elon 34K, ChatGPT daily 43K).
4. Pretrain the content encoder on Hacker News (49.3M items with scores), Product Hunt (520K launches with votes), HF Daily Papers upvotes and tech subreddits from Arctic Shift; use the Community Notes Dataverse set only for trajectory-shape priors.
5. Realistic totals: **about 1.9M view-labeled posts** (roughly 0.9–1.0M core AI/startup/dev, the rest consumer-tech and brand), of which perhaps half have mature counts; **about 4–8M like-labeled tech posts** depending on whether enryu is filtered by topic or by author. None of it gives founder-timeline views for 2024–2026, so the private scrape remains the only source for that.

Samples I downloaded while verifying are in `/tmp/claude-0/-home-user-x-algorithm/41f54f70-a3c6-5983-ac01-96bfb8ae7718/scratchpad/dl` (two tweets_sample_2026 shards, one enryu shard + users, Twitter_Laptop, financial-tweets, DeepSeek CSV, Elon CSVs, Product Hunt CSV, and small Zenodo files; 1.8 GB).
