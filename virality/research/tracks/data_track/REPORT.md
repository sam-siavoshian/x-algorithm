# Track "data": more public data for the startup use

2026-11-16. Everything is in the dataset under `tracks/data/`. Code changes are in `patch.diff` (nothing committed).

## Headline

- Three sources converted to the post schema and uploaded, each with guards checked on every file:
  `twitter100m` (11.17M posts by 17,343 accounts, bios and follower counts, likes only, 2020 to April 2023),
  `zenodo_brands` (523,246 posts by 182 official company accounts, 139,757 with views, bios, replies with exact
  parent ids, video files), `nitter_2026_replies` (5.26M posts of 40,998 accounts from the 2026 view corpus,
  1.68M of them replies with their own view counts).
- `startup_accounts.parquet`: 19,320 company and founder accounts with the evidence for each. 12,087 come from
  public launch lists (YC directory, Product Hunt, the brand set), 6,803 have a bio that reads as founder or
  company, 700 carry the launch track's label, 1,130 have two or more own announcements. Only 4,588 have posts in
  any source; the other 14,732 are the list of accounts a scrape should fetch.
- Company or founder launch posts with views go from 262 (my count from `launch_posts.parquet`; the brief says 205)
  to 1,014. Almost all of the gain (668) is the brand set: large consumer hardware and app companies, not startups.
- Launch posts from accounts under 1,000 followers go from 18 to between 341 and 1,195 depending on how strict the
  account test is, all from `twitter100m`: likes only, posted before May 2023. With views: 10 more, no more.
- Bios: 715 of the 6,797 training accounts (10.5%) now have one, covering 59,419 of the 465,675 rows.
- Replies: present now, but "link in the reply" on launch posts is still not measurable: 8 launch posts in the
  brand set and none in the 2026 view corpus have a first self-reply with a link. For ordinary posts of company
  accounts it is measurable: 1,041 root posts with views whose first self-reply within an hour carries a link, 26
  accounts with five or more.
- The largest remaining public gain needs no new source: the Community Archive itself has a `profile` table (bios)
  and reply rows for the corpus's own accounts. The current pipeline never loaded the first and drops the second.
  It needs the archive's public key (`CA_ANON_KEY`), which this session did not have.
- A defect found and fixed: `util.core.to_int` read string ids through a float, which changes the low digits of a
  19-digit post id. `sources/user_scrape.py` uses it for post, author, reply and quote ids, so the private scrape
  would have been ingested with wrong ids. Fixed in `to_int`; a test covers it.

## 1. Survey

Hugging Face: 2,195 datasets returned by about 100 search terms, column names of 1,559 read from the datasets-server.
Kaggle: 466 hits from 40 searches of the public list API; downloads answered 404 without a login in this session,
so no Kaggle file was opened (the list API also stopped answering after the search run). Zenodo: 158 records from
13 queries. Lists are in `survey/`.

| Dataset | What it has | Licence, collection | Measured | Decision |
|---|---|---|---|---|
| `enryu43/twitter100m_tweets` + `_users` (HF; Zenodo 10.5281/zenodo.15086029) | 88.08M posts, 145,842 accounts, to 2023-04-20; likes, replies, reposts, quotes; bios (133,832 non-empty), follower counts. No views, no media, no reply ids | none stated; scraped timelines for a blog analysis | all 41 shards | converted |
| Zenodo 21783628 "Beyond Luxury SMMA" | 193 official accounts of Chinese global brands, twitterapi.io pages: views, bookmarks, reply and conversation ids, quoted post, media with video files, author object with bio | none stated; collected by the authors through the twitterapi.io API | all 209 files | converted |
| `SinclairSchneider/tweets_sample_2026` (HF), replies | already a source for root posts; 43.99M of its 75.85M rows are replies, each with views | none stated; Nitter search over a newspaper vocabulary | all 73 shards | converted (replies and roots of 61,317 handles) |
| `datahiveai/ycombinator-companies` (HF) | 5,384 YC companies with company and founder Twitter URLs | CC BY-NC 2.0; scraped from the public YC directory, September 2025 | whole file | used: 2,887 company and 4,101 founder handles |
| `pranavjrat/top_producthunt_products` (HF) | 7,491 Product Hunt products with the product's Twitter link | GPL-3.0; scraped from product pages, June 2026 | whole file | used: 5,249 handles |
| `deberain/ChatGPT-Tweets` (HF) | 305,400 posts about ChatGPT, 2022-11-30 to 2023-02-24, 144,306 accounts, bio on 95%, follower counts | none stated; snscrape keyword search | whole file | bios used (1,036 corpus accounts); posts not converted: one post per account, counts read soon after posting |
| `makes-run/X-Social-Posts` (HF) | 24,319 rows, 2026, views on 20,602 | custom licence | whole set | rejected: replies on Indian political posts, bio column empty |
| `twitterapi-io/x-cashtag-mentions-2024-2026` (HF, also on Kaggle) | 43.3M rows | CC BY 4.0 | columns only | rejected: per-ticker daily aggregates, not posts |
| `calebheinzman/YC_Founders`, `TechCrunch_Battlefield_Founders` (HF) | 50 rows each | Apache-2.0 | whole files | too small |
| `ramyaks99/hackernews-showhn-github-enriched` (HF), Kaggle `volodymyrborysenko/every-show-hn-post` | Show HN posts; the first joins 46,076 to GitHub repositories | none; HN and GitHub APIs | columns | no X handles. A GitHub profile carries a `twitter_username` field, so the makers' handles could be fetched from the GitHub API; not done |
| Kaggle `mathisco01/product-hunt-launches`, `envercanbickin/product-hunt-product-launches-2026`, `logiover/product-hunt-daily-launches` | launches with votes; maker fields absent or redacted | unknown / MIT / CC BY-SA | metadata, one sample | no link to X accounts |
| Zenodo 23653155 (69 big-brand accounts, 13,779 posts with views) | known from the earlier survey | CC BY 4.0 | not re-opened | not converted: Amazon, Microsoft and similar, one image per post |
| `StephanAkkerman/financial-tweets`, the `SinclairSchneider` and `NLP-UniBW` siblings, Kaggle `bwandowando/*`, `pokeash/bitcoin-tweets-dataset-20252026` | views, 2025 to 2026 | mixed | earlier survey or metadata | off audience (finance, politics, sport, crypto) or pseudonymized authors |

No public dataset found has startup or founder timelines with views for 2023 to 2026, a list of launch posts linked
to Product Hunt or Show HN, or follower counts for the 2026 view corpus. `ItsXyron/Twitter-Master-DB` was not
touched. `emailmarketingdataset/twitter-users-dataset-dataset` looks like the same kind of leak and was not opened.

## 2. The three converted sources

Readers are in `xviral/sources/`: `twitter100m.py` (new functions `all_users`, `check_shard`,
`read_shard_accounts`), `twitterapi_io.py` (new), `nitter_sample.py` (new functions `read_shard_accounts`,
`link_self_replies`). Each source folder in the dataset has `counts.json` and the per-file check results.

### 2.1 `sources/twitter100m/`

| Check | Result |
|---|---|
| Author identity | `users.id` is the real X id (paulg 183749519, sama 1605); handles and ids unique in the users table |
| Timestamp | `date` agrees with the time in the post id to the second in 99.65% of rows; 305,704 rows that do not, and 98,892 ids from before 2010, are dropped |
| Count columns | among rows with 20 or more interactions, likes >= quotes in at least 99.86% and likes >= reposts in at least 97.6% of every shard; a shard under 95% is refused |
| Duplicates | none within shards |
| Overlap | 2,651 accounts are in the current 4.48M-post corpus (410 in the training rows); 3,160 converted posts are already in it |

Converted: every original and reply since 2020 of 17,343 accounts (7.46M originals, 3.71M replies): accounts with a
founder or company bio, a tech bio, a handle in the current corpus, or two or more first-person typed tech posts.
5,323 of them have under 1,000 followers, 4,999 between 1,000 and 10,000. Posts by year: 2020 1.21M, 2021 1.84M,
2022 4.10M, 2023 4.02M.

Limits. No views, no media, no line-break loss (text is intact). The follower count is one reading from April 2023,
not the count when a post was made. Replies are recognised by a leading handle; a self-reply inside a thread looks
like an original and cannot be linked to its parent. Counts are one reading taken around the last post date, so the
last days are immature (snapshot set to 2023-04-21 for the maturity filter). All of it pre-dates the current ranker.

Launch-type posts (first person, tech wording, `launch_labels.py`), since 2020:

| Set | Posts | Accounts | Under 1k followers | 1k to 10k | Read by hand |
|---|---|---|---|---|---|
| Every account | 25,245 | 11,145 | 4,978 | 7,887 | about 6 of 22 are product or software launches (under-1k sample without account evidence); the rest are books, events, shops, churches |
| Strict: startup account, or tech bio, or strong tech words in the post | 6,264 | 2,685 | 1,195 | 1,911 | 11 of 20 |
| Account in `startup_accounts` (any evidence) | 2,260 | 590 | 429 | 623 | 11 of 18 |
| Account in `startup_accounts` with list, rules or two kinds of evidence | 2,058 | 407 | 341 | 572 | not read separately |

`typed_own_posts.parquet` has every typed post with its flags; the timeline of the author is converted for 20,892
of the 25,245 and for 6,095 of the strict 6,264.

### 2.2 `sources/zenodo_brands/`

| Check | Result |
|---|---|
| Author identity | `author.id` is the real X id and matches the file name; 767 rows by other authors (34 files contain a few) dropped |
| Ids | read as exact integers (the first run went through `to_int` and corrupted them; fixed and re-run) |
| Timestamp | `createdAt` agrees with the id to the second; 2,923 rows with ids from before 2010 dropped |
| Count fields | where views exist, likes <= views in at least 99.6% of rows of every file; a file under 99% is refused |
| Views | posts created before 2023-01-01 get no view count (the field is 0 or partial there) |
| Snapshot | no scrape time in the files; the newest post is 2025-12-09, snapshot set to 2025-12-10 |

523,246 posts by 182 accounts (13 under 1,000 followers, 43 between 1,000 and 10,000, 87 between 10,000 and
100,000, 39 above), 2010-11 to 2025-12, 456,413 English. 139,757 have views (all from 2023 on, all but 9 older than
three days at the snapshot). 324,898 have media, 75,108 a video, 81,339 a video file URL. Line breaks are intact.
Bios for 178 accounts. Three handles overlap the current corpus.

Own typed posts (company rule of `launch_labels.py`): launch 1,765 (665 with views, 148 accounts), demo 224,
feature 186, hiring 9. Launch and open-source posts by follower tier (posts / with views / accounts): under 1k
15 / 10 / 7; 1k to 10k 151 / 86 / 31; 10k to 100k 873 / 342 / 75; 100k and above 729 / 230 / 35.

Replies: 67,872 (45,679 with views), 4,363 of them self-replies, 4,253 with the parent in the set
(`threads.parquet` has the parent id, the account replied to and a self-reply flag). 2,886 root posts have a
self-reply, 2,288 of those have views.

Limits. These are regional and global accounts of large Chinese consumer brands (phones, cars, appliances, games,
e-commerce): company voice, not startup voice, and much of it product marketing with giveaways. The follower count
is one reading at scrape time. Several accounts have a gap in 2023 and 2024 (OnePlus has posts to 2022 and from
mid-2025 only). No licence is stated on the record.

### 2.3 `sources/nitter_2026_replies/`

Same guards as the existing reader: after undoing the column shift, likes <= views in 99.995% of rows; timestamps
agree with the id to the minute in 100%. Authors keyed by handle with a synthetic id, as before.

Kept: every root post and reply of 61,317 handles (the 38,841 corpus accounts, 11,907 list handles, 11,296
founder, company or tech-bio accounts of `twitter100m`); 40,998 of them appear. 5,255,347 posts: 3.57M roots
(1.48M already in the corpus) and 1,680,408 replies (838,467 created before May 2026, the mature part).

| | Posts | Accounts | Mature |
|---|---|---|---|
| Replies by training accounts | 188,795 | 5,099 | 81,902 |
| Replies by corpus accounts | 1,665,772 | 30,542 | 831,631 |
| Self-replies (the only addressee is the author) | 30,711 | 7,723 | 16,532 |
| Self-replies by training accounts | 3,916 | | |

The source gives the handles replied to, not the parent post. `link_self_replies` takes the author's latest earlier
post within 30 minutes: a parent is recovered for 6,530 self-replies, a root post for 2,879. Of those roots, 227
have a link in the first self-reply (108 mature), and 7 accounts have three or more. Of 1,724 launch-typed roots of
the kept accounts, none has a recovered self-reply. The corpus is a sample of posts, so a root and its reply are
rarely both present.

`launch_any.parquet`: first-person announcements with strong tech wording from any account: 5,204 posts, 1,861
mature (launch 1,150, feature 411, hiring 147, open source 81, funding 59), 317 of the mature ones already in the
corpus, median 858 views. No follower counts, as everywhere in this source.

## 3. `startup_accounts.parquet`

One row per account, key `handle` (lower case). Built by `scripts/data_track/build_startup_accounts.py`.

| Evidence | Columns | Accounts |
|---|---|---|
| list: linked from the YC directory or a Product Hunt product page, or one of the 182 brand accounts | `ev_list`, `list_names`, `list_role`, `list_company`, `list_batch_or_launch`, `list_one_liner`, `list_website`, `in_brand_set` | 12,087 |
| bio: reads as a founder or as an organisation describing a product (`xviral/text/bio.py`) | `ev_bio`, `bio`, `bio_kind`, `bio_source`, `bio_date` | 6,803 (bio text for 7,441) |
| rules: the launch track's label from name and posting pattern | `ev_rules`, `launch_track_kind` | 700 (the 98 without a handle are not included) |
| posts: two or more own first-person announcements | `ev_posts`, `n_own_announcements`, per-source counts, `n_own_announcements_with_views`, `example_announcement` | 1,130 |

`kind` (11,066 company, 8,254 founder) comes from the list role, then the brand set, then the bio, then the launch
track; `kind_from` says which, `bio_disagrees` and `rules_disagree` flag conflicts (7 and 14). `evidence_n`: one
kind 18,035, two 1,172, three 111, four 2. Followers are known for 7,913 (4,519 under 1,000, 2,051 between 1,000
and 10,000), read in 2023 or late 2022 for most. Coverage columns say where the account's posts are:
`n_training_rows` (673 accounts), `n_posts_corpus`, `n_posts_twitter100m`, `n_posts_nitter2026`,
`n_replies_nitter2026`, `n_posts_brands`.

Precision, read by hand once, bio evidence alone: founder bios about 11 of 14 are people with their own product;
company bios about 7 of 14 are product companies (the rest are agencies, podcasts, community accounts). List
evidence is exact for YC; a Product Hunt page sometimes links a person, and the bio then decides the kind. Recall of
the bio rules is low: of the 79 launch-track accounts that now have a bio, the rules read 12 as founder or company
(11 agreeing with the launch track); many founder bios name no role ("@OpenAI", "Creator @datasetteproj").

`account_bios.parquet`: 267,393 handles with a bio from any source (bio, source, date, kind, followers), for
joining to the corpus by handle.

## 4. The gaps, before and after

| Gap | Before | After | From |
|---|---|---|---|
| Launch posts | 5,179 | +2,260 by accounts in `startup_accounts` (+6,264 strict, +25,245 loose) in `twitter100m`; +1,768 brand set; +1,030 mature in the 2026 view corpus not already held | all three |
| Company or founder launch posts with views | 262 (brief: 205) | 1,014: +5 by relabelling accounts in the corpus, +668 brand set, +79 from the 2026 view corpus | brand set mostly |
| Launch posts from accounts under 1,000 followers | 18 | +341 (list, rules or two kinds of evidence) to +429 (any evidence) to +1,195 (strict) without views; +10 with views (brand set) | `twitter100m` |
| Launch posts from accounts with 1,000 to 10,000 followers | 158 | +572 to +623 without views; +151 brand set (86 with views) | `twitter100m`, brand set |
| Bios | 0 accounts | 715 of 6,797 training accounts (10.5%; 59,419 of 465,675 rows), 415 of 3,905 audience accounts, 3,654 of 38,841 corpus accounts, 79 of the 700 launch-track accounts | `twitter100m` users, ChatGPT-Tweets |
| Replies | 0 | 188,795 replies with views by training accounts (81,902 mature), 67,872 by brand accounts | 2026 view corpus, brand set |
| Threads and "link in the reply" | not measurable | company accounts, ordinary posts: 1,041 roots with views and a link in the first self-reply within an hour, 26 accounts with five or more. Launch posts: 8. Corpus accounts: 108 mature roots, 7 accounts with three or more | brand set only |
| Company and founder accounts known | 798 by inference | 19,320 with evidence, 4,588 with posts in some source, 673 in the training rows | lists, bios |

Relabelling the current corpus with the larger list changes little: 85 launch posts by 54 accounts the launch track
had as "other" (5 with views). The accounts on the lists are mostly not in the corpus at all: of 6,987 YC handles,
1,376 have posts in any source and 99 are in the training rows.

What this means for use:
- `twitter100m` is the source for small-account launch posts and for account histories with bios: within-account
  lift on likes, by follower tier, with the account kind read from the bio. It cannot say anything about views,
  video or the current ranker.
- The brand set is the source for views on company launch posts with video, line breaks and threads, and for
  views against likes on company accounts. It should be a separate domain in any model or evidence table.
- The reply rows of the 2026 view corpus give reply view counts for corpus accounts (how much reach a reply gets
  against a post), not threads.

## 5. What public data cannot give

1. **Views on startup and founder launch posts, 2023 to 2026.** Outside the brand set there are 262 + 5 + 79.
   The owner's scrape of the 14,732 listed accounts without posts (6,987 YC handles first) would supply them, with
   follower counts at scrape time. `startup_accounts.parquet` filtered on the coverage columns is that list.
2. **Follower counts in the 2026 view corpus.** It has none; 33 of the 79 new launch posts have a count from another
   source, mostly from 2023. A scrape or one profile lookup per handle fixes it.
3. **Small accounts with views.** 10 launch posts under 1,000 followers. Small accounts appear in public data only
   before 2023 without views, or as single sampled posts without a history.
4. **Threads and link-in-reply on launch posts.** Needs the conversation of each launch post: the scrape must fetch
   the author's replies under their own posts (conversation id), which the brand set shows is a standard field of
   the twitterapi.io format. `sources/twitterapi_io.py` reads that format and writes the thread table.
5. **Bios for most corpus accounts.** 10.5% covered, from 2022 and 2023 readings. The Community Archive `profile`
   table has bios for its accounts: `ca-download` with `CA_ANON_KEY` set, table `profile`, then the existing join in
   `community_archive.py` fills `author_bio`. The same download without the `~is_reply` prefilter yields the
   archive accounts' replies with parent ids (`reply_to_tweet_id`), which is the direct route to threads for the
   corpus's own accounts. Not done here: the key was not available in this session and the keyless storage export
   answers "bucket not found".
6. **Anything only the account owner sees.** Profile clicks, link clicks, follows from a post, detail expands, and
   impressions by source: X analytics exports only (`sources/analytics.py`).
7. **Counts over time.** Every source is one reading per post. Early velocity needs repeated scrapes.
8. **Product Hunt and Show HN launches tied to their X posts.** No public table. The lists give handles and launch
   dates (`list_batch_or_launch`); the launch post has to be found in the account's timeline around that date once
   the timeline is scraped.

## 6. What did not work or was not done

- Kaggle downloads (404 without a login); no Kaggle file was read.
- Community Archive profiles and replies (no key in this session).
- The first conversion of the brand set had wrong post ids (`to_int` through a float); caught by a test, re-run.
- The first reply side table of the 2026 view corpus marked every self-reply as carrying a link (`external-link`
  is an empty string, not null); recomputed.
- No model was trained and no benchmark run: this track prepares data. Nothing here is merged into
  `work/filtered/posts.parquet`. To merge: add `twitterapi_io` to `schema.SOURCES` and the timestamp tolerance
  table, give `twitter100m` its 2020 floor as before, and keep the brand set out of the audience selection.
- All precision figures are one reading by one reader of 14 to 22 posts or bios each.
- Licences: none of the three converted sources states one; the YC list is non-commercial (CC BY-NC 2.0), the
  Product Hunt list GPL-3.0. All are scrapes of X or of public directories and fall under those sites' terms.

## Files (`tracks/data/`)

| Path | Content |
|---|---|
| `REPORT.md`, `patch.diff` | this report; code changes and new files |
| `startup_accounts.parquet` | 19,320 accounts with evidence and coverage |
| `account_bios.parquet` | 267,393 handles with a bio |
| `gaps.json` | every number of section 4 |
| `lists/list_handles.parquet` | 11,907 handles from the YC directory and Product Hunt with role, company, batch or launch date |
| `sources/twitter100m/` | `posts/*.parquet` (41 files, post schema), `accounts.parquet` (145,842 users with bio flags and per-account counts), `typed_own_posts.parquet`, `counts.json`, `shard_checks.json` |
| `sources/zenodo_brands/` | `posts.parquet`, `threads.parquet`, `typed_own_posts.parquet`, `counts.json`, `file_checks.json` |
| `sources/nitter_2026_replies/` | `posts/*.parquet` (73 files), `threads.parquet`, `self_reply_links.parquet`, `launch_any.parquet`, `launch_typed_roots_kept_accounts.parquet`, `counts.json`, `shard_stats.json` |
| `survey/` | Hugging Face candidates with columns, Kaggle and Zenodo hit lists |

Reproduce from `virality/` (scripts in `scripts/data_track/`): `survey_hf.py`, `survey_kaggle_zenodo.py`,
`build_lists.py`, `enryu_run.py` then `enryu_finalize.py`, `brands_run.py 2025-12-10` then `brands_finalize.py`,
`nitter_run.py` then `nitter_finalize.py`, `build_startup_accounts.py`, `gap_report.py`. Tests:
`tests/test_data_track_sources.py`.
