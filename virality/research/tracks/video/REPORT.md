# Track "video": what kind of launch video travels

2026-11-16. Deliverables in the dataset under `tracks/video/`: this report, `VIDEO_PLAYBOOK.md` (for a marketing
lead), `video_evidence.json` (shape of `launch_evidence.json`), `effects.json`, `model_pairs.json`,
`thumb_probe.json`, `validation.json`, `acquisition.json`, contact sheets under `sheets/`, `patch.diff` (new
scripts under `scripts/video/`, no existing code changed). Under `media/videos/`: `index.parquet`,
`thumb_features.parquet`, `thumb_emb_siglip2_base256.npy` with `thumb_emb_ids.parquet`, `README.md`.

## Headline

- **The video files were not acquired, so only the poster frame (first frame) was analysed.** Duration, pace,
  cuts, motion, captions through the video and audio are unmeasured. No audio tracks were published.
- 14,416 poster frames fetched and featurised: all video and GIF posts of the 798 company and founder accounts,
  the video posts on the benchmark pairs, and up to 40 video posts of each other account with an announcement video.
- Of 26 traits tested inside accounts, **one passes the rule fixed in advance**: a headline-size line of text in
  the first frame. Launch videos: 64.6% (55.6 to 71.2), 340 pairs, 57 accounts, likes ratio 1.35x (1.07 to 1.74),
  adjusted 1.26x (0.95 to 1.64). It passes all three tests among announcement videos of company and founder
  accounts (59.5%, 52.5 to 65.7; 1.27x; adjusted 1.29x, 1.02 to 1.68; 495 pairs, 68 accounts).
- Six more hold across all video posts of company and founder accounts but cannot be shown on launch videos alone:
  landscape over portrait (61.2%, 56.7 to 65.6; 1.56x), 1080 px over sub-720 px (60.6%; 1.70x), 16:9 (55.9%; 1.27x),
  and a title card or logo as the opening frame against product UI (56.0%; 1.24x), against a person (59.3%; 1.41x)
  and against everything else (55.1%; 1.21x).
- A face in the first frame shows no gain (46.3%, 42.5 to 49.7). The kind of opening (UI, person, title, graphics,
  real footage, bare frame) cannot be told apart among launch videos by level: all sit between 2.5x and 3.6x the
  account's usual with overlapping ranges.
- **Model: no gain.** Fine-tuning the `small` model on the rows with poster-frame features, with and without 36
  poster-frame inputs: +0.2 points (-0.7 to +1.0) on 2,149 held-out video-video pairs, +0.5 (-0.9 to +2.0) on 361
  launch pairs with a video. The benchmarks themselves hold too few video pairs to measure anything (58 of the
  2,030 test pairs).

## 1. Acquisition: what worked

| Route | Result |
|---|---|
| Poster frames from `pbs.twimg.com` (`media[].url` in the corpus row) | Worked. 14,557 requested at 5 a second: 14,416 fetched, 65 gone (404), 69 network or decode errors, 7 refused (403); the stop rule (8 refusals in a row) never triggered. |
| Video files through X's public embed endpoint (`cdn.syndication.twimg.com/tweet-result?id=...&token=...`, no login, cookie or key) | **Works technically; not run in bulk.** A test on 6 launch video posts, one request each a second apart, returned 6 of 6 with duration, aspect ratio, an HLS playlist and 3 to 5 mp4 variants on `video.twimg.com` (270p at 256 kbit/s up to 1080p at 10 Mbit/s). Durations seen: 7, 14, 15, 42, 70 and 260 seconds. The fetcher for all target posts was then refused by this session's permission classifier ("Unauthorized Access") before it started. The refusal was not worked around. |
| `video.twimg.com` variants directly | Not tried; the variant URLs come only from the route above. |

Consequences, stated plainly: no mp4 was downloaded. `media/videos/index.parquet` has `status = thumbnail_only`
for 14,416 posts and `thumbnail_failed` for 141; `duration`, `bitrate`, `has_audio` and `audio_path` are null
everywhere; `width` and `height` are the poster frame's size. There is no `media/videos/audio/`. The parent session
was told at the time so the audio worker would not wait. With permission for that fetch, metadata for about 8,000
posts takes under an hour at 2.5 requests a second. No bulk fetcher is in the patch, because none ran.

## 2. Features (CPU)

`scripts/video/thumb_features.py`, 448 px JPEG per frame, about 0.26 s a frame on 4 cores:

- SigLIP-2 base (patch 16, 256 px) image embedding, 768-d, and zero-shot scores against 34 prompts in 9 groups.
  The so400m model in `xviral/embed/siglip.py` was too slow for this machine; the base model was used.
- Pixel statistics: brightness, spread, share of the largest colour bin, colourfulness, edge density.
- Faces: OpenCV YuNet; count, largest face as a share of the frame height.
- Text: RapidOCR detector and recogniser at native size; boxes, words, area share, area in the lower third,
  tallest line.
- From the media record: aspect (poster size), upload resolution (`media[].width/height`), GIF or video, upload
  route (thumbnail URL path).

`label_thumbs.py` turns these into a first-frame class (highest prompt group, with logo merged into title, game,
film and generated imagery into graphic; a detected face moves a real or graphic frame to face; a frame with no
text that is near black or one flat colour is "bare") and boolean attributes. The class rule was fixed on the 900
oldest frames before any effect was computed.

**Validation by eye** (one reader, 40 frames per class drawn from announcement videos, a different sample from the
one the rule was set on; sheets in `sheets/`):

| Class or attribute | Correct of 40 | Share of launch videos |
|---|---|---|
| Product UI (app, site, editor, terminal, chat) | 36 | 31% |
| A person is the subject | 34 (28 as a talking head in the strict sense) | 20% |
| Title card or logo | 36 | 17% |
| Motion graphics, render, illustration, generated imagery | 33 | 11% |
| Real-world footage without a close-up person | 35 | 7% |
| Bare frame (class "other") | 30 | 13% |
| Chart, diagram or slide | not read: 163 frames in all, 1% | under 1% |
| A face is visible (attribute) | 36 | |
| Headline-size text line (attribute) | 31 | |
| Bare first frame (attribute) | 31 (27 strictly empty) | |
| Text in the lower third of a non-UI frame, as a proxy for burned-in captions | 14 | |

The caption proxy failed validation (name cards, titles and labels outnumber captions), so burned-in captions are
reported as unmeasured. Not validated by eye: large face, any text, text-heavy, dark frame.

What was asked for and could not be built without files: duration, cuts by frame differencing, motion,
sampled-frame embeddings, share of frames with a face, screen-recording likelihood over the video, captions over
time, audio.

## 3. Effects inside accounts

`scripts/video/video_effects.py`. The estimators are copied from `scripts/launch/effects.py`: pairs (posting order
balanced, at most 365 days apart, lifts differing 1.25x or more, at most 50 pairs an account), within-account
likes ratio, adjusted ratio with account fixed effects; intervals from resampling accounts. Outcome: likes over
the account's trailing median. Community Archive posts only. Every comparison is between video posts.

Rule, written into the script before any effect was computed: all three intervals exclude no effect among launch
videos of every account (LAUNCH, 846 posts, 438 accounts) or among announcement videos of company and founder
accounts (ANNOUNCE, 821 posts, 311 accounts), with at least 200 pairs and 25 accounts, and pairs and ratio point
the same way across all video posts of company and founder accounts (BROAD, 5,296 posts, 575 accounts).
`broad_only` (grade C): all three pass in BROAD only.

One correction was made after the first run and before anything was written up: the adjusted estimate first put
all class dummies in one regression, which compared each class with the small residual class instead of with
"every other first frame". It now runs one regression per trait on the trait's own comparison set (trait plus
first-line, layout, length, link, quote, problem-opening, portrait, sub-720, GIF and upload-route covariates). The
pairs and ratio numbers were not affected. In the first version the headline trait had no adjusted test at all;
the correction gave it one, which made the rule stricter for it. It still passes in ANNOUNCE.

| Trait (stronger version) | Status | LAUNCH win rate, pairs / accounts | LAUNCH ratio; adjusted | ANNOUNCE win rate; ratio; adjusted | BROAD win rate; ratio; adjusted |
|---|---|---|---|---|---|
| Headline-size text in the first frame | survives, B | 64.6% (55.6-71.2), 340 / 57 | 1.35 (1.07-1.74); 1.26 (0.95-1.64) | 59.5% (52.5-65.7); 1.27 (1.03-1.55); 1.29 (1.02-1.68) | 54.0% (51.1-57.0); 1.13 (1.02-1.26); 1.08 (0.99-1.19) |
| Landscape, not portrait | broad only, C | 47.9% (30.9-77.7), 66 / 16 | 1.84 (1.11-3.31); 1.77 (1.14-3.13) | 42.9% (27.5-68.1); 1.46 (0.93-2.65); 1.55 (1.08-2.33) | 61.2% (56.7-65.6); 1.56 (1.31-1.89); 1.47 (1.23-1.76) |
| 1080 px or more, not under 720 px | broad only, C | 38.1% (22.2-65.1), 58 / 16 | 0.83 (0.32-2.28); 0.96 (0.37-2.76) | 42.6% (27.8-61.1); 1.05 (0.50-2.26); 0.99 (0.39-2.11) | 60.6% (54.6-66.7); 1.70 (1.32-2.10); 1.46 (1.11-1.82) |
| 16:9, not another shape | broad only, C | 56.1% (47.8-66.1), 347 / 60 | 1.20 (0.89-1.65); 1.17 (0.84-1.58) | 52.8% (43.4-64.1); 1.22 (0.95-1.62); 1.14 (0.89-1.46) | 55.9% (53.1-58.9); 1.27 (1.16-1.41); 1.19 (1.07-1.30) |
| Title card or logo, not product UI | broad only, C | 60.2% (49.3-72.9), 113 / 31 | 1.32 (0.98-1.85); 1.38 (1.01-1.92) | 63.2% (53.4-74.1); 1.38 (1.03-1.92); 1.34 (1.04-1.88), 171 pairs | 56.0% (50.9-61.1); 1.24 (1.07-1.44); 1.18 (1.04-1.33) |
| Title card or logo, not a person | broad only, C | 45.0% (27.8-63.4), 61 / 20 | 0.99 (0.58-1.78); 0.97 (0.55-1.72) | 46.9% (31.1-62.7); 0.99 (0.59-1.68); 1.08 (0.65-1.80) | 59.3% (52.2-66.0); 1.41 (1.15-1.70); 1.27 (1.05-1.49) |
| Title card or logo, against every other opening | broad only, C | 53.4% (43.8-62.4), 320 / 53 | 1.13 (0.87-1.48); 1.15 (0.89-1.49) | 54.1% (47.4-62.3); 1.13 (0.89-1.45); 1.12 (0.90-1.46) | 55.1% (51.0-58.9); 1.21 (1.08-1.35); 1.14 (1.03-1.27) |
| No text in the lower third of a non-UI frame | broad only, no advice | 57.5% (44.2-70.6), 146 / 31 | 1.12 (0.78-1.59); 1.21 (0.84-1.71) | 54.4% (40.6-70.5); 1.18 (0.80-1.76); 1.28 (0.91-1.77) | 55.2% (50.7-59.8); 1.25 (1.08-1.44); 1.21 (1.05-1.39) |
| Not opening on product UI | leaning | 60.2% (53.1-69.0), 333 / 58 | 1.25 (0.995-1.56); 1.32 (1.04-1.67) | 60.2% (53.0-68.0); 1.29 (1.04-1.63); 1.14 (0.90-1.46) | 51.7% (48.7-54.8); 1.09 (0.98-1.21); 1.09 (0.99-1.20) |

`hd720` (720 px or more against under 720) duplicates the 1080 px row (61.1%, 1.63x) and is not counted
separately. The lower-third row passes in BROAD, but the attribute is not what it was built to be (see
validation), so it carries no advice.

No effect, or direction not stable (BROAD win rate for the trait as named): any readable text 49.2% (46.7-51.7);
text-heavy 51.0% (47.4-54.9); a visible face 46.3% (42.5-49.7), ratio 0.90 (0.79-1.03); large face 48.3%
(44.0-52.8); person-first 45.0% (41.3-48.6), ratio 0.86 (0.76-1.00), adjusted 0.88 (0.79-1.01); graphics-first
52.9% (48.8-56.9), although 64.0% (55.0-71.9) among launch videos with a ratio of 1.02; real-footage-first 51.9%
(47.2-56.8); bare first frame 53.8% (48.1-59.2); dark frame 51.4% (48.3-54.7); square against landscape 46.2%
(41.7-50.7), ratio 0.84 (0.72-0.96); GIF against video 43.4% (35.3-51.6), ratio 0.68 (0.48-0.90); Media Studio
upload 50.9% (42.3-61.3), and the opposite direction among launch videos (23.0%, 12.9-68.2, 78 pairs).
Too thin among launch videos: UI against person (66 pairs), UI against graphics (36), chart-first (29).

**Follower tiers** (BROAD, where pairs suffice). Landscape over portrait: under 100,000 followers 67.8%
(61.1-73.7), 2.25x (1.66-2.99), adjusted 2.10x; 100,000 and above 50.2% (41.7-59.3), 1.01x. 1080 px: under 100k
62.6% (54.9-70.3), 2.22x; above 56.5% (46.2-67.0), 1.24x (0.82-1.84). 16:9: under 100k 60.0% (55.5-64.7), 1.49x;
above 50.6%, 1.05x. Headline text: under 100k 52.6% (47.5-57.7), above 48.7% (42.6-54.9): not shown in either tier.
Title against a person: above 100k 67.5% (53.8-77.1), 1.73x; under 54.4% (43.0-67.3). Among launch videos the
tiers hold 290 posts (under 100k) and 222 posts (above): no trait reaches 200 pairs in either.

**Before and after 2026-04-23.** Launch videos after the cutoff: 151 posts, under 30 pairs for every trait, so
nothing is confirmed there. Headline text before: 66.5% (55.2-75.3), 226 pairs; after: 70.8% (50.0-85.4), 21
pairs. In BROAD, before the cutoff every C finding passes all three tests; after it (706 posts) only 16:9 has
pairs, ratio and adjusted in the same direction with two of three excluding no effect (59.2%, 49.8-68.6; 1.38x,
1.03-1.89; adjusted 1.41x, 1.08-1.87).

**Class levels** (launch videos, every account; mean likes over the account's usual): UI 3.22x (2.63-3.93, 259
posts), person 2.88x (2.27-3.65, 165), title 3.61x (2.86-4.69, 146), graphics 3.26x (2.41-4.34, 92), real footage
2.45x (1.70-3.65, 61), bare 3.54x (2.74-4.62, 112). Overlapping.

## 4. Model

`scripts/video/train_video.py`, `eval_video_pairs.py`, `thumb_probe.py`.

- Rows with features in the training table: 9,158 (7,189 train, 649 val, 1,320 test).
- Control and treatment both warm-start from `runs/ts46_s1/best.pt` and fine-tune the `small` preset for 600 steps
  (lr 2e-4, batch 64) with anchors restricted to the same 7,189 train rows; best step by loss on the 649 val rows.
  Treatment adds 36 inputs: a has-features flag, 16 principal components of the poster-frame embedding (fitted on
  train rows), class one-hots and attributes, with zero initial weights so step 0 equals the baseline. Two seeds
  each, averaged.
- Scored through the table path (not the serving path) on test-split rows.

| Pair set (same account, from 2026-04-23, real gap as stated) | Pairs / accounts | Baseline ts46_s1 | Control | With poster frame | Difference to control |
|---|---|---|---|---|---|
| Video-video, gap 1.25x | 2,149 / 212 | 57.9% (54.3-61.3) | 58.2% (54.9-61.5) | 58.4% (54.7-61.9) | +0.2 (-0.7 to +1.0) |
| Video-video, gap 2x | 1,584 / 186 | 59.9% (55.9-63.7) | 60.5% (56.6-64.2) | 60.5% (56.6-64.4) | +0.1 (-0.9 to +1.0) |
| Launch pairs with a video on either side, gap 1.25x | 361 / 47 | 64.8% (54.6-72.3) | 64.8% (54.7-71.5) | 65.4% (55.8-72.1) | +0.5 (-0.9 to +2.0) |
| The same, gap 2x | 249 / 41 | 69.5% (57.3-77.3) | 71.1% (60.6-78.4) | 71.5% (61.6-78.6) | +0.4 (-1.5 to +2.6) |
| All launch pairs in the table, gap 1.25x | 987 / 117 | 64.8% (60.3-68.4) | 66.3% (61.4-70.2) | 66.4% (61.5-70.4) | +0.1 (-0.5 to +0.8) |
| Same-topic benchmark pairs with a video, both posts in the table | 6 / 5 | not measurable | | | |

The two seeds disagree in sign (seed 1 +0.4, seed 2 -0.2 on video-video pairs). No checkpoint beats the baseline
beyond noise, so none was uploaded.

Probe without the model: a ridge regression on the lift, demeaned per account, trained on 10,880 video posts of
1,051 accounts before the cutoff and tested on 3,741 same-account video pairs of 303 accounts after it:
embedding alone 52.5% (49.6-55.3), attributes alone 49.7% (47.0-52.3), both 54.7% (52.0-57.5). The poster frame
carries a small signal on its own that the model's existing inputs already cover.

Benchmarks as asked: the 3,530 same-topic pairs hold 108 pairs with a video on either side (58 in the test part,
10 with video on both sides), and only 365 test pairs have both posts in the training table, 6 of them with a
video. The launch set scores 582 posts, 97 of them video. Confident-call tables were not produced: with a
difference of +0.2 on 2,149 pairs there is nothing to tabulate, and the two official benchmarks cannot resolve a
video effect of any plausible size.

## 5. Limits

- First frame only. A video is labelled by how it opens, not by what it is.
- For 467 of the 14,416 posts the poster is stored as an ordinary image, probably a thumbnail the uploader chose;
  for the rest it is taken to be the first frame, which was not checked against the files.
- One reader validated the labels; no second opinion. Label errors of 10 to 25% bias every class contrast toward
  no effect.
- 26 traits, one rule, no correction for multiple testing beyond requiring three tests and two sets of posts. The
  headline finding passes by a narrow margin on the adjusted test (lower bound 1.02 in ANNOUNCE, 0.95 in LAUNCH).
- Portrait and low resolution mark casual clips and re-uploads; the shape and resolution findings may be about
  production care, not the format.
- Likes only; views are absent where video is recorded.
- The fine-tune test ran through the table path and on 600 steps; the serving-path benchmark was not re-run.

## 6. What would change the answer

Permission to run the embed-endpoint fetch in a worker session. Everything else is in place: the target list
(`media/videos/index.parquet`), the per-post feature table to extend, the effects script (add duration bands,
cuts per minute, captions, audio as attributes), and the fine-tune harness.
