# Images track: does the picture matter for a startup's post, and which kind?

2026-11-16, one 4-core CPU machine, about 6 hours. Code: `scripts/images/` (+ small changes to `xviral/serve.py`,
`xviral/model/features.py`, `scripts/evidence/tac_serve_eval.py`, `scripts/launch/score_launch.py`), all in `patch.diff`.
Evidence: `image_evidence.json` (shape of `xviral/data/launch_evidence.json`), advice: `IMAGE_PLAYBOOK.md`.
Shared data: `media/images/` in the dataset (index, embeddings, attributes; no raw images).

## Headline

- **The kind of picture matters more than having one, and polish does not matter.** Inside company and founder
  accounts (782 accounts, 8,793 image posts), any image beats no media modestly (57.1% of 26,953 same-account pairs,
  1.23x likes, grade A). Against the account's other image posts, **charts and benchmark tables do 1.23x (A)**;
  **text-heavy announcement cards do 0.73x (A)** and are no better than no image; **screenshots of tweets, chats or
  emails do 0.80x (A)**. Product screenshots, designed graphics, document pages, people photos, other photos,
  illustrations and memes are indistinguishable from each other (0.91x to 1.06x, all intervals include 1).
- **Visual quality shows no effect** with tight intervals: aesthetic score top vs bottom third within type 1.01x
  (0.93-1.10), sharpness 1.01x, upload resolution 0.96x, colourfulness 1.05x, text cover 0.96x.
- **Launch posts:** an image over none is not established (1.16x, 0.96-1.42), consistent with the launch track (video is
  the lever); within launch posts a chart or benchmark table beat other images 68.5% (59-79), 1.60x, 365 pairs / 42
  accounts (too few accounts to grade).
- **"AI-generated" cannot be labelled reliably** (an open detector calls almost everything AI; by eye a third of the
  illustration class is recognisably AI). On the best available subset AI-style art trails designed graphics
  (0.80x, 0.64-1.02; 0.75x across all accounts; 0.73x on views): a lean, not a finding.
- **Model:** adding the image embedding and attributes to the small model (three members fine-tuned with and without,
  same rows) gives +2.2 points on the 321 same-topic benchmark pairs that involve an image (67.0% against 64.8%,
  interval 0.0 to +4.5; base ensemble 65.7%), nothing on 386 launch pairs with an image (+0.5, -3.0 to +4.2) and
  nothing on 69,078 ordinary same-account test pairs with an image (-0.2, -0.5 to +0.2). No checkpoint is put forward
  as beating the baseline.
- Views (2026 view corpus) confirm "image beats none" (1.25x, 13,733 pairs) but not the differences between kinds.

## 1. Images fetched

Priority order (`scripts/images/select.py`; first photo of every post that has a photo and no video):

| Priority | Posts | Accounts | Fetched |
|---|---|---|---|
| 0. Own announcements of company and founder accounts | 1,328 | 477 | 1,318 |
| 1. Own announcements of other accounts | 1,415 | 1,091 | 1,390 |
| 2. Posts of the same-topic benchmark pairs and launch test pairs | 1,026 | 299 | 994 |
| 3. Other posts of company and founder accounts (with a like baseline) | 9,697 | 739 | 9,654 |
| 4. Other posts of accounts with an announcement, newest 60 per account | 28,537 | 1,637 | 6,479 (6,534 requested) |
| 5. Other training-table posts | 61,850 | 4,968 | not reached |
| **Total** | | | **19,835 images, 2,636 accounts** |

The fetch ran for 67 minutes at 5 requests a second (`scripts/images/fetch.py`, the 680 px `small` rendition) and was
stopped by choice at 20,000 requests: representation on CPU (about 0.3 s an image) was the bottleneck, not the
fetch. 404 and 403 (deleted media) were 0.8% of requests; no 429 was seen. Everything in priorities 0 to 3 was
fetched; priority 4 was fetched in part (a random order inside the priority, so the cut is unrelated to outcomes).

## 2. Representation

`scripts/images/represent.py`, per image, CPU only:

- SigLIP base (`google/siglip-base-patch16-224`) image embedding, 768-d, unit norm; used for zero-shot types and as the model input.
- CLIP ViT-B/32 embedding (512-d) and the LAION aesthetic predictor v1 (linear head on that embedding, `shunk031/aesthetics-predictor-v1-vit-base-patch32` weights).
- Sharpness (variance of the Laplacian), edge density, brightness, contrast, saturation, colourfulness (Hasler-Suesstrunk),
  share of near-black and near-white pixels, three dominant colours (k-means on a 48 px thumbnail).
- Text: RapidOCR's text detector (PP-OCRv3 det, ONNX, no recognition) on a 512 px copy: number of text boxes, share of the
  image covered by text, tallest line. Run on shards 0 to 13 (priorities 0 to 3) only, for time.
- Faces: OpenCV YuNet (count, largest face area share).
- Original width and height from the corpus record; resolution = width x height of the original upload.

### Zero-shot type labels and how good they are

12 classes, 3 to 5 prompts each, mean prompt embedding, argmax (`scripts/images/prompts.py`; `logo` is counted as a
designed graphic). Prompts were fixed on round-1 contact sheets (shard 0, own announcements), then precision was read
by eye on an independent round-2 sample of 30 random images per class (`sheets/<class>_1.jpg` in the dataset):

| Class | Correct of 30 (round 2) | Main errors |
|---|---|---|
| designed graphic | 28 (93%) | an ad poster, a stadium photo with a logo |
| logo card (counted as designed graphic) | 28 (93%) | a photo of badges |
| text-heavy card | 26 (87%) | photos of signs, a whiteboard |
| chart or table | 26 (87%) | diagrams, timelines |
| product screenshot | 25 (83%) | diagrams, designed cards |
| people photo | 25 (83%) | illustrations of people, speaker cards |
| document or paper page | 24 (80%) | app screenshots, book covers |
| code or terminal screenshot | 23 (77%) | app UIs, banners styled as code |
| illustration or render | 22 (73%) | film stills, photos |
| other photo | 21 (70%) | photos where people are the subject, a map |
| meme | 20 (67%) | TV stills, tweets |
| message or social screenshot | 17 (57%) | other software screenshots (12 of the 13 errors) |

Every class is above the 25-inspected minimum; counts are in `labels_precision.json`. Class sizes over the 19,835
images: document 3,271, chart 3,027, social screenshot 2,318, other photo 1,772, product screenshot 1,623,
illustration 1,534, people photo 1,497, designed graphic 1,224 (incl. 691 logo cards), text card 1,123, meme 824,
code screenshot 1,622.

Round 1 changed three things: "product screenshot" had 14/30 (logo cards, posters and text-heavy cards landed there;
fixed by giving logos and posters their own prompts, 23/30 on the same shard after the change, 25/30 on round 2);
"meme" had 6/18 (meme-format prompts, 20/30 on round 2); "ai_art" caught 3D renders and illustration far more than
recognisably AI-generated images and was renamed "illustration_render". The zero-shot margin did not separate right from
wrong labels; the "confident" stricter set (top half of the margin within each class) is a robustness check, not a
validated high-precision set. "Social screenshot" (tweets, chats, emails) is the weakest class: 13 of its 30 were other
software screenshots, so read it as "screenshot of a message or of other software".

### "AI-generated": not reliable

Tried: an open detector (`Ateeqq/ai-vs-human-image-detector`, a SigLIP fine-tune) and a zero-shot score (AI prompts
against photo / designed prompts). Validation by eye (`ai_eye_labels.json`, sheets `ai_illus_0`, `illustration_render_1`,
`ai_photo_0`):

- Among "illustration / render" images, about a third are recognisably AI-generated, a quarter recognisably not, the
  rest cannot be told (21 AI, 17 not, 28 unsure of 66). Among photos, 1 of 36.
- The detector gives p > 0.99 to nearly everything (screenshots, graphics, real photos): it was trained on art against
  photos, not on social-media images. Inside illustrations it still ranks (AUC 0.80 on the 38 confident labels); the
  zero-shot score does not (AUC 0.53; it separates illustration from photo, nothing more).
- The only usable subset: illustration class and detector p >= 0.9999, 644 images; 17 of the 23 eye-labelled images
  that pass are AI (74%), and the threshold was read off those same labels (optimistic).

So the label is not reliable in general, and the effect of "AI slop" is reported only for this subset, as a lean.

## 3. Effects inside accounts

Method: `scripts/launch/effects.py`'s estimators, imported unchanged (`scripts/images/effects.py`). Outcome: likes against
the account's own trailing median (previous 30 posts). Three tests: same-account pairs with posting order balanced,
within-account ratio, and an account fixed-effects regression with the other image types, video, the text attributes
of the launch track and the post type held fixed. Intervals: 95%, accounts resampled. Likes from the Community Archive
rows (the source that records media reliably); views were too few everywhere (below). Posts with a photo whose image
was not fetched are left out. Scopes: every post of company and founder accounts (MAIN), the same with confident type
labels only (STRICT), every post of every account with an own announcement (ALL), launch posts, all own announcements,
ordinary posts, follower tiers, before / from 2026-04-23. Rule fixed before the run (`build_image_evidence.py`): stated
only when all three tests pass in MAIN or STRICT and one passes in the other, same direction; grade A if also confirmed
separately before and from the cutoff.

Image posts measured (Community Archive likes): 8,793 of 37,132 posts of company and founder accounts (782 accounts);
14,973 of 1.36M posts over every account with an announcement (3,653 accounts). Full tables: `effects.json` (every
scope, every test), `image_evidence.json` (graded entries). Main set, company and founder accounts (pairs / accounts;
ratio = likes with / without inside accounts with both; adjusted = regression with everything else held fixed):

| Comparison | Grade | Win rate | Pairs / accounts | Likes ratio | Adjusted | Before / from 2026-04-23 (win) |
|---|---|---|---|---|---|---|
| Any image vs no media | A | 57.1% (55.7-58.6) | 26,953 / 670 | 1.23 (1.16-1.30) | - | 58.6% / 55.6% |
| Chart or table vs no media | A | 61.2% (58.7-63.6) | 10,978 / 365 | 1.45 (1.28-1.65) | 1.32 (1.18-1.49) | 61.9% / 55.6% |
| Chart or table vs other images | A | 54.4% (51.5-57.2) | 7,618 / 353 | 1.23 (1.09-1.39) | - | 54.6% / 57.0% |
| Code screenshot vs no media | B | 59.2% (55.8-62.4) | 7,113 / 232 | 1.46 (1.27-1.70) | 1.32 (1.15-1.51) | 62.1% / 47.8% |
| Code screenshot vs other images | B- | 54.2% (50.7-57.7) | 5,073 / 228 | 1.20 (1.04-1.39) | - | 56.3% / 46.3% |
| Document page vs no media | A | 56.4% (53.6-59.1) | 10,278 / 330 | 1.27 (1.15-1.40) | 1.23 (1.12-1.35) | 56.5% / 56.0% |
| Product screenshot vs no media | A | 59.3% (56.2-62.3) | 8,607 / 315 | 1.23 (1.09-1.39) | 1.12 (1.01-1.24) | 59.9% / 60.8% |
| Designed graphic vs no media | B | 61.1% (58.0-64.2) | 6,991 / 280 | 1.25 (1.09-1.42) | 1.15 (1.04-1.29) | 64.2% / 55.7% |
| People photo vs no media | B | 55.6% (52.1-59.1) | 7,201 / 276 | 1.18 (1.03-1.35) | 1.16 (1.04-1.31) | 57.9% / 45.9% |
| Text card vs other images | A | 41.5% (36.8-46.6) | 4,301 / 231 | 0.73 (0.62-0.85) | - | 41.9% / 24.2% |
| Text card vs no media | none | 51.2% (46.5-56.0) | 5,725 / 234 | 0.90 (0.76-1.05) | 0.82 (0.70-0.95) | |
| Message / social screenshot vs other images | A | 44.2% (41.1-47.5) | 6,002 / 294 | 0.80 (0.71-0.89) | - | 44.9% / 36.9% |
| Face in the image vs image without | B | 46.1% (43.6-48.6) | 8,471 / 430 | 0.88 (0.80-0.96) | - | 46.6% / 50.2% |
| Product screenshot, designed graphic, document, people photo, other photo, illustration, meme vs other images | none | 46.5% to 54.6% | 2,695 to 7,304 each | 0.91 to 1.06, all intervals include 1 | | |
| Aesthetic score, top vs bottom third within type | none | 50.4% (47.9-52.8) | 7,766 / 464 | 1.01 (0.93-1.10) | 0.94 (0.88-1.02) | |
| Sharpness, top vs bottom third | none | 49.6% (47.2-52.0) | 7,406 / 465 | 1.01 (0.92-1.10) | | |
| Resolution of the upload, top vs bottom third | none | 49.2% (46.5-52.0) | 5,057 / 376 | 0.96 (0.85-1.08) | | |
| Colourfulness, top vs bottom third | none | 51.5% (48.9-54.0) | 7,684 / 447 | 1.05 (0.96-1.16) | | |
| Text cover (OCR), top vs bottom third | none | 49.1% (46.6-51.7) | 6,998 / 406 | 0.96 (0.87-1.05) | | |
| Two or more images vs one | none | 51.2% (48.0-54.3) | 6,594 / 312 | 1.07 (0.95-1.22) | | |
| AI-style (confident subset) vs designed graphic | lean | 48.5% (40.8-56.5) | 360 / 63 | 0.80 (0.64-1.02) | | |

The "adjusted" column for quality compares the top third with the middle third (the regression has both a top and a
bottom indicator); none of those passes either. Every A and B finding holds in the stricter confident-label set with
the same direction and a passing test (e.g. chart vs other images 57.2%, 1.32x; text card vs other images 40.2%, 0.73x).

**Launch posts and announcements.** Company and founder launch posts: 2,213 posts, 604 with a fetched image (579
accounts). Too few pairs to grade most comparisons there. The ones with a passing test: chart or table vs other images
68.5% (59.3-79.1), 1.60x (1.17-2.19), 365 pairs, 42 accounts (all own announcements: 68.7%, 1.64x, 589 / 59); chart vs
no media 68.7%, 1.66x (558 / 49). Any image vs none in launch posts: 59.4% (53.8-64.8), 1.16x (0.96-1.42), 989 / 125,
consistent with the launch track (image vs none not established; video is the launch lever). Product screenshot vs
other images in launch posts: 23.9% (16.4-37.0), 0.68x (0.40-1.13), 174 / 30: a lean against screenshots in launches,
opposite to ordinary posts (1.01x). Sharpness top vs bottom in launch posts: 28.9% (20.6-43.0), 0.65x (0.46-0.93), 216 /
34: lean only; ordinary posts show nothing (1.01x).

**Follower tiers.** The chart premium and the text-card penalty are carried by accounts under 100k followers (chart vs
other images 1.51x, 1.22-1.86, 143 accounts, against 0.97x at 100k+; text card vs other images 0.59x under 100k, 0.92x
at 100k+). Any image vs none: 1.30x under 100k, 1.16x at 100k+.

**Every account (ALL).** Same directions throughout, e.g. any image vs none 58.2%, 1.32x (1.27-1.37), 74,111 pairs,
1,873 accounts; chart vs other images 53.4%, 1.17x; text card vs other images 0.75x; message screenshot vs other images
0.87x. Illustration, meme, other photo and message screenshot vs no media pass only here (grade C): any image helps a
little over none.

**Views** (2026 view corpus, all accounts; raw orientation): any image vs none 54.0% of 13,733 pairs, 1.25x
(1.14-1.38); product screenshot vs none 1.42x (1.11-1.78); designed graphic vs none 1.56x (1.22-2.08). Kinds against
each other do not separate on views: chart vs other images 0.88x (0.69-1.11), text card vs other images 0.93x
(0.78-1.08). AI-style vs designed 0.73x (0.52-0.99), 292 pairs.

## 4. Model

**Setup** (`image_table.py`, `finetune.py`). The image-subset table: the 1,018 accounts with 5 or more fetched images
in the training table, every row of theirs (134,296 rows; 14,291 with an image: 10,626 train, 1,103 validation, 2,562
test). Two image inputs as auxiliary inputs of `PostEncoder` (the model already had the slot; `aux_dims`): the SigLIP
embedding (768) and 15 attributes (aesthetic, log sharpness, log resolution, log aspect ratio, text cover, log text
boxes, face, face area, brightness, saturation, colourfulness, dark and white share, edge density, several images),
standardised on the training split. The control table is the identical rows without the inputs. Each of the three
baseline members (ts46_s1, ts46_s2, compare_v2_full) was fine-tuned from its checkpoint for 2,000 steps (lr 2e-4,
warm-up 50) in both arms with the same seed; every run's best validation NLL was at step 500 and later steps overfit
(validation NLL 3.063 -> 3.114). A first pass at 400 steps gave identical arms (pairwise likes 61.10% both).

**Serving.** `xviral/model/features.transform` now accepts auxiliary inputs; `DraftScorer.aux_lookup` (past posts by
post_id) and `draft["aux"]` (the draft's own image) feed them; `tac_serve_eval.py --aux` and `score_launch.py --aux`
pass `media/images/aux_img.npz` and `aux_imgattr.npz`. Models without auxiliary inputs are unaffected.

**Results.** Ensembles of three per arm; paired differences with accounts resampled.

| Benchmark slice | Pairs | Image arm | Control | Base ensemble | Image - control |
|---|---|---|---|---|---|
| Same-topic benchmark, pairs with an image, 1.25x+ | 321 | 67.0% | 64.8% | 65.7% | +2.2 (0.0 to +4.5) |
| same, 2x+ | 217 | 68.7% | 66.4% | 67.3% | +2.3 (0.0 to +4.7) |
| same, 3x+ | 146 | 72.6% | 69.9% | 70.5% | +2.7 (0.0 to +5.8) |
| same, 5x+ | 90 | 82.2% | 82.2% | 83.3% | 0.0 (-3.6 to +2.9) |
| same, image on one side only | 198 | 75.3% | 72.2% | 72.2% | +3.0 (+1.0 to +5.4) |
| same, image on both sides | 123 | 53.7% | 52.9% | 55.3% | +0.8 (-3.9 to +5.9) |
| same, confident (60%+) calls | 13.7% / 12.5% / 15.0% of pairs | 75.0% | 75.0% | 72.9% | |
| Launch pairs with an image, 1.25x+ | 386 | 65.5% | 65.0% | 65.8% | +0.5 (-3.0 to +4.2) |
| launch, 2x+ | 239 | 67.4% | 66.1% | 67.0% | +1.3 (-3.5 to +6.2) |
| Ordinary same-account test pairs (in-table), at least one image | 69,078 / 553 accounts | 61.7% | 61.8% | | -0.2 (-0.5 to +0.2) |
| ordinary, both with an image | 4,569 / 337 | 61.9% | 61.3% | | +0.6 (-0.3 to +1.5) |
| ordinary, every test pair | 1,013,553 | 61.56% | 61.56% | | 0.0 |

Reading: the image input moves the benchmark by about two points where an image is involved, mostly on pairs where
only one side has an image, but the interval reaches zero, the slice is small (321 pairs) and the in-table ordinary
pairs (much larger) show nothing. The likeliest reading is that the model already knows "has a photo" from
`has_photo`/`media_count`, and the type information (chart vs text card, about 1.2x to 1.4x either way, on a minority
of posts) is too small a signal to move overall ranking. The slices were fixed in `bench_images.py` before scoring;
the benchmark pairs were never used for tuning. Not put forward as beating the baseline; worth a re-test with the
benchmark pairs' image coverage completed (only the first photo of each post was fetched) and with the type scores as
a 12-number input instead of the 768-d embedding.

## 5. What did not work, limits

- **CPU budget shaped the data.** Representation (about 0.3 s an image with OCR) was the bottleneck: the fetch was
  stopped at 20,000 of the planned 27,000 images; OCR ran only on priorities 0 to 3; priority 5 (generic training posts)
  was not fetched. Two torch jobs at once on 4 cores slowed both about eightfold (SigLIP 874 s per shard against 100 s);
  run one torch job at a time.
- **Only the first photo of each post** is represented; video thumbnails were not analysed.
- **Type labels are 57% to 93% precise.** The confident-label set (top half of zero-shot margin) is a robustness check;
  the margin itself did not separate right from wrong labels in round 1. "Message / social screenshot" (57%) is best
  read as "screenshot of a message or of other software".
- **AI-generated** is not reliably labelled (section 2). The 0.9999 detector threshold was read off the same 38 labels
  used to check it.
- **Quality tertiles** are within type, over all fetched images; the adjusted column for quality compares top with
  middle third.
- **Views** only in the 2026 view corpus (all accounts), where kinds of image do not separate; likes are the outcome
  everywhere else.
- **Associations inside accounts, not experiments.** About 30 comparisons per scope; grades and the confident-set check
  are the guard against chance. A text card may stand in for posts that had nothing else to show; a chart for posts
  that had a result to report (the "result in the first line" text attribute is held fixed in the adjusted tests, but
  not every confounder can be).
- **Accounts under 1,000 followers** are almost absent; the chart and text-card effects are carried by accounts under
  100k followers.

Files in the dataset: `tracks/images/` (this report, IMAGE_PLAYBOOK.md, image_evidence.json, effects.json,
model_results.json, labels_precision.json, ai_validation.json, ai_eye_labels.json, patch.diff) and `media/images/`
(index.parquet, embeddings.parquet, attributes.parquet, labels.parquet, ai_detector.parquet, aux_img.npz,
aux_imgattr.npz, contact sheets under sheets/).
