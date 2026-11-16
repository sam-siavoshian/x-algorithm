# Image playbook: what the data supports about pictures in startup and company posts on X

Built 2026-11-16 from the product's own corpus. Every number is measured on that corpus. Where the data cannot answer
a question, this page says so and gives no recommendation.

## How to read this

**What was measured.** 19,835 images (the first photo of a post) fetched for posts of accounts that make own
announcements: every photo post of the 782 company and founder accounts the launch track identified, plus a sample of
other accounts. Each image was given a type by a zero-shot image model (precision checked by eye, 30 images per type:
designed graphic 93%, text-heavy card 87%, chart or table 87%, product screenshot 83%, people photo 83%, document or
paper page 80%, code screenshot 77%, illustration or render 73%, other photo 70%, meme 67%, message or social
screenshot 57%) and measured for quality (aesthetic score, sharpness, resolution, colourfulness, amount of text).

**The yardstick.** As in the launch playbook: likes against the same account's own usual (median of its previous 30
posts). Every comparison is inside one account: two posts by the same account, one with the trait and one without.

**Three numbers per finding.** Win rate (of two posts by one account, how often the one with the trait did better
against the account's usual, posting order balanced; 50% = no effect), likes ratio inside accounts that have both, and
an adjusted ratio that holds the other image types, video, the text attributes and the post type fixed (where it is
defined). Brackets are 95% ranges from resampling accounts. Main set: every post of company and founder accounts
(37,132 posts, 8,793 with a fetched image). A finding is stated only if it passes a rule fixed before the run (all
tests pass in the main set or in a stricter set restricted to confident type labels, at least one passes in the other,
same direction). **A**: also confirmed separately before and from 23 April 2026. **B**: passes, but one period is too
small to confirm it on its own.

Likes are the outcome. Views exist only in the 2026 view corpus and are reported once, in section 1b, over all
accounts. These are associations inside accounts, not experiments. About 30
comparisons were tested; one weak pass by chance would not be surprising.

## 1. Does a picture matter?

**Yes, a little, for ordinary posts. Not established for launch posts.**

| | Win rate | Likes ratio | Pairs / accounts | Grade |
|---|---|---|---|---|
| Any image, against no media (every post) | 57.1% (55.7 to 58.6) | 1.23x (1.16 to 1.30) | 26,953 / 670 | A |
| Any image, against no media, launch posts only | 59.4% (53.8 to 64.8) | 1.16x (0.96 to 1.42) | 989 / 125 | not established |

**1b. On views** (2026 view corpus, every account with an own announcement; the only place views are numerous enough):
any image against no media 54.0% of 13,733 pairs, 1.25x views (1.14 to 1.38). The differences between kinds of image
in section 2 are not seen on views: chart or table against other images 0.88x (0.69 to 1.11), text-heavy card against
other images 0.93x (0.78 to 1.08). Treat section 2 as a finding about likes.

For a launch, the launch playbook's finding stands: a video or GIF is what separates the stronger launch posts (1.6x
against an image). An image is better than nothing on an ordinary post; the kind of image matters more than whether
there is one.

## 2. Which kind of picture

Against other images (the question a founder actually faces: which picture to attach):

| Grade | Kind | Win rate | Likes ratio vs other images | Pairs / accounts |
|---|---|---|---|---|
| A | **Chart, benchmark or results table** | 54.4% (51.5 to 57.2) | **1.23x (1.09 to 1.39)** | 7,618 / 353 |
| B- | Code or terminal screenshot | 54.2% (50.7 to 57.7) | 1.20x (1.04 to 1.39) | 5,073 / 228 (not confirmed after April) |
| none | Product screenshot | 49.9% (46.6 to 53.0) | 1.01x (0.91 to 1.13) | 5,981 / 311 |
| none | Designed graphic (logo, banner, mockup) | 54.6% (51.1 to 58.1) | 1.06x (0.93 to 1.19) | 5,138 / 279 |
| none | Document or paper page | 49.1% (46.6 to 51.9) | 1.00x (0.91 to 1.09) | 7,304 / 323 |
| none | Photo of people | 49.4% (45.8 to 53.0) | 1.01x (0.89 to 1.14) | 5,156 / 269 |
| none | Illustration, 3D render, AI-style art | 46.5% (41.9 to 51.1) | 0.91x (0.80 to 1.04) | 3,991 / 215 |
| none | Other photo (objects, places) | 47.1% (43.5 to 50.4) | 0.98x (0.83 to 1.13) | 4,743 / 244 |
| none | Meme | 47.5% (42.4 to 52.6) | 0.95x (0.78 to 1.19) | 2,695 / 132 |
| A | **Text-heavy announcement card** | other images win 58.5% (53.4 to 63.2) | **0.73x (0.62 to 0.85)** | 4,301 / 231 |
| A | **Screenshot of a tweet, chat or email** | other images win 55.8% (52.5 to 58.9) | **0.80x (0.71 to 0.89)** | 6,002 / 294 |
| B | An image with a face in it (any kind) | image posts without a face win 53.9% (51.4 to 56.4) | 0.88x (0.80 to 0.96) | 8,471 / 430 |

Against no media, the same kinds rank the same way: chart or table 1.45x (1.28 to 1.65; adjusted 1.32x), code
screenshot 1.46x (1.27 to 1.70; B), document page 1.27x (A), designed graphic 1.25x (B), product screenshot 1.23x (A;
adjusted 1.12x), photo of people 1.18x (B); a text-heavy card 0.90x (0.76 to 1.05; adjusted 0.82x, 0.70 to 0.95) and a
message screenshot 0.98x are no better than posting no image.

**For launch posts specifically** (company and founder accounts, too few pairs to grade on their own): a chart or
benchmark table beat the account's other image launch posts 68.5% of the time (59.3 to 79.1), 1.60x (1.17 to 2.19),
365 pairs, 42 accounts. Product screenshots trailed other images in launch posts (won 23.9% of 174 pairs, 30 accounts;
ratio 0.68x, 0.40 to 1.13): leaning, not established.

**Where it is strongest.** The chart effect is concentrated in accounts under 100,000 followers (vs other images 1.51x,
1.22 to 1.86, 143 accounts; at 100k+ 0.97x). The text-card penalty likewise (0.59x under 100k; 0.92x at 100k+).

## 3. Does visual quality matter?

**No.** Inside each kind of image, the top third against the bottom third of the same account's images:

| Quality measure | Win rate | Likes ratio | Pairs / accounts |
|---|---|---|---|
| Aesthetic score (LAION predictor) | 50.4% (47.9 to 52.8) | 1.01x (0.93 to 1.10) | 7,766 / 464 |
| Sharpness | 49.6% (47.2 to 52.0) | 1.01x (0.92 to 1.10) | 7,406 / 465 |
| Resolution of the upload | 49.2% (46.5 to 52.0) | 0.96x (0.85 to 1.08) | 5,057 / 376 |
| Colourfulness | 51.5% (48.9 to 54.0) | 1.05x (0.96 to 1.16) | 7,684 / 447 |
| Share of the image covered by text | 49.1% (46.6 to 51.7) | 0.96x (0.87 to 1.05) | 6,998 / 406 |
| Two or more images instead of one | 51.2% (48.0 to 54.3) | 1.07x (0.95 to 1.22) | 6,594 / 312 |

These are measured "no effect" results with narrow intervals, not missing data: on this corpus a prettier, sharper
or higher-resolution version of the same kind of picture does no better. What the picture shows matters; how polished
it is does not.

## 4. "AI slop"

**Cannot be labelled reliably; a weak sign that it does worse than a designed graphic, not established.** No tool available here tells AI-generated images apart
well enough: an open AI-image detector scores nearly every screenshot and graphic as AI (p > 0.99), and by eye only
about a third of the "illustration / render" images are recognisably AI-generated (another third cannot be told). On
the subset most likely to be AI-generated (detector p >= 0.9999 inside illustrations; 74% AI among 23 checked by eye),
AI-style images against designed graphics: 48.5% (40.8 to 56.5), 0.80x (0.64 to 1.02), 360 pairs, 63 accounts. In the
stricter set (confident type labels as well) the AI-style version won 28% (14 to 42), 0.58x (0.43 to 0.76), but that
is 125 pairs from 21 accounts, below the size this playbook grades. Illustrations and renders as a whole: 0.91x (0.80
to 1.04) against other images. Two other measurements point the same way: across every account with an announcement,
AI-style images against designed graphics did 0.75x on likes (0.58 to 0.97; 384 pairs, 73 accounts) and 0.73x on views
(0.52 to 0.99; 292 pairs). Consistent direction, small samples, a label that is right about three times in four: a
lean, not a finding. If anything, AI-looking art trails a designed graphic.

## 5. What to tell a startup

1. On an ordinary post, attach an image rather than none (1.23x, A). For a launch, attach a video (launch playbook).
2. **If you have a number, show it as a chart or table** (results, benchmarks, growth): 1.23x the account's other image
   posts (A), 1.45x a post with no image; strongest for accounts under 100k followers and in launch posts.
3. **Do not put the announcement in a text card.** A picture of words did 0.73x other images (A) and no better than no
   image; write the words in the post.
4. Screenshots of tweets, chats or emails do 0.80x other images (A).
5. Product screenshots, designed graphics, document pages, people photos, other photos and memes do alike against
   each other: pick by what you have. For AI-generated art there is a weak, unconfirmed sign that it trails a designed
   graphic (0.80x); prefer a designed graphic or a real screenshot when in doubt.
6. Do not spend on polish: aesthetic score, sharpness and resolution show no effect.
7. Faces: image posts with a face did 0.88x those without (B); a weak finding, do not build advice on it alone.

## What this cannot tell you

- Video thumbnails were not analysed (video is the launch track's subject).
- Views: too few posts with view counts.
- Accounts under 1,000 followers are almost absent.
- Type labels are 57% to 93% precise; the stricter confident-label set agrees with every stated finding, but a
  mislabelled tenth of a class dilutes its effect.
- The prediction model does not gain from the image itself yet: with the image embedding and attributes added it is
  +2.2 points on 321 benchmark pairs involving an image (interval 0.0 to +4.5) and unchanged on launch and ordinary
  pairs (REPORT.md, section 4). Advice on images should come from this page, not from the model's score.
