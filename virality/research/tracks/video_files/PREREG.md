# Track "video_files": rule and traits fixed before any effect was computed

Written 2026-11-17 while the feature extraction was running, before any outcome was joined to any feature.

## Outcome and estimators (unchanged from the launch and video tracks)

lift = log((likes+1) / (median likes of the account's previous 30 posts + 1)); Community Archive posts only.
Three estimates per trait, intervals from resampling accounts (2,000 draws; 500 for the regression):
pairs (same account, at most 365 days apart, exactly one with the trait, lifts differing 1.25x or more, win rate
averaged over "with it posted first" and "posted second", at most 50 pairs an account), ratio (within-account,
precision-weighted), adjusted (account fixed effects; covariates: first line names the product, three or more
lines, 280+ characters, link in the post, quote, problem opening, portrait, sub-720 px, and for audio traits the
visual kind of the video).

## Sets of posts (all video posts, GIFs excluded unless the trait is about GIFs)

- LAUNCH: launch and open-source video posts of every account. LAUNCH_CF: the same, company and founder accounts.
  STRICT: LAUNCH_CF posts whose opening has an announcement verb directly followed by a product name.
- ANNOUNCE: own-announcement video posts (launch, open source, feature, funding, milestone, hiring) of company and
  founder accounts.
- BROAD: every video post of company and founder accounts.
- BROAD under 100,000 followers and 100,000+; BROAD before and from 2026-04-23; the same splits for LAUNCH.

## Rule

- **survives**: pairs interval excludes 50%, ratio interval excludes 1 and adjusted interval excludes 1, all in the
  same direction, in LAUNCH, LAUNCH_CF or ANNOUNCE with at least 200 pairs and 25 accounts, and the pairs and ratio
  point estimates have the same direction in BROAD. Grade A if also confirmed (pairs interval excluding 50%) both
  before and from the cutoff; otherwise B.
- **broad_only** (grade C): all three exclude no effect in BROAD (200 pairs, 25 accounts) but not in a launch set.
- **leaning**: two of three in a launch set or in BROAD. **none**: otherwise. **too_thin**: under 200 pairs or 25
  accounts in every set. **not_measured**: the attribute failed validation by eye (precision under 70%).
- For an audio trait there is one more condition, because videos autoplay muted: it must also hold among pairs whose
  two videos have the same visual kind (`same_visual` pairs interval excluding 50%, or adjusted-with-visual-kind
  interval excluding 1 when same-visual pairs are under 200). An audio trait that passes without it is reported as
  "confounded with the picture".
- No correction for multiple testing beyond the three tests and two sets; the number of traits tested is reported.

## Traits and cut-offs

Length (true duration from meta.jsonl): bands under 15 s, 15 to 30, 30 to 60, 60 to 120, over 120. Tested as each
band against all other bands, each adjacent pair of bands, and under 60 s against 60 s or more.

Kind over the whole video (from 5 frames spread over the preview plus the frames at 0 s and 2.5 s, each given the
video track's first-frame class rule): screen recording (UI on 60%+ of body frames), talking head (person 60%+),
motion graphics or brand film (title, logo, graphic, real footage 60%+ together, with no UI), slideshow of stills
(not UI, at least 2 cuts, and 85%+ of the frame steps between cuts show no motion), mixed (everything else).
Attributes: share of all frames with a detected face (any; 50%+), share of body frames that are UI (any; 50%+).
The exact thresholds for "cut" and "no motion" are set by eye on contact sheets, without outcomes, and written in
`label_videos.py` before the effects run.

Pace: cuts per minute over the preview, bands 0, up to 6, 6 to 20, over 20 a minute; tested as any cut against none,
and over 12 a minute (one cut every 5 seconds) against fewer among videos with a cut.

Burned-in captions: readable text in the lower third on 3 or more of the 5 body frames, centred, whose wording
changes between frames; used only if 70%+ of a sample is correct by eye.

First 3 seconds: class of the frame at 0 s (the poster frame) and at 2.5 s. "Product in the first 3 seconds" =
either is product UI. Also: opens on a person; opens on a title card or logo; a cut inside the first 3 seconds.

Sound (videos with a duration of 1 s or more, GIFs excluded): audible against silent; speech against sound without
speech; music against sound without music; speech over music against speech alone, and against music alone; speech
alone against music alone; first word within 2 s; musical intro; speech rate above the median of videos with speech
(Whisper base and the voice detector run fast against the 130/170 cut-offs of the audio track, so the cut is the
sample median, fixed here); loud (-16 LUFS or louder); quiet (under -24 LUFS); loudness range 8 LU or more; music
120 beats a minute or more; 3+ onsets a second; spoken opening line (English): names the product, says a name the
post or account carries, states a problem or asks a question, contains a number, states a result, announcement verb.
