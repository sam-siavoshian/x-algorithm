# Launch video playbook: what the data supports

Built 2026-11-16 from the product's own corpus. It follows `LAUNCH_PLAYBOOK.md`, which found that a launch post
with a video or GIF beats the same account's launch posts without media in 65% of comparisons. This page asks what
kind of video. It states only what survived the checks.

## Read this first: only the opening frame was measured

The video files were not obtained. For each video post the corpus holds the poster frame, which is the still the
feed shows before the video plays and, for almost every post, the video's first frame. Everything below is about
that frame and about the video's shape and resolution. That the poster is the first frame was assumed, not checked
against the files.

**Not measured, and no advice is given on them:** length, pace and cuts, motion, captions through the video, sound,
voice-over, music, and what the video shows after its first frame. "Screen recording", "talking head" and so on
mean "the video opens on product UI", "opens on a person"; a video that opens on a title card and then shows the
product for a minute counts as a title-card opening.

## How to read the numbers

- 14,416 poster frames: every video and GIF post of the 798 company and founder accounts, plus other accounts'
  announcement videos. 846 of them are launch posts (438 accounts), 821 are announcements of company and founder
  accounts (311 accounts), 5,296 are any video post by a company or founder account (575 accounts).
- Each post is compared with the same account's usual: likes divided by the median of its previous 30 posts.
  Every comparison is between two video posts of one account.
- *Win rate*: of two video posts by one account, one with the trait and one without, how often the one with it did
  better. 50% is no effect. *Likes ratio*: likes with the trait over likes without, inside accounts that have both.
  *Adjusted*: the same with the post's wording, layout, link, video shape and upload type held fixed.
  Brackets are 95% ranges from resampling accounts.
- A finding is stated only if all three numbers exclude "no effect" among launch videos or among announcement
  videos, and point the same way across all videos of these accounts. This rule was fixed before any number was
  computed. 26 traits were tested, so one passing by chance would not be surprising.
- Grade B: passes the rule, not provable on its own after 23 April 2026. Grade C: holds across all video posts of
  company and founder accounts, but launch videos alone are too few to show it.
- Likes only. These are associations inside accounts, not experiments.

## 1. The one finding for launch videos

**Open the video on a headline-size line of text: the product name or the claim (B).**

| Set of video posts | Win rate | Likes ratio | Adjusted | Pairs / accounts |
|---|---|---|---|---|
| Launch posts, every account | 64.6% (55.6 to 71.2) | 1.35x (1.07 to 1.74) | 1.26x (0.95 to 1.64) | 340 / 57 |
| Announcements, company and founder accounts | 59.5% (52.5 to 65.7) | 1.27x (1.03 to 1.55) | 1.29x (1.02 to 1.68) | 495 / 68 |
| Every video post, company and founder accounts | 54.0% (51.1 to 57.0) | 1.13x (1.02 to 1.26) | 1.08x (0.99 to 1.19) | 5,242 / 283 |

What it means: the first frame carries a line of text at least 8% of the frame height tall. In practice that is a
title card ("Introducing Atlas"), a product name over footage, or a headline. It matches the launch playbook's
strongest finding for the post text (name the product in the first line).

How far it holds:
- It passes all three tests among announcement videos. Among launch videos two of three pass; the adjusted ratio
  just includes no effect.
- Before 23 April 2026: 66.5% (55.2 to 75.3), 226 pairs. Since then only 21 pairs exist (70.8%, 50.0 to 85.4).
- It is not shown separately for small or large accounts: across all videos, under 100,000 followers 52.6%
  (47.5 to 57.7), above 48.7% (42.6 to 54.9).
- The text detector is right about 4 times in 5 (31 of 40 frames read by eye carry a headline-size line).

## 2. Holds for these accounts' videos in general, not shown for launch videos alone (C)

Measured on every video post of company and founder accounts. Launch videos alone give too few pairs.

| In the stronger version | Win rate | Likes ratio | Adjusted | Pairs / accounts |
|---|---|---|---|---|
| Landscape, not portrait | 61.2% (56.7 to 65.6) | 1.56x (1.31 to 1.89) | 1.47x (1.23 to 1.76) | 2,980 / 185 |
| 1080 px or more, not under 720 px | 60.6% (54.6 to 66.7) | 1.70x (1.32 to 2.10) | 1.46x (1.11 to 1.82) | 1,898 / 160 |
| 16:9, not another shape | 55.9% (53.1 to 58.9) | 1.27x (1.16 to 1.41) | 1.19x (1.07 to 1.30) | 6,791 / 364 |
| Opens on a title card or logo, not on product UI | 56.0% (50.9 to 61.1) | 1.24x (1.07 to 1.44) | 1.18x (1.04 to 1.33) | 2,057 / 171 |
| Opens on a title card or logo, not on a person | 59.3% (52.2 to 66.0) | 1.41x (1.15 to 1.70) | 1.27x (1.05 to 1.49) | 1,543 / 149 |
| Opens on a title card or logo, against every other opening | 55.1% (51.0 to 58.9) | 1.21x (1.08 to 1.35) | 1.14x (1.03 to 1.27) | 4,042 / 229 |

1. **Landscape, not portrait.** This is an effect for accounts under 100,000 followers: 67.8% (61.1 to 73.7),
   likes ratio 2.25x (1.66 to 2.99). For accounts above 100,000 there is none (50.2%, 1.01x). Among launch videos
   only 66 pairs exist; the pair test cannot tell (47.9%, 30.9 to 77.7) while the likes ratio agrees (1.84x, 1.11
   to 3.31). Read it with care: portrait videos from these accounts are mostly phone clips, so this may be
   "produced video against casual clip" as much as the shape itself.
2. **Full resolution.** 1,697 of the video posts are 1080 px or more and 315 are under 720 px, so this says
   "do not post a low-resolution video", not "higher is better". Strongest under 100,000 followers (62.6%, 2.22x).
3. **16:9.** Also in the same direction after 23 April 2026 (59.2%, 49.8 to 68.6; likes ratio 1.38x, 1.03 to 1.89).
4. **A title card before the product UI.** The same comparison among announcement videos: 63.2% (53.4 to 74.1),
   likes ratio 1.38x (1.03 to 1.92), 171 pairs from 38 accounts. That is below the 200 pairs the rule asks for, so
   it is graded C, but it agrees with finding 1.

## 3. A leaning, not a finding

**Opening on raw product UI.** A third of launch videos open on product UI. They won 39.8% of pairs against the
same account's other launch videos (so the other opening won 60.2%, 53.1 to 69.0; 333 pairs, 58 accounts). The
likes ratio (1.25x, 0.995 to 1.56) just includes no effect; among announcement videos the pairs and the ratio pass
and the adjusted ratio does not. Across all videos there is no difference (51.7%). Two of three tests pass in each
launch set and never all three, so no recommendation is made beyond finding 1.

## 4. Tested, no effect found

| Trait of the first frame or file | Win rate, all videos | Pairs / accounts |
|---|---|---|
| Any readable text in the first frame | 49.2% (46.7 to 51.7) | 7,284 / 380 |
| Text covers 8% or more of the frame, against under 2% | 51.0% (47.4 to 54.9) | 4,067 / 274 |
| A face is visible | 46.3% (42.5 to 49.7); likes ratio 0.90x (0.79 to 1.03) | 5,583 / 310 |
| A large face (a quarter of the frame height), against no face | 48.3% (44.0 to 52.8) | 3,409 / 217 |
| Opens on a person, against every other opening | 45.0% (41.3 to 48.6); adjusted 0.88x (0.79 to 1.01), includes no effect | 5,189 / 279 |
| Opens on motion graphics, a render or generated imagery | 52.9% (48.8 to 56.9) | 3,739 / 215 |
| Opens on real-world footage (hardware, a room, outdoors) | 51.9% (47.2 to 56.8) | 3,059 / 174 |
| Opens on a bare frame (black, blank, a small mark) | 53.8% (48.1 to 59.2) | 2,437 / 151 |
| A dark first frame | 51.4% (48.3 to 54.7) | 5,657 / 312 |
| Square, against landscape | 46.2% (41.7 to 50.7); likes ratio 0.84x (0.72 to 0.96) | 3,039 / 193 |
| A GIF, against a video | 43.4% (35.3 to 51.6); likes ratio 0.68x (0.48 to 0.90) | 1,268 / 83 |
| Uploaded through Media Studio, against an ordinary upload | 50.9% (42.3 to 61.3) | 2,265 / 122 |

Among launch videos, how the six kinds of opening do against the account's usual cannot be told apart:
product UI 3.2x (2.6 to 3.9), a person 2.9x (2.3 to 3.7), title card or logo 3.6x (2.9 to 4.7), graphics 3.3x
(2.4 to 4.3), real-world footage 2.5x (1.7 to 3.7), a bare frame 3.5x (2.7 to 4.6). All launch videos do well; the
ranges overlap.

A face in the first frame is common advice. This corpus does not support it for these accounts: a visible face
shows no gain, and a person-first opening does slightly worse than the account's other videos across all video
posts (not provable among launch videos).

## 5. What could not be measured

| Question | Why not |
|---|---|
| Short against long | No video files. The public embed route returned duration and the files on a test of 6 posts; the bulk fetch was not permitted in the worker session. |
| Pace, cuts per minute, motion | Same. |
| Burned-in captions | Same. A proxy from the first frame (text in the lower third) was checked by eye and is a caption in only 14 of 40 frames, so it was dropped. |
| Sound, voice-over, music | Same. No audio was obtained. |
| Screen recording across the whole video, share of frames with a face | Same. Only the first frame was classified. |
| Any effect on views | View counts exist only in the part of the corpus that does not record video. |
| Charts or slides as the opening | 50 such videos among company and founder accounts. |
| Launch videos since 23 April 2026 | 151 launch videos; every launch finding has under 30 pairs in that period. |

## 6. Does the first frame improve the product's predictor?

No. The predictor was re-fitted on the video posts that have a poster frame, once with 36 extra inputs describing
that frame and once without, from the same starting point, two seeds each.

| Held-out pairs (same account, posted from 23 April 2026) | Without | With the first frame | Difference |
|---|---|---|---|
| Two video posts, 2,149 pairs, 212 accounts | 58.2% | 58.4% | +0.2 (-0.7 to +1.0) |
| The same, likes differ 2x or more, 1,584 pairs | 60.5% | 60.5% | +0.1 (-0.9 to +1.0) |
| Launch pairs with a video on either side, 361 pairs, 47 accounts | 64.8% | 65.4% | +0.5 (-0.9 to +2.0) |

On its own, the first frame picks the better of two videos by one account 54.7% of the time (52.0 to 57.5; 3,741
held-out pairs). That is a real but small signal, and it adds nothing to what the predictor already uses. The
same-topic benchmark has 58 held-out pairs with a video on either side, too few to measure anything.

## Limits

- First frame only; see the top of this page.
- Observational. Accounts choose how to open a video; "inside the same account, videos opened this way did
  better" is not proof that re-cutting a video produces the gain.
- The kind of opening was assigned by an image model and checked by eye on 40 frames per kind (one reader):
  product UI 36 of 40 correct, a person 34, title card or logo 36, graphics 33, real-world footage 35, bare frame
  30. A visible face: 36 of 40. Mislabelled frames pull every difference toward "no effect".
- The accounts are mostly technology and AI companies with large followings; 290 of the 846 launch videos are from
  accounts known to be under 100,000 followers.
