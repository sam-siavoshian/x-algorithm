# Track "video_files": what kind of video, and what kind of sound, travels

2026-11-17. Deliverables in the dataset under `tracks/video_files/`: this report, `VIDEO_AUDIO_PLAYBOOK.md` (for a
marketing lead), `video_file_evidence.json` and `audio_evidence.json` (shape of `launch_evidence.json`),
`effects.json` (every estimate), `model_pairs.json`, `validation.json`, `PREREG.md`, `video_features.parquet` and
`video_emb.npy` (per-video tables), `frame_features_raw.parquet`, contact sheets under `sheets/`, `patch.diff`.

## Headline

- 5,975 video posts with an outcome were analysed over their whole length (2 frames a second, first 180 s) and, for
  the 4,248 with an audio track, over the first 90 s of sound. 53 traits were tested inside accounts with the launch
  track's method and a rule written down beforehand (`PREREG.md`).
- **Nothing passes the rule on launch videos alone** (779 launch videos, 409 accounts; 765 announcement videos of
  company and founder accounts). Most traits have 150 to 400 launch pairs and intervals of plus or minus 10 to 15
  points. No grade A or B finding.
- **Fourteen traits hold across all video posts of company and founder accounts** (4,975 posts, 556 accounts; grade
  C: true of these accounts' videos in general, not shown for launch videos):
  - Length: 60 to 120 seconds does best (56.4%, 52.8 to 59.8; 1.31x likes; adjusted 1.25x, 1.13 to 1.40). Under
    15 seconds does worst (43.7%; 0.76x). Launch videos point the same way (60 to 120 s: 61.7%, 46.4 to 75.4; 1.37x,
    1.10 to 1.74; adjusted 1.30x, 1.02 to 1.63: two of three tests).
  - Kind: a person on camera for most of the video does worse than everything else (45.2%, 41.9 to 48.6; 0.81x), and
    worse than a brand film (42.7%; the film gets 1.40x). This is an effect of accounts with 100,000+ followers (37.1% there,
    52.7% under 100,000). Screen recordings are average (51.1%; 0.98x). A face on screen for half the video or more:
    45.6%, 0.83x.
  - Pace: at least one scene change beats one continuous shot (55.6%; 1.26x; adjusted 1.20x); more than 20 changes a
    minute beats fewer (55.3%; 1.20x; adjusted 1.23x). A scene change inside the first 3 seconds: 53.3%, 1.14x in
    general, but the opposite direction among announcements (45.1%, 38.0 to 50.9), so it carries no advice.
  - Burned-in captions: videos with them do slightly worse (44.7%, 40.9 to 48.7; 0.87x; adjusted 0.87x, 0.78 to
    1.00). The label is 72% precise and captions go with talking-head videos, so read this as "no gain found".
  - Sound: among videos with sound, those without speech beat those with speech (58.0%, 54.2 to 61.8; 1.36x;
    adjusted with the visual kind fixed 1.23x, 1.08 to 1.41; 57.1% among pairs of the same visual kind). Music alone
    against speech alone: 62.1% (56.6 to 68.1), 1.66x, adjusted 1.48x, same-kind pairs 62.7% (51.6 to 72.1). A wide
    loudness range (8 LU+): 54.9%, 1.19x, adjusted 1.18x.
- **Does sound matter at all?** Having sound against silence is not established: 53.1% (49.9 to 56.5), 1.15x, and
  1.02x (0.90 to 1.14) once length is also held fixed (silent videos are short clips). A video with speech does no
  better than a silent one (49.4%). What separates videos is whether the sound is a voice or music, and a music bed
  marks a produced film more finely than the five visual kinds do; with autoplay muted, that is the likelier reading.
- No effect, with tight intervals in the broad set: screen recording against the rest, product UI in the first 3
  seconds, a face on screen at any point, loudness, music tempo, speech rate, time to the first word, and every trait
  of the spoken opening line.
- **Model: no gain that survives.** 46 new inputs (16 components of a pooled frame embedding, 29 video and audio
  attributes, a flag) against a same-rows control, two seeds: +1.3 points on 848 held-out video-video pairs (-0.3 to
  +2.9; both seeds positive), -0.4 (-1.1 to +0.2) on 6,397 video-against-other pairs, -2.4 (-6.6 to +0.5) on 290
  launch pairs with a video. No checkpoint uploaded.

## 1. Data and features

| Step | Result |
|---|---|
| Files in the dataset | 7,420 previews, 5,291 audio tracks, metadata for 7,420 fetched posts (451 GIFs, 1,678 videos with no audio track) |
| Posts with an outcome (lift against the account's trailing median, Community Archive, rebuilt with `scripts/launch/build_table.py`) | 5,976 of the 7,420; 5,875 Community Archive. The other 1,444 have under 8 previous posts or were captured too early |
| Visual features | 5,999 videos: all with an outcome plus 24. The remaining 1,421 were not processed for time (5.6 s of CPU a video) |
| Audio features | 4,248 of the 4,260 tracks of the posts with an outcome (12 failed to decode) |
| Like count at fetch against the corpus | median ratio 0.995, 10th to 90th percentile 0.96 to 1.04: the corpus counts are final |

Sets of posts (GIFs excluded): launch videos of every account 779 posts / 409 accounts; of company and founder
accounts 593 / 251 (strict subset 357 / 166); announcement videos of company and founder accounts 765 / 299; every
video post of company and founder accounts 4,975 / 556 (1,700 under 100,000 followers, 1,272 at 100,000+, the rest
unknown; 662 from 2026-04-23).

**Visual** (`scripts/video_files/frame_features.py`, `label_videos.py`). On every frame: grey difference and
colour-histogram distance to the previous frame, faces (YuNet). On 6 sampled frames (2.5 s and the mid-points of 5
equal parts): SigLIP-2 base embedding, zero-shot scores against the video track's 34 prompts, and for the 5 body
frames text detection and recognition in the lower 40% of the frame. The frame at 0 s is the video track's poster
frame. Each sampled frame gets the video track's first-frame class rule; the video's kind is the class on 60% or more
of the 5 body frames.

Shares among all video posts of company and founder accounts: screen recording 41%, person on camera 24%, brand film
(motion graphics, titles, filmed or generated footage, no UI) 22%, mixed 12%, slideshow under 1%. Length: under 15 s
22%, 15 to 30 s 21%, 30 to 60 s 23%, 60 to 120 s 19%, over 120 s 15%. No scene change 23%; over 20 a minute 28%.
Captions 14%. Sound: silent 37%, music only 24%, speech over music 19%, speech only 18%, other 3%.

**Validation by eye** (one reader, contact sheets, 40 videos a label; `validation.json`, `sheets/`):

| Label | Correct of 40 | Note |
|---|---|---|
| Screen recording | 37 | misses: animated diagram, filmed QR code, animated graph |
| Person on camera | 37 (32 as talking head or interview in the strict sense) | 3 screen recordings with a webcam inset; 5 acted or generated scenes with people |
| Brand film | 36 | a broad class; for AI companies it includes showcases of generated video and robots |
| Mixed | 27 | residual class: 13 are plain screen recordings or films that the 5 frames split |
| Slideshow of stills | not validated | 60 videos in all (36 in the broad set): too rare to test |
| Burned-in captions | 29 (72%) | the rest are animated titles or UI text; recall not measured |
| Scene changes | 42 of 54 flagged steps on 9 sheets are real cuts or abrupt changes | see below |

**What the pace measure resolves.** At 2 frames a second a "scene change" is a half-second step whose two frames
differ strongly in both brightness pattern and colour histogram. It separates no change, occasional change and
constant change. It counts fast animation and cross-fades as changes, misses zooms and page changes inside
white-background screen recordings (a navigation-heavy demo can read as "no cut"), counts two cuts under half a
second apart once, and tops out at 120 a minute. A first version that merged consecutive flagged steps undercounted a
fast-cut film (8 for about 22 real cuts in 24 s) and was dropped before any outcome was read. Exact cuts per minute
are not available from these files.

**Audio** (`scripts/video_files/audio_features.py`, using `xviral/media/audio.py`): loudness (EBU R128), voice
detection, AudioSet tagger, sound class, Whisper transcript with word times, opening-line traits by the regular
expressions of `xviral/text/post_type.py`. Three reductions for CPU time: at most 4 ten-second tagger windows a file,
Whisper `base` on the first 30 s only, no percussive share. Speech rate reads high with this setup (median 195 words a
minute), so the cut is the sample median, as fixed in `PREREG.md`. Nothing was checked by ear. Spoken product names
are mostly missed by the "names the product" expression because transcripts lower-case and misspell them (10% of
opening lines match); "says a name the post or account carries" (12%) is the usable version.

## 2. Effects inside accounts

`scripts/video_files/effects.py`, `build_evidence.py`. Estimators imported from `scripts/audio/effects.py`: pairs
(posting order balanced, at most 365 days apart, lifts differing 1.25x or more, at most 50 an account), within-account
likes ratio, adjusted ratio with account fixed effects; intervals from resampling accounts. Rule in `PREREG.md`.

One correction after the first run on the visual traits and before anything was written up: the adjusted estimate
now also holds the post type fixed, as the launch track's regression does. Without it, in the set of all video posts,
a trait more common in launch videos inherits the launch lift. It changed one status: a title card or logo in the
first 3 seconds went from grade C to leaning (adjusted 1.07x, 0.97 to 1.18).

L = launch videos, every account. B = every video post of company and founder accounts. Adjusted for audio traits
includes the visual kind.

| Trait (version that does better) | Status | L win rate, pairs / accounts | L ratio | B win rate, pairs / accounts | B ratio | B adjusted |
|---|---|---|---|---|---|---|
| 60 to 120 s, against other lengths | C | 61.7% (46.4-75.4), 288 / 49 | 1.37 (1.10-1.74) | 56.4% (52.8-59.8), 5,045 / 291 | 1.31 (1.17-1.48) | 1.25 (1.13-1.40) |
| Not under 15 s | C | 55.6% (39.7-73.1), 155 / 29 | 1.00 (0.62-1.59) | 56.3% (52.8-59.6), 4,963 / 283 | 1.32 (1.17-1.48) | 1.23 (1.08-1.38) |
| 60 s or more, against under 60 s | C | 56.5% (43.5-70.2), 316 / 52 | 1.12 (0.90-1.40) | 53.7% (50.3-57.1), 5,889 / 337 | 1.23 (1.11-1.38) | 1.17 (1.06-1.29) |
| Not a person on camera | C | 56.8% (40.6-67.1), 211 / 38 | 0.94 (0.65-1.32) | 54.8% (51.4-58.1), 4,960 / 275 | 1.23 (1.09-1.38) | 1.19 (1.06-1.32) |
| Brand film, against a person on camera | C | 62.0% (38.2-83.6), 50 / 14 | 1.21 (0.72-1.72) | 57.3% (53.0-61.7), 2,198 / 186 | 1.40 (1.20-1.63) | 1.36 (1.18-1.57) |
| A face in under half the frames | C | 58.8% (44.7-69.9), 215 / 34 | 1.15 (0.81-1.57) | 54.4% (50.5-58.3), 5,165 / 288 | 1.21 (1.06-1.37) | 1.19 (1.06-1.35) |
| At least one scene change | C | 56.6% (43.5-69.1), 227 / 42 | 0.86 (0.54-1.30) | 55.6% (52.3-58.8), 5,518 / 306 | 1.26 (1.12-1.41) | 1.20 (1.07-1.32) |
| Over 20 scene changes a minute | C | 60.0% (44.5-72.1), 256 / 54 | 1.09 (0.81-1.43) | 55.3% (52.0-58.7), 5,803 / 314 | 1.20 (1.07-1.34) | 1.23 (1.10-1.41) |
| No burned-in captions | C, weak label | 59.4% (46.5-70.5), 186 / 40 | 1.04 (0.73-1.39) | 55.3% (51.3-59.1), 4,083 / 232 | 1.15 (1.01-1.29) | 1.15 (1.00-1.29) |
| Scene change inside the first 3 s | C, no advice | 50.6% (42.4-58.2), 408 / 84 | 0.93 (0.75-1.14) | 53.3% (50.5-56.1), 6,897 / 377 | 1.14 (1.03-1.25) | 1.14 (1.04-1.25) |
| Sound without speech, against speech | C | 63.5% (50.1-73.3), 171 / 42 | 1.04 (0.71-1.49) | 58.0% (54.2-61.8), 3,142 / 225 | 1.36 (1.18-1.58) | 1.23 (1.08-1.41) |
| Music alone, against speech alone | C | 77.6% (58.2-91.2), 40 / 13 | 1.68 (1.02-3.01) | 62.1% (56.6-68.1), 1,405 / 143 | 1.66 (1.35-1.99) | 1.48 (1.20-1.80) |
| Loudness range 8 LU or more | C | 61.1% (50.9-71.8), 188 / 38 | 1.41 (1.05-1.86) | 54.9% (50.7-58.9), 3,261 / 239 | 1.19 (1.05-1.37) | 1.18 (1.04-1.32) |
| Brand film, against the rest | leaning | 60.3% (46.1-74.0), 261 / 39 | 1.22 (0.89-1.64) | 52.4% (49.1-55.5), 4,969 / 277 | 1.16 (1.04-1.29) | 1.19 (1.06-1.32) |
| Sound, against silent | leaning | 58.4% (48.5-70.1), 245 / 48 | 1.05 (0.75-1.49) | 53.1% (49.9-56.5), 5,259 / 283 | 1.15 (1.02-1.29) | 1.14 (1.03-1.28) |
| No product UI in the first 3 s | leaning | 59.7% (51.4-67.8), 342 / 57 | 1.23 (0.96-1.56) | 50.1% (46.8-53.6), 5,697 / 304 | 1.05 (0.94-1.16) | 1.06 (0.95-1.17) |
| Title card or logo in the first 3 s | leaning | 54.3% (44.7-61.6), 369 / 61 | 1.02 (0.83-1.26) | 54.9% (51.7-58.1), 4,654 / 253 | 1.17 (1.04-1.30) | 1.07 (0.97-1.18) |
| Musical intro before speech | leaning | 59.0% (35.4-83.9), 25 / 12 | 0.85 (0.46-1.50) | 58.0% (51.7-64.9), 964 / 94 | 1.37 (1.09-1.72) | 1.24 (0.99-1.55) |

"No product UI in the first 3 seconds" is the whole-video counterpart of the video track's "not opening on product
UI": the same two tests pass among announcements (59.9%, 51.3 to 68.4; 1.31x, 1.03 to 1.68) and nothing shows in the
broad set, so it stays leaning.

**Sound with the picture held fixed** (broad set):

| Trait | All pairs | Pairs of the same visual kind | Adjusted with kind | Adjusted with kind and length |
|---|---|---|---|---|
| Sound, against silent | 53.1% (49.9-56.5) | 57.0% (52.7-61.4), 2,787 | 1.14 (1.03-1.28) | 1.02 (0.90-1.14) |
| Speech, against silent | 49.4% (45.0-53.6) | 52.2% (46.3-58.4), 1,454 | 1.02 (0.88-1.18) | 0.88 (0.75-1.05) |
| No speech, against speech (videos with sound) | 58.0% (54.2-61.8) | 57.1% (51.5-62.7), 1,296 | 1.23 (1.08-1.41) | 1.39 (1.20-1.61) |
| Music alone, against speech alone | 62.1% (56.6-68.1) | 62.7% (51.6-72.1), 493 | 1.48 (1.20-1.80) | 1.59 (1.27-2.02) |
| Music alone, against speech over music | 59.0% (53.7-63.8) | 56.1% (49.6-63.0), 804 | 1.21 (1.04-1.41) | 1.46 (1.24-1.70) |
| Loudness range 8 LU+ | 54.9% (50.7-58.9) | 57.4% (53.4-61.3), 1,758 | 1.18 (1.04-1.32) | 1.16 (1.03-1.31) |
| Opening line says a name the post carries | 57.1% (51.9-62.1) | 53.1% (47.3-58.9), 623 | 1.29 (1.07-1.54) | 1.28 (1.09-1.57) |

Two audio traits pass the three tests but not the same-picture check and are marked `confounded_with_picture`: music
alone against speech over music, and the opening line saying the product's name.

**No effect or not established** (broad set, win rate for the trait as named): screen recording 51.1% (47.9-54.2),
ratio 0.98; product UI in 3 of 5 frames 51.1%; product UI at any point 52.1%, adjusted 1.06 (0.96-1.16); a face at
any point 51.9%; mixed kinds 50.2%; 30 to 60 s 52.8% (50.0-55.7), ratio 1.05; 15 to 30 s 47.9%; over 120 s 48.4%;
product UI in the first 3 s 49.9%; a person in the first 3 s 46.5% (43.0-50.0); over 12 changes a minute among
videos with a cut 53.7%, adjusted 1.12 (0.99-1.25); music against sound without music 53.9% (49.8-58.0), adjusted
1.15 (0.98-1.33); speech over music against speech alone 54.0%, adjusted 1.14 (0.95-1.37); first word within 2 s
43.6% (38.2-49.4), ratio 0.87 (0.71-1.08); faster speech 46.7%; loud (-16 LUFS+) 52.4%; quiet (under -24 LUFS)
48.2%; music at 120 bpm+ 52.6% (49.4-55.7); busy music 49.0%; spoken opening line names the product (expression)
53.8%, adjusted 1.14 (0.94-1.42); states a problem 46.4%; contains a number 53.5%; announcement verb 55.9%
(48.0-63.1), adjusted 1.16 (0.99-1.40). Too thin: slideshows (36 posts), spoken result (243 pairs, 18 accounts).

**Follower tiers** (broad set). The person-on-camera penalty is in large accounts: not-person 62.9% (56.0-69.6),
1.56x at 100,000+; 47.3% (41.1-54.4), 0.92x under 100,000. 60 to 120 s is clearer in small accounts: under 100,000
56.0% (49.6-62.6), 1.50x (1.19-1.92), adjusted 1.36x (1.12-1.65); 100,000+ 53.2%, 1.13x (0.90-1.37). Avoiding
under-15-second clips, fast cutting and no captions pass in 100,000+ accounts only. Music alone against speech alone
holds in both: 61.5%, 1.82x (1.18-2.75) under 100,000; 66.2%, 1.84x (1.34-2.36) at 100,000+. Launch videos by tier
(170 posts under 100,000 followers and 160 at 100,000+; 141 launch videos from the cutoff): no trait reaches 200 pairs.

**Before and from 2026-04-23.** From the cutoff the broad set has 662 posts; no finding is confirmed there on its
own. Directions agree for length (60 to 120 s: 55.1%, 43.7-64.3), cuts (56.9%, 49.9-64.5), captions, face share and
music against speech (58.5%, ratio 1.72, 1.08-2.94); the person-on-camera penalty is absent after the cutoff
(not-person 51.5%, ratio 0.99). Sound against silent is stronger after it (63.4%, 54.9-72.0; 1.62x) and is the one
place a "silent does worse" signal appears; before the cutoff it is 52.9% (49.5-56.4).

**Levels** (launch videos, every account; mean likes over the account's usual). Length: under 15 s 2.92x, 15 to 30 s
3.19x, 30 to 60 s 3.13x, 60 to 120 s 3.37x, over 120 s 3.52x. Kind: screen 3.63x (3.06-4.36), mixed 3.17x, film
3.05x, person 2.78x (2.12-3.56). Sound: music only 3.55x (2.95-4.36), silent 3.25x, speech over music 3.21x, speech
only 2.53x (1.97-3.26). All overlapping.

## 3. Model

`scripts/video_files/train_vf.py`, `eval_vf_pairs.py`; harness of the video track. Rows with features in the training
table: 4,889 (3,870 train, 314 validation, 705 test). Control and treatment warm-start from `runs/ts46_s1/best.pt`
and fine-tune the `small` preset for 600 steps (lr 2e-4) with anchors restricted to the same 3,870 train rows; best
step by loss on the 314 validation rows. Treatment adds 46 inputs with zero initial weights: a flag, 16 principal
components of the pooled embedding of 5 frames (fitted on train rows), 29 attributes (length, kind, UI and face
share, pace, captions, first 3 seconds, shape, sound class, loudness, speech timing). Two seeds, averaged. All four
runs picked step 200; validation loss rises after it (2.83 to 2.87 at step 600 for the control), so 600 steps
overfits these rows.

Validation loss at the chosen step: control 2.847 and 2.825; with video inputs 2.819 and 2.802.

Pairs: same account, likes differing 1.25x or more, at most 365 days apart, at most 50 an account, table path.

| Pair set | Pairs / accounts | Baseline | Control | With video and audio | Difference to control |
|---|---|---|---|---|---|
| Video-video, train rows | 7,182 / 375 | 63.2% | 66.4% (64.6-68.2) | 67.6% (65.9-69.3) | +1.1 (+0.7 to +1.7) |
| Video-video, validation rows | 652 / 40 | 64.9% | 64.7% (59.7-70.1) | 65.2% (59.5-70.6) | +0.5 (-1.4 to +2.4) |
| Video-video, test rows | 848 / 112 | 58.4% | 59.7% (53.8-65.1) | 61.0% (55.3-66.3) | +1.3 (-0.3 to +2.9) |
| The same, gap 2x | 598 / 94 | 61.5% | 62.5% (55.3-69.5) | 63.7% (57.0-70.6) | +1.2 (-0.8 to +3.0) |
| Video against another post, test rows | 6,397 / 260 | 65.9% | 66.5% (63.7-69.2) | 66.1% (63.3-68.8) | -0.4 (-1.1 to +0.2) |
| The same, gap 2x | 5,021 / 258 | 68.9% | 69.5% | 69.4% | -0.1 (-0.9 to +0.6) |
| Launch pairs with a video, test rows | 290 / 42 | 63.4% | 64.5% (56.0-71.6) | 62.1% (51.8-70.1) | -2.4 (-6.6 to +0.5) |
| The same, gap 2x | 201 / 36 | 67.7% | 70.2% | 66.7% | -3.5 (-8.0 to -0.6) |

Single seeds on test video-video pairs: +1.8 (-0.1 to +3.8) and +1.9 (+0.3 to +3.5) against the control average. The
train gain (+1.1) is the same size as the test gain, so this is not memorisation, but the interval on held-out pairs
includes zero, video-against-other pairs do not move, and launch pairs with a video get worse. Confident-call tables
were not produced: there is no gain to tabulate, and the serving-path benchmarks hold too few video pairs (58 of
2,030 test pairs) to measure one.

## 4. What did not work

- Launch-only evidence: too few launch videos for any whole-video or audio trait.
- The first scene-change counter (merging consecutive changed steps): undercounts fast cutting; replaced.
- Slideshow of stills as a kind: 60 videos.
- The spoken "names the product" expression on transcripts: needs capitalised names; transcripts do not give them.
- A first effects run without post type in the adjustment (corrected, see section 2).
- Processing all 7,420 videos: the 1,421 without an outcome were skipped for time; they would add about 1,100 rows
  to the model test only.

## 5. Honest limits

- Observational, likes only, inside accounts. Grade C findings are about these accounts' videos in general; the
  launch sets cannot confirm or refute them, and none is confirmed on its own after the cutoff.
- 53 traits, three tests each. With 53 traits two or three would pass one 95% test by chance; requiring three tests
  and 200 pairs limits that, but the weakest C findings (captions, cut in the first 3 seconds, loudness range) could
  be chance.
- Labels: one reader, 40 videos a class, no second opinion. Kind is read from 5 frames a video; the "film" class
  mixes motion graphics with filmed and generated footage. Label errors bias contrasts toward no effect.
- Pace is coarse (2 frames a second) and blind to changes inside white screen recordings.
- Audio covers the first 90 seconds, 4 tagger windows, 30 seconds of transcript, and was never checked by ear.
- The sound findings hold with five visual kinds fixed. They do not prove that viewers hear anything; a music bed
  and a wide loudness range are marks of an edited film that the visual kinds only partly capture.
- Videos of large accounts dominate the broad set's pairs (cap of 50 pairs an account limits, not removes, this).

## 6. Reproduce

From `virality/` with the dataset in `data/` and the two tars extracted to `data/vf/audio_x`, `data/vf/preview_x`:
`scripts/launch/build_table.py` (needs `data/launch/accounts.parquet` = `tracks/launch/accounts_labeled.parquet`),
then under `scripts/video_files/`: `frame_features.py <i> 4` (four copies), `audio_features.py <i> 4`,
`label_videos.py`, `cut_sheet.py` and `kind_sheet.py` (validation sheets), `effects.py`, `build_evidence.py`,
`train_vf.py --arm baseline|control|vf`, `eval_vf_pairs.py`.
