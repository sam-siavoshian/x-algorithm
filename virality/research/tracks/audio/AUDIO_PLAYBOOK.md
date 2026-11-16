# Audio playbook: what the data supports about the sound of a launch video on X

Built 2026-11-16. This page states only what survives measurement on the product's own corpus.

## What survives

**Nothing.** No recommendation about the sound of a launch video is supported, because no effect of sound could be
measured: the corpus holds a thumbnail for each video post and no video or audio file, and the audio of the posts was
not fetched (see "Why" below). Every entry in `audio_evidence.json` has the status `not_measured` and
`supported: false`.

This is "could not be measured", not "sound makes no difference". The two must not be confused in the product:
the checklist should say nothing about voiceover, music, loudness, speech rate or the spoken opening line.

## The questions, and what can be said about each today

| Question | Status |
|---|---|
| Voiceover, music only, both, or silence | Not measured |
| A human voice or a text-to-speech voice | Not measured, and not measurable with this pipeline: no detector was validated |
| Speech in the first two seconds, or a musical intro | Not measured |
| Speech rate | Not measured |
| Music tempo and energy | Not measured |
| Loudness and dynamic range | Not measured |
| What the first spoken line says (product name, a problem, a number, a result) | Not measured |
| Does any of it matter once the kind of picture is held fixed | Not measured |

## How much the corpus could show if the audio were in hand

Counted on the real posts, with traits dealt at random to learn the width of the intervals (`power.json`):

| Set of video posts | Videos | Accounts with two or more | A trait on half of the videos: pairs / accounts | Smallest win rate it could tell from 50% |
|---|---|---|---|---|
| Launch posts, company and founder accounts | 613 | 114 | about 475 / 79 | about 57%, likes ratio about 1.25x |
| All own announcements, every account | 1,109 | 194 | about 690 / 125 | about 56%, likes ratio about 1.20x |
| The same, accounts under 100,000 followers | 393 | 71 | about 156 / 41 | about 60%, likes ratio about 1.46x |
| The same, 100,000 followers or more | 303 | 45 | about 300 / 32 | about 60%, likes ratio about 1.33x |
| The same, from 23 April 2026 | 200 | 31 | about 60 / 18 | about 66%: no separate confirmation possible |

For comparison, the launch playbook's findings are win rates of 56% to 72%. So:

- A coarse trait that splits all videos (has a voiceover or not, has music or not, loud or quiet) could be detected
  if its effect is as large as the mid-sized findings of the launch playbook (about 57% or more).
- A trait that exists only inside a subset (speech in the first two seconds, speech rate and the spoken opening line
  need a voiceover; tempo needs music) has about a quarter of the pairs: around 120 pairs from 36 accounts in the main
  set, under the 200-pair minimum of the launch rule. Only an effect of about 63% or more would show, and it would be
  graded "too thin" in the main set.
- Nothing about sound could be confirmed separately after 23 April 2026.

So even with every audio file, the realistic outcome is one or two coarse findings at most, and "no effect found"
or "too thin" for the fine questions. More posts would be needed for those: the video track's index lists 7,236
videos from 1,208 accounts when ordinary posts of company and founder accounts are included.

## Why sound might not matter, and why that is still untested

Videos in the X feed start playing without sound. Likes on a launch post can therefore come from people who never
heard it. That is a reason to expect small effects of sound on likes, not evidence of it. The analysis that would
test it is written and ready (`same_visual` in `scripts/audio/effects.py`: same account, two videos with the same kind
of first frame, one with the audio trait and one without).

## Why the audio is missing

- The corpus stores `video_url` as empty for every video post.
- A public route exists: X's embed endpoint returned mp4 files for 3 of 3 corpus posts tried, with no login. One
  video was downloaded and analysed as a test.
- The bulk download of the 2,027 videos was not run. This session's permission system refused it, and the video
  track did not acquire files either. Nobody with authority over that decision has approved it yet.

## What unlocks it

The audio track of the 2,027 video posts listed in `tracks/launch/launch_posts.parquet` (1,465 own announcements and
562 demo posts; GIFs excluded), as one file per post named `<post_id>.mp4` (or `.m4a`, `.flac`): the lowest-bitrate
variant is enough. Then three commands produce the features, the effects and this playbook's evidence table:

    python scripts/audio/extract.py <folder>           # about 3 to 6 hours on one 4-core machine
    python scripts/audio/effects.py
    python scripts/audio/build_audio_evidence.py

## Limits

- The feature code was checked on 15 recordings with a known answer (read speech, four pieces of music, silence, and
  mixes of speech over music): 15 of 15 classified correctly. It was run on one real launch video. Its accuracy on
  launch videos at large is unknown.
- Text-to-speech against a human voice is left out: no detector could be validated by listening.
