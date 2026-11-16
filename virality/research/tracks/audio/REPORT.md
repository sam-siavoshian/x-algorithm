# Track "audio": what kind of sound travels on a launch video

2026-11-16. Deliverables in this folder: `AUDIO_PLAYBOOK.md`, `audio_evidence.json` (shape of
`launch_evidence.json`), `power.json`, `validation.json`, `sample_features.parquet` (one real video), `patch.diff`.

## Headline

- **No effect of sound was measured.** The audio of the corpus's video posts was not obtained, so every audio trait
  is unknown for every post. All 22 entries of `audio_evidence.json` are `not_measured`; the playbook recommends
  nothing.
- A public route to the files exists (3 of 3 posts tried), but the bulk download was not run: this session's
  permission system refused it, and the video track published an index with no files (`media/videos/README.md`:
  "Video FILES WERE NOT ACQUIRED").
- The feature code, the effects analysis and the evidence builder are written and tested, and run end to end on the
  real post table with placeholder traits. With the audio in a folder, three commands finish the track.
- If the audio were in hand, the corpus could detect only coarse, mid-sized effects: 613 launch videos from 252
  company and founder accounts give about 475 same-account pairs for a trait on half the videos, an interval of
  plus or minus 7 points on the win rate. Traits that exist only with a voiceover or only with music would have about
  120 pairs, under the launch rule's 200-pair minimum.

## 1. Acquisition

| Step | Result |
|---|---|
| Wait for the video track's `media/videos/index.parquet` and `media/videos/audio/` | Index appeared at 07:29 UTC, 10 minutes after start: 7,565 posts (7,236 video, 329 GIF), every row `thumbnail_only` or `pending`, `has_audio` and `audio_path` null everywhere. No `audio/` folder. Its README states that no video file was acquired. No `tracks/video/REPORT.md` existed when this report was written, so the reason is not known here. |
| Test a public route on a handful of posts | `GET https://cdn.syndication.twimg.com/tweet-result?id=<post id>&token=<any>` (X's embed endpoint, no login, cookie or key) returned HTTP 200 for 3 of 3 corpus posts: two videos with mp4 variants at 256 kbit/s to 10 Mbit/s plus an HLS playlist, and one GIF with a single mp4. It also returns the current like count and, for GIFs, a view count field. One 254-second video was downloaded at the lowest bitrate (2.8 MB, AAC audio at 32 kbit/s) and analysed. |
| Fetch the 2,027 videos | **Not run.** The command was refused by this session's permission system as unauthorized access. I did not look for a way around the refusal. No further request was sent to X after it. |

Requests sent to X in total: 4 metadata lookups and 1 file download. No refusal (403 or 429) was received from X.

## 2. Features (`xviral/media/audio.py`, tested)

One call, `features(path)`, CPU only:

| Feature | How |
|---|---|
| Has an audio track; audible | ffprobe; integrated loudness above -50 LUFS and under 95% of 50 ms frames below -50 dBFS |
| Loudness, loudness range, true peak | ffmpeg `ebur128` (EBU R128) |
| Speech share | Silero voice-activity detector (the copy in faster-whisper), kept only inside windows where the tagger also hears speech, so sung vocals do not count |
| Music share | AudioSet tagger (`bookbot/distil-ast-audioset`, 44M parameters) on 10 s windows, P(Music) at least 0.3 |
| Sound class | silent, speech only, music only, speech and music, other (speech in 15%+, music in 30%+ of the analysed time) |
| Time to first speech; speech in the first 2 s; musical intro | Whisper `small` word timestamps (first 75 s) |
| Speech rate | transcribed words per minute of voiced time |
| Music tempo and energy | librosa beat tracker, onset rate, percussive share of energy, beat strength, on the music windows |
| Voice level over the music bed | level of voiced frames minus the other non-silent frames |
| Transcript and opening line | first sentence (two or more words, at most 30); traits by the regular expressions of `xviral/text/post_type.py` unchanged (names the product, problem first, number), plus result, announcement verb, and "says a name the post or account carries" |

Cost: about 0.1 s of CPU per second of audio without speech and 0.25 to 0.45 s with speech, on 4 cores.

**Check on audio with a known answer** (`scripts/audio/validate_samples.py`, `validation.json`): read speech, four
pieces of music, silence, noise at -80 dBFS, speech 15, 5 and 0 dB above two pieces of music, a 6-second musical
intro before speech, and 3 seconds of silence before speech.

| Check | Result |
|---|---|
| Sound class | 15 of 15 correct |
| Time to first speech | 5.82 s for a 6.0 s intro; 2.93 s for 3.0 s of silence; 0.0 s elsewhere |
| Speech rate, same reading in 9 conditions | 148 to 174 words a minute: the measure moves by about 10% with the music bed, so the fixed cut-offs (130 and 170) are coarse |
| Tempo, same piece alone and under speech | 136 and 136 (drum and bass); 112 and 108 to 112 (orchestral) |
| Music threshold | pure speech scores P(Music) at most 0.05; speech 15 dB above a quiet orchestral bed scores 0.36 to 0.66. The first threshold tried (0.5) missed that case and was lowered to 0.3 |

On the one real video (a music product's launch film): music only, -17.0 LUFS, loudness range 11.6 LU, 123 beats a
minute, no transcript (the vocals are sung). Plausible, and a single case.

**Synthetic voice: left out.** The brief allowed it only if a detector could be validated by ear on a sample. I
cannot listen to audio, and no real launch-video speech was available to validate on, so no detector was run and
there is no such column.

## 3. Effects (`scripts/audio/effects.py`, tested on placeholder traits only)

The launch track's method, restricted to video posts (GIFs excluded): same-account pairs with posting order
balanced, the within-account likes ratio, and a regression with account fixed effects, 11 text traits and the visual
type. Scopes: launch posts of company and founder accounts (main), the strict subset, launch posts of every account,
all own announcements, under and over 100,000 followers, before and from 2026-04-23. 22 traits with cut-offs fixed
before any video was analysed (listed in `audio_evidence.json` under `_meta.traits_fixed_in_advance`).

The muted-autoplay question has its own estimate, `same_visual`: pairs whose two videos have the same first-frame
class from the video track's `thumb_features.parquet`. That file was not published when this report was written; the
loader looks for a class column under four likely names and must be pointed at the right one when it exists.

Checks done without audio:
- A planted 1.5x effect on a random half of the 1,109 announcement videos is recovered: win rate 64.4% (58.0 to
  69.8), likes ratio 1.54x (1.30 to 1.81), adjusted 1.54x (1.30 to 1.88).
- `scripts/audio/dry_run.py` deals placeholder traits at random and runs every scope and trait on the real post
  table (1,561 video posts joined). In one such run, 5 of the 88 pair intervals of two scopes (plain and
  same-visual) excluded 50% by chance, as expected at the 95% level, which is why the three-test rule is kept.

## 4. What the corpus could show (`scripts/audio/power.py`, `power.json`)

Video posts with a baseline in the Community Archive source; traits dealt at random 40 times; medians.

| Scope | Videos (GIFs dropped) | Accounts | With 2+ videos | Trait on half of all videos: pairs / accounts, win-rate interval, ratio interval | Trait on half of a half subset |
|---|---|---|---|---|---|
| Launch, company and founder | 613 (29) | 252 | 114 | 475 / 79, ±7.3 points, ×/÷ 1.25 | 118 / 36, ±13.0, ×/÷ 1.42 |
| Launch, strict subset | 370 (18) | 168 | 66 | 236 / 47, ±9.8, ×/÷ 1.35 | 62 / 23, ±16.3, ×/÷ 1.61 |
| Launch, every account | 812 (40) | 421 | 138 | 488 / 89, ±7.9, ×/÷ 1.25 | 127 / 39, ±11.7, ×/÷ 1.41 |
| Announcements, company and founder | 792 (33) | 302 | 146 | 651 / 102, ±6.2, ×/÷ 1.21 | 176 / 49, ±10.8, ×/÷ 1.38 |
| Announcements, every account | 1,109 (47) | 549 | 194 | 691 / 125, ±6.2, ×/÷ 1.20 | 191 / 58, ±10.3, ×/÷ 1.35 |
| Announcements, under 100k followers | 393 (17) | 234 | 71 | 156 / 41, ±10.4, ×/÷ 1.46 | 40 / 17, ±19.4, ×/÷ 1.94 |
| Announcements, 100k+ followers | 303 (15) | 109 | 45 | 301 / 32, ±10.0, ×/÷ 1.33 | 92 / 17, ±15.1, ×/÷ 1.52 |
| Announcements, before 2026-04-23 | 909 (45) | 482 | 163 | 482 / 104, ±7.3, ×/÷ 1.23 | 121 / 47, ±11.6, ×/÷ 1.40 |
| Announcements, from 2026-04-23 | 200 (2) | 136 | 31 | 61 / 18, ±16.4, ×/÷ 1.59 | 14 / 7, ±31 |

An effect smaller than the half-width cannot be told from none. The launch track's findings are 6 to 22 points
above 50%, so a whole-video trait of that size could show in the main set; a trait inside the voiceover or music
subset could not pass the 200-pair bar there; nothing could be confirmed after the cutoff.

## What did not work

- Getting audio from the video track: it has none.
- Fetching the videos myself: refused by the session's permission system.
- The first music threshold (0.5) missed quiet music under speech; fixed on the controlled mixes, not on outcomes.

## What would unlock the analysis

1. Approval, by whoever owns that decision, to download the videos by the public embed route, or the files from any
   other source. Needed: one file per post named `<post_id>.mp4` (lowest bitrate is enough; audio-only is enough) for
   the 2,027 video posts in `tracks/launch/launch_posts.parquet` (`has_video`, not GIF): 1,465 own announcements,
   of which 1,109 enter the effects, and 562 demo posts. About one request a second gives the set in under an hour;
   the test file was 2.8 MB for 254 seconds.
2. `python scripts/audio/extract.py <folder>` (3 to 6 hours on 4 cores), `python scripts/audio/effects.py`,
   `python scripts/audio/build_audio_evidence.py`.
3. The video track's first-frame classes, for the same-visual test.
4. For the fine questions (opening line, speech rate, tempo), more posts: the video track's index lists 7,236 videos
   from 1,208 accounts including ordinary posts of company and founder accounts.

## Honest limits

- No finding. Nothing in this track says sound matters or does not.
- The feature code is validated on clean public recordings and controlled mixes, and on one real video. Launch videos
  have sound effects, sung vocals, several speakers and non-English speech that the samples do not cover.
- `effects.py` and `build_audio_evidence.py` have only seen placeholder traits.
- The tagger and Whisper models are downloaded from Hugging Face at first use (about 180 MB and 480 MB).
- New package requirements beyond the `encoders` extra: `librosa`, `soundfile`, `transformers` (already listed).
