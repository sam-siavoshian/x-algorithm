# Why startup and AI-company posts spread on X: graded outside evidence, tested on our corpus

Built 2026-11-16. Two halves. Sections 1 to 4 grade what is known outside this project (214 claims, each with its source,
sample, whether it controlled for the account, and a grade). Sections 5 to 7 turn the claims into measurable traits and
test them on our own launch posts with the launch track's method. Section 8 lists the traits the model should receive.

## How to read this

**Grades for outside claims.**
- *Controlled study*: a randomized experiment, a natural experiment, or a matched same-author design.
- *Large observational*: 10,000 posts or more, no randomization.
- *Small observational*: fewer than that.
- *Anecdote*: a single case, a founder or vendor write-up, press.
- Two extra labels that are not strength grades: *platform code* (read off X's published ranking code; deterministic, but
  says what is scored, not how large the effect is on a post) and *platform statement* (said by X or its staff, no data).

**Counts.** 214 records: 27 controlled, 48 large observational, 18 small observational, 78 anecdote, 23 platform code,
16 platform statement, 4 results from this project's earlier corpus work (observational, listed for cross-reference).
43 are specific to AI companies; none of those 43 is a controlled study of X posts. 17 records carry a figure that was
not re-checked against the source page; each is marked in the appendix.

**Our tests.** Launch-type posts (launch and open-source release) of 530 company and founder accounts, 1,980 posts;
3,654 posts of 1,657 accounts when every account is pooled. Outcome: likes divided by the median likes of the account's
previous 30 posts. Every comparison is inside one account, posting order balanced, intervals from resampling accounts.
A finding counts only if it passes the rule the launch track fixed in advance (stated in section 5). Likes only: view
counts exist for 205 of these posts.

## 1. What the outside evidence says, in ten lines

1. The account and luck dominate. With full knowledge of author and content, models explain under half of the variance
   in how far a post goes, and most of that is the author's history ({{R:C03}}). Any claim not measured inside accounts
   is mostly measuring who posted.
2. Wording matters, modestly. Same author, same link, different wording: the better wording can be picked 65.6% of the
   time ({{R:W01}}). That is the ceiling the best controlled study found, and it is the design this project uses.
3. Early engagement causes later engagement, with diminishing returns. One random early upvote raised final scores by
   25% ({{R:C07}}); a small early push more than doubled later support, four times the push added little ({{R:C08}}).
   Half of a tweet's impressions arrive within about 80 minutes ({{R:T01}}). All of this is from other platforms or
   older Twitter; nobody has run it on X launch posts.
4. Links: independent large studies find link posts get 4 to 8 times fewer views per follower ({{R:K01}}); X's head of
   product says links are not penalised ({{X:S03}}); the 2026 code has no link penalty ({{X:C15}}).
5. Video: X's marketing says video wins; the 2026 code gives video actions zero weight ({{X:C09}}); Buffer's 2025 data
   has text ahead of video on engagement rate ({{X:P05}}). None of these compares inside accounts.
6. Timing: four vendors, four different best times, all pooled across accounts ({{X:P08}}, {{X:P09}}, {{X:P10}},
   {{X:P11}}). The code has no posting-hour term ({{X:C18}}).
7. For brand posts, emotion plus information beats information alone, and price or deal content lowers engagement
   ({{R:B01}}, 106,316 posts with firm fixed effects, Facebook).
8. X's code: the score is a weighted sum of predicted actions (reply 5, quote 5, repost 1, like 0.7; report -234, mute
   -59, not interested -53) ({{X:C01}}, {{X:C02}}); an author's second and third posts in one feed request are scored
   at 0.625 and 0.44 ({{X:C04}}); only originals and quotes reach non-followers ({{X:C06}}); engagement bait is
   labelled spam ({{X:C14}}).
9. Everything specific to startup launches is anecdote: 74 of 82 playbook records are single cases told by winners or
   by vendors of launch videos. Reported view counts often include paid distribution ({{PB034}}, {{PB073}}).
10. For open-source and AI work there is one controlled result: a burst of tweets about a repository causes about 7%
    more GitHub stars ({{R:B08}}), and one matched-control result: AI papers shared by two curator accounts had 2 to 3
    times the citations ({{R:B11}}).

## 2. General evidence, by theme

### 2.1 Wording
@@rows R:W01 R:W02 R:W03 R:W04 R:W05 R:W06 R:W07 R:W08 R:W10 R:P01

What it means for a launch post: informative, specific first lines (proper nouns, numbers) are the controlled finding
that maps most directly; it matches this project's strongest launch result (first line names the product). Headline
A/B archives warn that lessons from one test transfer poorly to the next story ({{R:W10}}).

### 2.2 Emotion
@@rows R:E01 R:E02 R:E03 R:E04 R:E05 R:E06 R:C11

Almost all of this is political or news content. The best-known finding (moral-emotional words) has a published
critique showing an arbitrary letter-count model predicts as well ({{R:E03}}). Not a basis for launch advice.

### 2.3 Images and video
@@rows R:M01 R:M02 R:M03 R:M06 X:S13 X:C09 X:P05 X:P16 X:P03

No peer-reviewed, author-controlled study of demo videos, GIFs or screenshots on X exists ({{R:M06}}).

### 2.4 Links, hashtags, mentions, asks
@@rows R:K01 R:K03 R:K04 X:P04 X:S01 X:S02 X:S03 X:S04 X:C15 X:P22 X:S10 X:C14 X:S08 R:K06

### 2.5 Timing and decay
@@rows R:T01 R:T02 X:P08 X:P09 X:P10 X:P11 X:C08 X:C18 PB070

### 2.6 Early amplification, seeding, cascades
@@rows R:C01 R:C02 R:C03 R:C05 R:C06 R:C07 R:C08 R:C09 R:C12 R:C13

### 2.7 Brand and product posts
@@rows R:B01 R:B02 R:B03 R:B04 R:B07 R:P02 R:P03 R:K07

### 2.8 What X's code scores
@@rows X:C01 X:C02 X:C03 X:C04 X:C05 X:C06 X:C07 X:C10 X:C11 X:C12 X:C13 X:C22

Advice that still quotes the 2023 weights (reply 13.5, "author replies 75x", profile click 12) is out of date
({{X:C22}}).

## 3. Specific to AI startups

None of this is controlled. It is listed because it is what the launches that travelled did, and it defines the traits
tested in section 5.

### 3.1 Benchmark charts and "first / best" claims
@@rows PB004 PB005 PB006 PB037 PB027

### 3.2 Demo video of the model doing something
@@rows PB027 PB036 PB041 PB045 PB032 PB072 PB050 PB051

### 3.3 Waitlists, invites, research previews
@@rows PB012 PB036 PB029 PB014 PB046 PB047

### 3.4 Open-weights and open-source releases
@@rows PB006 PB007 PB008 PB009 PB010 PB025 PB030 R:B08 R:B10

### 3.5 Employees, investors and well-known accounts
@@rows PB020 PB028 PB015 PB079 PB080 PB055 PB073 R:B11

### 3.6 Teasers and follow-ups
@@rows PB001 PB011 PB081 PB082 PB018 PB013 X:S12

### 3.7 Founder threads, build in public, Y Combinator, Product Hunt
@@rows PB017 PB078 PB048 PB049 PB019 PB021 PB023 PB024 PB056 PB057 PB058 PB060 PB061

### 3.8 Controversy and paid reach
@@rows PB033 PB040 PB034 PB035 PB075

## 4. Where the outside sources contradict each other

| Topic | One side | Other side |
|---|---|---|
| Links | 4 to 8 times fewer views per follower ({{R:K01}}); zero median engagement for non-Premium link posts ({{X:P04}}); Musk 2023 and 2024 ({{X:S01}}, {{X:S02}}) | "Links are not deboosted" ({{X:S03}}, {{X:S04}}); no penalty in the 2026 code ({{X:C15}}) |
| Video | "10x engagement" ({{X:S13}}); every launch playbook | Zero video weights in code ({{X:C09}}); text ahead of video ({{X:P05}}, {{PB060}}) |
| Best time | Tue / Wed 9 to 10am ({{X:P08}}) | Tue to Thu noon to 6pm ({{X:P09}}); 9pm ({{X:P11}}); no hour term in code ({{X:C18}}) |
| Premium | About 10x reach, pooled ({{X:P01}}) | No significant boost in a 2023 feed audit ({{R:C13}}); no multiplier in 2026 code ({{X:C10}}) |
| Asking for engagement | Share requests win in same-author pairs, 2014 ({{R:W02}}) | Engagement bait labelled spam in 2026 code ({{X:C14}}); Grok to push solicitation down ({{X:S08}}) |
| Images | Images and faces raise engagement ({{R:M01}}) | Media use associated with lost algorithmic exposure, except video ({{R:M03}}) |
| Moral-emotional words | +20% retweets per word ({{R:E02}}) | Arbitrary letter count predicts as well ({{R:E03}}) |

## 5. Tested on our data

**Method.** Scripts `scripts/marketing/` (traits in `traits.py`, tests in `effects_mk.py`, the rule in
`build_marketing_evidence.py`), reusing the launch track's table, outcome, estimators and rule unchanged. Each trait is
a regular expression on the post text, a Pacific-time clock reading, or a fact about the account's other posts and
about quote posts by other corpus accounts. Trait definitions were written, and samples read, before any outcome was
looked at.

**Three numbers per trait**, as in the launch playbook: win rate in same-account pairs where exactly one post has the
trait (50% is no effect), likes ratio inside accounts that have both, and the ratio holding the other traits fixed
(the launch track's 20 traits plus the 24 new text traits, account fixed effects).

**Rule, fixed in advance.** Main set: company and founder accounts. Strict set: posts opening with an announcement
verb followed by a product name. A trait is *confirmed* or *contradicted* only if every test excludes "no effect" in
the main or the strict set, at least one does in the other, in the same direction. Under 200 pairs or 25 accounts in
the main set, the rule moves one level wider (every account pooled) and the result can be at most grade C; if still
too thin, the claim is *untestable here*. *Not confirmed* means tested with enough data and nothing passed. Grades A to
C as in the launch playbook (A: also confirmed separately before and after 23 April 2026).

About 45 new comparisons were run on top of the launch track's 30. One or two passing by chance would not be
surprising.

### 5.1 Confirmed

@@hyp confirmed

Reading the new ones:

- **Open source / open weights (B).** Launch posts that say so got 1.33x the likes of the same account's other launch
  posts. It holds for accounts under and over 100,000 followers and inside model releases (57.4%, likes ratio 1.45x,
  1.10 to 1.90). Before April 2026: 61.7% (53.3 to 70.2). Since: 53.4% (43.4 to 63.9), same direction, not provable on
  its own. On reposts the win rate is 57.7% (50.9 to 65.1).
- **A "first / best / most capable" claim (A).** 61.4%, 1.53x, adjusted 1.30x; confirmed in the strict set and
  separately before (60.9%, 54.3 to 67.6) and after April 2026 (63.1%, 52.5 to 73.3). **Caution:** accounts write "our
  most capable model yet" for their flagship releases and not for minor ones, so this trait also marks how big the
  news is. The adjusted figure holds model releases fixed and the effect stays, but the data cannot separate the
  wording from the size of the launch. It is evidence that the model needs this input, not that adding the phrase to
  a draft lifts it.
- **Nothing posted in the 24 hours before (B).** 57.5%, 1.43x. Extends the launch track's 2-hour finding and matches
  the repeat-author decay in X's code. Before April 2026 58.8% (53.0 to 64.2); since 57.4% (47.4 to 68.0).
- **Model releases** do 1.46x the same account's other launches (A). A topic, not a choice.

The remaining rows are launch-track results restated against their outside sources.

### 5.2 Contradicted

@@hyp contradicted

- **Credits, discounts, deals, prizes (C, pooled).** The version without the offer won 72.5% (57.4 to 82.5) of 211
  pairs from 27 accounts; likes ratio 0.55x. This contradicts the playbook advice and agrees with the one large study
  of brand posts ({{R:B01}}). Limits: 47 posts, about half of the matches read by hand were real offers (the rest were
  credit grants and programmes, which are smaller news), and the result comes from before April 2026 (18 pairs since).
- **A second launch within a week (B, fragile).** A launch-type post that follows another by 1 to 7 days got 0.79x
  (0.68 to 0.90) the likes of the same account's other launch posts. The win-rate interval only just excludes 50%
  (38.1 to 50.0), it does not hold after April 2026 (50.9%, 40.6 to 58.9, with the trait), and it is absent for
  accounts over 100,000 followers. The playbook claim is about total reach across a launch week, which this does not
  test; what it says is that each later post in a cluster should be expected to do less than a stand-alone launch.
- **"Text beats video" (Buffer).** Not for launch posts measured inside accounts: video won 65.4% against no media.
- **"Pain point first".** The playbooks say it about the video. For the post text the opposite holds: the
  announcement first won 72.5%.

### 5.3 Tested, not confirmed

@@hyp not_confirmed

What stands out:

- **Weekday morning Pacific time: nothing.** Seven comparisons, none passes; in the strict set the weekday-morning
  version lost (45.4%, 39.1 to 51.2). The likes ratio for the 8 to 10am slot is 1.13x (0.98 to 1.28), which rules out
  the 3 to 5 times claimed by the vendor sample of 79 tweets ({{PB070}}). 61% of these launch posts were already
  posted on a weekday morning Pacific, so the comparison is against the minority that were not.
- **A call to action, "free", a waitlist, an availability line: nothing.** Each has an interval that includes no
  effect and a point estimate near 50%. Waitlist against available-now: 52.9% (36.3 to 68.0), too wide to say more.
- **A teaser beforehand: nothing detectable.** 54.9% (41.1 to 67.8) among company and founder accounts, 55.1% (46.6 to
  63.7) pooled, likes ratio 1.09x (0.87 to 1.40). About half of the detected teasers, read by hand, were teasers of
  the launch that followed. The teaser posts themselves get 0.94x (0.79 to 1.11) the account's usual.
- **Benchmark claims and numbers anywhere in the text: a lean, not a finding.** Benchmark claim 54.5% (48.7 to 60.5),
  adjusted 1.10x (0.93 to 1.30). One lead that was not specified in advance and so is not a finding: for accounts
  under 100,000 followers the benchmark version won 63.7% (55.2 to 73.6), adjusted 1.51x; for larger accounts 52.0%.
- **Naming a competitor: 59.1% (42.6 to 74.4)**, 47 posts. Direction as the comparative-advertising literature says,
  interval far too wide.
- **Video in a model release: 65.3% (55.4 to 73.4)** in pairs, but the likes ratio 1.34x (0.95 to 1.82) includes no
  effect; on reposts it passes (69.5%, 61.6 to 75.9; 1.75x, 1.12 to 2.67). Consistent with the launch track's video
  finding, not a separate confirmation.
- **"Excited to announce" opener: leaning against** (40.4%, 31.5 to 50.0), not established.

### 5.4 Untestable here

@@hyp untestable

On the amplification and follow-up rows, the numbers are large and are **not effects**:

- Launches quoted by another corpus account within the first hour did 1.76x (1.54 to 2.02) the same account's other
  launches (69.2% of 1,471 pairs); with a same-team quote 1.54x. There is a dose pattern inside accounts: 0.85x of
  the account's average launch with no first-hour quoter, 1.10x with one, 1.78x with two, 2.60x with three or more.
  A launch that is taking off is quoted sooner and by more people, so this cannot be read as the result of arranging
  quotes. The one comparison that removes part of that problem, same-team quote within the hour against a same-team
  quote later the same day, shows nothing: 45.5% (32.0 to 64.3), 169 pairs from 17 accounts, too thin to conclude.
  Only quotes by accounts that are themselves in the corpus are visible (41% of launches have one within 48 hours).
- The controlled studies say early support does cause later support ({{R:C07}}, {{R:C08}}). Our data neither
  confirms nor contradicts that for X launches. It should enter the model as a signal observed after posting, not as
  advice.
- Follow-up: a launch the account later quote-posts did 3.2x, which is again a consequence. The follow-up posts
  themselves are ordinary: an account's quote of its own launch 1 to 7 days later got 1.09x (0.75 to 1.62) its usual
  (63 posts) and about a fifth of the launch's likes. Other posts by the account in the 24 hours after a launch got
  0.86x (0.76 to 0.96) its usual, back to 1.02x on days 2 to 4: no halo from a launch onto what follows.
- Posting another announcement 1 to 4 days after shows no association at all with the launch's result (49.4%, 42.5 to
  55.9).

## 6. Outside claims our data contradicts or fails to support, stated plainly

| Outside claim | Outside grade | Our data |
|---|---|---|
| Launch on a weekday morning Pacific, ideally Tuesday 8 to 10am | Large observational, pooled across accounts; one vendor sample of 79 | No effect in seven comparisons; a 3 to 5x gain is ruled out |
| Open the launch on the pain point | Anecdote (video vendors) | Contradicted for the post text: announcement first wins 72.5% |
| Text beats video on X | Large observational, pooled | Contradicted for launch posts inside accounts: video wins 65.4% |
| Offer credits, a discount or a prize | Anecdote | Contradicted (grade C): 0.55x |
| Launch again and again within days | Anecdote | Each later post does 0.79x; total reach untested |
| Gate access with a waitlist | Anecdote (AI consumer apps) | No effect found (51.3%, 36.6 to 68.2) |
| End with a call to action | Anecdote | No effect found (52.6%, 45.6 to 59.5) |
| Say it is free | Controlled, but on purchase choice | No effect found on likes (50.1%, 39.4 to 59.8) |
| Tease the day before | Anecdote; X's own launch framework | No effect found (54.9%, 41.1 to 67.8) |
| Lead with a benchmark | Anecdote | Lean only (54.5%, 48.7 to 60.5) |
| Adding an image raises engagement | Large observational | Not established for launch posts (1.17x, 0.95 to 1.45) |
| Hashtags add 21% | Anecdote, no traceable study | Too few to test; the direction is against |

And the outside claims our data supports: links in the post cost something (but 1.29x on likes, nowhere near 4 to
8x); specific, informative first lines win; posts from one account close together suppress each other; open-source
releases travel; a launch video beats none.

"No effect found" on likes does not show a tactic is useless for sign-ups or sales. A waitlist or a call to action may
do its work on the people who already saw the post. This corpus measures how far the post travels, in likes.

## 7. What the corpus cannot say

- Video length: `duration_ms` is empty for all 852 video launch posts.
- Anything in replies: link in reply, the body of a thread, replies from well-known accounts, the author replying.
- Reposts by other accounts, paid promotion, clipping campaigns.
- What an image shows (benchmark chart, screenshot, face): left to the media track; the text proxy showed nothing.
- Founder account against company account for the same launch (no bios).
- Effects on views, and anything for accounts under 1,000 followers.
- Engagement-bait requests: 10 launch posts in the main set.

## 8. Traits the model should receive as inputs

Full list with sources in `model_traits.json`.

- **Measured effect here:** post type; first line names the product; three or more lines; media kind; problem-first
  opener; quote or original; link in the post; number in the first line; open-source / open-weights wording; "first /
  best / most capable" claim; credits or discount offer; hours since the account's previous post (2-hour and 24-hour
  marks); the account's launch-type posts in the previous 7 days.
- **Context to condition on, not levers:** model release; company or founder account; follower tier; period (several
  effects are weaker after April 2026).
- **Outside evidence only, to be filled when the data exists:** first-hour quote posts by other accounts (count,
  same-team count, followers of the quoters), as a signal after posting and never as advice; video duration; image
  content; link in the first reply; engagement-bait request.
- **Do not add on this evidence:** Pacific hour or weekday; call to action; "free"; waitlist; benchmark claim in the
  text; number anywhere in the text; teaser; celebration emoji; "you" in the first line; availability line; demo
  wording.

## Limits

- Observational. "Inside the same account, launch posts with this trait did better" is not proof that editing a draft
  produces the gain. Traits such as open source, "most capable" and model release describe what is being launched as
  much as how it is written.
- Likes only. Accounts are mostly technology and AI accounts with more than 10,000 followers.
- Text traits are regular expressions checked by reading seven samples each, once. Clean: call to action, free, open
  source, benchmark, named comparison, superlative, availability, waitlist (research previews count as waitlist).
  About half correct: credits and discounts, named customers, backstory, coming soon, backed by. Teasers: about half.
- About a quarter of "launch" labels are wrong (launch track); the strict set controls for that.
- The outside review was assembled by web search on one day. 17 figures were not re-checked against the source page.
  Several vendor pages were read through search summaries. X's launch guide could not be opened; its framework is
  reported as "Tease, Reveal, Sustain".

## Appendix: every outside claim

@@appendix
