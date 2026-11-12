# Reasons and fixes: what the product may claim, and how each claim is earned

Question: how do we tell a user "this draft will underperform because X, change it to Y" without it being generic language-model advice, and how sure can we be?

Sources: two literature reviews run on 2026-11-11 (links at the end), X's code in this repository, and three measurements on our own corpus (`scripts/evidence/`, results in `11_within_account_effects.json` and `11_same_topic_pair_effects.json`).

## 1. The honest ceiling

No one can be sure a single post will do better after a change, and the product must not say so.

- When news sites reran the identical headline test, the winner changed about a quarter of the time (Chartbeat, 141,000 tests). Noise alone caps the hit rate.
- On pairs of tweets by the same author about the same link with different wording, the best published model picked the winner 65.6% of the time and people 61.3% (Tan, Lee, Pang 2014). A model trained without those controls scored 55.3% on the same pairs.
- On 32,487 randomized headline tests, language models pick the best headline 40 to 47% of the time against 33% by chance (LOLA, Upworthy archive).
- Meta's 2026 study: the offline predictor's top-ranked ad creative finished 43rd of 50 live. Generating many candidates and measuring worked; trusting the predictor's single pick did not.
- There is no published randomized experiment on X from 2022 to 2026 for any post attribute. Everything said about "what works on X" is code, platform statements, or observational.

What can be earned is a track record: among the comparisons where we speak, the version we preferred wins a stated share of the time, measured on held-out data and then on customers' real posts.

## 2. Three kinds of statement, three standards of proof

| Kind | Example | Why it is true | When shown |
|---|---|---|---|
| Rule from X's code | "This is your third post in 5 hours. In one feed request your second post is scored at 0.625 and the third at 0.44." | Read off the repository; deterministic | Always, with the file it comes from |
| The account's own record | "Your 14 posts with an image: median 3,400 views. Your 52 without: 1,200." | Arithmetic on the user's own history, which they can check | When both groups are large enough and the interval excludes no difference |
| Tested change | "Version B (image added) is predicted to beat version A. For changes of this type, when we are this confident, the preferred version won 68% of 1,240 held-out comparisons." | The model's comparison, gated by a measured hit rate for that type of change | Only when every gate in section 4 passes; otherwise "no change we can stand behind" |

The numbers inside the examples in this section and in section 8 are illustrative; the real ones come from the benchmark in section 6 once the model is trained.

Reasons are never free text from a language model. A reason is either a rule that fired, a fact about the account's history, or the name of the one thing that differs between two scored versions.

## 3. Reasons by construction: one change at a time

Explaining a prediction after the fact (feature importances over an embedding) is unreliable and not checkable. Instead the product creates the explanation: it takes the draft, applies one edit operator, scores both versions against the same history in one pass, and reports the difference. The reason is the operator.

Edit operators, each a defined transformation: add an image; remove or move the link; split into lines; lengthen with a concrete detail; shorten the opening; change the posting time; wait until N hours after the previous post; convert quote to original or the reverse; remove hashtags or mentions.

A language model's only jobs are to carry out an operator on the text ("make the first line shorter, change nothing else") and to phrase the result. It never decides what is better. Multi-attribute rewrites are allowed as candidates, but then the claim is only "this version is preferred", with no single reason attached: published work shows attribution agrees with randomized results for one-word changes and diverges as more changes are stacked (Lemaire, Yin, Netzer).

## 4. Gates before a tested change is shown

1. **The type of change has a measured effect.** Each operator has its own hit rate on held-out, same-account, same-topic pairs (section 5). An operator with no measured effect is never recommended, whatever the model says.
2. **Calibrated confidence.** The model outputs P(B beats A). That probability is calibrated on held-out same-account pairs, and the threshold for speaking is set so the error among spoken comparisons stays under a stated level (selective prediction; Learn then Test).
3. **Agreement.** Every member of the ensemble prefers the same version.
4. **Support.** The rewritten version is close to things this account, or accounts like it, have posted. A version unlike anything in the data is where a model is confidently wrong.
5. **Meaning preserved.** The rewrite says the same thing as the draft.

If any gate fails the product says so: "no confident difference". YouTube's title test and Chartbeat both report "inconclusive" instead of forcing a winner; so do we.

## 5. What our own data says about each change (2026-11-11)

Method A, within account: for each account with at least five posts with and five without an attribute, compare the account's own averages, then average over accounts. 4,221 audience accounts, training period only.

Method B, same account and same topic: pairs of posts by one account that quote the same post or carry the same link with different wording, within 30 days, likes differing by at least 25%. 13,459 pairs from 2,023 accounts in the 4.48M-post corpus. This is the Tan, Lee, Pang design; the topic and the author are held fixed, so what remains is the wording, the format and the timing.

| Change | A: views | A: likes | B: version with it wins | Read |
|---|---|---|---|---|
| Image or video attached | 1.24x (1.17 to 1.32) | 1.36x | 67.0% (63.5 to 70.3), 2,247 pairs; 62% once posting order is balanced | Real. Agrees with the one controlled study in the literature |
| Posted within 2 hours of the previous post | 0.72x (0.68 to 0.76) | 0.64x | not testable this way | Real, and X's code gives the mechanism (same-author decay) |
| Three or more lines | no view data | 1.80x | 61.6% (58.3 to 64.8), 1,830 pairs | Real on likes; unmeasured on views |
| Text at least 1.5x longer | 1.74x (long posts) | 1.22x | 58.9% (57.6 to 60.3), 5,424 pairs | Real but modest. Matches Tan et al.: the more informative version wins |
| Link in the first line | no view data | 0.78x | 38.2% (31.8 to 45.3), 309 pairs | Leading with the link loses about 6 times in 10 |
| External link present at all | 0.97x (0.90 to 1.06) | 1.03x | not applicable | No detectable penalty in 2026 data. Matches X's statements since October 2025 that links are not deboosted |
| Link at the very end | 1.10x (0.90 to 1.33) | 1.24x | 48.6% (40.7 to 56.8), 175 pairs | Not established |
| Opens with a question | 1.20x | 0.81x | 48.9% (45.6 to 52.0), 1,261 pairs | Nothing. The two outcomes disagree in A and the controlled test is flat |
| Opens in first person | 1.11x | 1.16x | 51.7% (49.4 to 54.1) | Nothing |
| Has a hashtag | 1.15x | 0.99x | 53.5% (45.8 to 61.6), 202 pairs | Nothing |
| Mentions another account | 1.21x | 0.90x | 50.8% (46.6 to 54.6) | Nothing |
| Opens with a number | 0.46x | 0.50x | too few pairs | Confounded: these are list items and status updates, a different kind of post |

With posting order balanced (`scripts/evidence/build_edit_evidence.py`, the table the product reads): image or video 64.9% (61.5 to 68.2); three or more lines 61.5% (58.2 to 64.5); link not in the first line 61.3% (54.1 to 68.0); text 1.5x longer 58.7% (57.1 to 60.1); link at the very end 48.5%; opens with a question 48.9%; first person 51.7%; hashtag 53.4% (45.3 to 61.1); mention 50.9%. The first four are the supported kinds of change.

Three things this table teaches:

- **Within-account averages are not enough.** "Opens with a question" looks like +20% on views and -19% on likes. With the topic held fixed it is exactly 50/50. A product that recommended from method A, or from a model that learned method A's patterns, would hand out advice that does nothing.
- **Order is a confounder.** Of two posts about the same topic within 48 hours, the later one wins 64% of the time. The image result is 59% when the image version went first and 71% when it went second; balanced, about 62 to 65%. Every operator's hit rate must be computed with order balanced.
- **The well-known advice is mostly unsupported.** Questions, hashtags, mentions and first-person openings show nothing. What holds up is small in number: attach an image, do not lead with the link, give the post room (lines, detail), and space posts out.

## 6. How the tested-change hit rate gets built

1. `scripts/evidence/tac_pairs.py` builds the same-account, same-topic pairs from the whole corpus (42,473 within 30 days; 2,964 with views on both sides). Pairs after the test cutoff are held out.
2. The trained model scores both sides of every held-out pair. Report accuracy overall, per operator (pairs that differ in exactly that attribute), and by confidence band. This is the benchmark for the product's central claim; 66% is the published reference point.
3. Calibrate P(B beats A) on the validation pairs and fix the speaking threshold.
   **First result (2026-11-11, MODEL.md 9.3).** 579 held-out pairs scored through the draft path with the small all-accounts model: 60.8% correct overall, 68.3% on the 39% of pairs where calibrated confidence is at least 60%, 73.8% at 70 to 80% and 78.6% above 80%. On pairs that differ by an image the model is right 79.2% of the time. Likes only; the upper confidence bands hold under 100 pairs.
4. Add the Upworthy archive (32,487 randomized headline tests, CC BY 4.0) as an outside, randomized check on the wording operators. Expect weak transfer: clicks on headlines are not views in a feed.

## 7. The only thing that makes it sure: the live loop

Offline evidence stays observational. Certainty comes from the product measuring itself:

- Record every recommendation, whether the user took it, and what the post then did.
- Randomized encouragement: for a random share of posts where two versions are close, randomize which one is marked preferred. Pooled over accounts with the model's prediction as a covariate, this estimates each operator's true effect.
- Sample sizes (80% power, log views, within-account spread of about 1.0): a 2x lift needs about 33 posts per arm, 50% about 96, 20% about 473, 10% about 1,729. One account can never verify a 10 to 20% lift on its own; the evidence has to be pooled across customers.
- Publish the record in the product: "when we said likely better, it was, in 64% of 412 cases", per operator, with a reliability chart. Tag each reason "rule", "your record", or "tested, N cases".
- Re-measure after every change to X's code. The link penalty is the example: 4 to 8 times fewer views per follower in 2022 to 2024, confirmed by X; denied by X from October 2025; undetectable in our 2026 data.

## 8. What the product says, and does not say

Says: "Version B is likely to do better (confidence band: high). It differs from your draft in one way: an image is attached. For this kind of change at this confidence, the preferred version won 68% of 1,240 held-out comparisons. Two rules from X's code also apply to your draft: ..."

Does not say: "this will go viral", "guaranteed", a lift number that has not been calibrated against real outcomes, or a reason for which there is no rule, no record and no measured operator.

## Sources

Literature: Tan, Lee, Pang 2014 (arxiv.org/abs/1405.1438); Upworthy Research Archive (nature.com/articles/s41597-021-00934-7); LOLA (arxiv.org/abs/2406.02611); Hagar and Diakopoulos 2021 on Chartbeat tests (doi.org/10.1080/21670811.2021.1984266); Meta offline-to-online creative study (arxiv.org/abs/2607.23696); Meta AdLlama (arxiv.org/abs/2507.21983); Martin et al. 2016 on limits of prediction (arxiv.org/abs/1602.01013); Learn then Test (arxiv.org/abs/2110.01052); selective classification (arxiv.org/abs/1705.08500); Lemaire, Yin, Netzer on causal text attribution; Loghmani 2025 on spurious correlations in observational fine-tuning (arxiv.org/abs/2506.00152); link-penalty measurement 2022 to 2024 (arxiv.org/abs/2410.17390); impression half-life (arxiv.org/abs/2302.09654); Bojinov et al. 2023 on switchback designs (arxiv.org/abs/2009.00148).

Items the reviews could not verify from a primary page are not used above.
