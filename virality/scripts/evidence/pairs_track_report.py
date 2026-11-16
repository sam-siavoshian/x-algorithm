#!/usr/bin/env python3
"""Write tracks/pairs/REPORT.md from the result files of the controlled-pairs track.   python scripts/evidence/pairs_track_report.py <pairs dir> <out.md>"""
import json
import sys

D, OUT = sys.argv[1], sys.argv[2]
G = json.load(open(f"{D}/final_gap_table.json"))
R = json.load(open(f"{D}/results.json"))
DEV = json.load(open(f"{D}/dev_eval.json"))
C = json.load(open(f"{D}/pair_counts.json"))
N = json.load(open(f"{D}/label_noise.json"))
BC = json.load(open(f"{D}/benchmark_by_cosine.json"))
M = G["models"]
p = lambda x, d=1: "-" if x is None else f"{100 * x:.{d}f}%"  # noqa: E731
ci = lambda c: f"{100 * c[0]:.1f} to {100 * c[1]:.1f}"  # noqa: E731
L: list[str] = []
w = L.append


def gap_rows(name: str, label: str) -> None:
    for g, x in M[name]["by_gap"].items():
        w(f"| {label} | {g} | {x['pairs']:,} | {p(x['accuracy'])} | {p(x['confident_share'], 0)} ({x['confident_pairs']}) | {p(x['confident_accuracy'])} ({ci(x['confident_accuracy_ci'])}) |")


def bands(name: str) -> str:
    return ", ".join(f"{k.replace('-', ' to ')}: {p(v['accuracy'])} of {v['pairs']}" for k, v in M[name]["by_gap"]["1.25x"]["bands"].items())


def vs(a: str, b: str) -> str:
    v = G["versus"][f"{a},{b}"]
    return f"{100 * v['accuracy_gap']:+.1f} ({100 * v['accuracy_gap_ci'][0]:+.1f} to {100 * v['accuracy_gap_ci'][1]:+.1f})"


def vsc(a: str, b: str) -> str:
    v = G["versus"][f"{a},{b}"]
    return f"{100 * v['confident_gap']:+.1f} ({100 * v['confident_gap_ci'][0]:+.1f} to {100 * v['confident_gap_ci'][1]:+.1f})"


e, b = M["ft_w10_ensemble"], M["baseline_ensemble"]
w("# Track \"pairs\": more controlled pairs, used to train the surrogate itself")
w("")
w("Worker report, 2026-11-16. All numbers are likes; the benchmark is the fixed 3,530 same-account, same-topic pairs of MODEL.md 9.4")
w(f"({M['base_ts46_s1']['pairs']:,} with like counts at least 25% apart). No benchmark pair was trained or tuned on.")
w("")
w("## Headline")
w("")
c = M["ft_w0_ensemble"]
x, y, z = e["by_gap"]["2.0x"], b["by_gap"]["2.0x"], c["by_gap"]["2.0x"]
w("The hypothesis holds weakly: a small gain that is real at the level of the three-model ensemble, well short of the goal, and it does not come from the newly mined pairs.")
w("")
w(f"- **Ensemble.** Three 46-column models each fine-tuned for 600 steps with the pair loss (weight 10): {p(e['accuracy'])} of all pairs correct ({ci(e['accuracy_ci'])}) against {p(b['accuracy'])} for the baseline ensemble,")
w(f"  {vs('ft_w10_ensemble', 'baseline_ensemble')} points paired, interval from resampling accounts. The same three models fine-tuned the same way without the pair loss score {p(c['accuracy'])}; against that control the gain is {vs('ft_w10_ensemble', 'ft_w0_ensemble')}.")
w(f"- **Confident calls.** On pairs 2x or more apart: {p(x['confident_share'], 0)} of pairs at {p(x['confident_accuracy'])} correct, against {p(y['confident_share'], 0)} at {p(y['confident_accuracy'])} (baseline) and {p(z['confident_share'], 0)} at {p(z['confident_accuracy'])} (control).")
x3, x5 = e["by_gap"]["3.0x"], e["by_gap"]["5.0x"]
w(f"  At 3x: {p(x3['confident_share'], 0)} at {p(x3['confident_accuracy'])}; at 5x: {p(x5['confident_share'], 0)} at {p(x5['confident_accuracy'])}. The goal is 40% coverage at 75 to 80% on 2x pairs: coverage moved a third of the way, accuracy barely.")
w(f"- **Single model.** The best fine-tune of `ts46_s1` is {p(M['ft_w10']['accuracy'])} against {p(M['base_ts46_s1']['accuracy'])}, {vs('ft_w10', 'base_ts46_s1')}: not detectable alone. The three members gain +0.4, +0.9 and +0.9 against their own starting points.")
w(f"- **The mined pairs add nothing.** Fine-tuning on the 1,471 same-link-or-quote pairs alone does as well as on all 11,507. On mined pairs of held-out accounts accuracy stays at {p(DEV['base_ts46_s1']['dev']['all_1.25x']['accuracy'])} before and after.")
w(f"- **From scratch is worse.** Training from scratch with the pair loss gives {p(M['scratch_w1']['accuracy'])} ({vs('scratch_w1', 'base_ts46_s1')} points). It memorised the training pairs")
w(f"  ({p(DEV['scratch_w1']['train']['all_1.25x']['accuracy'])} correct on them) and stayed at {p(DEV['scratch_w1']['dev']['all_1.25x']['accuracy'])} on held-out pairs.")
i = N["identical"]["capped_20_per_account"]
w(f"- **Why so little.** When one account posts the same text twice, the like counts differ by 2x or more {p(1 - i['within_2.0x'], 0)} of the time. Pairs of one message reworded are barely more spread out than that,")
w("  so their labels are mostly luck and their ceiling is about 72%. Benchmark-style pairs (two takes on one link) have a ceiling near 80% overall and 85% at 2x.")
w("- **Checkpoints.** `runs/ft_w10`, `runs/ft_w10_s2`, `runs/ft_w10_v2full` together beat the baseline ensemble by the margin above. Caveats before shipping them are under \"Honest limits\".")
w("")
w("## Step 1. Mining more controlled pairs")
w("")
w("`scripts/evidence/mine_pairs.py`, `scripts/evidence/select_pairs.py`. Counts in `pair_counts.json`.")
w("")
w("Embedding all 4.48M posts with bge-small takes many hours on four cores, so mining ran in two stages. First a word-overlap prefilter: within each account,")
w("every pair of posts at most 14 days apart, both at least 6 words, with TF-IDF cosine of 0.25 or more over word 1- and 2-grams (links removed): 2,145,761 candidate pairs")
w("over 321,490 posts. Then those posts were embedded with bge-small (plain text, links removed, no author line) and each pair got its embedding cosine. Pairs with no shared")
w("words at all are therefore missed. The same-link and same-quoted-post pairs of the benchmark definition were added whatever their cosine.")
w("")
w("Threshold, from reading eight random pairs per band (pairs whose posts sit in at most three candidate pairs):")
w("")
w("| Cosine | What the sampled pairs are |")
w("|---|---|")
w("| 0.95 and over | 8 of 8 the same message reworded (a sentence added, a word changed, the same story from another outlet) |")
w("| 0.90 to 0.95 | 6 of 8 the same message; 2 the same topic with different content |")
w("| 0.85 to 0.90 | about half the same message; the rest one template with different content (market recaps, action alerts) |")
w("| 0.80 to 0.85 | 2 of 8; mostly templates and replies |")
w("| below 0.80 | same account voice or same topic at most |")
w("")
w("So 0.90 and over is the tier \"strict\" (same message reworded) and 0.85 to 0.90 the tier \"loose\". In a second sample taken from the final training set, 10 of 12 strict pairs")
w("were the same message reworded (the two misses were a weekly podcast title and a daily reading list), and about 6 of 12 loose pairs.")
w("The benchmark's own pairs are looser than either tier: their median cosine is 0.70, they are two takes on one link or quoted post, not one message reworded.")
w("")
w("| Stage | Pairs | Before 2026-04-23 | After | Straddling | Accounts | Likes 1.25x apart | 2x apart |")
w("|---|---|---|---|---|---|---|---|")
names = {"0 same link": "Same link or quoted post, 48 hours (benchmark definition)", "0b": "Same link or quoted post, 14 days", "1 candidates with cosine >= 0.85": "Cosine 0.85 and over, texts differ",
         "1 candidates with cosine >= 0.90": "Cosine 0.90 and over, texts differ", "2 ": "After length and reply filters (cosine 0.85 and over, or shared link or quote)",
         "3 ": "After dropping template posts (a post in more than 3 mined pairs)", "4 after": "After the cap of 40 pairs per account and period",
         "4 tier tac": "of which shared link or quoted post", "4 tier strict": "of which strict (0.90 and over)", "4 tier loose": "of which loose (0.85 to 0.90)",
         "5 training": "Training pool: before the cutoff, no benchmark post, no account with a benchmark calibration pair", "5 tier tac": "of which shared link or quoted post",
         "5 tier strict": "of which strict", "5 tier loose": "of which loose"}
for k, v in C.items():
    lab = next((n for pre, n in names.items() if k.startswith(pre)), None)
    if lab:
        w(f"| {lab} | {v['pairs']:,} | {v['before_cutoff']:,} | {v['after_cutoff']:,} | {v['straddling']:,} | {v['accounts']:,} | {v['likes_differ_1.25x']:,} | {v['likes_differ_2x']:,} |")
w("")
w("Reading the table: raw near-duplicates are plentiful (403,338 pairs at cosine 0.90 before the cutoff) but nine in ten come from posts that repeat a template dozens of times.")
w("After removing those, 51,246 pre-cutoff pairs remain against 7,158 by the benchmark definition, about seven times more. The training pool is 45,786 pairs, 32,664 with likes 1.25x apart.")
w("")
w("Excluded from training: every pair containing a benchmark post, and every account that has a benchmark calibration pair (their histories contain benchmark posts). That second rule")
w("costs half of the pre-cutoff link-or-quote pairs (4,530 to 2,171), because the calibration half of the benchmark was drawn from exactly those accounts.")
w("")
w("Reposted-own-content pairs (identical text) were mined and used in step 2 as the noise measurement; they carry no wording signal and were not trained on. Thread-opener variants were")
w("not mined separately: the filtered corpus holds originals and quotes, and an opener reworded is already a near-duplicate pair.")
w("")
w("**What was actually trained on.** The pair table is built through the serving path (step 3), which needs bge-small embeddings of each pair's history, about 60 posts per second on this machine.")
w("The full pool needed 468,000 new embeddings (about two hours), so accounts were chosen within a budget of 105,000: all accounts with a link-or-quote pair first, then by pairs per post to embed.")
tr, dv = DEV["base_ts46_s1"]["train"], DEV["base_ts46_s1"]["dev"]
w(f"Result: 13,518 pairs from 3,037 accounts, all with likes 1.25x apart: {tr['all_1.25x']['pairs']:,} for training ({tr['tac_1.25x']['pairs']:,} link-or-quote, {tr['strict_1.25x']['pairs']:,} strict, {tr['loose_1.25x']['pairs']:,} loose)")
w(f"and {dv['all_1.25x']['pairs']:,} from 15% of the accounts held out ({dv['tac_1.25x']['pairs']}, {dv['strict_1.25x']['pairs']:,}, {dv['loose_1.25x']['pairs']}). The selection favours accounts with short histories (median 28 posts of history, against 48 for link-or-quote pairs).")
w("")
w("## Step 2. Label noise: the ceiling for any model")
w("")
w("`scripts/evidence/label_noise.py`, `label_noise.json`. Same account, same normalised text (lower case, links removed), at least 6 words, same media state and kind, at most 14 days apart.")
w("At serving time nothing distinguishes the two sides, so their spread is noise no model can predict.")
w("")
w("| Identical-text pairs | Pairs | Likes within 1.25x | within 2x | within 3x | within 5x |")
w("|---|---|---|---|---|---|")
I = N["identical"]
for k, lab in (("all", f"All ({I['all']['accounts']:,} accounts)"), ("capped_20_per_account", "At most 20 per account"), ("one_pair_per_account", "One per account"), ("within_48h", "At most 48 hours apart"),
               ("2_to_14_days", "2 to 14 days apart"), ("after_cutoff", "After the cutoff")):
    v = I[k]
    w(f"| {lab} | {v['pairs']:,} | {p(v['within_1.25x'], 0)} | {p(v['within_2.0x'], 0)} | {p(v['within_3.0x'], 0)} | {p(v['within_5.0x'], 0)} |")
for k, v in I["by_larger_like_count"].items():
    lo, hi = k.split("-")
    w(f"| Larger like count {lo} to {hi if int(hi) < 10 ** 6 else 'up'} | {v['pairs']:,} | {p(v['within_1.25x'], 0)} | {p(v['within_2.0x'], 0)} | {p(v['within_3.0x'], 0)} | {p(v['within_5.0x'], 0)} |")
w("")
w(f"The same text posted twice lands within 2x only {p(I['capped_20_per_account']['within_2.0x'], 0)} of the time, and beyond 5x {p(1 - I['capped_20_per_account']['within_5.0x'], 0)} of the time. The noise grows with the size of the post:")
w("above 1,000 likes, identical posts are within 2x only 28% of the time. The later copy wins 45% of the time (the benchmark's 64% for the later version does not come from reposting).")
w("")
w("Reworded pairs next to that (at most 20 per account):")
w("")
w("| Pairs | Count | Likes within 2x | within 3x | Spread of log gap |")
w("|---|---|---|---|---|")
w(f"| Identical text | {I['capped_20_per_account']['pairs']:,} | {p(I['capped_20_per_account']['within_2.0x'], 0)} | {p(I['capped_20_per_account']['within_3.0x'], 0)} | {I['capped_20_per_account']['sd_log']:.2f} |")
for k, v in N["reworded_by_cosine_capped_20_per_account"].items():
    w(f"| Texts differ, cosine {k.replace('-', ' to ').replace('1.01', '1.00')} | {v['pairs']:,} | {p(v['within_2.0x'], 0)} | {p(v['within_3.0x'], 0)} | {v['sd_log']:.2f} |")
o = N["ceiling_link_or_quote_pairs_48h"]["observed"]
w(f"| Same link or quoted post, 48 hours | {o['pairs']:,} | {p(o['within_2.0x'], 0)} | {p(o['within_3.0x'], 0)} | {o['sd_log']:.2f} |")
w("")
w("Near-duplicate pairs are spread almost exactly like identical ones. Link-or-quote pairs are clearly wider: those two posts differ in more than wording.")
w("")
w("**Ceiling by size of gap.** The observed log gap is modelled as effect plus noise, the noise drawn from the identical pairs and the effect from a Laplace distribution holding the")
w("variance the pairs have beyond noise. The ceiling is how often the observed winner is the side with the better effect: the score of a model that knew every effect exactly.")
w("")
w("| Real gap at least | Benchmark-style pairs (link or quote, 48 h): share, ceiling | Reworded, cosine 0.90 and over: share, ceiling | Cosine 0.85 to 0.90: share, ceiling |")
w("|---|---|---|---|")
for g in ("1.25x", "2.0x", "3.0x", "5.0x"):
    cells = [f"{p(N[k]['laplace'][g]['share_of_pairs'], 0)}, {p(N[k]['laplace'][g]['ceiling'], 0)}" for k in ("ceiling_link_or_quote_pairs_48h", "ceiling_reworded_cosine_0.90_plus", "ceiling_reworded_cosine_0.85_to_0.90")]
    w(f"| {g} | " + " | ".join(cells) + " |")
w("")
w("With a normal instead of a Laplace effect the ceilings are 2 to 3 points higher. Two readings. On the benchmark's kind of pair a perfect model would score about 80% overall and 85% at 2x,")
w("so the target (75 to 80% on confident calls) is under the ceiling, and the present 60% is far from it. On reworded-message pairs the ceiling is about 71 to 75% and does not rise with the gap:")
w("a large gap between two wordings of one message is mostly luck. Those are the pairs this track added.")
w("Caveat: the model assumes reworded pairs carry the same noise as identical reposts; if platforms treat exact reposts differently, the noise is misjudged in an unknown direction.")
w("")
w("## Step 3. Fine-tuning the surrogate with a loss on the mined pairs")
w("")
w("Code: `xviral/model/pairs.py`, `scripts/build_pair_table.py`, `scripts/train_pairs.py`, small changes in `xviral/model/train.py` and `config.py` (all in `patch.diff`).")
w("")
w("- **Pair table.** Each pair goes through `DraftScorer.featurize` exactly as in the benchmark: the two posts as alternative drafts at the time of the earlier one, the account's earlier posts")
w("  as history (48, the model's window), neither outcome visible. Training episodes are therefore the serving episode, not consecutive posts with their own features.")
w("- **Loss.** RankNet on the difference of the like head's predicted mean for the two isolated candidates, weight proportional to the log like ratio (capped at 5x), pairs under 1.25x unused,")
w("  added to the ordinary loss with weight `w_ctl`. One batch of 64 pairs per step next to the ordinary batch of 64 episodes. A margin variant exists (`--ctl-loss margin`) and was not run for lack of time.")
w("- **Runs.** From `runs/ts46_s1/best.pt`, 600 steps, learning rate 2e-4, warm-up 50, weights 0 (control), 1, 3, 10, 30; one run on the link-or-quote pairs alone; weights 10 and 0 also from `ts46_s2` and `compare_v2/full`")
w("  to form a fine-tuned and a control three-model ensemble. Checkpoint choice inside a run uses the held-out controlled pairs and the validation loss, never the benchmark.")
w("- **Benchmark.** `tac_serve_eval.py` with a new `--pairs` option that scores the stored list of 3,530 pairs. My run of the unchanged `ts46_s1` reproduces the stored per-pair scores (correlation 0.99999, same accuracy).")
w("")
w("| Model | Trained pairs | Benchmark, all pairs | vs start, points | 2x and over | Confident share | Correct there | vs start | Held-out pairs | Training pairs | Test pairwise likes | Test like NLL |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
bt = R["baseline_test_metrics"]
rows = [("base_ts46_s1", "ts46_s1 (start)", "0"), ("ft_w0", "control, weight 0", "0"), ("ft_w1", "weight 1", "11,507"), ("ft_w3", "weight 3", "11,507"), ("ft_w10", "weight 10", "11,507"), ("ft_w30", "weight 30", "11,507"),
        ("ft_tac_w3", "link-or-quote pairs only, weight 3", "1,471"), ("ft_half_w1", "half the pairs, weight 1", "5,595"), ("scratch_w1", "from scratch, weight 1 (step 4)", "11,507"),
        ("ft_w0_s2", "from ts46_s2: control, weight 0", "0"), ("ft_w10_s2", "from ts46_s2: weight 10", "11,507"), ("ft_w0_v2full", "from compare_v2/full: control, weight 0", "0"), ("ft_w10_v2full", "from compare_v2/full: weight 10", "11,507")]
for n, lab, k in rows:
    m = M[n]
    t = R["runs"][n].get("test") or bt
    dv_, tr_ = DEV.get(n, {}).get("dev", {}).get("all_1.25x"), DEV.get(n, {}).get("train", {}).get("all_1.25x")
    w(f"| {lab} | {k} | {p(m['accuracy'])} | {vs(n, 'base_ts46_s1') if n != 'base_ts46_s1' else '-'} | {p(m['by_gap']['2.0x']['accuracy'])} | {p(m['confident']['share'], 0)} | {p(m['confident']['accuracy'])} | "
      f"{vsc(n, 'base_ts46_s1') if n != 'base_ts46_s1' else '-'} | {p(dv_['accuracy']) if dv_ else 'not run'} | {p(tr_['accuracy']) if tr_ else 'not run'} | {t['pairwise_likes']:.4f} | {t['nll_likes']:.3f} |")
w("")
w(f"Benchmark columns: {M['base_ts46_s1']['pairs']:,} pairs; intervals are 95% from resampling accounts, paired against `ts46_s1` (for the last four rows that is not their own starting model; those comparisons follow below). \"Confident\" is stated confidence of 60% or more, scale cross-fitted over accounts.")
w(f"Held-out pairs: {dv['all_1.25x']['pairs']:,} mined pairs from 444 accounts never used for the pair loss. Test columns: the ordinary audience test split (95,357 rows).")
w("")
w("Against each model's own start and its own control (points, paired):")
w("")
w("| Member | Weight 10 vs start | Control vs start | Weight 10 vs control | Same, among confident calls |")
w("|---|---|---|---|---|")
for lab, f, c0, st in (("ts46_s1", "ft_w10", "ft_w0", "base_ts46_s1"), ("ts46_s2", "ft_w10_s2", "ft_w0_s2", "ts46_s2"), ("compare_v2/full", "ft_w10_v2full", "ft_w0_v2full", "v2full")):
    w(f"| {lab} | {vs(f, st)} | {vs(c0, st)} | {vs(f, c0)} | {vsc(f, c0)} |")
w(f"| Ensemble of the three | {vs('ft_w10_ensemble', 'baseline_ensemble')} | {vs('ft_w0_ensemble', 'baseline_ensemble')} | {vs('ft_w10_ensemble', 'ft_w0_ensemble')} | {vsc('ft_w10_ensemble', 'ft_w0_ensemble')} |")
w("")
w("What the tables say:")
w("")
w(f"1. **No transfer to held-out mined pairs.** Accuracy on them is {p(DEV['base_ts46_s1']['dev']['all_1.25x']['accuracy'])} before and {p(min(DEV[n]['dev']['all_1.25x']['accuracy'] for n in ('ft_w1', 'ft_w3', 'ft_w10', 'ft_w30')))} to "
  f"{p(max(DEV[n]['dev']['all_1.25x']['accuracy'] for n in ('ft_w1', 'ft_w3', 'ft_w10', 'ft_w30')))} after, at every weight, while accuracy on the training pairs rises from {p(DEV['base_ts46_s1']['train']['all_1.25x']['accuracy'])} to "
  f"{p(max(DEV[n]['train']['all_1.25x']['accuracy'] for n in ('ft_w1', 'ft_w3', 'ft_w10', 'ft_w30')))}.")
w("2. **A small gain on the benchmark, visible only when models are pooled.** No single fine-tune clears its interval against its start. The three together do, against the baseline ensemble and against the control ensemble, by about 1.4 points.")
w("   Fine-tuning without the pair loss changes nothing (control ensemble equals the baseline to a tenth of a point), so the gain is the pair loss, not the extra steps.")
w("3. **More confident calls, slightly more often right.** From weight 3 up, confident calls cover 31 to 33% of pairs instead of 27% for the single model, and 34% instead of 27% for the ensemble, at 1.5 to 2 points higher accuracy.")
w("   The model separates the two versions further apart where it was already leaning the right way; it does not get many more pairs right.")
w("4. **1,471 link-or-quote pairs do as much as all 11,507.** The 10,036 mined rewordings add nothing measurable on top. That matches step 2: their labels are close to noise.")
w("   What helps is training on the benchmark's own kind of pair, scored the way the product scores, and only about 1,700 such pairs were available for training.")
w("5. **Weight.** 1 does nothing; 3, 10 and 30 are alike on the benchmark. Weight 10 was used for the other two members.")
w("6. **Ordinary metrics hold.** Test pairwise accuracy on likes stays at 0.598 to 0.600. The like NLL drifts up with the weight (2.966 at the start, 2.989 at weight 10, 3.001 at weight 30), a small loss of calibration.")
w("   (The validation NLL printed at the start of each run, 2.942, is on a different 20,000-row subset than the in-run figure, 2.957; the control shows the in-run level is unchanged.)")
w("")
w("By held-out tier (pairs 1.25x apart; start model, then weight 10):")
w("")
w("| Tier | Held-out pairs | Start | Weight 10 | Link-or-quote only | From scratch |")
w("|---|---|---|---|---|---|")
for t_, lab in (("tac", "Same link or quoted post"), ("strict", "Strict (cosine 0.90 and over)"), ("loose", "Loose (0.85 to 0.90)"), ("all", "All")):
    w(f"| {lab} | {DEV['base_ts46_s1']['dev'][t_ + '_1.25x']['pairs']:,} | " + " | ".join(p(DEV[n]['dev'][t_ + '_1.25x']['accuracy']) for n in ("base_ts46_s1", "ft_w10", "ft_tac_w3", "scratch_w1")) + " |")
w("")
w("The start model is at 53% on reworded-message pairs and 59% on link-or-quote pairs, against ceilings of about 71% and 80%.")
w("")
w("### Next to the baseline: accuracy by size of real gap")
w("")
w("| Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there (95% interval) |")
w("|---|---|---|---|---|---|")
gap_rows("baseline_ensemble", "Baseline: three 46-column models")
gap_rows("ft_w10_ensemble", "The same three, fine-tuned with the pair loss (weight 10)")
gap_rows("ft_w0_ensemble", "The same three, fine-tuned without it (control)")
gap_rows("base_ts46_s1", "ts46_s1 alone")
gap_rows("ft_w10", "ts46_s1 fine-tuned, weight 10")
gap_rows("ft_tac_w3", "ts46_s1 fine-tuned, link-or-quote pairs only")
gap_rows("scratch_w1", "From scratch with the pair loss")
w("")
w("The gap is measured on (likes + 1), so the pair counts at 2x and 3x (2,071 and 1,390) are slightly above the README's 2,030 and 1,377; the baseline rows otherwise match it.")
w("")
w("By stated confidence, pairs 1.25x apart (correct share of pair count):")
w("")
for n, lab in (("baseline_ensemble", "Baseline ensemble"), ("ft_w10_ensemble", "Fine-tuned ensemble, pair loss"), ("ft_w0_ensemble", "Fine-tuned ensemble, control"), ("base_ts46_s1", "ts46_s1"), ("ft_w10", "ts46_s1, weight 10"), ("scratch_w1", "From scratch")):
    w(f"- {lab}: {bands(n)}")
w("")
w("The bands above 70% hold 200 to 300 pairs, and the 80% band 40 or fewer: read those as plus or minus 6 to 15 points.")
w("")
w("## Step 4. From scratch with the pair loss")
w("")
s = R["runs"]["scratch_w1"]
w("Same recipe as `ts46_s1` (small preset, 46 columns, seed 1, 4,000 steps, learning rate 1e-3) with the pair loss at weight 1 from the first step.")
w(f"Ordinary validation ends where the baseline does (pairwise likes {s['val_pairwise_likes']:.3f}, test like NLL {s['test']['nll_likes']:.3f}), but the benchmark falls to {p(M['scratch_w1']['accuracy'])}")
w(f"({vs('scratch_w1', 'base_ts46_s1')} points, below the 58.2 to 60.9% range of the ten existing single models) and confident calls shrink to {p(M['scratch_w1']['confident']['share'], 0)} of pairs.")
w(f"The model reaches {p(DEV['scratch_w1']['train']['all_1.25x']['accuracy'])} on its training pairs and {p(DEV['scratch_w1']['dev']['all_1.25x']['accuracy'])} on held-out ones: with 4,000 steps of 64 pairs it sees each pair about 22 times and learns")
w("them by heart, apparently at the expense of what the like head otherwise learns about content. One run, one seed; the direction is clear, the size is not.")
w("")
w("## What worked, what did not")
w("")
w("Worked: fine-tuning with the pair loss on serving-path episodes, modestly (+1.4 points for the ensemble, confident coverage from 27% to 34%); the noise measurement, which explains both the size of this result")
w("and the earlier failure of the direct pair model; the mining recipe and its template filter as a way to find rewordings; scoring the benchmark from a stored pair list with reused embeddings (71 seconds per model once the caches exist).")
w("")
w("Did not work: the mined rewordings as training signal (no effect beyond the 1,471 link-or-quote pairs); held-out mined pairs (no gain at any weight); the pair loss from scratch (3 points worse); weight 1 (no effect).")
w("")
w("## Honest limits")
w("")
w("- **The weight for the ensemble was chosen with the benchmark in view.** Held-out mined pairs could not separate the weights, and weights 1, 3 and 10 had been scored on the benchmark for `ts46_s1` when 10 was picked.")
w("  The other two members were then fine-tuned once at that weight and scored once; their gains do not carry that choice. The ensemble intervals (+0.2 to +2.6, and +0.1 to +2.5 against the control) only just exclude zero.")
w("- One seed per configuration. Single-model differences under about 1.5 points are inside the account-resampling intervals. `compare_v2/full` carries most of the ensemble's gain against control (+1.8); `ts46_s1` and `ts46_s2` add +0.3 and +0.4.")
w("- Only 13,518 of the 32,664 usable pool pairs were trained on (encoder time), chosen toward accounts that were cheap to embed, so with shorter histories than the benchmark's accounts. Held-out accuracy did not differ")
w("  between half the pairs and all of them (54.3% and 54.5%), which suggests the remaining 19,000 would not change the picture, but that was not run.")
w("- Half of the pre-cutoff link-or-quote pairs were unavailable for training because their accounts hold the benchmark's calibration pairs. Since those pairs are what helped, a benchmark that kept only the 2,030 test-period pairs would free about 2,400 more.")
w("- The margin loss, longer fine-tunes, a lower pair weight from scratch, and the pair loss on the ts0 models (which lead this benchmark by about 2 points) were not run.")
w("- Post-cutoff mined pairs (3,458 strict, 2,344 loose) exist as a second test set for reworded messages and were not scored; they would need about 40 minutes of embedding.")
w("- Mining can only find rewordings that share words; a paraphrase with none is missed.")
w("- The ceiling rests on identical reposts being a fair sample of noise, and on a symmetric effect distribution.")
w("- The like head's NLL on the ordinary test split rises by 0.02 at weight 10. The advice layer's confidence scale must be refitted for fine-tuned models (the cross-fitted scale drops from 2.95 to 1.90 for the ensemble).")
w("")
w("- Tests: 113 pass, 1 skipped, 2 fail (`tests/test_serve.py::test_advice_gates` and `test_operators_change_one_thing`). Both fail on a missing file, `xviral/data/edit_evidence.json`, which is not in the checkout; neither touches the code changed here.")
w("")
w("## What I would do next")
w("")
w("1. Re-test the weight-10 fine-tune with fresh seeds and on the ts0 models before shipping it; the gain is small enough that one more ensemble could confirm or erase it.")
w("2. Stop mining rewordings. The limit there is the label, not the count: a ceiling near 72% with almost no room between noise and signal.")
w("3. Get more pairs of the benchmark's own kind, which are the ones that helped (training already used the 14-day version): free the calibration accounts, and drop the cap of 40 per account for this tier.")
w("4. Reduce label noise at the source: the product's own randomized comparisons (research/11, section 7), or accounts that post one text many times, where the average over copies is a far cleaner label.")
w("")
w("## Files in `tracks/pairs/`")
w("")
w("- `REPORT.md`, `patch.diff` (code changes and new files against the checked-out branch).")
w("- `results.json` (every run: benchmark, held-out pairs, validation, test), `final_gap_table.json` and `.txt` (benchmark by gap and confidence band, all models and ensembles), `dev_eval.json` (held-out and training pairs by tier), `baseline_gap_table.json`.")
w("- `pair_counts.json`, `label_noise.json`, `label_noise_identical.parquet`, `benchmark_by_cosine.json`.")
w("- `train_pool.parquet` (45,786 pairs with cosine and tier), `train_pairs.parquet` (the 13,518 trained and held out), `after_cutoff_pairs.parquet`, `benchmark_pairs.parquet`.")
w("- `pair_table/` (the serving-path feature table of the 13,518 pairs, ready for `scripts/train_pairs.py`).")
w("- `bench/cache_{0..3}.pkl` (feature caches for the 3,530 benchmark pairs: any new checkpoint scores in about a minute with `scripts/bench_pairs.sh`), `bench/<run>/shard_*.parquet` (per-pair scores of every run).")
w("- `runs/<run>/best.pt` and `metrics.json` for each run. `ft_w10`, `ft_w10_s2`, `ft_w10_v2full` are the fine-tuned ensemble; `ft_w0*` the controls.")
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
