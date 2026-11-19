#!/usr/bin/env python3
"""Track "sota": every table of the report from the runs on disk.

    python scripts/sota/report.py --out data/sota/results [--cands thurstone,soft,...] [--extra label=name1+name2 ...]

For the control and each candidate (two seeds, data/sota/runs/<cand>_s{1,2}): curves (train / validation / test at every
validation step), the chosen checkpoint's numbers, and paired differences against the control of the same seed on
  ordinary same-account test pairs (table rows after the cutoff; data/sota/test/<run>.npy),
  development pairs (data/sota/dev/<run>.npz), benchmark pairs (data/bench3/<run>/), launch pairs (data/sota/launch/<run>.parquet),
with 95% intervals from resampling accounts. The confidence scale on launch pairs is the one fitted on the benchmark.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/sota"); sys.path.insert(0, "scripts")
import sota_lib as S
from xviral.model import evaluate
L = S.L
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/sota/results"); ap.add_argument("--cands", default="thurstone,soft,ridge,histdrop,mask,snap,listnet")
ap.add_argument("--extra", action="append", default=[], help="label=treated1+treated2|control1+control2: an extra paired comparison of two ensembles")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
RUNS = Path("data/sota/runs")
cands = ["ctl"] + [c for c in a.cands.split(",") if all((RUNS / f"{c}_s{s}" / "metrics.json").exists() for s in (1, 2))]
pct = lambda x: "-" if x is None else f"{100 * x:.1f}"  # noqa: E731
res, md = {"candidates": {}, "versus": {}}, []

# --- curves ---
md += ["## Curves: train / validation / test at every validation step", "",
       "Same-account pairs on likes, ordinary posts (train: 6,000 fitted rows; validation: held-out accounts, selects the step by NLL; test: 6,000 rows after the cutoff), "
       "like-head NLL, and accuracy on the development pairs with a 2x gap. Two seeds, `s1 / s2`.", "",
       "| Run | Step | Pairs train | Pairs val | Pairs test | NLL train | NLL val (all heads) | NLL test | Dev pairs 2x |", "|---|---|---|---|---|---|---|---|---|"]
for c in cands:
    M = [json.load(open(RUNS / f"{c}_s{s}" / "metrics.json")) for s in (1, 2)]
    res["candidates"][c] = {"best_step": [m["best_step"] for m in M], "history": [m["history"] for m in M], "final": [m["final"] for m in M], "train_seconds": [m["train_seconds"] for m in M]}
    for i in range(len(M[0]["history"])):
        h = [m["history"][i] for m in M]
        two = lambda k, f=100, d=1: " / ".join(f"{f * x[k]:.{d}f}" for x in h)  # noqa: E731
        star = " / ".join("chosen" if x["step"] == m["best_step"] else "" for x, m in zip(h, M)).strip(" /")
        md.append(f"| {c} | {h[0]['step']}{' (' + star + ')' if star else ''} | {two('train_pairwise_likes')} | {two('pairwise_likes')} | {two('test_pairwise_likes')} | "
                  f"{two('train_nll_likes', 1, 3)} | {two('val_nll', 1, 3)} | {two('test_nll_likes', 1, 3)} | {two('dev_acc_2.0x')} |")

# --- per-pair arrays ---
def get(fn, names):
    try:
        return fn(names)
    except Exception as e:  # noqa: BLE001
        return None
rows_t = np.load("data/sota/test/rows.npy") if os.path.exists("data/sota/test/rows.npy") else None
if rows_t is not None:
    from train_one import FEATURE_SETS
    from xviral.model.features import FeatureTable
    from xviral.model.config import TrainCfg
    table = FeatureTable.load("data/work/model/table", "cpu")
    author, created = table.t["author_idx"][rows_t].numpy(), table.t["created_s"][rows_t].numpy()
    y = np.log10(table.t["counts"][rows_t][:, 0].clamp(min=0).float().numpy() + 1.0)
    tc = TrainCfg()
    I, J = evaluate.same_author_pairs(author, created, y, tc.pair_window_days * 86400, tc.pair_margin)
    pa = author[I]
    def ordinary(names):
        s = np.mean([np.load(f"data/sota/test/{n}.npy") for n in names], 0)
        return (((s[I] > s[J]) == (y[I] > y[J])) & (s[I] != s[J])).astype(float) + 0.5 * (s[I] == s[J])

def summarize(names):
    out = {}
    B, D, Ln = get(S.bench, names), get(S.dev, names), get(S.launch, names)
    if rows_t is not None and all(os.path.exists(f"data/sota/test/{n}.npy") for n in names):
        c = ordinary(names)
        out["ordinary"] = {"pairs": int(len(c)), "accuracy": float(c.mean()), "ci": L.boot(pa, [c, np.ones(len(c))], lambda p, q: p / q, n=500), "_c": c}
    for key, A in (("bench", B), ("dev", D), ("launch", Ln)):
        if A is None:
            continue
        conf = S.with_scale(A, B["scale"]) if key == "launch" and B is not None else A["conf"]
        A["conf_used"] = conf
        t = L.table(A, conf=conf)
        acc40, ci40, n40 = S.cov_at(A, 2.0, 0.4)
        out[key] = {"table": t, "scale": A["scale"], "surest40_2x": {"accuracy": acc40, "ci": ci40, "pairs": n40}, "_A": A}
    return out

def versus(x, y):
    out = {}
    if "ordinary" in x and "ordinary" in y:
        dlt = x["ordinary"]["_c"] - y["ordinary"]["_c"]
        out["ordinary"] = {"gap": float(dlt.mean()), "ci": L.boot(pa, [dlt, np.ones(len(dlt))], lambda p, q: p / q, n=500)}
    for key in ("bench", "dev", "launch"):
        if key in x and key in y:
            A, B = x[key]["_A"], y[key]["_A"]
            v = L.versus(A, B, conf_a=A["conf_used"], conf_b=B["conf_used"])
            big = np.abs(A["d"]) >= np.log(2)
            def m40(Z):
                m = np.zeros(int(big.sum())); m[np.argsort(-np.abs(Z["s"][big]), kind="stable")[:int(round(0.4 * big.sum()))]] = 1; return m
            ca, cb, ma, mb = A["correct"][big], B["correct"][big], m40(A), m40(B)
            v["surest40_2x_gap"] = {"gap": float((ca * ma).sum() / ma.sum() - (cb * mb).sum() / mb.sum()),
                                    "ci": L.boot(A["auth"][big], [ca * ma, ma, cb * mb, mb], lambda p, q, r, s: p / q - r / s)}
            out[key] = v
    return out

def strip(o):
    return {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v) for k, v in o.items()}

SUM = {}
for c in cands:
    for tag, names in (("s1", [f"{c}_s1"]), ("s2", [f"{c}_s2"]), ("both", [f"{c}_s1", f"{c}_s2"])):
        SUM[(c, tag)] = summarize(names)
        res["candidates"][c][f"scores_{tag}"] = strip(SUM[(c, tag)])

def cell(o, key, g="1.25x"):
    if key not in o: return "-"
    x = o[key]["table"][g]; return f"{pct(x['accuracy'])} ({pct(x['accuracy_ci'][0])} to {pct(x['accuracy_ci'][1])})"
def conf(o, key, g="2.0x"):
    if key not in o: return "-"
    x = o[key]["table"][g]; return f"{100 * x['confident_share']:.0f}% at {pct(x['confident_accuracy'])}"
def s40(o, key):
    return "-" if key not in o else f"{pct(o[key]['surest40_2x']['accuracy'])}"

md += ["", "## Results: the chosen checkpoint of every run", "",
       "Ordinary pairs: same account, within 7 days, likes (train: 6,000 fitted rows; validation: held-out accounts; test: all audience rows after the cutoff). "
       "Benchmark: 2,852 same-topic pairs 1.25x apart (2,030 at 2x). Launch: 850 pairs. `both` = the two seeds' scores averaged. "
       "Confident = stated 60% or more; the scale is cross-fitted over accounts on the benchmark and reused on launch pairs.", "",
       "| Run | Step | Pairs train | Pairs val | Pairs test | Dev pairs 1.25x / 2x | Benchmark, all | Benchmark 2x | Benchmark 2x confident | Surest 40% of 2x | Launch, all | Launch 2x confident |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c in cands:
    for i, tag in enumerate(("s1", "s2", "both")):
        o = SUM[(c, tag)]
        if tag != "both":
            f = res["candidates"][c]["final"][i]
            pre = f"| {c} {tag} | {res['candidates'][c]['best_step'][i]} | {pct(f['train_sample']['pairwise_likes'])} | {pct(f['val']['pairwise_likes'])} | {pct(f['test']['pairwise_likes'])} | {pct(f['dev']['ctl_acc_1.25x'])} / {pct(f['dev']['ctl_acc_2.0x'])} |"
        else:
            od = o.get("ordinary"); dv = o.get("dev")
            pre = f"| **{c} both** | | | | {pct(od['accuracy']) if od else '-'} | {(pct(dv['table']['1.25x']['accuracy']) + ' / ' + pct(dv['table']['2.0x']['accuracy'])) if dv else '-'} |"
        md.append(f"{pre} {cell(o, 'bench')} | {cell(o, 'bench', '2.0x')} | {conf(o, 'bench')} | {s40(o, 'bench')} | {cell(o, 'launch')} | {conf(o, 'launch')} |")

md += ["", "## Candidate minus control, paired, 95% intervals from resampling accounts", "",
       "`mean of seeds`: the per-pair difference averaged over the two seed-matched pairs of runs. `ensembles`: two-seed candidate ensemble against two-seed control ensemble. Points.", "",
       "| Candidate | Ordinary test pairs | Dev pairs, all | Dev 2x confident calls per 100 | Benchmark, all (mean of seeds) | Benchmark, all (ensembles) | Benchmark 2x (ensembles) | Benchmark 2x confident share | Benchmark 2x confident calls per 100 | Surest 40% of 2x | Launch, all |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
def vrow(label, X, Y, members=None):
    v = versus(X, Y)
    res["versus"][label] = v
    g = lambda key, g_, f: L.pm(v[key][g_][f], v[key][g_][f + "_ci"]) if key in v else "-"  # noqa: E731
    mm = "-"
    if members:
        As, Bs = members
        if all("bench" in x for x in As + Bs):
            r = L.mean_member_gap([x["bench"]["_A"] for x in As], [x["bench"]["_A"] for x in Bs])
            res["versus"][label]["bench_mean_member_gap"] = r
            mm = L.pm(r["1.25x"]["gap"], r["1.25x"]["ci"])
    per100 = lambda key: (f"{v[key]['2.0x']['confident_correct_per100_gap']:+.1f} ({v[key]['2.0x']['confident_correct_per100_gap_ci'][0]:+.1f} to {v[key]['2.0x']['confident_correct_per100_gap_ci'][1]:+.1f})" if key in v else "-")  # noqa: E731
    md.append(f"| {label} | {L.pm(v['ordinary']['gap'], v['ordinary']['ci']) if 'ordinary' in v else '-'} | {g('dev', '1.25x', 'accuracy_gap')} | {per100('dev')} | {mm} | {g('bench', '1.25x', 'accuracy_gap')} | {g('bench', '2.0x', 'accuracy_gap')} | "
              f"{g('bench', '2.0x', 'confident_share_gap')} | {per100('bench')} | {L.pm(v['bench']['surest40_2x_gap']['gap'], v['bench']['surest40_2x_gap']['ci']) if 'bench' in v else '-'} | {g('launch', '1.25x', 'accuracy_gap')} |")
for c in cands[1:]:
    vrow(c, SUM[(c, "both")], SUM[("ctl", "both")], ([SUM[(c, "s1")], SUM[(c, "s2")]], [SUM[("ctl", "s1")], SUM[("ctl", "s2")]]))
for e in a.extra:
    label, rest = e.split("=", 1); x, y_ = rest.split("|")
    X, Y = summarize(x.split("+")), summarize(y_.split("+"))
    res["candidates"][label] = {"scores_both": strip(X), "members": x.split("+")}
    o = X; od = o.get("ordinary"); dv = o.get("dev")
    md.insert(md.index("## Candidate minus control, paired, 95% intervals from resampling accounts") - 1,
              f"| **{label}** | | | | {pct(od['accuracy']) if od else '-'} | {(pct(dv['table']['1.25x']['accuracy']) + ' / ' + pct(dv['table']['2.0x']['accuracy'])) if dv else '-'} | {cell(o, 'bench')} | {cell(o, 'bench', '2.0x')} | {conf(o, 'bench')} | {s40(o, 'bench')} | {cell(o, 'launch')} | {conf(o, 'launch')} |")
    vrow(f"{label} vs {y_}", X, Y)
json.dump(res, open(f"{a.out}/results.json", "w"), indent=1, default=float)
open(f"{a.out}/tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
