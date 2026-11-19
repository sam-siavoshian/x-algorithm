#!/usr/bin/env python3
"""Per-kind confidence scales for a model set, fitted on pre-cutoff pairs only (the selective track's shipped rule:
P(the pick is right) = sigmoid(scale * |gap in log10 predicted likes|), one scale per kind of comparison).

    python scripts/combine/fit_selective.py --name te3l_broad2 --members a,b --describe "..." --out data/combine/selective_rule.json [--add-to xviral/data/selective_rule.json]

Fitted on all pairs 1.25x apart of the development pairs (same_topic) and of the launch pairs from before the cutoff
(launch). Also written: a time-and-account split check (scale fitted on the judge track's fit cell: both posts before T,
account half A; read on its validation cell: later post on or after T, account half B), and the same check with accounts
held out only. The test sets are not read.
"""
import argparse, json, sys
import numpy as np, polars as pl
sys.path.insert(0, "scripts/selective"); sys.path.insert(0, "scripts/combine")
from sel_models import fit_scale  # noqa: E402
from pair_eval import boot  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--members", required=True); ap.add_argument("--describe", default="")
ap.add_argument("--launch-members", default="", help="other members for launch comparisons (default: the same members)"); ap.add_argument("--describe-launch", default="")
ap.add_argument("--pairs", default="data/combine/pairs"); ap.add_argument("--out", required=True); ap.add_argument("--add-to", default=None); a = ap.parse_args()
L = np.log
kinds = {}
for kind, s in (("same_topic", "dev"), ("launch", "launchpre")):
    T = pl.read_parquet(f"{a.pairs}/{s}.parquet")
    mem = (a.launch_members or a.members) if kind == "launch" else a.members
    g = np.mean([T["s3"].to_numpy() if m == "s3" else T[f"g_{m}"].to_numpy() for m in mem.split(",")], axis=0)
    d, auth, cell = T["d"].to_numpy(), T["author_id"].to_numpy(), T["cell"].to_numpy()
    clear = np.abs(d) >= L(1.25); c = ((g * d) > 0).astype(float); two = np.abs(d) >= L(2)
    k = fit_scale(np.abs(g[clear]), c[clear])["scale"]

    def read(k_, m):
        conf = (1 / (1 + np.exp(-k_ * np.abs(g))) >= 0.6).astype(float)
        mm = m & two
        idx = np.flatnonzero(mm); top = np.zeros(len(g)); top[idx[np.argsort(-np.abs(g[idx]), kind="stable")[: int(round(.4 * len(idx)))]]] = 1
        one = np.ones(int(mm.sum()))
        return {"pairs_2x": int(mm.sum()), "all_calls": float(c[mm].mean()), "stated_60_share": float(conf[mm].mean()), "stated_60_correct": float((c * conf)[mm].sum() / max(conf[mm].sum(), 1)),
                "stated_60_correct_ci": boot(auth[mm], [(c * conf)[mm], conf[mm]], lambda x, n: x / n), "surest_40pct_correct": float((c * top)[mm].sum() / max(top[mm].sum(), 1)),
                "surest_40pct_ci": boot(auth[mm], [(c * top)[mm], top[mm]], lambda x, n: x / n), "_": len(one)}
    F, V = cell == "fit", cell == "val"
    kf = fit_scale(np.abs(g[F & clear]), c[F & clear])["scale"]
    half = np.array([hash(int(x)) % 2 == 0 for x in auth])
    ka = fit_scale(np.abs(g[half & clear]), c[half & clear])["scale"]
    kinds[kind] = {"model": {"type": "logistic_gap", "scale": round(float(k), 4), "bias": 0.0}, "fallback": {"type": "logistic_gap", "scale": round(float(k), 4), "bias": 0.0},
                   "members": mem.split(","),
                   "fitted_on": {"set": s, "pairs": int(clear.sum()), "pairs_2x": int(two.sum()), "accounts": int(len(set(auth[clear]))), "period": "both posts before 2026-04-23",
                                 "all_calls_correct": float(c[clear].mean()), "mean_abs_gap_log10": float(np.abs(g[clear]).mean())},
                   "in_sample": read(k, clear),
                   "time_and_account_split": {"scale_fitted_on_fit_cell": round(float(kf), 4), "fit_pairs": int((F & clear).sum()), "train": read(kf, F & clear), "validation": read(kf, V & clear)},
                   "account_split": {"scale_fitted_on_half_of_accounts": round(float(ka), 4), "train": read(ka, half & clear), "validation": read(ka, ~half & clear)}}
entry = {"members": (a.describe or a.members) + (("; launch comparisons: " + (a.describe_launch or a.launch_members)) if a.launch_members else ""),
         "member_names": a.members.split(","), "launch_member_names": (a.launch_members or a.members).split(","), "kinds": kinds}
rule = {"version": 1, "fitted": "2026-11-19", "source": "research/tracks/combine/REPORT.md, scripts/combine/fit_selective.py", "speak": 0.6,
        "score": "mean over members of log10(predicted median likes + 1), version a minus version b; P(pick is right) = sigmoid(scale * |gap|)",
        "note": "One scale per kind of comparison, fitted on pairs from before 2026-04-23. A scale belongs to the model set it was fitted with.",
        "default_ensemble": a.name, "ensembles": {a.name: entry}}
json.dump(rule, open(a.out, "w"), indent=1)
if a.add_to:
    base = json.load(open(a.add_to)); base["ensembles"][a.name] = entry
    json.dump(base, open(a.add_to, "w"), indent=1)
for kind, v in kinds.items():
    t, s_ = v["time_and_account_split"], v["account_split"]
    print(f"{kind}: scale {v['model']['scale']:.2f} on {v['fitted_on']['pairs']} pairs ({v['fitted_on']['accounts']} accounts); time-and-account split scale {t['scale_fitted_on_fit_cell']:.2f}: "
          f"train surest 40% {100 * t['train']['surest_40pct_correct']:.1f}%, stated 60%+ {100 * t['train']['stated_60_share']:.0f}% at {100 * t['train']['stated_60_correct']:.1f}% | "
          f"validation surest 40% {100 * t['validation']['surest_40pct_correct']:.1f}%, stated 60%+ {100 * t['validation']['stated_60_share']:.0f}% at {100 * t['validation']['stated_60_correct']:.1f}% ({t['validation']['pairs_2x']} pairs 2x apart)")
