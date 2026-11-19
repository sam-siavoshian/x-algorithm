#!/usr/bin/env python3
"""Final scoring of the llm_judge track. Blends are fitted on development pairs only and applied once to the 3,530 benchmark
pairs (2,852 with likes 1.25x apart) and the 850 launch pairs. Writes data/llm/final.json, final.md and per-pair score tables."""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/llm_judge")
from lib import *
from sklearn.linear_model import LogisticRegression

LUNA_CUT, SOL_CUT, TEST_CUT = 1779062400000, 1777507200000, 1776902400000   # 2026-05-18, 2026-04-30, 2026-04-23 UTC
O = "data/llm/out/"
md = []
def say(*x):
    s = " ".join(str(v) for v in x); print(s); md.append(s)
pc = lambda x: f"{100 * x:.1f}%"
def rng_(c):
    return f"({100 * c[0]:.1f} to {100 * c[1]:.1f})"

def load(name):
    P = pairs(name).filter(pl.col("d").abs() >= L125)
    P = P.join(load_judge(f"{O}{name}__n30__gpt-6-luna__logprob.jsonl").select("post_id", "post_id_b", pl.col("j").alias("luna"), pl.col("pos").alias("luna_pos")), on=["post_id", "post_id_b"], how="left")
    try:
        P = P.join(load_judge(f"{O}{name}__n30_m__gpt-6-luna__logprob.jsonl").select("post_id", "post_id_b", pl.col("j").alias("luna_m")), on=["post_id", "post_id_b"], how="left")
    except FileNotFoundError:
        pass
    try:
        P = P.join(load_judge(f"{O}{name}__n0_m__gpt-6-luna__logprob.jsonl").select("post_id", "post_id_b", pl.col("j").alias("luna_n0m")), on=["post_id", "post_id_b"], how="left")
    except FileNotFoundError:
        pass
    try:
        P = P.join(load_judge(f"{O}{name}__n30__gpt-6.1-sol__json-low.jsonl").select("post_id", "post_id_b", pl.col("j").alias("sol")), on=["post_id", "post_id_b"], how="left")
    except FileNotFoundError:
        P = P.with_columns(pl.lit(None, dtype=pl.Float64).alias("sol"))
    P = P.join(pl.read_parquet(f"data/llm/emb_head_{name}.parquet"), on=["post_id", "post_id_b"], how="left")
    return P.with_columns(pl.col("luna").fill_null(0.0)).sort("post_id", "post_id_b")

def z(P, cols):
    """Inputs divided by their spread within the pair set (no labels used), so one blend carries across sets whose score scales differ."""
    return np.stack([P[c].to_numpy() / (np.sqrt(np.mean(P[c].to_numpy() ** 2)) + 1e-12) for c in cols], 1)

def fit_blend(P, cols):
    Z, y = z(P, cols), P["d"].to_numpy() > 0
    m = LogisticRegression(C=100.0, fit_intercept=False).fit(np.vstack([Z, -Z]), np.concatenate([y, ~y]).astype(int))
    return m.coef_[0]

def table(P, scores: dict, title, conf_dev: dict | None = None):
    """Accuracy by gap, confident coverage (scale cross-fitted over accounts inside this set, as compare_runs.py), surest 40% at 2x."""
    d, a = P["d"].to_numpy(), P["author_id"].to_numpy()
    out = {}
    say(f"\n**{title}** ({len(d)} pairs, {len(set(a))} accounts)\n")
    say("| Model | Real gap at least | Pairs | Correct | Confident (60%+) share (pairs) | Correct there | Surest 40% correct |")
    say("|---|---|---|---|---|---|---|")
    for n, s in scores.items():
        conf, k = crossfit_conf(s, d, a)
        c = np.where(s == 0, 0.5, (s * d) > 0).astype(float)
        out[n] = {"scale": k}
        for g in (1.25, 2, 3, 5):
            m = np.abs(d) >= np.log(g)
            cm = m & (conf >= 0.6)
            idx = np.flatnonzero(m); top = idx[np.argsort(-np.abs(s[idx]), kind="stable")[: int(round(0.4 * len(idx)))]]; t40 = np.zeros(len(d), bool); t40[top] = True
            r = {"pairs": int(m.sum()), "acc": float(c[m].mean()), "ci": ci(a, lambda i: c[i][m[i]].mean()), "conf_share": float(cm.sum() / m.sum()), "conf_pairs": int(cm.sum()),
                 "conf_acc": float(c[cm].mean()) if cm.sum() else None, "conf_ci": ci(a, lambda i: c[i][cm[i]].mean() if cm[i].sum() else np.nan) if cm.sum() else None,
                 "top40_acc": float(c[t40].mean()), "top40_ci": ci(a, lambda i: c[i][t40[i]].mean() if t40[i].sum() else np.nan)}
            out[n][f"{g}x"] = r
            say(f"| {n} | {g}x | {r['pairs']:,} | {pc(r['acc'])} {rng_(r['ci'])} | {100 * r['conf_share']:.0f}% ({r['conf_pairs']}) | " + (f"{pc(r['conf_acc'])} {rng_(r['conf_ci'])}" if cm.sum() else "-") + f" | {pc(r['top40_acc'])} {rng_(r['top40_ci'])} |")
        out[n]["_c"], out[n]["_conf"], out[n]["_s"] = c, conf, s
    return out

def paired(P, T, x, y):
    d, a = P["d"].to_numpy(), P["author_id"].to_numpy()
    r = {}
    for g in (1.25, 2):
        m = np.abs(d) >= np.log(g)
        dc = T[x]["_c"] - T[y]["_c"]; ma, mb = m & (T[x]["_conf"] >= .6), m & (T[y]["_conf"] >= .6)
        cc = lambda i: (T[x]["_c"][i] * ma[i]).sum() / m[i].sum() - (T[y]["_c"][i] * mb[i]).sum() / m[i].sum()
        def t40(i, k):
            ii = i[m[i]]; s = np.abs(T[k]["_s"][ii]); top = ii[np.argsort(-s, kind="stable")[: int(round(.4 * len(ii)))]]; return T[k]["_c"][top].mean()
        e = np.arange(len(d))
        r[f"{g}x"] = {"acc": [float(dc[m].mean())] + ci(a, lambda i: dc[i][m[i]].mean()), "conf_share": [float(ma.sum() / m.sum() - mb.sum() / m.sum())] + ci(a, lambda i: ma[i].sum() / m[i].sum() - mb[i].sum() / m[i].sum()),
                      "correct_conf_per_100": [float(100 * cc(e))] + [100 * v for v in ci(a, cc)], "top40": [float(t40(e, x) - t40(e, y))] + ci(a, lambda i: t40(i, x) - t40(i, y))}
        q = r[f"{g}x"]
        say(f"- {x} minus {y}, pairs {g}x apart ({int(m.sum())}): correct {100 * q['acc'][0]:+.1f} ({100 * q['acc'][1]:+.1f} to {100 * q['acc'][2]:+.1f}); confident share {100 * q['conf_share'][0]:+.1f} ({100 * q['conf_share'][1]:+.1f} to {100 * q['conf_share'][2]:+.1f}); "
            f"correct confident calls per 100 pairs {q['correct_conf_per_100'][0]:+.1f} ({q['correct_conf_per_100'][1]:+.1f} to {q['correct_conf_per_100'][2]:+.1f}); surest 40% correct {100 * q['top40'][0]:+.1f} ({100 * q['top40'][1]:+.1f} to {100 * q['top40'][2]:+.1f})")
    return r

def strip(T):
    return {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in T.items()}

def slices(P, cols, title, sl):
    d, a = P["d"].to_numpy(), P["author_id"].to_numpy(); out = {}
    say(f"\n**{title}**\n"); say("| Slice | Pairs | Accounts | " + " | ".join(cols) + " |"); say("|---|---|---|" + "---|" * len(cols))
    for n, m in sl.items():
        if m.sum() < 30:
            say(f"| {n} | {int(m.sum())} | too few | " + " | ".join("-" for _ in cols) + " |"); continue
        row = {}
        for cname, s in cols.items():
            row[cname] = acc(np.where(s == 0, 1e-12 * np.sign(d) * 0, s), d, a, m) if False else None
            c = np.where(s == 0, .5, (s * d) > 0).astype(float); row[cname] = {"acc": float(c[m].mean()), "ci": ci(a, lambda i: c[i][m[i]].mean() if m[i].sum() else np.nan, n=500)}
        out[n] = {"pairs": int(m.sum()), **row}
        say(f"| {n} | {int(m.sum())} | {len(set(a[m]))} | " + " | ".join(f"{pc(row[c]['acc'])} {rng_(row[c]['ci'])}" for c in cols) + " |")
    return out

R = {}
dev, bench, launch = load("dev"), load("bench"), load("launch")
# ---------- development: train / validation of blends
say("## Development pairs: blend fitted on half A, scored on half B\n")
A, B = dev.filter(pl.col("half") == "A"), dev.filter(pl.col("half") == "B")
w = fit_blend(A, ["s3", "luna"]); say("weights (surrogate, luna judge), inputs in units of their own spread:", np.round(w, 3).tolist())
R["dev_split"] = {}
for nm, Q in (("half A (train)", A), ("half B (validation)", B)):
    Zq = z(Q, ["s3", "luna"])
    T = table(Q, {"surrogate (baseline three)": Q["s3"].to_numpy(), "luna judge": Q["luna"].to_numpy(), "blend surrogate + luna": Zq @ w, "embedding head (out-of-fold)": Q["e"].to_numpy()}, "Development " + nm)
    R["dev_split"][nm] = {"table": strip(T), "paired": paired(Q, T, "blend surrogate + luna", "surrogate (baseline three)")}
ds = dev.filter(pl.col("sol").is_not_null())
As, Bs = ds.filter(pl.col("half") == "A"), ds.filter(pl.col("half") == "B")
w3h = fit_blend(As, ["s3", "luna", "sol"]); w2h = fit_blend(As, ["s3", "sol"])
say("\nSample with the stronger model (every 4th development pair). Weights fitted on half A: surrogate+sol", np.round(w2h, 3).tolist(), "; surrogate+luna+sol", np.round(w3h, 3).tolist())
for nm, Q in (("half A sample (train)", As), ("half B sample (validation)", Bs)):
    T = table(Q, {"surrogate (baseline three)": Q["s3"].to_numpy(), "luna judge": Q["luna"].to_numpy(), "sol judge": Q["sol"].to_numpy(), "blend surrogate + sol": z(Q, ["s3", "sol"]) @ w2h, "blend surrogate + luna + sol": z(Q, ["s3", "luna", "sol"]) @ w3h}, "Development " + nm)
    R["dev_split"][nm] = {"table": strip(T), "paired": paired(Q, T, "blend surrogate + sol", "surrogate (baseline three)")}
# ---------- final blends on all development pairs
W = {"s3+luna": fit_blend(dev, ["s3", "luna"]), "s3+sol": fit_blend(ds, ["s3", "sol"]), "s3+luna+sol": fit_blend(ds, ["s3", "luna", "sol"]), "s3+luna+emb": fit_blend(dev, ["s3", "luna", "e"]), "s10+luna": fit_blend(dev, ["s10", "luna"])}
R["blend_weights"] = {k: v.tolist() for k, v in W.items()}
say("\n## Test sets, scored once\n\nBlend weights fitted on all development pairs (inputs in units of their own spread within each set):", json.dumps({k: np.round(v, 3).tolist() for k, v in W.items()}))
for name, P in (("bench", bench), ("launch", launch)):
    sc = {"surrogate (baseline three)": P["s3"].to_numpy(), "luna judge": P["luna"].to_numpy(), "blend surrogate + luna": z(P, ["s3", "luna"]) @ W["s3+luna"],
          "ten untouched models": P["s10"].to_numpy(), "blend ten models + luna": z(P, ["s10", "luna"]) @ W["s10+luna"], "embedding head": P["e"].to_numpy(), "blend surrogate + luna + embedding head": z(P, ["s3", "luna", "e"]) @ W["s3+luna+emb"]}
    if "luna_m" in P.columns:
        sc["luna judge, names masked"] = P["luna_m"].fill_null(0.0).to_numpy()
    T = table(P, sc, {"bench": "Benchmark, all pairs 1.25x apart", "launch": "Launch pairs"}[name]); say("")
    R[name] = {"table": strip(T), "paired": {f"{x}|{y}": paired(P, T, x, y) for x, y in (("blend surrogate + luna", "surrogate (baseline three)"), ("luna judge", "surrogate (baseline three)"), ("blend ten models + luna", "ten untouched models"))}}
    S = P.filter(pl.col("sol").is_not_null())
    if S.height:
        sc = {"surrogate (baseline three)": S["s3"].to_numpy(), "luna judge": S["luna"].to_numpy(), "sol judge": S["sol"].to_numpy(), "blend surrogate + luna": z(S, ["s3", "luna"]) @ W["s3+luna"],
              "blend surrogate + sol": z(S, ["s3", "sol"]) @ W["s3+sol"], "blend surrogate + luna + sol": z(S, ["s3", "luna", "sol"]) @ W["s3+luna+sol"]}
        T2 = table(S, sc, {"bench": "Benchmark, random sample scored by the stronger model", "launch": "Launch pairs, random sample scored by the stronger model"}[name]); say("")
        R[name + "_sol_sample"] = {"table": strip(T2), "paired": {f"{x}|{y}": paired(S, T2, x, y) for x, y in (("sol judge", "surrogate (baseline three)"), ("blend surrogate + sol", "surrogate (baseline three)"), ("blend surrogate + luna + sol", "surrogate (baseline three)"), ("sol judge", "luna judge"))}}
    # agreement and complementarity
    d = P["d"].to_numpy(); j, s = P["luna"].to_numpy(), P["s3"].to_numpy(); cj, cs = (j * d) > 0, (s * d) > 0; ag = (j * s) > 0
    R[name]["agreement"] = {"agree": float(ag.mean()), "corr": float(np.corrcoef(j, s)[0, 1]), "both_right": float((cj & cs).mean()), "judge_only": float((cj & ~cs).mean()), "surrogate_only": float((~cj & cs).mean()), "neither": float((~cj & ~cs).mean()),
                            "acc_when_agree": float(cs[ag].mean()), "pairs_agree": int(ag.sum()), "surrogate_acc_when_disagree": float(cs[~ag].mean()), "judge_acc_when_disagree": float(cj[~ag].mean())}
    q = R[name]["agreement"]
    say(f"Agreement, luna judge and surrogate ({name}): same side on {pc(q['agree'])} of pairs, score correlation {q['corr']:.2f}. Both right {pc(q['both_right'])}, judge only {pc(q['judge_only'])}, surrogate only {pc(q['surrogate_only'])}, neither {pc(q['neither'])}. "
        f"When they agree ({q['pairs_agree']} pairs) the call is right {pc(q['acc_when_agree'])}; when they disagree the surrogate is right {pc(q['surrogate_acc_when_disagree'])}, the judge {pc(q['judge_acc_when_disagree'])}.")
    if S.height:
        d2 = S["d"].to_numpy(); j2, s2 = S["sol"].to_numpy(), S["s3"].to_numpy(); cj2, cs2 = (j2 * d2) > 0, (s2 * d2) > 0; ag2 = (j2 * s2) > 0
        R[name + "_sol_sample"]["agreement"] = {"agree": float(ag2.mean()), "corr": float(np.corrcoef(j2, s2)[0, 1]), "corr_with_luna": float(np.corrcoef(j2, S["luna"].to_numpy())[0, 1]), "acc_when_agree": float(cs2[ag2].mean()), "pairs_agree": int(ag2.sum()),
                                                "surrogate_acc_when_disagree": float(cs2[~ag2 & (j2 != 0)].mean()), "judge_acc_when_disagree": float(cj2[~ag2 & (j2 != 0)].mean())}
        q = R[name + "_sol_sample"]["agreement"]
        say(f"Agreement, sol judge and surrogate ({name} sample): same side {pc(q['agree'])}, correlation {q['corr']:.2f} (with luna {q['corr_with_luna']:.2f}). Agree ({q['pairs_agree']} pairs): right {pc(q['acc_when_agree'])}; disagree: surrogate right {pc(q['surrogate_acc_when_disagree'])}, judge {pc(q['judge_acc_when_disagree'])}.")
    # calibration of the judge's own stated probability (order-averaged), no fitting
    for jn, col in (("luna", "luna"), ("sol", "sol")):
        Q = P.filter(pl.col(col).is_not_null() & (pl.col(col) != 0)); pj = 1 / (1 + np.exp(-np.abs(Q[col].to_numpy()))); cq = (Q[col].to_numpy() * Q["d"].to_numpy()) > 0
        cal = [{"stated": f"{lo:.2f}-{hi:.2f}", "pairs": int(((pj >= lo) & (pj < hi)).sum()), "mean_stated": float(pj[(pj >= lo) & (pj < hi)].mean()) if ((pj >= lo) & (pj < hi)).sum() else None,
                "correct": float(cq[(pj >= lo) & (pj < hi)].mean()) if ((pj >= lo) & (pj < hi)).sum() else None} for lo, hi in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .99), (.99, 1.01))]
        R[name].setdefault("calibration", {})[jn] = cal
        say(f"Calibration of the {jn} judge's own probability ({name}): " + "; ".join(f"stated {c['stated']}: {pc(c['correct'])} of {c['pairs']}" for c in cal if c["pairs"]))
    # leakage and other slices
    t0 = np.minimum(P["t_a"].to_numpy(), P["t_b"].to_numpy()); fol = P["followers"].to_numpy().astype(float); later = (P["d"].to_numpy() > 0) == (P["t_a"].to_numpy() > P["t_b"].to_numpy())
    sl = {"all": np.ones(P.height, bool), "both posts before 2026-04-23": np.maximum(P["t_a"].to_numpy(), P["t_b"].to_numpy()) < TEST_CUT, "2026-04-23 to 2026-05-17 (before luna's knowledge cutoff)": (t0 >= TEST_CUT) & (t0 < LUNA_CUT),
          "from 2026-05-18 (after luna's knowledge cutoff)": t0 >= LUNA_CUT, "from 2026-04-30 (after sol's knowledge cutoff)": t0 >= SOL_CUT,
          "under 50k followers": fol < 50_000, "50k followers and over": fol >= 50_000, "followers unknown": np.isnan(fol), "under 50k and after luna's cutoff": (fol < 50_000) & (t0 >= LUNA_CUT),
          "later post won": later, "earlier post won": ~later}
    cols = {"surrogate": P["s3"].to_numpy(), "luna judge": P["luna"].to_numpy(), "blend surrogate + luna": z(P, ["s3", "luna"]) @ W["s3+luna"]}
    if "luna_m" in P.columns:
        cols["luna judge, names masked"] = P["luna_m"].fill_null(0.0).to_numpy()
    if "luna_n0m" in P.columns:
        cols["luna judge, no history, names masked"] = P["luna_n0m"].fill_null(0.0).to_numpy()
    R[name]["slices"] = slices(P, cols, f"Slices, {name}, all pairs 1.25x apart", sl)
    if S.height:
        ok = P["sol"].is_not_null().to_numpy()
        R[name + "_sol_sample"]["slices"] = slices(S, {"surrogate": S["s3"].to_numpy(), "luna judge": S["luna"].to_numpy(), "sol judge": S["sol"].to_numpy(), "blend surrogate + sol": z(S, ["s3", "sol"]) @ W["s3+sol"]}, f"Slices, {name}, sample scored by the stronger model", {k: v[ok] for k, v in sl.items()})
    # the product-relevant slice: both posts written after the judge's knowledge cutoff
    for cn, cut, jcol, wkey in (("luna", LUNA_CUT, "luna", "s3+luna"), ("sol", SOL_CUT, "sol", "s3+sol")):
        for lab, msk in ((f"after {cn}'s knowledge cutoff", t0 >= cut), (f"before {cn}'s knowledge cutoff", t0 < cut)):
            Q = P.filter(pl.Series(msk) & pl.col(jcol).is_not_null())
            if Q.height < 100 or (cn == "sol" and not S.height):
                continue
            T3 = table(Q, {"surrogate (baseline three)": Q["s3"].to_numpy(), f"{cn} judge": Q[jcol].to_numpy(), f"blend surrogate + {cn}": z(Q, ["s3", jcol]) @ W[wkey]}, f"{name}: pairs {lab}")
            R[name].setdefault("by_cutoff", {})[f"{cn}, {lab}"] = {"table": strip(T3), "paired": {x: paired(Q, T3, x, "surrogate (baseline three)") for x in (f"{cn} judge", f"blend surrogate + {cn}")}}
    # clean slice: wording really differs (text similarity under 0.9; near-identical pairs are corrected reposts) and both posts are after the judge's cutoff
    try:
        sim = P.join(pl.read_parquet(f"data/llm/textsim_{name}.parquet"), on=["post_id", "post_id_b"], how="left")["text_similarity"].to_numpy()
        for cn, cut, jcol, wkey in (("luna", LUNA_CUT, "luna", "s3+luna"), ("sol", SOL_CUT, "sol", "s3+sol")):
            for lab, msk in ((f"different wording, after {cn}'s knowledge cutoff", (sim < .9) & (t0 >= cut)), (f"different wording, any date", sim < .9), (f"near-identical texts (corrected reposts)", sim >= .9)):
                Q = P.filter(pl.Series(msk) & pl.col(jcol).is_not_null())
                if Q.height < 100 or (cn == "sol" and not S.height):
                    continue
                sc3 = {"surrogate (baseline three)": Q["s3"].to_numpy(), f"{cn} judge": Q[jcol].to_numpy(), f"blend surrogate + {cn}": z(Q, ["s3", jcol]) @ W[wkey]}
                T4 = table(Q, sc3, f"{name}: {lab} ({cn})")
                R[name].setdefault("clean", {})[f"{cn}, {lab}"] = {"table": strip(T4), "paired": {x: paired(Q, T4, x, "surrogate (baseline three)") for x in (f"{cn} judge", f"blend surrogate + {cn}")}}
    except FileNotFoundError:
        pass
    keep = [c for c in ("part", "post_id", "post_id_b", "author_id", "likes", "likes_b", "d", "s3", "s10", "luna", "luna_m", "luna_n0m", "sol", "e", "luna_pos", "followers", "t_a", "t_b") if c in P.columns]
    P.select(keep).with_columns(pl.Series("blend_s3_luna", z(P, ["s3", "luna"]) @ W["s3+luna"])).write_parquet(f"data/llm/scores_{name}.parquet")
dev.select([c for c in ("post_id", "post_id_b", "author_id", "half", "likes", "likes_b", "d", "s3", "s10", "luna", "sol", "e", "followers", "t_a", "t_b") if c in dev.columns]).write_parquet("data/llm/scores_dev.parquet")
json.dump(R, open("data/llm/final.json", "w"), indent=1)
open("data/llm/final.md", "w").write("\n".join(md) + "\n")
