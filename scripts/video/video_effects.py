#!/usr/bin/env python3
"""What kind of launch video does better, inside accounts, from the poster frame (first frame the feed shows).

    python scripts/video/video_effects.py     # data/launch/table.parquet + data/video/labels.parquet -> data/video/effects.json

Method of scripts/launch/effects.py, unchanged (pairs_est / ratio_est / adjusted are copied from it): outcome is
lift = log((likes+1)/(account's trailing median likes+1)); every comparison is between video posts of ONE account;
pairs are posting-order balanced, at most 365 days apart, lifts differing 1.25x or more, at most 50 pairs per account;
intervals resample accounts. Only Community Archive posts (the only source that records video).

Every attribute here is read from the poster frame or the post's media record. Duration, cuts, pace, captions over
time and audio are NOT measured: the video files were not acquired.

Rule fixed before any effect was computed: a trait is stated as a finding only if, for likes, the pair win rate
interval excludes 50%, the within-account ratio interval excludes 1 and the adjusted ratio interval excludes 1 in the
scope `launch_all` (launch-type video posts, every account) or in `announce_cf` (video posts that are own
announcements of company and founder accounts), AND the point estimates have the same direction in `allvideo_cf`
(every video post of company and founder accounts). Each finding is then re-measured before and after 2026-04-23.
"""
import json, sys, numpy as np, polars as pl
rng = np.random.default_rng(0)
CUT = 1776902400000; LAUNCH = ["launch", "open_source"]; ANN = ["launch", "open_source", "feature", "funding", "milestone", "hiring"]

def boot_ci(stat, n, B=2000):
    idx = rng.integers(0, n, (B, n))
    b = np.array([stat(i) for i in idx]); b = b[np.isfinite(b)]
    return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) > B / 2 else (float("nan"), float("nan"))

def pairs_est(q, y):
    q = q.filter(pl.col(y).is_not_null()).select("author_id", "created_at_ms", "post_id", "a", pl.col(y).alias("y"))
    w, n = q.filter(pl.col("a")), q.filter(~pl.col("a"))
    p = w.join(n, on="author_id", suffix="_n").filter(((pl.col("created_at_ms") - pl.col("created_at_ms_n")).abs() <= 365 * 86_400_000) & ((pl.col("y") - pl.col("y_n")).abs() >= np.log(1.25)))
    if p.height == 0: return None
    p = p.with_columns((pl.col("y") > pl.col("y_n")).alias("win"), (pl.col("created_at_ms") < pl.col("created_at_ms_n")).alias("wf"))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 50)
    n1, n2 = p.filter(pl.col("wf")).height, p.filter(~pl.col("wf")).height
    per = p.group_by("author_id").agg((pl.col("win") & pl.col("wf")).sum().alias("w1"), pl.col("wf").sum().alias("n1"), (pl.col("win") & ~pl.col("wf")).sum().alias("w2"), (~pl.col("wf")).sum().alias("n2")).sort("author_id")
    w1, m1, w2, m2 = (per[c].to_numpy().astype(float) for c in ("w1", "n1", "w2", "n2"))
    def bal(i):
        a, b = m1[i].sum(), m2[i].sum()
        if a < 1 or b < 1: return np.nan
        return 0.5 * (w1[i].sum() / a + w2[i].sum() / b)
    rate = bal(np.arange(len(w1))); lo, hi = boot_ci(bal, len(w1))
    return {"win_rate": None if not np.isfinite(rate) else round(float(rate), 4), "lo": round(lo, 4), "hi": round(hi, 4), "pairs": int(p.height), "accounts": int(per.height), "pairs_with_first": int(n1), "pairs_with_second": int(n2),
            "raw_win_rate": round(float((w1.sum() + w2.sum()) / (m1.sum() + m2.sum())), 4)}

def ratio_est(q, y):
    q = q.filter(pl.col(y).is_not_null())
    per = q.group_by("author_id").agg(pl.col(y).filter(pl.col("a")).mean().alias("m1"), pl.col("a").sum().alias("n1"), pl.col(y).filter(~pl.col("a")).mean().alias("m0"), (~pl.col("a")).sum().alias("n0")).filter((pl.col("n1") > 0) & (pl.col("n0") > 0)).sort("author_id")
    if per.height < 5: return None
    dlt = (per["m1"] - per["m0"]).to_numpy(); w = (per["n1"] * per["n0"] / (per["n1"] + per["n0"])).to_numpy().astype(float)
    st = lambda i: float(np.sum(w[i] * dlt[i]) / np.sum(w[i]))
    lo, hi = boot_ci(st, len(w))
    return {"ratio": round(float(np.exp(st(np.arange(len(w))))), 3), "ratio_lo": round(float(np.exp(lo)), 3), "ratio_hi": round(float(np.exp(hi)), 3), "accounts_with_both": int(per.height),
            "posts_with": int(per["n1"].sum()), "posts_without": int(per["n0"].sum())}

def adjusted(q, y, names):
    q = q.filter(pl.col(y).is_not_null())
    extra = [pl.col("ptype") == t for t in sorted(set(q["ptype"].unique()) - {"launch"})]
    q = q.with_columns([pl.col(c).cast(pl.Float64) for c in names] + [e.cast(pl.Float64).alias(f"_t{j}") for j, e in enumerate(extra)])
    names = names + [f"_t{j}" for j in range(len(extra))]
    q = q.filter(pl.len().over("author_id") >= 2)
    X = q.select(names).to_numpy(); Y = q[y].to_numpy(); g = q["author_id"].to_numpy()
    u, inv = np.unique(g, return_inverse=True)
    def dm(M):
        s = np.zeros((len(u),) + M.shape[1:]); np.add.at(s, inv, M); c = np.bincount(inv).astype(float)
        return M - (s / c.reshape(-1, *([1] * (M.ndim - 1))))[inv]
    Xd, Yd = dm(X), dm(Y)
    beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
    rows = [np.where(inv == k)[0] for k in range(len(u))]
    B = []
    for _ in range(500):
        pick = rng.integers(0, len(u), len(u)); ii = np.concatenate([rows[k] for k in pick])
        B.append(np.linalg.lstsq(Xd[ii], Yd[ii], rcond=None)[0])
    B = np.array(B)
    return {n: {"ratio": round(float(np.exp(beta[j])), 3), "ratio_lo": round(float(np.exp(np.quantile(B[:, j], 0.025))), 3), "ratio_hi": round(float(np.exp(np.quantile(B[:, j], 0.975))), 3)} for j, n in enumerate(names) if not n.startswith("_t")} | {
        "_n": {"posts": int(len(Y)), "accounts": int(len(u))}}

T = pl.read_parquet("data/launch/table.parquet").filter((pl.col("source") == "community_archive") & pl.col("has_video") & pl.col("lift_likes").is_not_null())
V = pl.read_parquet("data/video/labels.parquet")          # post_id, cls, attributes (scripts/video/label_thumbs.py)
T = T.join(V, on="post_id", how="inner")
T = T.with_columns((pl.col("chars") >= 280).alias("long280"), (pl.col("lines") >= 3).alias("lines3"), (pl.col("ext_link") & ~pl.col("points_to_reply")).alias("link_in_post"))
c = pl.col
CLS = ["ui", "face", "title", "graphic", "real", "chart"]
# name: (description, with, restrict-to)
A = {f"first_{k}": (d, c("cls") == k, None) for k, d in {
        "ui": "the first frame shows product UI (app, website, editor, terminal, chat), against every other first frame",
        "face": "the first frame shows a person (talking head, interview, selfie), against every other first frame",
        "title": "the first frame is a title card or a logo (text or mark on a plain background), against every other first frame",
        "graphic": "the first frame is motion graphics, a render, an illustration or generated imagery, against every other first frame",
        "real": "the first frame is real-world footage without a person in close-up (hardware, robot, stage, room, outdoors), against every other first frame",
        "chart": "the first frame is a chart, diagram or slide, against every other first frame"}.items()}
A |= {
    "first_ui_vs_face": ("first frame product UI, against a person", c("cls") == "ui", c("cls").is_in(["ui", "face"])),
    "first_ui_vs_title": ("first frame product UI, against a title card or logo", c("cls") == "ui", c("cls").is_in(["ui", "title"])),
    "first_ui_vs_graphic": ("first frame product UI, against motion graphics or a render", c("cls") == "ui", c("cls").is_in(["ui", "graphic"])),
    "first_face_vs_title": ("first frame a person, against a title card or logo", c("cls") == "face", c("cls").is_in(["face", "title"])),
    "first_real_vs_ui": ("first frame real-world footage or a person, against product UI", c("cls").is_in(["real", "face"]), c("cls").is_in(["real", "face", "ui"])),
    "has_face": ("a face is detected in the first frame", c("has_face"), None),
    "big_face": ("a face at least a quarter of the frame height, against no face", c("big_face"), c("big_face") | ~c("has_face")),
    "empty_first": ("the first frame is empty: near black or one flat colour, no text (a fade-in)", c("empty_first"), None),
    "dark_first": ("the first frame is dark overall (mean brightness under 40 of 255), against brighter", c("dark_first"), None),
    "text_on_frame": ("readable text in the first frame (2 or more text boxes)", c("text_on_frame"), None),
    "text_heavy": ("text covers 8% or more of the first frame, against under 2%", c("text_heavy"), c("text_heavy") | c("text_light")),
    "big_headline": ("a text line at least 8% of the frame height tall (headline-size) in the first frame", c("big_headline"), None),
    "lower_text_nonui": ("text in the lower third of a first frame that is not product UI (a caption-like line), against no text there, non-UI frames only", c("lower_text"), c("cls") != "ui"),
    "portrait": ("portrait video (taller than wide), against landscape", c("portrait"), c("portrait") | c("landscape")),
    "square": ("square-ish video (0.8 to 1.2), against landscape", c("square"), c("square") | c("landscape")),
    "wide_16_9": ("16:9 (1.7 to 1.85 wide), against every other shape", c("wide_16_9"), None),
    "hd1080": ("poster frame 1080 px or more on the short side, against under 720", c("hd1080"), c("hd1080") | c("sub720")),
    "hd720": ("poster frame 720 px or more on the short side, against under 720", ~c("sub720"), None),
    "gif": ("a GIF (silent loop), against a video", c("mtype") == "gif", None),
    "studio_upload": ("uploaded through X Media Studio / the ads tools (amplify_video), against an ordinary upload; videos only", c("route") == "amplify", c("route").is_in(["amplify", "ext"])),
}
# Adjusted: one regression per trait, on the trait's own comparison set: the trait plus these covariates, account fixed
# effects. (A first version put all class dummies in one regression, which measured each class against the small
# residual class instead of against "every other first frame"; it was replaced before any finding was stated.)
COV = ["names_product_first_line", "lines3", "long280", "link_in_post", "is_quote", "open_problem", "portrait", "sub720", "is_gif", "studio"]
OWN = {"portrait": ["portrait"], "square": ["portrait"], "wide_16_9": ["portrait"], "hd1080": ["sub720"], "hd720": ["sub720"], "gif": ["is_gif", "studio"], "studio_upload": ["studio", "is_gif"]}
T = T.with_columns([(c("cls") == k).alias("c_" + k) for k in CLS] + [(c("mtype") == "gif").alias("is_gif"), (c("route") == "amplify").alias("studio")])
SCOPES = {
    "launch_all": ("launch-type video posts (launch, open-source release) of every account", c("ptype").is_in(LAUNCH)),
    "launch_cf": ("launch-type video posts of company and founder accounts", c("ptype").is_in(LAUNCH) & (c("akind") != "other")),
    "announce_cf": ("video posts that are own announcements (launch, open source, feature, funding, milestone, hiring) of company and founder accounts", c("ptype").is_in(ANN) & (c("akind") != "other")),
    "announce_all": ("video posts that are own announcements, every account", c("ptype").is_in(ANN)),
    "allvideo_cf": ("every video post of company and founder accounts", c("akind") != "other"),
    "allvideo_all": ("every video post with a fetched poster frame (company and founder accounts, plus up to 40 per other account that has an announcement video)", pl.lit(True)),
    "launch_all_under_100k": ("launch-type video posts, accounts with a known follower count under 100,000", c("ptype").is_in(LAUNCH) & c("tier").is_in(["<1k", "1k-10k", "10k-100k"])),
    "launch_all_100k_plus": ("launch-type video posts, accounts with 100,000 followers or more", c("ptype").is_in(LAUNCH) & (c("tier") == "100k+")),
    "allvideo_cf_under_100k": ("every video post of company and founder accounts under 100,000 followers", (c("akind") != "other") & c("tier").is_in(["<1k", "1k-10k", "10k-100k"])),
    "allvideo_cf_100k_plus": ("every video post of company and founder accounts with 100,000 followers or more", (c("akind") != "other") & (c("tier") == "100k+")),
    "launch_all_before_cutoff": ("launch-type video posts, every account, before 2026-04-23", c("ptype").is_in(LAUNCH) & (c("created_at_ms") < CUT)),
    "launch_all_after_cutoff": ("launch-type video posts, every account, on or after 2026-04-23", c("ptype").is_in(LAUNCH) & (c("created_at_ms") >= CUT)),
    "announce_cf_before_cutoff": ("own-announcement video posts of company and founder accounts, before 2026-04-23", c("ptype").is_in(ANN) & (c("akind") != "other") & (c("created_at_ms") < CUT)),
    "announce_cf_after_cutoff": ("own-announcement video posts of company and founder accounts, on or after 2026-04-23", c("ptype").is_in(ANN) & (c("akind") != "other") & (c("created_at_ms") >= CUT)),
    "allvideo_cf_before_cutoff": ("every video post of company and founder accounts, before 2026-04-23", (c("akind") != "other") & (c("created_at_ms") < CUT)),
    "allvideo_cf_after_cutoff": ("every video post of company and founder accounts, on or after 2026-04-23", (c("akind") != "other") & (c("created_at_ms") >= CUT)),
}
out = {"scopes": {}, "class_lift": {}}
for sk, (sdesc, sf) in SCOPES.items():
    S = T.filter(sf)
    res = {"description": sdesc, "posts": S.height, "accounts": S["author_id"].n_unique(), "effects": {}, "adjusted_likes": {}}
    for k, (desc, fa, restrict) in A.items():
        q = (S.filter(restrict) if restrict is not None else S).with_columns(fa.fill_null(False).alias("a"))
        res["effects"][k] = {"description": desc, "likes": {"pairs": pairs_est(q, "lift_likes"), "within_account": ratio_est(q, "lift_likes"),
                             "posts_with": q.filter(c("a")).height, "posts_without": q.filter(~c("a")).height}}
        try:
            cov = [x for x in COV if x not in OWN.get(k, [])]
            if q.filter(c("a")).height >= 15 and q.filter(~c("a")).height >= 15:
                r = adjusted(q, "lift_likes", ["a"] + cov); res["adjusted_likes"][k] = r["a"] | {"posts": r["_n"]["posts"], "accounts": r["_n"]["accounts"]}
        except Exception:  # noqa: BLE001
            res["adjusted_likes"][k] = None
    out["scopes"][sk] = res
    print(f"\n== {sk}: {S.height} posts, {res['accounts']} accounts")
    print(f"{'attribute':<22}{'win':>6} {'interval':>13} {'pairs':>6} {'acc':>4} | {'ratio':>5} {'interval':>12} {'acc':>4} {'n+':>5} {'n-':>5} | {'adj':>5} {'interval':>12}")
    for k, e in res["effects"].items():
        p, r = e["likes"]["pairs"], e["likes"]["within_account"]; a = res["adjusted_likes"].get(k)
        ps = f"{100*p['win_rate']:5.1f}% {100*p['lo']:5.1f}-{100*p['hi']:<5.1f} {p['pairs']:>6} {p['accounts']:>4}" if p and p["win_rate"] is not None else " " * 32
        rs = f"{r['ratio']:5.2f} {r['ratio_lo']:5.2f}-{r['ratio_hi']:<5.2f} {r['accounts_with_both']:>4} {r['posts_with']:>5} {r['posts_without']:>5}" if r else " " * 35
        as_ = f"{a['ratio']:5.2f} {a['ratio_lo']:5.2f}-{a['ratio_hi']:<5.2f}" if a else ""
        print(f"{k:<22}{ps} | {rs} | {as_}")
# how each first-frame class does against the account's own usual (account-weighted mean of lift, bootstrap over accounts)
for sk in ("launch_all", "launch_cf", "announce_cf", "allvideo_cf"):
    S = T.filter(SCOPES[sk][1]); out["class_lift"][sk] = {}
    for k in CLS + ["other"]:
        z = S.filter(c("cls") == k)
        if z.height < 15: continue
        per = z.group_by("author_id").agg(c("lift_likes").mean().alias("m"), (c("lift_likes") > 0).mean().alias("b"), pl.len().alias("n")).sort("author_id")
        m, b, n = (per[x].to_numpy().astype(float) for x in ("m", "b", "n"))
        f1 = lambda i: np.sum(m[i] * n[i]) / np.sum(n[i]); f2 = lambda i: np.sum(b[i] * n[i]) / np.sum(n[i]); l1, h1 = boot_ci(f1, len(m)); l2, h2 = boot_ci(f2, len(m))
        out["class_lift"][sk][k] = {"posts": z.height, "accounts": per.height, "share_of_scope": round(z.height / S.height, 4), "geo_mean_ratio": round(float(np.exp(f1(np.arange(len(m))))), 3), "lo": round(float(np.exp(l1)), 3), "hi": round(float(np.exp(h1)), 3),
                                    "beat_own_median": round(float(f2(np.arange(len(m)))), 4), "beat_lo": round(l2, 4), "beat_hi": round(h2, 4)}
        x = out["class_lift"][sk][k]; print(f"{sk:<12}{k:<8} posts {x['posts']:>5} ({100*x['share_of_scope']:.0f}%) acc {x['accounts']:>4} mean {x['geo_mean_ratio']:.2f}x ({x['lo']:.2f}-{x['hi']:.2f}) beat {100*x['beat_own_median']:.0f}% ({100*x['beat_lo']:.0f}-{100*x['beat_hi']:.0f})")
json.dump(out, open("data/video/effects.json", "w"), indent=1)
