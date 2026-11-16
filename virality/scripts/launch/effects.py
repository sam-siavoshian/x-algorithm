#!/usr/bin/env python3
"""What separates launch posts that beat the account's own usual result from those that do not.

    python scripts/launch/effects.py      # reads data/launch/table.parquet, writes data/launch/effects.json

Every number is within account. Outcome: lift = log((likes+1)/(account's trailing median likes+1)); views likewise.
Three estimates per attribute, each with a 95% interval from resampling accounts (2,000 draws):
  pairs     all pairs of launch-type posts by one account, at most 365 days apart, where exactly one has the attribute
            and the two lifts differ by 1.25x or more: how often the one with it has the higher lift, computed
            separately for "with it posted first" and "with it posted second" and averaged (posting order balanced),
            at most 50 pairs per account.
  ratio     accounts that have launch posts both with and without it: mean lift with minus without per account,
            precision-weighted over accounts, as a ratio of likes.
  adjusted  the same within-account contrast from one regression with every attribute in it (account fixed effects),
            so that e.g. video is not credited with the effect of length.
"""
import json, sys, numpy as np, polars as pl
rng = np.random.default_rng(0)
T = pl.read_parquet("data/launch/table.parquet")
T = T.with_columns((pl.col("chars") >= 280).alias("long280"), (pl.col("chars") >= 600).alias("long600"), (pl.col("chars") < 140).alias("short140"), (pl.col("lines") >= 3).alias("lines3"),
                   (pl.col("dow") >= 6).alias("weekend"), pl.col("dow").is_in([2, 3, 4]).alias("tue_thu"), pl.col("hour_utc").is_between(13, 17).alias("h13_17"), pl.col("hour_utc").is_between(18, 23).alias("h18_23"),
                   pl.col("hour_utc").is_between(0, 12).alias("h00_12"),
                   (pl.col("gap_prev_h") < 2).fill_null(False).alias("prev2h"), (pl.col("gap_next_h") < 2).fill_null(False).alias("next2h"),
                   ((pl.col("gap_prev_h") < 2).fill_null(False) | (pl.col("gap_next_h") < 2).fill_null(False)).alias("near2h"),
                   ((pl.col("gap_prev_h") < 24).fill_null(False)).alias("prev24h"),
                   (pl.col("media_kind") == "video").alias("video"), (pl.col("media_kind") == "image").alias("image"), (pl.col("media_kind") != "none").alias("media"),
                   (pl.col("ext_link") & ~pl.col("points_to_reply")).alias("link_in_post"), (~pl.col("ext_link") & ~pl.col("points_to_reply")).alias("no_link"))
STRICT = r"(?:[Ii]ntroducing|[Aa]nnouncing|[Uu]nveiling|[Mm]eet|[Ll]aunching|[Ll]aunch(?:ed)?|[Rr]eleasing|[Rr]eleased?|[Oo]pen[- ]?sourc(?:ed|ing)|INTRODUCING|ANNOUNCING)[:\s]+(?:the |our |my |a |an )?(?:new |all-new )?[@A-Z0-9][\w.@-]+"
T = T.with_columns(pl.col("text").fill_null("").str.replace_all(r"https?://\S+", "").str.slice(0, 300).str.contains(STRICT).alias("strict"))
LAUNCH = ["launch", "open_source"]; CUT = 1776902400000; ANN = ["launch", "open_source", "feature", "funding", "milestone", "hiring"]
SCOPES = {
    "launch_cf": ("launch-type posts (launch, open-source release) of company and founder accounts", pl.col("ptype").is_in(LAUNCH) & (pl.col("akind") != "other")),
    "launch_all": ("first-person launch-type posts of every account in the corpus", pl.col("ptype").is_in(LAUNCH)),
    "launch_strict_cf": ("launch-type posts of company and founder accounts whose opening has an announcement verb directly followed by a product name (label noise removed)", pl.col("ptype").is_in(LAUNCH) & (pl.col("akind") != "other") & pl.col("strict")),
    "launch_strict_all": ("the same strict launch posts, every account", pl.col("ptype").is_in(LAUNCH) & pl.col("strict")),
    "launch_all_under_100k": ("first-person launch-type posts of accounts with a known follower count under 100,000", pl.col("ptype").is_in(LAUNCH) & pl.col("tier").is_in(["<1k", "1k-10k", "10k-100k"])),
    "launch_all_100k_plus": ("first-person launch-type posts of accounts with 100,000 followers or more", pl.col("ptype").is_in(LAUNCH) & (pl.col("tier") == "100k+")),
    "launch_all_before_cutoff": ("first-person launch-type posts of every account, posted before 2026-04-23", pl.col("ptype").is_in(LAUNCH) & (pl.col("created_at_ms") < CUT)),
    "launch_all_after_cutoff": ("first-person launch-type posts of every account, posted on or after 2026-04-23 (the model's test period)", pl.col("ptype").is_in(LAUNCH) & (pl.col("created_at_ms") >= CUT)),
    "launch_cf_before_cutoff": ("launch-type posts of company and founder accounts, before 2026-04-23", pl.col("ptype").is_in(LAUNCH) & (pl.col("akind") != "other") & (pl.col("created_at_ms") < CUT)),
    "launch_cf_after_cutoff": ("launch-type posts of company and founder accounts, on or after 2026-04-23", pl.col("ptype").is_in(LAUNCH) & (pl.col("akind") != "other") & (pl.col("created_at_ms") >= CUT)),
    "announce_cf": ("every own announcement (launch, open source, feature, funding, milestone, hiring) of company and founder accounts", pl.col("ptype").is_in(ANN) & (pl.col("akind") != "other")),
    "announce_all": ("every own announcement of every account", pl.col("ptype").is_in(ANN)),
}
CA = pl.col("source") == "community_archive"     # video is only recorded reliably in this source
# name: (description, with, restrict-to (None = all), comparison description)
A = {
    "video_vs_none": ("a video or GIF is attached, against no media", pl.col("video"), CA & (pl.col("video") | ~pl.col("media"))),
    "image_vs_none": ("an image is attached, against no media", pl.col("image"), CA & (pl.col("image") | ~pl.col("media"))),
    "video_vs_image": ("a video or GIF is attached, against an image", pl.col("video"), CA & pl.col("media")),
    "media_vs_none": ("an image or video is attached, against no media", pl.col("media"), None),
    "link_in_post_vs_none": ("an external link is in the post, against no link", pl.col("link_in_post"), pl.col("link_in_post") | pl.col("no_link")),
    "points_to_reply_vs_link_in_post": ("the post says the link is in the reply or below, against the link in the post", pl.col("points_to_reply"), pl.col("points_to_reply") | pl.col("link_in_post")),
    "points_to_reply_vs_none": ("the post says the link is in the reply or below, against no link and no pointer", pl.col("points_to_reply"), pl.col("points_to_reply") | pl.col("no_link")),
    "thread_marker": ("the post is marked as the start of a thread (🧵, 1/, 'thread', ends pointing down)", pl.col("thread_marker"), None),
    "open_announce": ("the first line carries the announcement verb (introducing, announcing, launching, meet, excited to ...)", pl.col("open_announce"), None),
    "open_problem": ("the first line states a problem or asks a question", pl.col("open_problem"), None),
    "open_result": ("the first line states a result (a benchmark, x times faster, a percentage, #1)", pl.col("open_result"), None),
    "open_number": ("the first line contains a number", pl.col("open_number"), None),
    "open_question": ("the first line contains a question", pl.col("open_question"), None),
    "names_product_first_line": ("the first line names the product (a proper name, version or handle)", pl.col("names_product_first_line"), None),
    "long280": ("the text is 280 characters or longer", pl.col("long280"), None),
    "long600": ("the text is 600 characters or longer, against under 280", pl.col("long600"), pl.col("long600") | ~pl.col("long280")),
    "lines3": ("the text is laid out on three or more lines", pl.col("lines3"), None),
    "tags": ("the post tags at least one account", pl.col("tags"), None),
    "tags3": ("the post tags three or more accounts, against none", pl.col("tags3"), pl.col("tags3") | ~pl.col("tags")),
    "hashtag": ("the post has a hashtag", pl.col("hashtag"), None),
    "emoji": ("the post has an emoji", pl.col("emoji"), None),
    "is_quote": ("the announcement is a quote post", pl.col("is_quote"), None),
    "weekend": ("posted on Saturday or Sunday (UTC)", pl.col("weekend"), None),
    "tue_thu": ("posted Tuesday to Thursday (UTC), against Monday and Friday", pl.col("tue_thu"), ~pl.col("weekend")),
    "h13_17": ("posted 13:00 to 17:59 UTC (US morning), against any other hour", pl.col("h13_17"), None),
    "h18_23": ("posted 18:00 to 23:59 UTC, against 13:00 to 17:59 UTC", pl.col("h18_23"), pl.col("h18_23") | pl.col("h13_17")),
    "h00_12": ("posted 00:00 to 12:59 UTC, against 13:00 to 17:59 UTC", pl.col("h00_12"), pl.col("h00_12") | pl.col("h13_17")),
    "prev2h": ("posted within 2 hours after the account's previous post", pl.col("prev2h"), None),
    "next2h": ("the account posts again within 2 hours", pl.col("next2h"), None),
    "near2h": ("another post by the account within 2 hours before or after", pl.col("near2h"), None),
}
NO_VIEWS = {"video_vs_none", "image_vs_none", "video_vs_image", "lines3", "open_announce", "open_problem", "open_result", "open_number", "open_question", "names_product_first_line"}
ADJ = ["video", "image", "link_in_post", "points_to_reply", "thread_marker", "open_announce", "open_problem", "open_result", "open_number", "names_product_first_line", "long280", "lines3", "tags", "hashtag",
       "is_quote", "weekend", "h18_23", "h00_12", "prev2h", "next2h"]

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
    per = q.group_by("author_id").agg(pl.col(y).filter(pl.col("a")).mean().alias("m1"), pl.col("a").sum().alias("n1"), pl.col(y).filter(~pl.col("a")).mean().alias("m0"), (~pl.col("a")).sum().alias("n0"),
                                      (pl.col(y).filter(pl.col("a")) > 0).mean().alias("b1"), (pl.col(y).filter(~pl.col("a")) > 0).mean().alias("b0")).filter((pl.col("n1") > 0) & (pl.col("n0") > 0)).sort("author_id")
    if per.height < 5: return None
    dlt = (per["m1"] - per["m0"]).to_numpy(); w = (per["n1"] * per["n0"] / (per["n1"] + per["n0"])).to_numpy().astype(float); db = (per["b1"] - per["b0"]).to_numpy()
    st = lambda i: float(np.sum(w[i] * dlt[i]) / np.sum(w[i]))
    lo, hi = boot_ci(st, len(w)); sb = lambda i: float(np.sum(w[i] * db[i]) / np.sum(w[i])); blo, bhi = boot_ci(sb, len(w))
    return {"ratio": round(float(np.exp(st(np.arange(len(w))))), 3), "ratio_lo": round(float(np.exp(lo)), 3), "ratio_hi": round(float(np.exp(hi)), 3), "accounts_with_both": int(per.height),
            "posts_with": int(per["n1"].sum()), "posts_without": int(per["n0"].sum()), "beat_own_median_diff": round(sb(np.arange(len(w))), 4), "beat_diff_lo": round(blo, 4), "beat_diff_hi": round(bhi, 4)}

def pooled(q, y):
    q = q.filter(pl.col(y).is_not_null())
    r = {}
    for nm, f in (("with", pl.col("a")), ("without", ~pl.col("a"))):
        z = q.filter(f)
        r[nm] = {"posts": z.height, "accounts": z["author_id"].n_unique(), "median_ratio_to_own_median": None if z.height == 0 else round(float(np.exp(z[y].median())), 3),
                 "beat_own_median": None if z.height == 0 else round(float((z[y] > 0).mean()), 4)}
    return r

def adjusted(S, y, names):
    q = S.filter(pl.col(y).is_not_null() & CA)
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

out = {"scopes": {}, "types": {}}
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
if ONLY: out = json.load(open("data/launch/effects.json"))
for sk, (sdesc, sf) in SCOPES.items():
    if ONLY and sk not in ONLY: continue
    S = T.filter(sf & pl.col("lift_likes").is_not_null())
    res = {"description": sdesc, "posts": S.height, "posts_community_archive": S.filter(CA).height, "accounts_community_archive": S.filter(CA)["author_id"].n_unique(), "accounts": S["author_id"].n_unique(), "posts_with_views": S.filter(pl.col("lift_views").is_not_null()).height, "effects": {}}
    for k, (desc, fa, restrict) in A.items():
        q = (S.filter(restrict) if restrict is not None else S).with_columns(fa.fill_null(False).alias("a"))
        e = {"description": desc}
        # Likes: Community Archive posts only. The 2026 view corpus stores text without line breaks and does not record video,
        # so first-line, layout and video attributes cannot be read there. Views exist only in the view corpus.
        for y, yn in (("lift_likes", "likes"), ("lift_views", "views")):
            qq = q.filter(CA) if yn == "likes" else q.filter(~CA)
            if yn == "views" and (k in NO_VIEWS or qq.filter(pl.col(y).is_not_null()).height < 60): continue
            e[yn] = {"pairs": pairs_est(qq, y), "within_account": ratio_est(qq, y), "pooled": pooled(qq, y)}
        res["effects"][k] = e
    res["adjusted_likes"] = adjusted(S, "lift_likes", ADJ)
    out["scopes"][sk] = res
    print(f"\n== {sk}: {S.height} posts, {res['accounts']} accounts, {res['posts_with_views']} with views")
    print(f"{'attribute':<34}{'win':>6} {'interval':>13} {'pairs':>6} {'acc':>4} | {'ratio':>5} {'interval':>12} {'acc':>4} {'n+':>5} {'n-':>5} | {'adj':>5} {'interval':>12}")
    for k, e in res["effects"].items():
        p, r = e["likes"]["pairs"], e["likes"]["within_account"]; a = res["adjusted_likes"].get({"video_vs_none": "video", "image_vs_none": "image", "link_in_post_vs_none": "link_in_post", "points_to_reply_vs_none": "points_to_reply"}.get(k, k))
        ps = f"{100*p['win_rate']:5.1f}% {100*p['lo']:5.1f}-{100*p['hi']:<5.1f} {p['pairs']:>6} {p['accounts']:>4}" if p and p["win_rate"] is not None else " " * 32
        rs = f"{r['ratio']:5.2f} {r['ratio_lo']:5.2f}-{r['ratio_hi']:<5.2f} {r['accounts_with_both']:>4} {r['posts_with']:>5} {r['posts_without']:>5}" if r else " " * 35
        as_ = f"{a['ratio']:5.2f} {a['ratio_lo']:5.2f}-{a['ratio_hi']:<5.2f}" if a else ""
        print(f"{k:<34}{ps} | {rs} | {as_}")
# how each type does against the account's usual
for scope, f in () if ONLY else (("company", pl.col("akind") == "company"), ("founder", pl.col("akind") == "founder"), ("company_and_founder", pl.col("akind") != "other"), ("other_accounts", pl.col("akind") == "other")):
    out["types"][scope] = {}
    for pt in ANN + ["demo"]:
        for y, yn in (("lift_likes", "likes"), ("lift_views", "views")):
            z = T.filter(f & (pl.col("ptype") == pt) & pl.col(y).is_not_null())
            if z.height < 20: continue
            per = z.group_by("author_id").agg(pl.col(y).mean().alias("m"), (pl.col(y) > 0).mean().alias("b"), pl.col("top10_likes").mean().alias("t"), pl.len().alias("n")).sort("author_id")
            m, b, t, n = (per[c].to_numpy().astype(float) for c in ("m", "b", "t", "n"))
            f1 = lambda i: np.sum(m[i] * n[i]) / np.sum(n[i]); f2 = lambda i: np.sum(b[i] * n[i]) / np.sum(n[i]); f3 = lambda i: np.sum(t[i] * n[i]) / np.sum(n[i])
            l1, h1 = boot_ci(f1, len(m)); l2, h2 = boot_ci(f2, len(m)); l3, h3 = boot_ci(f3, len(m))
            out["types"][scope].setdefault(pt, {})[yn] = {"posts": z.height, "accounts": per.height, "median_ratio": round(float(np.exp(z[y].median())), 3), "geo_mean_ratio": round(float(np.exp(f1(np.arange(len(m))))), 3),
                "geo_mean_lo": round(float(np.exp(l1)), 3), "geo_mean_hi": round(float(np.exp(h1)), 3), "beat_own_median": round(float(f2(np.arange(len(m)))), 4), "beat_lo": round(l2, 4), "beat_hi": round(h2, 4),
                **({"top10_of_own": round(float(f3(np.arange(len(m)))), 4), "top10_lo": round(l3, 4), "top10_hi": round(h3, 4)} if yn == "likes" else {})}
print("\n== how each type does against the account's own trailing median")
for scope, d in out["types"].items():
    for pt, v in d.items():
        for yn, x in v.items():
            print(f"{scope:<20}{pt:<12}{yn:<6} posts {x['posts']:>5} acc {x['accounts']:>4}  median {x['median_ratio']:5.2f}x  mean {x['geo_mean_ratio']:5.2f}x ({x['geo_mean_lo']:.2f}-{x['geo_mean_hi']:.2f})  beat {100*x['beat_own_median']:.0f}% ({100*x['beat_lo']:.0f}-{100*x['beat_hi']:.0f})" + (f"  top10 {100*x['top10_of_own']:.0f}%" if "top10_of_own" in x else ""))
json.dump(out, open("data/launch/effects.json", "w"), indent=1)
