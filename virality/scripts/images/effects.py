#!/usr/bin/env python3
"""Does the picture matter, and which kind? Within-account effects of image type and image quality on likes.

    python scripts/images/effects.py      # reads data/launch/table.parquet + data/images/attributes.parquet, writes data/images/effects.json

Method: scripts/launch/effects.py unchanged (its pairs / ratio / adjusted estimators are imported from there): lift =
log((likes+1)/(account's trailing median likes+1)); pairs within account, posting order balanced, at most 50 per account,
likes differing 1.25x or more, at most 365 days apart; precision-weighted within-account ratio; account fixed-effects
regression with the text attributes held fixed; 95% intervals from resampling accounts. Likes from the Community
Archive rows only (the source that records media reliably); views from the 2026 view corpus where 60+ posts.

Comparisons (each a separate attribute): every image type against no media; every image type against the other image
posts; quality tertiles inside image posts (aesthetic score, sharpness, original resolution, text cover), top against
bottom tertile, tertile cut points fixed on all fetched images; "AI-generated" against designed graphics (confident
subset only, see REPORT.md); faces against none; one image against several.

Rule fixed before looking at results (same as the launch track): an effect is stated when, in its primary scope, all
three tests exclude no effect, and in the confident-label subset (zero-shot margin in the top half of its class) at
least one test passes in the same direction; graded A when it is also confirmed (same direction, one test passing)
separately before and from 2026-04-23, B otherwise. Anything else is reported as not established.
"""
import json, sys, numpy as np, polars as pl
src = open("scripts/launch/effects.py").read()
lib = src[src.index("def boot_ci"):src.index("out = {\"scopes\"")]          # the estimators only, not the launch run
G = {"np": np, "pl": pl, "rng": np.random.default_rng(0)}
exec(compile("CA = pl.col('source') == 'community_archive'\n" + lib, "scripts/launch/effects.py", "exec"), G)
pairs_est, ratio_est, pooled = G["pairs_est"], G["ratio_est"], G["pooled"]
CA = pl.col("source") == "community_archive"
CUT = 1776902400000
T = pl.read_parquet("data/launch/table.parquet")
A = pl.read_parquet("data/images/attributes.parquet").drop("author_id", "media_idx", "priority")
TYPES = [c for c in json.load(open("data/images/type_names.json"))]
T = T.join(A, on="post_id", how="left").with_columns(pl.col("itype").is_not_null().alias("fetched"))
# posts with a photo whose image we do not have are left out of every comparison involving images
T = T.filter((pl.col("media_kind") == "none") | pl.col("fetched") | (pl.col("media_kind") == "video"))
T = T.with_columns((pl.col("chars") >= 280).alias("long280"), (pl.col("lines") >= 3).alias("lines3"), (pl.col("ext_link") & ~pl.col("points_to_reply")).alias("link_in_post"),
                   (pl.col("media_kind") == "none").alias("nomedia"), pl.col("fetched").alias("img"))
CTRL = ["long280", "lines3", "link_in_post", "points_to_reply", "thread_marker", "open_announce", "open_problem", "open_number", "names_product_first_line", "tags", "hashtag", "is_quote", "emoji"]
ANN = ["launch", "open_source", "feature", "funding", "milestone", "hiring"]; LAUNCH = ["launch", "open_source"]
cf = pl.col("akind") != "other"
SCOPES = {
    "launch_cf": ("launch-type posts of company and founder accounts", pl.col("ptype").is_in(LAUNCH) & cf),
    "launch_all": ("launch-type posts of every account", pl.col("ptype").is_in(LAUNCH)),
    "announce_cf": ("every own announcement of company and founder accounts", pl.col("ptype").is_in(ANN) & cf),
    "announce_all": ("every own announcement of every account", pl.col("ptype").is_in(ANN)),
    "posts_cf": ("every post of company and founder accounts (announcements and ordinary posts)", cf),
    "posts_all": ("every post of every account that has at least one own announcement", pl.lit(True)),
    "ordinary_cf": ("ordinary posts (not an own announcement) of company and founder accounts", cf & ~pl.col("ptype").is_in(ANN + ["demo"])),
    "ordinary_all": ("ordinary posts (not an own announcement) of every account with an own announcement", ~pl.col("ptype").is_in(ANN + ["demo"])),
}
for base in ("announce_all", "posts_cf", "posts_all"):
    d, f = SCOPES[base]
    SCOPES[base + "_under_100k"] = (d + ", accounts under 100k followers", f & pl.col("tier").is_in(["<1k", "1k-10k", "10k-100k"]))
    SCOPES[base + "_100k_plus"] = (d + ", accounts with 100k followers or more", f & (pl.col("tier") == "100k+"))
    SCOPES[base + "_before_cutoff"] = (d + ", posted before 2026-04-23", f & (pl.col("created_at_ms") < CUT))
    SCOPES[base + "_after_cutoff"] = (d + ", posted on or after 2026-04-23", f & (pl.col("created_at_ms") >= CUT))
ATTR = {}
for t in TYPES:
    ATTR[f"{t}_vs_none"] = (f"the image is a {t.replace('_', ' ')}, against no media", pl.col("itype") == t, (pl.col("itype") == t) | pl.col("nomedia"))
    ATTR[f"{t}_vs_other_images"] = (f"the image is a {t.replace('_', ' ')}, against any other kind of image", pl.col("itype") == t, pl.col("img"))
ATTR["image_vs_none"] = ("any fetched image, against no media", pl.col("img"), pl.col("img") | pl.col("nomedia"))
for q in ("aesthetic", "sharpness", "resolution", "text_area_share", "colourfulness"):
    ATTR[f"{q}_top_vs_bottom"] = (f"{q} in the top third of all fetched images, against the bottom third (image posts only)", pl.col(f"{q}_tert") == 2, pl.col("img") & (pl.col(f"{q}_tert") != 1))
ATTR["face_vs_no_face"] = ("a face is detected in the image, against image posts without one", pl.col("faces") > 0, pl.col("img"))
ATTR["several_images"] = ("two or more images, against one", pl.col("n_media") >= 2, pl.col("img"))
ATTR["ai_vs_designed"] = ("the image is AI-generated art (confident subset), against a designed graphic", pl.col("ai_confident"), pl.col("img") & (pl.col("ai_confident") | (pl.col("itype") == "designed_graphic")))
ATTR["ai_vs_other_images"] = ("the image is AI-generated art (confident subset), against any other image", pl.col("ai_confident"), pl.col("img"))

def adjusted(S, y, names):
    q = S.filter(pl.col(y).is_not_null() & CA)
    q = q.with_columns([pl.col(c).fill_null(False).cast(pl.Float64) for c in names]).filter(pl.len().over("author_id") >= 2)
    X = q.select(names).to_numpy(); Y = q[y].to_numpy(); u, inv = np.unique(q["author_id"].to_numpy(), return_inverse=True)
    keep = X.std(0) > 0; X = X[:, keep]; nm = [n for n, k in zip(names, keep) if k]
    def dm(M):
        s = np.zeros((len(u),) + M.shape[1:]); np.add.at(s, inv, M); c = np.bincount(inv).astype(float); return M - (s / c.reshape(-1, *([1] * (M.ndim - 1))))[inv]
    Xd, Yd = dm(X), dm(Y); beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]; rows = [np.where(inv == k)[0] for k in range(len(u))]; B = []
    rr = np.random.default_rng(0)
    for _ in range(400):
        ii = np.concatenate([rows[k] for k in rr.integers(0, len(u), len(u))]); B.append(np.linalg.lstsq(Xd[ii], Yd[ii], rcond=None)[0])
    B = np.array(B)
    return {n: {"ratio": round(float(np.exp(beta[j])), 3), "ratio_lo": round(float(np.exp(np.quantile(B[:, j], .025))), 3), "ratio_hi": round(float(np.exp(np.quantile(B[:, j], .975))), 3),
                "posts_with": int(X[:, j].sum())} for j, n in enumerate(nm)} | {"_n": {"posts": int(len(Y)), "accounts": int(len(u))}}

T = T.with_columns([(pl.col("itype") == t).alias(f"is_{t}") for t in TYPES] + [pl.col("media_kind").eq("video").alias("is_video")]
                   + [(pl.col("ptype") == t).alias(f"pt_{t}") for t in ANN + ["demo"]])
CTRL = CTRL + [f"pt_{t}" for t in ANN + ["demo"]]          # post type held fixed: image posts are more often announcements
TYPE_ADJ = [f"is_{t}" for t in TYPES] + ["is_video"] + CTRL          # against no media (the omitted level)
QUAL_ADJ = [f"{q}_hi" for q in ("aesthetic", "sharpness", "resolution", "text_area_share")] + [f"{q}_lo" for q in ("aesthetic", "sharpness", "resolution", "text_area_share")] + [f"is_{t}" for t in TYPES[1:]] + CTRL
T = T.with_columns([pl.col(f"{q}_tert").eq(2).alias(f"{q}_hi") for q in ("aesthetic", "sharpness", "resolution", "text_area_share", "colourfulness")] +
                   [pl.col(f"{q}_tert").eq(0).alias(f"{q}_lo") for q in ("aesthetic", "sharpness", "resolution", "text_area_share", "colourfulness")])
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
out = {"scopes": {}}
for sk, (sdesc, sf) in SCOPES.items():
    if ONLY and sk not in ONLY: continue
    for conf_sub in (("all", "confident") if sk in ("posts_cf", "launch_cf", "announce_cf", "ordinary_cf") else ("all",)):
        S = T.filter(sf & pl.col("lift_likes").is_not_null())
        if conf_sub == "confident": S = S.filter(~pl.col("img") | pl.col("confident"))
        res = {"description": sdesc + ("" if conf_sub == "all" else "; image posts restricted to confident type labels"), "posts": S.height, "accounts": S["author_id"].n_unique(),
               "image_posts": S.filter(pl.col("img")).height, "image_posts_community_archive": S.filter(pl.col("img") & CA).height, "effects": {}}
        for k, (desc, fa, restrict) in ATTR.items():
            q = S.filter(restrict).with_columns(fa.fill_null(False).alias("a"))
            e = {"description": desc}
            for y, yn in (("lift_likes", "likes"), ("lift_views", "views")):
                qq = q.filter(CA) if yn == "likes" else q.filter(~CA)
                if qq.filter(pl.col("a") & pl.col(y).is_not_null()).height < (10 if yn == "likes" else 60): continue
                e[yn] = {"pairs": pairs_est(qq, y), "within_account": ratio_est(qq, y), "pooled": pooled(qq, y)}
            res["effects"][k] = e
        res["adjusted_types_vs_none"] = adjusted(S, "lift_likes", TYPE_ADJ)
        res["adjusted_quality_image_posts"] = adjusted(S.filter(pl.col("img")), "lift_likes", QUAL_ADJ)
        out["scopes"][sk + ("" if conf_sub == "all" else "__confident")] = res
        print(f"\n== {sk} [{conf_sub}]: {S.height} posts, {res['accounts']} accounts, {res['image_posts_community_archive']} image posts (CA)", flush=True)
        for k, e in res["effects"].items():
            if "likes" not in e: continue
            p, r = e["likes"]["pairs"], e["likes"]["within_account"]
            ps = f"{100*p['win_rate']:5.1f}% {100*p['lo']:5.1f}-{100*p['hi']:<5.1f} {p['pairs']:>6} {p['accounts']:>4}" if p and p.get("win_rate") is not None else " " * 32
            rs = f"{r['ratio']:5.2f} {r['ratio_lo']:5.2f}-{r['ratio_hi']:<5.2f} {r['accounts_with_both']:>4} {r['posts_with']:>5}" if r else ""
            print(f"  {k:<36}{ps} | {rs}")
import os
json.dump(out, open(os.environ.get("EFFECTS_OUT", "data/images/effects.json"), "w"), indent=1)
