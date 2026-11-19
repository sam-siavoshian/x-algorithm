#!/usr/bin/env python3
"""Build the items the language-model judge sees, for three pair sets: dev (2,805 pre-cutoff development pairs),
bench (3,530 benchmark pairs), launch (850 launch pairs, rebuilt as scripts/launch/model_check.py does).

    python scripts/llm_judge/prep.py     # writes data/llm/items_{dev,bench,launch}.jsonl and data/llm/pairs_{set}.parquet

Each item: the two posts as the product would see two drafts (text, media kind, quoted text, link domain; no time, no
outcome), and up to 200 of the account's posts created before the earlier of the two, with like counts. Surrogate
score differences (baseline three models, and ten untouched) are joined on."""
import glob, json, re, numpy as np, polars as pl

D = "data"
CUT = 1776902400000  # 2026-04-23 UTC
B3 = ["ts46_s1", "ts46_s2", "compare_v2_full"]
B10 = B3 + ["ts0_s0", "ts0_s1", "ts20_s1", "ts20_s2", "compare_full", "compare_audience_only_training", "compare_v2_audience_only_training"]
DEVNAME = {"compare_full": "cmp_full", "compare_audience_only_training": "cmp_aud", "compare_v2_audience_only_training": "cmpv2_aud"}
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "author_followers", "created_at_ms", "text", "kind", "quoted_text", "media", "urls", "likes", "source"]
TCO = re.compile(r"https?://t\.co/\w+")


def render(r: dict) -> dict:
    t = r["text"] or ""
    dom = [u.get("final_domain") or "" for u in (r["urls"] or []) if not u.get("is_media")]
    t = TCO.sub("", t).strip()
    m = sorted({x["type"] for x in (r["media"] or []) if x.get("type")})
    return {"text": t, "media": ("video" if ("video" in m or "animated_gif" in m) else "image" if m else ""), "links": [d for d in dom if d][:2],
            "quoted": (r["quoted_text"] or "")[:400] if r["kind"] == "quote" else "", "likes": r["likes"], "t": r["created_at_ms"], "id": r["post_id"]}


def main():
    dev = pl.read_parquet(f"{D}/tracks/replicate/dev_table/pairs.parquet").select("post_id", "post_id_b", "author_id", "likes", "likes_b")
    for nm, cols in (("s3", B3), ("s10", B10)):
        dev = dev.with_columns(pl.Series(nm, np.mean([np.load(f"{D}/tracks/replicate/dev_scores/start_{DEVNAME.get(m, m)}.npy") for m in cols], 0)))
    dev = dev.with_columns(pl.lit("dev").alias("part"))
    bp = pl.read_parquet(f"{D}/tracks/pairs/benchmark_pairs.parquet")
    for m in B10:
        R = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{D}/bench2/{m}/shard_*.parquet"))]).select("post_id", "post_id_b", "s_likes")
        f1 = bp.join(R, on=["post_id", "post_id_b"], how="left")
        f2 = bp.join(R.rename({"post_id": "post_id_b", "post_id_b": "post_id"}).with_columns(-pl.col("s_likes")), on=["post_id", "post_id_b"], how="left")
        bp = bp.with_columns(pl.coalesce(f1["s_likes"], f2["s_likes"]).alias("l_" + m))
    bench = bp.with_columns(pl.mean_horizontal(["l_" + m for m in B3]).alias("s3"), pl.mean_horizontal(["l_" + m for m in B10]).alias("s10")).select("post_id", "post_id_b", "author_id", "likes", "likes_b", "s3", "s10", "part")
    # launch pairs, as model_check.py
    T = pl.read_parquet(f"{D}/tracks/launch/launch_posts.parquet").select("post_id", "author_id", "likes", "created_at_ms", "akind", "ptype")
    S = pl.read_parquet(f"{D}/tracks/launch/launch_scores.parquet").select("post_id", "pred_likes")
    S10 = None
    for m in B10:
        x = pl.read_parquet(f"{D}/tracks/replicate/launch_scores/start_{DEVNAME.get(m, m)}.parquet").with_columns((pl.col("pred_likes") + 1).log10().alias("m_" + m)).drop("pred_likes")
        S10 = x if S10 is None else S10.join(x, on="post_id")
    S10 = S10.with_columns(pl.mean_horizontal(["m_" + m for m in B3]).alias("r3"), pl.mean_horizontal(["m_" + m for m in B10]).alias("r10")).select("post_id", "r3", "r10")
    P = T.join(S, on="post_id").join(S10, on="post_id", how="left")
    p = P.join(P, on="author_id", suffix="_b").filter(pl.col("post_id") < pl.col("post_id_b"))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), (pl.col("pred_likes") + 1).log10().sub((pl.col("pred_likes_b") + 1).log10()).alias("s3"),
                       (pl.col("r10") - pl.col("r10_b")).alias("s10"), (pl.col("r3") - pl.col("r3_b")).alias("s3r"))
    p = p.filter(pl.col("d").abs() >= np.log(1.25))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 30)
    launch = p.select("post_id", "post_id_b", "author_id", "likes", "likes_b", "s3", "s10", "s3r", pl.lit("launch").alias("part"), "akind")
    sets = {"dev": dev, "bench": bench, "launch": launch}
    authors = pl.concat([s["author_id"] for s in sets.values()]).unique()
    posts = pl.scan_parquet(f"{D}/work/filtered/posts.parquet").filter(pl.col("author_id").is_in(authors.implode())).select(COLS).collect()
    print("posts of pair accounts", posts.height, "accounts", authors.len())
    by_id = {r["post_id"]: r for r in posts.filter(pl.col("post_id").is_in(pl.concat([pl.concat([s["post_id"], s["post_id_b"]]) for s in sets.values()]).unique().implode())).iter_rows(named=True)}
    hist = {a[0]: g.sort("created_at_ms") for a, g in posts.group_by("author_id")}
    for name, s in sets.items():
        rows, out = [], open(f"{D}/llm/items_{name}.jsonl", "w")
        for r in s.iter_rows(named=True):
            a, b = by_id.get(r["post_id"]), by_id.get(r["post_id_b"])
            if a is None or b is None:
                rows.append(None); continue
            t0 = min(a["created_at_ms"], b["created_at_ms"])
            h = hist[r["author_id"]]
            h = h.filter((pl.col("created_at_ms") < t0) & ~pl.col("post_id").is_in([r["post_id"], r["post_id_b"]])).tail(200)
            fol = [x for x in (a["author_followers"], b["author_followers"]) if x is not None]
            item = {"post_id": r["post_id"], "post_id_b": r["post_id_b"], "author_id": r["author_id"], "handle": a["author_handle"], "name": a["author_display_name"],
                    "followers": max(fol) if fol else None, "a": render(a), "b": render(b), "history": [render(x) for x in h.iter_rows(named=True)]}
            out.write(json.dumps(item) + "\n")
            rows.append({"t_a": a["created_at_ms"], "t_b": b["created_at_ms"], "followers": item["followers"], "n_hist": h.height, "same_text": item["a"]["text"] == item["b"]["text"]})
        ok = [x is not None for x in rows]
        s = s.filter(pl.Series(ok)).hstack(pl.DataFrame([x for x in rows if x is not None]))
        s.write_parquet(f"{D}/llm/pairs_{name}.parquet")
        d = np.log1p(s["likes"].to_numpy()) - np.log1p(s["likes_b"].to_numpy())
        k = np.abs(d) >= np.log(1.25)
        print(name, "pairs", s.height, "missing", len(ok) - sum(ok), "clear", int(k.sum()), "2x", int((np.abs(d) >= np.log(2)).sum()), "s3 acc", round(float(((s["s3"].to_numpy() * d) > 0)[k].mean()), 4),
              "s10 acc", round(float(((s["s10"].to_numpy() * d) > 0)[k].mean()), 4), "later wins", round(float(((d > 0) == (s["t_a"].to_numpy() > s["t_b"].to_numpy()))[k].mean()), 3),
              "median hist", float(s["n_hist"].median()), "followers known", round(float(s["followers"].is_not_null().mean()), 3), "after 2026-05-18", int((np.minimum(s["t_a"].to_numpy(), s["t_b"].to_numpy()) >= 1779062400000).sum()))


if __name__ == "__main__":
    main()
