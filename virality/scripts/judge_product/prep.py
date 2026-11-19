#!/usr/bin/env python3
"""Development and test tables for the judge_product track.

    python scripts/judge_product/prep.py

1. Items for the 3,655 launch pairs from before the cutoff (the selective track's launch development pairs), in the format
   scripts/llm_judge/prep.py writes for the other sets: data/llm/items_launchpre.jsonl, data/llm/pairs_launchpre.parquet.
2. For all four sets the surrogate's gap on the product's own scale: `sg` = the baseline three's gap in log10 predicted likes
   through the serving path (tracks/selective/feat_*_base3.parquet), oriented post_id minus post_id_b.
3. The development split: `cell` = fit (both posts before T, account half A), val (later post on or after T, account half B),
   or other; T is 2025-10-01 for same-topic pairs and 2026-01-01 for launch pairs. Samples for the stronger model."""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/llm_judge")
from lib import half_of
from prep import render, COLS

D = "data"
T_SPLIT = {"dev": 1759276800000, "launchpre": 1767225600000}     # 2025-10-01, 2026-01-01 UTC


def sg_of(P: pl.DataFrame, feat: str) -> pl.DataFrame:
    F = pl.read_parquet(f"{D}/tracks/selective/{feat}.parquet").select("post_id", "post_id_b", "s_base3")
    a = P.join(F, on=["post_id", "post_id_b"], how="left")["s_base3"]
    b = P.join(F.select(pl.col("post_id").alias("post_id_b"), pl.col("post_id_b").alias("post_id"), (-pl.col("s_base3")).alias("s_base3")), on=["post_id", "post_id_b"], how="left")["s_base3"]
    return P.with_columns(a.fill_null(b).alias("sg"))


def main():
    F = pl.read_parquet(f"{D}/tracks/selective/feat_launch_pre_base3.parquet").select("post_id", "post_id_b", "author_id", pl.col("likes_a").alias("likes"), "likes_b", "akind", pl.col("s_base3").alias("s3"))
    posts = pl.scan_parquet(f"{D}/work/filtered/posts.parquet").filter(pl.col("author_id").is_in(F["author_id"].unique().implode())).select(COLS).collect()
    ids = pl.concat([F["post_id"], F["post_id_b"]]).unique()
    by_id = {r["post_id"]: r for r in posts.filter(pl.col("post_id").is_in(ids.implode())).iter_rows(named=True)}
    hist = {a[0]: g.sort("created_at_ms") for a, g in posts.group_by("author_id")}
    rows, out = [], open(f"{D}/llm/items_launchpre.jsonl", "w")
    for r in F.iter_rows(named=True):
        a, b = by_id.get(r["post_id"]), by_id.get(r["post_id_b"])
        if a is None or b is None:
            rows.append(None); continue
        t0 = min(a["created_at_ms"], b["created_at_ms"])
        h = hist[r["author_id"]]
        h = h.filter((pl.col("created_at_ms") < t0) & ~pl.col("post_id").is_in([r["post_id"], r["post_id_b"]])).tail(200)
        fol = [x for x in (a["author_followers"], b["author_followers"]) if x is not None]
        item = {"post_id": r["post_id"], "post_id_b": r["post_id_b"], "author_id": r["author_id"], "handle": a["author_handle"], "name": a["author_display_name"],
                "followers": max(fol) if fol else None, "a": render(a), "b": render(b), "history": [render(x) for x in h.iter_rows(named=True)]}
        assert item["a"]["likes"] == r["likes"] and item["b"]["likes"] == r["likes_b"]
        out.write(json.dumps(item) + "\n")
        rows.append({"t_a": a["created_at_ms"], "t_b": b["created_at_ms"], "followers": item["followers"], "n_hist": h.height, "same_text": item["a"]["text"] == item["b"]["text"]})
    out.close()
    ok = [x is not None for x in rows]
    P = F.filter(pl.Series(ok)).hstack(pl.DataFrame([x for x in rows if x is not None])).with_columns(pl.lit("launchpre").alias("part"))
    P.write_parquet(f"{D}/llm/pairs_launchpre.parquet")
    print("launchpre pairs", P.height, "missing", len(ok) - sum(ok), "median history", P["n_hist"].median(), "followers known", round(float(P["followers"].is_not_null().mean()), 3))

    rng = np.random.default_rng(7)
    for name, feat in (("dev", "feat_dev_base3"), ("launchpre", "feat_launch_pre_base3"), ("bench", "feat_bench_base3"), ("launch", "feat_launch_test_base3")):
        P = sg_of(pl.read_parquet(f"{D}/llm/pairs_{name}.parquet"), feat)
        d = np.log1p(P["likes"].to_numpy().astype(float)) - np.log1p(P["likes_b"].to_numpy().astype(float))
        P = P.with_columns(pl.Series("d", d), pl.Series("half", [half_of(a) for a in P["author_id"].to_list()]))
        if name in T_SPLIT:
            late = np.maximum(P["t_a"].to_numpy(), P["t_b"].to_numpy()) >= T_SPLIT[name]
            hb = P["half"].to_numpy() == "B"
            cell = np.where(~late & ~hb, "fit", np.where(late & hb, "val", np.where(late, "late_A", "early_B")))
            P = P.with_columns(pl.Series("cell", cell))
            print(name, {c: int((cell == c).sum()) for c in ("fit", "val", "late_A", "early_B")}, "accounts", {c: P.filter(pl.col("cell") == c)["author_id"].n_unique() for c in ("fit", "val")})
        k = np.abs(d) >= np.log(1.25); m = k & P["sg"].is_not_null().to_numpy()
        print(name, "pairs", P.height, "clear", int(k.sum()), "with sg", int(m.sum()), "sg acc", round(float(((P["sg"].to_numpy()[m] * d[m]) > 0).mean()), 4), "s3 acc", round(float(((P["s3"].to_numpy()[k] * d[k]) > 0).mean()), 4))
        P.write_parquet(f"{D}/llm/jp_pairs_{name}.parquet")
    # samples for the stronger model: launch development pairs, 300 from each cell; same-topic validation cell, 300 more
    L = pl.read_parquet(f"{D}/llm/jp_pairs_launchpre.parquet")
    keys = []
    for c in ("fit", "val"):
        Q = L.filter(pl.col("cell") == c)
        keys += [[a, b] for a, b in Q.select("post_id", "post_id_b").sample(min(300, Q.height), seed=7).iter_rows()]
    json.dump(keys, open(f"{D}/llm/sample_launchpre_sol.json", "w"))
    json.dump(keys[:300], open(f"{D}/llm/sample_launchpre_cfg.json", "w"))      # the fit-cell 300: history length and prompt variants
    Dv = pl.read_parquet(f"{D}/llm/jp_pairs_dev.parquet").filter(pl.col("cell") == "val")
    json.dump([[a, b] for a, b in Dv.select("post_id", "post_id_b").sample(min(300, Dv.height), seed=7).iter_rows()], open(f"{D}/llm/sample_dev_val_sol.json", "w"))
    print("samples written")


if __name__ == "__main__":
    main()
