#!/usr/bin/env python3
"""The expanded model table: the existing table's rows copied bit for bit, the new rows appended with the same transforms.

    python scripts/expand/build_table.py --old data/work/model/table --new-rows data/expand/new_rows.parquet --emb data/expand/emb --out data/expand/table

Recipe
  numeric and categorical inputs of the new rows   model.features._arrays with the existing table's stored statistics
                                                   (means, deviations, missing-indicator columns, categorical layout), so
                                                   the two tables share one input space and one model can be scored on either
  existing rows                                    every array copied from the old table by post id; nothing recomputed
  embeddings                                       old from the old table, new from scripts/expand/embed_rows.py
  semantic ids                                     new rows assigned with the existing codebook (work/emb/sid_codebook.npz)
  neighbours                                       new rows: features.neighbors.global_neighbors over old and new rows together
                                                   (whole days strictly before the post); old rows keep their stored neighbours
  self-similarity                                  new rows: neighbors.author_centroid_and_selfsim over the account's old and new rows
  blanked in the new rows                          verified flag and video durations: constant or absent in every existing row
  one more categorical input, src_regime           how the source sampled the account (stream sample, search sample, full timeline);
                                                   train with --drop-cat src_regime to leave it out
Rows are ordered by (account, time, post id) as model.features expects; the account index and bounds are rebuilt.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from xviral.embed import sid as sidmod
from xviral.features.neighbors import author_centroid_and_selfsim, global_neighbors
from xviral.model import features

log = logging.getLogger("table")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/work/model/table")
    ap.add_argument("--rows", default="data/work/dataset/rows.parquet")
    ap.add_argument("--new-rows", default="data/expand/new_rows.parquet")
    ap.add_argument("--emb", default="data/expand/emb")
    ap.add_argument("--codebook", default="data/work/emb/sid_codebook.npz")
    ap.add_argument("--out", default="data/expand/table")
    ap.add_argument("--dry-run", action="store_true", help="build with only the new rows that already have an embedding")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    old, out = Path(args.old), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta0 = json.loads((old / "meta.json").read_text())
    K = np.load(old / "nbr_idx.npy", mmap_mode="r").shape[1]

    # Existing rows in table order, with the columns the neighbour search reads.
    ncols = ["post_id", "author_id", "created_at_ms", "source", "log_reach_ratio", "log_like_ratio", "rate_fav", "rate_reply", "follower_tier_idx"]
    old_ids = np.load(old / "post_id.npy")
    ro = pl.DataFrame({"post_id": old_ids}).join(pl.read_parquet(args.rows, columns=ncols), on="post_id", how="left", maintain_order="left")
    assert ro["author_id"].null_count() == 0
    chk = ro.with_row_index("_i").sort("author_id", "created_at_ms", "post_id")["_i"].to_numpy()
    assert (chk == np.arange(len(old_ids))).all(), "the old table is not in (account, time, post id) order of rows.parquet"

    new = pl.read_parquet(args.new_rows)
    parts = [np.load(f) for f in sorted(glob.glob(str(Path(args.emb) / "part-*.npz")))]
    eid, ev = np.concatenate([p["post_id"] for p in parts]), np.concatenate([p["emb"] for p in parts])
    pos = {int(p): i for i, p in enumerate(eid)}
    idx = np.array([pos.get(int(p), -1) for p in new["post_id"].to_list()])
    if args.dry_run:                # a test on the rows embedded so far
        new, idx = new.filter(pl.Series(idx >= 0)), idx[idx >= 0]
    assert (idx >= 0).all(), f"{int((idx < 0).sum())} new rows have no embedding"
    e_new = ev[idx].astype(np.float32)
    del parts, ev
    e_old = np.asarray(np.load(old / "emb.npy", mmap_mode="r"), dtype=np.float32)
    n_old, n_new = len(old_ids), new.height
    log.info("rows: %d existing, %d new", n_old, n_new)

    # Semantic ids with the existing codebook; checked on existing rows first.
    probe = np.random.default_rng(0).choice(n_old, 3000, replace=False)
    same = (sidmod.phoenix_codes(sidmod.assign(e_old[probe], args.codebook)).astype(np.int16) == np.load(old / "sid.npy", mmap_mode="r")[probe]).all(1).mean()
    log.info("semantic ids recomputed for 3000 existing rows: %.4f identical to the stored ones", same)
    sid_new = sidmod.phoenix_codes(sidmod.assign(e_new, args.codebook)).astype(np.int16)

    # Neighbours of the new rows, searched over everything.
    um = pl.concat([ro.select(ncols), new.select(ncols)], how="vertical_relaxed")
    E = np.concatenate([e_old, e_new])
    nb = global_neighbors(um, E, k=K).slice(n_old, n_new)
    log.info("neighbours done; %d new rows have at least one", int((nb["nbr_post_id"].list.len() > 0).sum()))
    au = new["author_id"].unique()
    m_sub = um.with_row_index("_i").filter(pl.col("author_id").is_in(au.implode()))
    _, ss = author_centroid_and_selfsim(m_sub.drop("_i"), E[m_sub["_i"].to_numpy()])
    new = new.drop([c for c in ("nbr_density_7d", "nbr_best_reach_7d", "nbr_best_like_7d", "self_sim_48h", "src_regime") if c in new.columns])
    new = new.join(nb.select("post_id", "nbr_post_id", "nbr_cos", "nbr_density_7d", "nbr_best_reach_7d", "nbr_best_like_7d"), on="post_id", how="left").join(ss.select("post_id", "self_sim_48h"), on="post_id", how="left")
    del E, um

    # Inputs that carry no information in the existing rows are blanked in the new rows, as an existing source would
    # deliver them: the verified flag (null in every existing row) and video durations (no existing source has them, so a
    # video is "duration unknown" there). Left in, they would mark the source of a row and enter on a scale that was
    # never fitted (the stored deviation of a constant column is 1). new_rows.parquet keeps the real values.
    new = new.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("verified"), pl.lit(None, dtype=pl.Int64).alias("video_duration_ms"),
        pl.lit(False).alias("video_gt_10s"), pl.lit(False).alias("video_index_eligible"), pl.lit(False).alias("thunder_has_video"),
        pl.when(pl.col("has_video")).then(pl.lit("gt_60s")).otherwise(pl.lit("none")).alias("video_duration_bucket"))
    meta2 = {**meta0, "cat_names": meta0["cat_names"] + ["src_regime"]}
    new = features.with_src_regime(new).with_row_index("_in")
    arr, stats, srt = features._arrays(new, meta2)
    assert stats["num_names"] == meta0["num_names"]
    order_new = srt["_in"].to_numpy()
    arr["emb"], arr["has_emb"], arr["sid"] = e_new[order_new].astype(np.float16), np.ones(n_new, bool), sid_new[order_new]
    reg_old = features.with_src_regime(ro.select("author_id", "source")).select(
        pl.col("src_regime").replace_strict({s: i for i, s in enumerate(dict(features.CAT_SPEC)["src_regime"])}, default=0, return_dtype=pl.Int32))["src_regime"].to_numpy()

    # Merge in (account, time, post id) order.
    a_all = np.concatenate([ro["author_id"].to_numpy(), srt["author_id"].to_numpy()])
    t_all = np.concatenate([ro["created_at_ms"].to_numpy(), srt["created_at_ms"].to_numpy()])
    p_all = np.concatenate([old_ids, arr["post_id"]])
    order = np.lexsort((p_all, t_all, a_all))
    inv = np.empty(len(order), np.int64)
    inv[order] = np.arange(len(order))
    n = len(order)
    res: dict[str, np.ndarray] = {}
    for k in ("post_id", "created_s", "split", "w_row", "audience", "num", "fmt", "counts", "followers", "y_like", "y_reach", "y_likes", "y_views", "rates", "has_emb", "sid"):
        o = np.load(old / f"{k}.npy")
        assert o.dtype == arr[k].dtype and o.shape[1:] == arr[k].shape[1:], (k, o.dtype, arr[k].dtype, o.shape, arr[k].shape)
        res[k] = np.concatenate([o, arr[k]])[order]
    res["cat"] = np.concatenate([np.concatenate([np.load(old / "cat.npy"), reg_old[:, None].astype(np.int32)], axis=1), arr["cat"]])[order]
    a_sorted = a_all[order]
    change = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    ai = np.cumsum(change) - 1
    res["author_idx"], res["a_start"], res["a_end"] = ai.astype(np.int32), starts[ai].astype(np.int64), ends[ai].astype(np.int64)
    # Neighbour indices: old ones re-pointed to the merged order, new ones looked up by post id.
    nbr = np.full((n, K), -1, np.int32)
    cos = np.zeros((n, K), np.float32)
    o_idx = np.load(old / "nbr_idx.npy")
    nbr[inv[:n_old]] = np.where(o_idx >= 0, inv[np.clip(o_idx, 0, None)], -1).astype(np.int32)
    cos[inv[:n_old]] = np.load(old / "nbr_cos.npy")
    where = {int(p): int(inv[i]) for i, p in enumerate(p_all)}
    for j, (ids_, cs_) in enumerate(zip(srt["nbr_post_id"].to_list(), srt["nbr_cos"].to_list())):
        if ids_:
            r = inv[n_old + j]
            nbr[r, :len(ids_[:K])] = [where[int(q)] for q in ids_[:K]]
            cos[r, :len(cs_[:K])] = cs_[:K]
    res["nbr_idx"], res["nbr_cos"] = nbr, cos
    mm = np.lib.format.open_memmap(out / "emb.npy", mode="w+", dtype=np.float16, shape=(n, e_old.shape[1]))
    for i in range(0, n, 100_000):
        o = order[i:i + 100_000]
        blk = np.empty((len(o), e_old.shape[1]), np.float16)
        blk[o < n_old] = e_old[o[o < n_old]].astype(np.float16)
        blk[o >= n_old] = arr["emb"][o[o >= n_old] - n_old]
        mm[i:i + len(o)] = blk
    mm.flush()
    del mm
    for k, v in res.items():
        np.save(out / f"{k}.npy", v)
    texts = pl.read_parquet(old / "texts.parquet")["text"].to_list() + features.render_texts(srt)
    pl.DataFrame({"post_id": res["post_id"], "text": [texts[i] for i in order]}).write_parquet(out / "texts.parquet", compression="zstd")
    split, counts = res["split"], res["counts"]
    meta = {**meta0, "n": n, "n_authors": int(len(starts)), "cat_names": stats["cat_names"], "cat_cards": stats["cat_cards"],
            "splits": {k: int((split == v).sum()) for k, v in features.SPLITS.items()}, "has_views": int((counts[:, 4] >= 0).sum()),
            "has_likes": int((counts[:, 0] >= 0).sum()), "audience_rows": int(res["audience"].sum()),
            "expand": {"existing_rows": n_old, "new_rows": n_new, "statistics_from": str(old), "new_rows_file": args.new_rows}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    # Row information for evaluation slices.
    info = pl.concat([ro.select("post_id", "author_id", pl.lit(False).alias("is_new"), pl.lit(None, dtype=pl.String).alias("expand_src")),
                      srt.select("post_id", "author_id", pl.lit(True).alias("is_new"), "expand_src")])[order.tolist()]
    info.write_parquet(out / "row_info.parquet")
    # Existing rows must be identical to the old table.
    back = inv[:n_old]
    for k in ("num", "counts", "split", "w_row", "audience", "y_likes", "rates", "sid", "fmt", "followers"):
        assert np.array_equal(res[k][back], np.load(old / f"{k}.npy"), equal_nan=True), k
    assert np.array_equal(res["cat"][back][:, :-1], np.load(old / "cat.npy"))
    assert np.array_equal(np.load(out / "emb.npy", mmap_mode="r")[back[:5000]], np.load(old / "emb.npy", mmap_mode="r")[:5000])
    miss = np.isnan(res["num"]).sum()
    log.info("table written: %d rows, %d accounts, splits %s, views on %d, NaN inputs %d; existing rows identical to the old table", n, len(starts), meta["splits"], meta["has_views"], int(miss))
    newm = ~np.isin(res["post_id"], old_ids)
    const = [nm for i, nm in enumerate(meta["num_names"]) if res["num"][~newm, i].std() < 1e-6 and res["num"][newm, i].std() > 1e-6]
    log.info("inputs constant in existing rows but varying in new rows (should be none): %s", const)
    print(json.dumps({"n": n, "new": int(newm.sum()), "new_with_neighbour": int((nbr[newm] >= 0).any(1).sum()), "old_with_neighbour": int((nbr[~newm] >= 0).any(1).sum()),
                      "regime_counts": {dict(features.CAT_SPEC)["src_regime"][i]: int(c) for i, c in enumerate(np.bincount(res["cat"][:, -1], minlength=4))}, "meta": {k: meta[k] for k in ("splits", "has_views", "audience_rows", "cat_names", "cat_cards", "n_authors")}}, indent=1))


if __name__ == "__main__":
    main()
