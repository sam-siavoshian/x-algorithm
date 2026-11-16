#!/usr/bin/env python3
"""Label noise on the pair task: when one account posts the same text twice, how far apart are the like counts?

    python scripts/evidence/label_noise.py --posts <work>/filtered/posts.parquet --out noise.json [--scored <dir>/scored_joined.parquet]

Identical pairs: same account, same normalised text (lower case, links removed, white space collapsed), at least
--min-words words, same media state (both with or both without media) and same kind, at most --max-gap-days apart.
Nothing the model sees at serving time differs between the two sides (both drafts are scored at the same moment
against the same history), so the spread of their like counts is noise no model can predict. With --scored (the candidate pairs
written by select_pairs.py), reworded pairs are reported next to them by cosine band.

The ceiling by size of gap: the observed log gap of a reworded pair is modelled as effect + noise, with the noise
drawn from the identical pairs and the effect from a Laplace distribution whose variance is what the reworded
pairs have beyond the noise. The ceiling at gap g is the share of pairs with observed gap of at least g whose
observed winner is the side with the positive effect: what a model that knew every effect exactly would score.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

NORM = (pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "").str.replace_all(r"\s+", " ").str.strip_chars())
GAPS = (1.25, 2.0, 3.0, 5.0)


def spread(d: np.ndarray) -> dict:
    a = np.abs(d)
    return {"pairs": int(len(a)), **{f"within_{g}x": float((a < np.log(g)).mean()) for g in GAPS}, "median_ratio": float(np.exp(np.median(a))),
            "sd_log": float(d.std())}


def ceiling(noise: np.ndarray, obs: np.ndarray, seed: int = 0, n: int = 2_000_000) -> dict:
    rng = np.random.default_rng(seed)
    var_e = max(obs.var() - noise.var(), 0.0)
    out = {"sd_log_observed": float(obs.std()), "sd_log_noise": float(noise.std()), "sd_log_effect": float(np.sqrt(var_e))}
    N = rng.choice(noise, n) * rng.choice([-1.0, 1.0], n)
    for name, E in (("laplace", rng.laplace(0, np.sqrt(var_e / 2), n)), ("normal", rng.normal(0, np.sqrt(var_e), n))):
        O = E + N
        out[name] = {f"{g}x": {"share_of_pairs": float((np.abs(O) >= np.log(g)).mean()), "ceiling": float(((E * O) > 0)[np.abs(O) >= np.log(g)].mean())} for g in (1.0,) + GAPS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scored", default=None)
    ap.add_argument("--before", default="2026-04-23")
    ap.add_argument("--max-gap-days", type=float, default=14.0)
    ap.add_argument("--min-words", type=int, default=6)
    args = ap.parse_args()
    from xviral.util.core import date_to_ms

    cut = date_to_ms(args.before)
    d = pl.scan_parquet(args.posts).select("post_id", "author_id", "created_at_ms", "text", "kind", "likes", (pl.col("media").list.len() > 0).alias("m"),
                                           "quoted_post_id").collect()
    d = d.with_columns(NORM.alias("nt")).with_columns(pl.col("nt").str.count_matches(r"\S+").alias("nw"))
    k = d.filter(pl.col("nw") >= args.min_words).with_columns(pl.col("nt").hash().alias("h"))
    grp = k.group_by("author_id", "h").len().filter(pl.col("len").is_between(2, 6))
    k = k.join(grp.select("author_id", "h"), on=["author_id", "h"], how="semi")
    p = k.join(k, on=["author_id", "h"], suffix="_b").filter((pl.col("created_at_ms") < pl.col("created_at_ms_b")) & (pl.col("nt") == pl.col("nt_b")))
    p = p.filter(((pl.col("created_at_ms_b") - pl.col("created_at_ms")) <= args.max_gap_days * 86_400_000) & (pl.col("m") == pl.col("m_b")) & (pl.col("kind") == pl.col("kind_b"))
                 & (pl.col("quoted_post_id").fill_null(0) == pl.col("quoted_post_id_b").fill_null(0)))
    p = p.with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"), ((pl.col("created_at_ms_b") - pl.col("created_at_ms")) / 3_600_000).alias("gap_h"),
                       pl.max_horizontal("likes", "likes_b").alias("mx"))
    p.select("post_id", "post_id_b", "author_id", "d", "gap_h", "likes", "likes_b", "nw").write_parquet(args.out.replace(".json", "_identical.parquet"))
    res: dict = {"definition": __doc__.split("\n\n")[2]}
    res["identical"] = {"all": spread(p["d"].to_numpy()) | {"accounts": p["author_id"].n_unique(), "later_wins": float((p.filter(pl.col("d") != 0)["d"] < 0).mean())},
                        "before_cutoff": spread(p.filter(pl.col("created_at_ms_b") < cut)["d"].to_numpy()),
                        "after_cutoff": spread(p.filter(pl.col("created_at_ms") >= cut)["d"].to_numpy()),
                        "within_48h": spread(p.filter(pl.col("gap_h") <= 48)["d"].to_numpy()),
                        "2_to_14_days": spread(p.filter(pl.col("gap_h") > 48)["d"].to_numpy()),
                        "one_pair_per_account": spread(p.sample(fraction=1.0, shuffle=True, seed=0).unique("author_id", keep="first")["d"].to_numpy()),
                        "by_larger_like_count": {f"{lo}-{hi}": spread(p.filter(pl.col("mx").is_between(lo, hi))["d"].to_numpy())
                                                 for lo, hi in ((0, 9), (10, 99), (100, 999), (1000, 10 ** 9))}}
    # Weight accounts equally-ish: cap each account at 20 pairs for the noise sample used in the ceiling.
    cap = p.sample(fraction=1.0, shuffle=True, seed=0).with_columns(pl.int_range(pl.len()).over("author_id").alias("r")).filter(pl.col("r") < 20)
    res["identical"]["capped_20_per_account"] = spread(cap["d"].to_numpy()) | {"accounts": cap["author_id"].n_unique()}
    noise = cap["d"].to_numpy()
    if args.scored:
        # scored_joined.parquet of select_pairs.py: candidate pairs with cosine, like counts, media state and kind of both sides.
        s = pl.read_parquet(args.scored).with_columns(((pl.col("likes") + 1).log() - (pl.col("likes_b") + 1).log()).alias("d"))
        s = s.filter((pl.col("nw") >= args.min_words) & (pl.col("nw_b") >= args.min_words))
        same_fmt = (pl.col("m") == pl.col("m_b")) & (pl.col("kind") == pl.col("kind_b"))
        one = lambda f: f.sample(fraction=1.0, shuffle=True, seed=0).with_columns(pl.int_range(pl.len()).over("author_id").alias("r")).filter(pl.col("r") < 20)  # noqa: E731
        res["reworded_by_cosine_capped_20_per_account"] = {
            f"{lo:.2f}-{hi:.2f}": spread(one(s.filter(pl.col("cos").is_between(lo, hi, closed="left")))["d"].to_numpy())
            | {"same_media_and_kind": spread(one(s.filter(pl.col("cos").is_between(lo, hi, closed="left") & same_fmt))["d"].to_numpy())}
            for lo, hi in ((0.97, 1.01), (0.93, 0.97), (0.90, 0.93), (0.85, 0.90), (0.80, 0.85), (0.70, 0.80), (0.60, 0.70))}
        gap48 = (pl.col("created_at_ms_b") - pl.col("created_at_ms")) <= 48 * 3_600_000
        for name, f in (("link_or_quote_pairs_48h", pl.col("tac").is_not_null() & gap48), ("reworded_cosine_0.90_plus", pl.col("cos") >= 0.90),
                        ("reworded_cosine_0.85_to_0.90", pl.col("cos").is_between(0.85, 0.90, closed="left"))):
            obs = one(s.filter(f))["d"].to_numpy()
            res[f"ceiling_{name}"] = {"pairs": int(len(obs)), "observed": spread(obs)} | ceiling(noise, obs)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
