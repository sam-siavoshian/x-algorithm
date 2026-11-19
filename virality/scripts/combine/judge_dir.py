#!/usr/bin/env python3
"""A working folder for scripts/judge_product/{fit_rule,test_eval}.py in which the surrogate's gap (`sg`) is another
model set's, so the blend rule can be refitted for it from the stored judge answers (no new judge calls).

    python scripts/combine/judge_dir.py --out data/combine/llm_base3/                         # the current product's gap, unchanged (a check)
    python scripts/combine/judge_dir.py --out data/combine/llm_new/ --members a,b [--sets dev,launchpre]

Judge answers: tracks/llm_judge/out and tracks/judge_product/out, the n30 runs of gpt-6-luna and gpt-6.1-sol, merged per
file name (a later answer for the same pair and order replaces an earlier one, as the loader does).
--sets limits which pair tables are written: with dev,launchpre only, the test pairs cannot be read by mistake.
"""
import argparse, glob, os, shutil
import numpy as np, polars as pl
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--members", default=""); ap.add_argument("--pairs", default="data/combine/pairs")
ap.add_argument("--launch-members", default="", help="other members for the launch sets (default: the same members)")
ap.add_argument("--sets", default="dev,launchpre,bench,launch"); a = ap.parse_args()
os.makedirs(a.out + "out", exist_ok=True)
for s in a.sets.split(","):
    P = pl.read_parquet(f"data/tracks/judge_product/jp_pairs_{s}.parquet")
    if a.members:
        G = pl.read_parquet(f"{a.pairs}/{s}.parquet")
        assert (G["post_id"] == P["post_id"]).all() and (G["post_id_b"] == P["post_id_b"]).all()
        mem = (a.launch_members or a.members) if s in ("launchpre", "launch") else a.members
        g = np.mean([G[f"g_{m}"].to_numpy() for m in mem.split(",")], axis=0)
        assert not np.isnan(g).any(), s
        P = P.with_columns(pl.Series("sg", g), pl.col("sg").alias("sg_base3"))
    P.write_parquet(f"{a.out}jp_pairs_{s}.parquet")
    shutil.copy(f"data/tracks/judge_product/jp_textsim_{s}.parquet", f"{a.out}jp_textsim_{s}.parquet")
    for model, mode in (("gpt-6-luna", "logprob"), ("gpt-6.1-sol", "json-low")):
        name = f"{s}__n30__{model}__{mode}.jsonl"
        with open(a.out + "out/" + name, "w") as w:
            for d in ("data/tracks/llm_judge/out/", "data/tracks/judge_product/out/"):
                if os.path.exists(d + name):
                    w.write(open(d + name).read())
print("written", a.out, sorted(os.listdir(a.out)), len(os.listdir(a.out + "out")))
