#!/usr/bin/env python3
"""The bundle's serving path gives the gaps the offline evaluation used: benchmark units scored through
service.load_bundle (its DraftScorer, feature cache of the benchmark, no embedding call) against the gaps stored in
data/combine/pairs/bench.parquet, and the launch members' output against a scorer built from their checkpoints alone.

    python scripts/combine/parity.py --bundle data/combine/bundle [--n 40]
"""
import argparse, json, math, pickle
import numpy as np, polars as pl
from xviral.app import service
from xviral.serve import DraftScorer

ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default="data/combine/bundle"); ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]


class NoCall:
    local, provider, dim = False, "none", 1024

    def encode(self, texts, batch_size=None):
        raise RuntimeError("a text was not in the feature cache")

    def describe(self):
        return {"provider": "none", "model": "none", "dimensions": 1024, "endpoint": None}

    def take_usage(self):
        return {}


svc = service.load_bundle(a.bundle, embedder=NoCall())
sc = svc.scorer
cfg = json.load(open(a.bundle + "/bundle.json"))
U = pl.read_parquet("data/tracks/selective/units_bench.parquet").sort("unit_id").with_row_index("_r").filter(pl.col("_r") % 4 == 0).drop("_r").head(a.n)
d = pl.scan_parquet("data/work/filtered/posts.parquet").select(COLS).filter(pl.col("author_id").is_in(U["author_id"].unique().to_list())).collect()
by_author = {k: g.sort("created_at_ms") for (k,), g in d.group_by("author_id")}
by_id = {r["post_id"]: r for r in d.filter(pl.col("post_id").is_in(U["post_ids"].explode().to_list())).iter_rows(named=True)}
cache = pickle.load(open("data/combine/caches/te3l/bench_0.pkl", "rb"))
P = pl.read_parquet("data/combine/pairs/bench.parquet")
names = [c.split("/")[-1][:-3] for c in cfg["checkpoints"]]
lnames = [c.split("/")[-1][:-3] for c in cfg["comparison_members"]["launch"]["checkpoints"]]
alone = DraftScorer([a.bundle + "/" + c for c in cfg["comparison_members"]["launch"]["checkpoints"]], encoder=NoCall(), sid_codebook=a.bundle + "/" + cfg["sid_codebook"])
lg = lambda m: math.log10(m["likes_q50"] + 1)      # noqa: E731
worst = {"forecast_members_vs_offline": 0.0, "launch_members_vs_offline": 0.0, "launch_members_vs_own_scorer": 0.0}
n = 0
for r in U.iter_rows(named=True):
    g = by_author[r["author_id"]]; xs = [by_id[p] for p in r["post_ids"]]
    hist = g.filter(pl.col("created_at_ms") < r["when_ms"]).tail(56)
    acct = {"author_id": r["author_id"], "handle": xs[0]["author_handle"], "display_name": xs[0]["author_display_name"], "followers": xs[0]["author_followers"], "following": xs[0]["author_following"]}
    drafts = [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]} for x in xs]
    sc.feature_cache = cache; alone.feature_cache = cache
    reps = sc.score(hist, drafts, acct, r["when_ms"]); ref = alone.score(hist, drafts, acct, r["when_ms"])
    row = P.filter((pl.col("post_id") == r["post_ids"][0]) & (pl.col("post_id_b") == r["post_ids"][1]))
    sign = 1.0
    if not row.height:
        row = P.filter((pl.col("post_id") == r["post_ids"][1]) & (pl.col("post_id_b") == r["post_ids"][0])); sign = -1.0
    row = row.row(0, named=True)
    gap = np.mean([lg(x) - lg(y) for x, y in zip(reps[0]["members"], reps[1]["members"])])
    lgap = np.mean([lg(x) - lg(y) for x, y in zip(reps[0]["kind_reports"]["launch"]["members"], reps[1]["kind_reports"]["launch"]["members"])])
    rgap = np.mean([lg(x) - lg(y) for x, y in zip(ref[0]["members"], ref[1]["members"])])
    worst["forecast_members_vs_offline"] = max(worst["forecast_members_vs_offline"], abs(gap - sign * np.mean([row[f"g_{m}"] for m in names])))
    worst["launch_members_vs_offline"] = max(worst["launch_members_vs_offline"], abs(lgap - sign * np.mean([row[f"g_{m}"] for m in lnames])))
    worst["launch_members_vs_own_scorer"] = max(worst["launch_members_vs_own_scorer"], abs(lgap - rgap))
    n += 1
print(json.dumps({"units": n, "largest_difference_log10": worst, "type_correction_loaded": bool(sc.type_correction), "selective": svc.advisor.selective.rule["ensemble"],
                  "launch_members": len(sc.kind_scorers["launch"].model.models)}))
