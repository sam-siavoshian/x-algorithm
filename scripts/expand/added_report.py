#!/usr/bin/env python3
"""What the expansion added: rows by source, account kind, follower tier, post type (xviral/text/launch_labels.py) and
how many carry views; the same cuts for the existing rows next to it. Writes data/expand/added.json and added.md."""
import json
import polars as pl
from xviral.text import launch_labels as ll
D = "data/"
cols = ["post_id", "author_id", "author_handle", "created_at_ms", "text", "source", "likes", "views", "followers_at_post", "follower_tier", "split", "in_audience", "has_video"]
new = pl.read_parquet(D + "expand/new_rows.parquet", columns=cols + ["expand_src", "acct_kind", "kind_from", "in_brand_set", "acct_in_original"])
old = pl.read_parquet(D + "work/dataset/rows.parquet", columns=cols)
sa = pl.read_parquet(D + "tracks/data/startup_accounts.parquet").select("handle", pl.col("kind").alias("sa_kind"), pl.col("kind_from").alias("sa_from"))
lt = pl.read_parquet(D + "tracks/launch/accounts_labeled.parquet").select("author_id", pl.col("akind").alias("lt_kind"))
old = old.with_columns(pl.col("author_handle").str.to_lowercase().alias("handle")).join(sa, on="handle", how="left").join(lt, on="author_id", how="left") \
    .with_columns(pl.coalesce("lt_kind", "sa_kind").fill_null("other").alias("acct_kind"), pl.col("source").alias("expand_src"))
def typed(d):
    d = ll.add_types(d.with_columns(pl.col("acct_kind").alias("akind"), pl.col("has_video").fill_null(False)))
    return d.with_columns(pl.when(ll.own_expr()).then(pl.col("ptype_raw")).otherwise(pl.lit("not an own announcement")).alias("ptype"),
                          pl.col("follower_tier").fill_null("unknown"), pl.when(pl.col("followers_at_post").is_null() | (pl.col("followers_at_post") <= 0)).then(pl.lit("unknown")).otherwise(pl.col("follower_tier")).alias("tier"))
new, old = typed(new), typed(old)
pl.concat([new.select("post_id", "ptype", pl.lit(True).alias("is_new")), old.select("post_id", "ptype", pl.lit(False).alias("is_new"))]).filter(pl.col("ptype") != "not an own announcement").write_parquet(D + "expand/typed_posts.parquet")
def cut(d, by):
    g = d.group_by(by).agg(pl.len().alias("posts"), pl.col("author_id").n_unique().alias("accounts"), pl.col("views").is_not_null().sum().alias("with_views"),
                           (pl.col("split") == "train").sum().alias("train"), (pl.col("split") == "val").sum().alias("val"), (pl.col("split") == "test").sum().alias("test"),
                           (pl.col("ptype").is_in(["launch", "open_source"])).sum().alias("launch_posts"), ((pl.col("ptype").is_in(["launch", "open_source"])) & pl.col("views").is_not_null()).sum().alias("launch_with_views"))
    return g.sort(by).to_dicts()
out = {"new": {}, "existing": {}}
for name, d in (("new", new), ("existing", old)):
    out[name]["total"] = cut(d.with_columns(pl.lit("all").alias("all")), "all")
    for by in ("expand_src", "acct_kind", "tier", "ptype"):
        out[name]["by_" + by] = cut(d, by)
    out[name]["by_source_and_kind"] = cut(d.with_columns((pl.col("expand_src") + " | " + pl.col("acct_kind")).alias("k")), "k")
    out[name]["by_kind_and_tier"] = cut(d.filter(pl.col("acct_kind").is_in(["company", "founder"])).with_columns((pl.col("acct_kind") + " | " + pl.col("tier")).alias("k")), "k")
    out[name]["launch_posts_by_kind_and_tier"] = cut(d.filter(pl.col("ptype").is_in(["launch", "open_source"]) & pl.col("acct_kind").is_in(["company", "founder"])).with_columns((pl.col("acct_kind") + " | " + pl.col("tier")).alias("k")), "k")
    out[name]["by_year"] = cut(d.with_columns(pl.from_epoch("created_at_ms", time_unit="ms").dt.year().cast(pl.String).alias("y")), "y")
out["new"]["by_evidence"] = cut(new.with_columns(pl.col("kind_from").fill_null("?")), "kind_from")
out["new"]["accounts_already_in_training"] = {"accounts": new.filter(pl.col("acct_in_original"))["author_id"].n_unique(), "posts": new.filter(pl.col("acct_in_original")).height,
                                              "by_source": cut(new.filter(pl.col("acct_in_original")), "expand_src")}
out["new"]["posts_per_account"] = {k: float(v) for k, v in zip(("p10", "p50", "p90", "max"), [new.group_by("author_id").len()["len"].quantile(q) for q in (.1, .5, .9, 1.0)])}
json.dump(out, open(D + "expand/added.json", "w"), indent=1, default=str)
def md(rows, key):
    h = f"| {key} | Posts | Accounts | With views | Train | Val | Test | Launch-type own posts | of them with views |\n|---|---|---|---|---|---|---|---|---|\n"
    return h + "\n".join(f"| {r[list(r)[0]]} | {r['posts']:,} | {r['accounts']:,} | {r['with_views']:,} | {r['train']:,} | {r['val']:,} | {r['test']:,} | {r['launch_posts']:,} | {r['launch_with_views']:,} |" for r in rows) + "\n"
txt = ""
for name in ("new", "existing"):
    txt += f"\n## {name} rows\n\n" + md(out[name]["total"], "All")
    for by, lab in (("by_expand_src", "Source"), ("by_acct_kind", "Account kind"), ("by_tier", "Follower tier"), ("by_ptype", "Post type"), ("by_source_and_kind", "Source and kind"), ("by_kind_and_tier", "Kind and tier"), ("launch_posts_by_kind_and_tier", "Launch posts: kind and tier"), ("by_year", "Year")):
        txt += "\n" + md(out[name][by], lab)
txt += "\n## new rows by the evidence for the account kind\n\n" + md(out["new"]["by_evidence"], "Evidence")
open(D + "expand/added.md", "w").write(txt)
print(txt[:6000]); print(out["new"]["accounts_already_in_training"], out["new"]["posts_per_account"])
