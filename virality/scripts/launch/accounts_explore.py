import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(120); pl.Config.set_tbl_width_chars(300)
d=pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id","author_id","author_handle","author_display_name","author_followers","created_at_ms","text","kind","likes","views","source","urls","snapshot_at_ms").collect()
t=pl.col("text").fill_null("")
low=t.str.to_lowercase()
d=d.with_columns(
  low.str.contains(r"\b(we|we're|we've|our|us)\b").alias("we"),
  low.str.contains(r"\b(i|i'm|i've|my|me)\b").alias("i"),
  low.str.contains(r"\b(introducing|announcing|now available|is now live|is live|we're excited|we are excited|excited to (announce|share|launch)|launching|just (launched|shipped|released)|new in |rolling out|now (generally )?available|is here)\b").alias("ann"),
)
a=d.group_by("author_id").agg(pl.len().alias("n"), pl.col("author_handle").last().alias("handle"), pl.col("author_display_name").last().alias("name"), pl.col("author_followers").max().alias("f"),
   pl.col("we").mean().alias("we"), pl.col("i").mean().alias("i"), pl.col("ann").mean().alias("ann"), (pl.col("kind")=="quote").mean().alias("q"), pl.col("source").n_unique().alias("ns"), pl.col("views").is_not_null().sum().alias("nv"))
a.write_parquet("data/launch/acct_stats0.parquet")
print(a.select(pl.len(), (pl.col("handle")=="").sum(), pl.col("n").quantile(0.5).alias("n50"), pl.col("n").quantile(0.9).alias("n90"), pl.col("n").max().alias("nmax")))
print(a.select(pl.col("we").quantile(0.5).alias("we50"), pl.col("we").quantile(0.9).alias("we90"), pl.col("i").quantile(0.5).alias("i50"), pl.col("i").quantile(0.1).alias("i10"), pl.col("ann").quantile(0.9).alias("ann90"), pl.col("ann").quantile(0.99).alias("ann99")))
b=a.filter(pl.col("n")>=15)
print("high ann"); print(b.sort("ann",descending=True).head(60).select("handle","name","f","n","we","i","ann"))
print("we>i*2, random"); print(b.filter((pl.col("we")>0.25)&(pl.col("i")<0.08)).sample(60,seed=2).select("handle","name","f","n","we","i","ann"))
print(b.filter((pl.col("we")>0.25)&(pl.col("i")<0.08)).height, b.filter((pl.col("we")>0.15)&(pl.col("i")<0.1)).height, b.height)
