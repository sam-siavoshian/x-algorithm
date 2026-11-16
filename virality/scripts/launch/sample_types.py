import sys, polars as pl
sys.path.insert(0, "scripts/launch")
from labels import ORDER, own_expr
seed=int(sys.argv[1]); n=int(sys.argv[2]); types=sys.argv[3].split(",") if len(sys.argv)>3 else ORDER
a=pl.read_parquet("data/launch/accounts.parquet").filter(pl.col("akind")!="other").select("author_id","akind")
t=pl.read_parquet("data/launch/ptype_raw.parquet").join(a,on="author_id").filter(own_expr())
print(t.group_by("akind","ptype_raw").len().sort("akind","len"))
d=pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id","author_handle","text").collect().join(t,on="post_id")
for k in types:
    print("=====",k)
    for r in d.filter(pl.col("ptype_raw")==k).sample(min(n,d.filter(pl.col("ptype_raw")==k).height),seed=seed).iter_rows(named=True):
        print(f"[{r['akind'][0]}:{r['author_handle'][:12]}]", r["text"][:150].replace("\n"," / "))
