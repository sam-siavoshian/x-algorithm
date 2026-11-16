"""Company and founder accounts, from display name and posting patterns (author_bio is empty in this corpus).

company  organisation voice: under 10% of posts use I/my, the display name is not a person's name, at least 3 own
         product posts (typed and tech-worded) making up 6%+ of posts, 30%+ of posts tech-worded.
founder  an individual who announces their own work: at least 3 typed, tech-worded, first-person posts (4%+ of posts).
Excluded from both: news/digest/fan names, breaking-news aggregators, token-promotion accounts.
"""
import sys, polars as pl, gender_guesser.detector as gg
sys.path.insert(0, "scripts/launch")
from labels import clean
D = gg.Detector(case_sensitive=False)
def person_name(name):
    import re
    toks = re.findall(r"[A-Za-zÀ-ÿ'’-]+", (name or "").split("|")[0].split("(")[0].split("@")[0])
    if not (2 <= len(toks) <= 3): return False
    if any(t.lower() in ("ai", "labs", "lab", "inc", "hq", "app", "developers", "devs", "research", "institute", "network", "protocol", "cloud", "studio", "team", "official", "foundation", "capital", "ventures", "community", "the") for t in toks): return False
    return D.get_gender(toks[0]) != "unknown"

d = pl.scan_parquet("data/work/filtered/posts.parquet").select("post_id", "author_id", "text").collect()
lc = clean()
x = d.with_columns(lc.alias("lc")).group_by("author_id").agg(
    pl.col("lc").str.contains(r"(^\s*(🚨|breaking|just in|headlines|news:)|\bbreaking:)").mean().alias("agg_share"),
    pl.col("lc").str.contains(r"\b(airdrop|giveaway|presale|whitelist|\$[a-z]{2,6}\b|memecoin|100x|gm gm|tge)\b").mean().alias("token_share"))
a0 = pl.read_parquet("data/launch/acct_stats0.parquet")
t = pl.read_parquet("data/launch/ptype_raw.parquet")
own = (pl.col("ptype_raw") != "other") & ((pl.col("ptype_raw") != "demo") | pl.col("has_video")) & pl.col("techy") & ~pl.col("reprint")
g = t.group_by("author_id").agg(own.sum().alias("typed"), (own & pl.col("fp_head")).sum().alias("typed_self"), pl.col("techy").mean().alias("tech_share"))
a = a0.join(g, on="author_id").join(x, on="author_id").with_columns((pl.col("typed") / pl.col("n")).alias("typed_share"), (pl.col("typed_self") / pl.col("n")).alias("typed_self_share"),
        pl.col("name").map_elements(person_name, return_dtype=pl.Boolean).alias("person"))
NEWSY = r"(?i)\b(news|daily|times|tv|radio|party|council|press|journal|magazine|post|herald|tribune|headlines|breaking|updates?|alerts?|insider|report|reports|digest|watch|tracker|newsletter|newswire|podcast|fm|media|memes?|fan|fans|archive|bot|dept|ministry|department)\b"
a = a.with_columns(pl.col("name").str.contains(NEWSY).fill_null(False).alias("newsy"))
ok = (pl.col("n") >= 15) & (pl.col("tech_share") >= 0.30) & ~pl.col("newsy") & (pl.col("agg_share") < 0.10) & (pl.col("token_share") < 0.20)
company = ok & (pl.col("i") < 0.10) & ~pl.col("person") & (pl.col("typed") >= 3) & (pl.col("typed_share") >= 0.06) & ((pl.col("we") >= 0.12) | (pl.col("typed_share") >= 0.15))
founder = ok & (pl.col("typed_self") >= 2) & (pl.col("typed_self_share") >= 0.03)
a = a.with_columns(pl.when(company).then(pl.lit("company")).when(founder).then(pl.lit("founder")).otherwise(pl.lit("other")).alias("akind"))
print(a.group_by("akind").agg(pl.len(), pl.col("n").sum(), pl.col("typed").sum(), pl.col("f").is_not_null().sum().alias("has_f"), (pl.col("nv") > 0).sum().alias("has_views")))
a.write_parquet("data/launch/accounts.parquet")
