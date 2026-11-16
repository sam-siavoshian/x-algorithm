"""Company and founder X handles from public launch lists: the YC directory and Product Hunt top products.

  datahiveai/ycombinator-companies (HF, CC BY-NC 2.0; scraped from the public YC directory, Sept 2025): company and founder Twitter URLs.
  pranavjrat/top_producthunt_products (HF, GPL-3.0; scraped from producthunt.com product pages, June 2026): the product's Twitter link.
Writes data/track_data/lists/list_handles.parquet: handle, role (company | founder), list, name, company, batch_or_launch, one_liner, website.
"""
import json, re, os
import polars as pl

R = "data/raw/lists/"; OUT = "data/track_data/lists"; os.makedirs(OUT, exist_ok=True)
BAD = {"home", "share", "intent", "search", "hashtag", "i", "explore", "settings", "login", "signup", "twitter", "x", "status", "about", "privacy", "tos", "messages", "notifications", "compose", "en", "www"}

def handle(url):
    if not url: return None
    m = re.search(r"(?i)(?:twitter\.com|x\.com)/(?:#!/)?@?([A-Za-z0-9_]{1,15})(?:[/?#]|$)", str(url).strip())
    h = m.group(1).lower() if m else None
    return None if (h is None or h in BAD) else h

rows = []
yc = pl.read_csv(R + "datahiveai__ycombinator-companies/yc-companies.csv", infer_schema_length=0)
fcols = sorted({c.rsplit(" ", 2)[0] for c in yc.columns if re.match(r"Founder \d+ Twitter URL", c)})
for r in yc.iter_rows(named=True):
    base = {"list": "yc_directory", "company": r["Company Name"], "batch_or_launch": r["Batch"], "one_liner": r["One Liner"], "website": r["Website"], "status": r["Status"], "industry": r["Industry"]}
    h = handle(r.get("Twitter URL"))
    if h: rows.append({**base, "handle": h, "role": "company", "name": r["Company Name"]})
    for c in [c for c in yc.columns if re.match(r"Founder \d+ Twitter URL", c)]:
        h2 = handle(r.get(c)); n = c.split(" Twitter")[0]
        if h2 and h2 != h: rows.append({**base, "handle": h2, "role": "founder", "name": r.get(n + " Name") or r.get(n + " Full Name")})
ph = pl.read_csv(R + "pranavjrat__top_producthunt_products/dataset.csv", infer_schema_length=0)
for r in ph.iter_rows(named=True):
    try: s = json.loads(r["social_links"] or "{}")
    except Exception: s = {}
    h = handle(s.get("twitter") or s.get("x"))
    if h: rows.append({"list": "producthunt_top", "company": r["title"], "batch_or_launch": r["launch_date"], "one_liner": r["tagline"], "website": r["website_link"], "status": None, "industry": r["categories"],
                       "handle": h, "role": "company", "name": r["title"]})
df = pl.DataFrame(rows, infer_schema_length=None)
df.write_parquet(OUT + "/list_handles.parquet")
print(yc.height, "YC companies;", ph.height, "PH products")
print(df.group_by("list", "role").agg(pl.len(), pl.col("handle").n_unique()).sort("list", "role"))
print("unique handles", df["handle"].n_unique())
c = pl.read_parquet("data/track_data/corpus_handles_filtered.parquet"); t = pl.read_parquet("data/track_data/corpus_handles_train.parquet")
u = df.unique("handle")
print("in filtered corpus", u.join(c, left_on="handle", right_on="h", how="semi").group_by("role").len().to_dicts(), "in training rows", u.join(t, left_on="handle", right_on="h", how="semi").group_by("role").len().to_dicts())
