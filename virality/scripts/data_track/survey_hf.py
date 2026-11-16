"""Enumerate Hugging Face datasets of X/Twitter posts by keyword and read their column names from the datasets-server.

Writes data/track_data/survey/hf_candidates.parquet: id, downloads, last_modified, licence, columns, and flags for
view / bio / reply / follower columns. No rows are downloaded here.
"""
import json, os, re, sys, time, concurrent.futures as cf
import requests, polars as pl
from huggingface_hub import HfApi

OUT = "data/track_data/survey"; os.makedirs(OUT, exist_ok=True)
TERMS = ["twitter", "tweets", "tweet", "x posts", "x.com", "x_dataset", "twitter user", "twitter profiles", "twitter bio", "twitter threads", "twitter replies",
         "tweets views", "viewCount", "impression", "nitter", "twscrape", "snscrape", "twitterapi", "apify twitter", "x scrape", "x-posts", "xposts", "tweet engagement",
         "founder tweets", "startup tweets", "ai tweets", "tech twitter", "crypto twitter", "elon tweets", "influencer tweets", "kol tweets", "build in public", "indie hackers",
         "product hunt", "producthunt", "show hn", "hacker news launch", "y combinator", "yc companies", "ycombinator", "startup", "startups twitter", "crunchbase", "founders",
         "twitter 2024", "twitter 2025", "twitter 2026", "tweets 2025", "tweets 2026", "x 2025", "grok", "viral tweets", "tweet virality", "tweet popularity", "social media engagement",
         "twitter timelines", "user timelines", "twitter accounts", "kaito", "bittensor x", "masa twitter", "twitter archive", "community archive", "tweet thread", "twitter conversation"]
api = HfApi()
seen = {}
for t in TERMS:
    try:
        for d in api.list_datasets(search=t, limit=400, full=False):
            seen.setdefault(d.id, {"id": d.id, "downloads": d.downloads or 0, "last_modified": str(d.last_modified or ""), "tags": d.tags or [], "terms": []})["terms"].append(t)
    except Exception as e:
        print("search fail", t, e, file=sys.stderr)
print(len(seen), "datasets", file=sys.stderr)
NAME = re.compile(r"(?i)(tw(ee|i)t|x[_-]?(post|data|scrape)|nitter|producthunt|product[_-]hunt|show[_-]?hn|hacker|ycomb|yc[_-]|startup|founder|crunchbase|kol|x_dataset|indie)")
cands = [v for k, v in seen.items() if NAME.search(k)]
print(len(cands), "name-matching", file=sys.stderr)
S = requests.Session(); H = {"Authorization": "Bearer " + os.environ.get("HF_TOKEN", "hf_proxyinjected")}

def info(v):
    for _ in range(3):
        try:
            r = S.get("https://datasets-server.huggingface.co/info", params={"dataset": v["id"]}, headers=H, timeout=30)
            if r.status_code == 429: time.sleep(5); continue
            if r.status_code != 200: v["err"] = str(r.status_code); return v
            di = r.json().get("dataset_info", {}); cols = set(); rows = 0
            for cfg in di.values():
                cols |= set((cfg.get("features") or {}).keys())
                for sp in (cfg.get("splits") or {}).values(): rows += sp.get("num_examples") or 0
            v["columns"] = sorted(cols); v["rows"] = rows; return v
        except Exception as e:
            v["err"] = type(e).__name__
    return v

with cf.ThreadPoolExecutor(6) as ex: cands = list(ex.map(info, cands))
def flag(cols, pat): return any(re.search(pat, c, re.I) for c in cols or [])
rows = []
for v in cands:
    c = v.get("columns") or []
    lic = [t.split(":", 1)[1] for t in v["tags"] if t.startswith("license:")]
    rows.append({"id": v["id"], "downloads": v["downloads"], "last_modified": v["last_modified"][:10], "licence": ",".join(lic), "rows": v.get("rows"), "err": v.get("err"),
                 "has_views": flag(c, r"view|impression|see_count"), "has_likes": flag(c, r"like|fav"), "has_bio": flag(c, r"bio|description|profile"),
                 "has_reply": flag(c, r"reply|conversation|thread|parent"), "has_followers": flag(c, r"follower"), "columns": ", ".join(c)[:1500], "terms": ",".join(v["terms"])[:200]})
df = pl.DataFrame(rows, infer_schema_length=None); df.write_parquet(f"{OUT}/hf_candidates.parquet")
json.dump(sorted(seen), open(f"{OUT}/hf_all_ids.json", "w"))
print(df.height, df["has_views"].sum(), df["has_bio"].sum())
