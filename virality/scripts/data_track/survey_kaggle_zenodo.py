"""Search Kaggle (public list API, no login) and Zenodo for X/Twitter post datasets; write the hit lists."""
import requests, json, os, time, sys
OUT = "data/track_data/survey"; os.makedirs(OUT, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "dataset-survey/1.0"
KT = ["twitter views", "tweets viewCount", "x posts 2025", "x posts 2026", "tweets 2025", "tweets 2026", "tweets 2024", "twitter founders", "startup tweets", "ai tweets", "tech twitter",
      "twitter profiles bio", "twitter replies", "twitter threads", "product hunt", "show hn", "y combinator twitter", "openai tweets", "sam altman tweets", "elon musk tweets 2025",
      "twitter influencers", "x.com posts", "tweets impressions", "twitter engagement", "build in public", "indie hackers", "llm tweets", "chatgpt tweets", "deepseek tweets", "claude tweets",
      "grok tweets", "nvidia tweets", "crypto twitter kol", "ai agents tweets", "saas twitter", "twitter company accounts", "brand tweets", "tech ceo tweets", "vc tweets", "startup launch"]
k = {}
for t in KT:
    for page in (1, 2):
        try:
            r = S.get("https://www.kaggle.com/api/v1/datasets/list", params={"search": t, "page": page, "sortBy": "updated"}, timeout=30)
            if r.status_code != 200: print("kaggle", t, r.status_code, file=sys.stderr); break
            js = r.json()
            if not js: break
            for d in js:
                k.setdefault(d["ref"], {"ref": d["ref"], "title": d.get("title"), "size": d.get("totalBytes"), "updated": (d.get("lastUpdated") or "")[:10], "licence": d.get("licenseName"),
                                        "downloads": d.get("downloadCount"), "subtitle": d.get("subtitle"), "terms": []})["terms"].append(t)
        except Exception as e:
            print("kaggle fail", t, e, file=sys.stderr); break
        time.sleep(0.4)
json.dump(list(k.values()), open(f"{OUT}/kaggle_hits.json", "w"), indent=0)
print("kaggle", len(k))
z = {}
for t in ["twitter views dataset", "tweets viewCount", "X posts 2024", "X posts 2025", "tweets startup", "twitter brand accounts", "tweets artificial intelligence 2024", "twitterapi.io", "twscrape", "tweets founders", "tweets 2025 dataset", "product hunt", "tweets impressions"]:
    try:
        r = S.get("https://zenodo.org/api/records", params={"q": t, "size": 25, "type": "dataset", "sort": "mostrecent"}, timeout=40)
        for h in r.json().get("hits", {}).get("hits", []):
            z.setdefault(h["id"], {"id": h["id"], "title": h["metadata"].get("title"), "date": h["metadata"].get("publication_date"), "licence": (h["metadata"].get("license") or {}).get("id"),
                                   "files": [(f["key"], f["size"]) for f in h.get("files", [])][:6], "terms": []})["terms"].append(t)
    except Exception as e:
        print("zenodo fail", t, e, file=sys.stderr)
    time.sleep(1)
json.dump(list(z.values()), open(f"{OUT}/zenodo_hits.json", "w"), indent=0)
print("zenodo", len(z))
