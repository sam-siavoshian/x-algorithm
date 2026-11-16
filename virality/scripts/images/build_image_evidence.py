#!/usr/bin/env python3
"""Image evidence table, in the shape of xviral/data/launch_evidence.json: one entry per comparison, oriented so that
win_rate is the win rate of the version the entry describes.

    python scripts/images/build_image_evidence.py     # reads data/images/effects.json (+ labels_precision.json, model json), writes xviral/data/image_evidence.json

An entry is `supported` only when it survives every check, decided by a fixed rule (set before the effects were run):
  three tests per set of posts, each with a 95% interval from resampling accounts:
    pairs     win rate in same-account pairs, posting order balanced                  (interval excludes 50%)
    ratio     likes ratio with / without inside accounts that have both               (interval excludes 1)
    adjusted  the same ratio with the other image types and the text attributes held fixed (interval excludes 1; where defined)
  sets: MAIN = every post of company and founder accounts; STRICT = the same with image posts restricted to confident
  type labels (zero-shot margin in the top half of the class); ALL = every post of every account with an own announcement.
  survives   every test passes in MAIN or in STRICT, at least one passes in the other, same direction in both
  main_only  every test passes in MAIN; same direction in STRICT but no test passes there
  wider_only every test passes only in ALL
  leaning    same direction everywhere, some test passes somewhere
  none       nothing passes, or the direction flips
  too_thin   under 100 pairs or under 20 accounts in MAIN
Each entry also carries the launch-post and announcement subsets, follower tiers and a time split (MAIN, before and from
2026-04-23). Grade: A survives and is confirmed in both periods; B survives; B- main_only; C wider_only. Entries without
a grade carry no advice.
"""
import glob, json, os
E = {}
for f in sorted(glob.glob("data/images/effects_*.json")): E.update(json.load(open(f))["scopes"])     # effects.py run in parts
json.dump({"scopes": E}, open("data/images/effects.json", "w"))
MAIN, STRICT, ALL = "posts_cf", "posts_cf__confident", "posts_all"
def adj_key(k):
    if k.endswith("_vs_none") and k != "image_vs_none": return ("adjusted_types_vs_none", "is_" + k[:-8])
    if k.endswith("_top_vs_bottom"): return ("adjusted_quality_image_posts", k[:-14] + "_hi")
    return (None, None)
def tests(scope, k):
    if scope not in E or k not in E[scope]["effects"] or "likes" not in E[scope]["effects"][k]: return {"pairs": None, "ratio": None, "adjusted": None}, 0, {}
    e = E[scope]["effects"][k]["likes"]; p, r = e["pairs"], e["within_account"]
    blk, nm = adj_key(k); a = E[scope].get(blk, {}).get(nm) if blk else None
    t = {"pairs": p if p and p.get("win_rate") is not None else None, "ratio": r, "adjusted": a}
    d = 0 if not t["pairs"] else (1 if t["pairs"]["win_rate"] > 0.5 else -1)
    ok = {}
    if t["pairs"] and d: ok["pairs"] = (t["pairs"]["lo"] > 0.5) if d > 0 else (t["pairs"]["hi"] < 0.5)
    if r and d: ok["ratio"] = (r["ratio_lo"] > 1) if d > 0 else (r["ratio_hi"] < 1)
    if a and d: ok["adjusted"] = (a["ratio_lo"] > 1) if d > 0 else (a["ratio_hi"] < 1)
    return t, d, ok
def orient(t, flip):
    o = {}
    if t["pairs"]:
        p = t["pairs"]; w, lo, hi = (1 - p["win_rate"], 1 - p["hi"], 1 - p["lo"]) if flip else (p["win_rate"], p["lo"], p["hi"])
        o.update(win_rate=round(w, 4), lo=round(lo, 4), hi=round(hi, 4), pairs=p["pairs"], accounts=p["accounts"])
    for nm, src in (("likes_ratio", t["ratio"]), ("adjusted_likes_ratio", t["adjusted"])):
        if src:
            r, lo, hi = (round(1 / src["ratio"], 3), round(1 / src["ratio_hi"], 3), round(1 / src["ratio_lo"], 3)) if flip else (src["ratio"], src["ratio_lo"], src["ratio_hi"])
            o[nm] = {"ratio": r, "lo": lo, "hi": hi} | ({"accounts_with_both": src["accounts_with_both"], "posts_with": src["posts_without" if flip else "posts_with"], "posts_without": src["posts_with" if flip else "posts_without"]} if "accounts_with_both" in src else {})
    return o
out = {}
for k, e in E[MAIN]["effects"].items():
    tm, dm_, okm = tests(MAIN, k); ts, ds, oks = tests(STRICT, k); ta, da, oka = tests(ALL, k)
    allm, alls, alla = bool(okm) and all(okm.values()), bool(oks) and all(oks.values()), bool(oka) and all(oka.values())
    thin = not tm["pairs"] or tm["pairs"]["pairs"] < 100 or tm["pairs"]["accounts"] < 20
    same = dm_ == ds and dm_ != 0
    if thin and not alla: status = "too_thin"
    elif same and ((allm and any(oks.values())) or (alls and any(okm.values()))): status = "survives"
    elif same and allm: status = "main_only"
    elif alla and da == dm_ and (ds == dm_ or not ts["pairs"]): status = "wider_only"
    elif same and da == dm_ and (any(okm.values()) or any(oks.values()) or any(oka.values())): status = "leaning"
    else: status = "none"
    flip = dm_ < 0
    m = orient(tm, flip)
    desc = e["description"] if not flip else "NOT: " + e["description"]
    entry = {"description": desc, **{x: m.get(x) for x in ("win_rate", "lo", "hi", "pairs", "accounts")}, "supported": status == "survives", "status": status,
             "outcome": "likes relative to the account's trailing median (previous 30 posts)", "design": "same account, every post of company and founder accounts, posting order balanced",
             "likes_ratio": m.get("likes_ratio"), "adjusted_likes_ratio": m.get("adjusted_likes_ratio"), "tests_passed_main": okm,
             "strict_subset": orient(ts, flip) | {"tests_passed": oks}, "all_accounts": orient(ta, flip) | {"tests_passed": oka}, "measured_as": k, "flipped": flip}
    for sc, nm in (("ordinary_cf", "ordinary_posts_company_founder"), ("ordinary_all", "ordinary_posts_all_accounts"), ("launch_cf", "launch_posts_company_founder"),("announce_cf", "announcements_company_founder"), ("launch_all", "launch_posts_all_accounts"), ("announce_all", "announcements_all_accounts"),
                   ("posts_cf_under_100k", "company_founder_under_100k"), ("posts_cf_100k_plus", "company_founder_100k_plus"), ("posts_all_under_100k", "all_accounts_under_100k"), ("posts_all_100k_plus", "all_accounts_100k_plus")):
        tt, dd, okk = tests(sc, k); entry[nm] = orient(tt, flip) | {"tests_passed": okk, "same_direction": dd == dm_}
    per = {}
    for sc, nm in (("posts_cf_before_cutoff", "before_2026_04_23"), ("posts_cf_after_cutoff", "from_2026_04_23")):
        tt, dd, okk = tests(sc, k); per[nm] = orient(tt, flip) | {"tests_passed": okk, "same_direction": dd == dm_, "confirmed": dd == dm_ and any(okk.values())}
    entry["periods"] = per | {"design": "every post of company and founder accounts (MAIN), split at the model's test cutoff; each period estimated on its own"}
    entry["replicated_in_both_periods"] = bool(per["before_2026_04_23"]["confirmed"] and per["from_2026_04_23"]["confirmed"])
    entry["grade"] = ("A" if status == "survives" and entry["replicated_in_both_periods"] else "B" if status == "survives" else "B-" if status == "main_only" else "C" if status == "wider_only" else None)
    v = E.get(ALL, {"effects": {k: {}}})["effects"][k].get("views")
    entry["views"] = "too few posts with view counts" if v is None else {"pairs": v["pairs"], "within_account": v["within_account"], "note": "all accounts, 2026 view corpus"}
    out[k] = entry
for nm, f in (("_label_precision", "data/images/labels_precision.json"), ("_ai_generated", "data/images/ai_validation.json"), ("_model", "data/images/model_results.json")):
    if os.path.exists(f): out[nm] = json.load(open(f))
out["_scopes"] = {k: {x: v[x] for x in ("description", "posts", "accounts", "image_posts", "image_posts_community_archive")} for k, v in E.items()}
out["_meta"] = {"built": "2026-11-16", "corpus": "work/filtered/posts.parquet; images fetched from pbs.twimg.com (first photo of a post, 680 px rendition)",
                "rule": __doc__.split("An entry is")[1].strip(), "baseline": "median likes of the account's previous 30 posts (at least 8); posts captured under 3 days after posting dropped",
                "method": "scripts/launch/effects.py estimators, unchanged", "posts_without_a_fetched_image": "posts with a photo whose image was not fetched are left out of every comparison"}
os.makedirs("xviral/data", exist_ok=True)
json.dump(out, open("xviral/data/image_evidence.json", "w"), indent=1); json.dump(out, open("data/images/image_evidence.json", "w"), indent=1)
print(f"{'entry':<40}{'grade':<4}{'status':<12}{'win':>6} {'interval':>12} {'pairs':>6} {'acc':>4}  {'ratio':>5} {'interval':>11}  {'adj':>5} {'interval':>11} | launch_cf | before/after")
for k, v in out.items():
    if k.startswith("_"): continue
    lr, ar, L = v["likes_ratio"], v["adjusted_likes_ratio"], v["launch_posts_company_founder"]; pr = v["periods"]
    fp = lambda x: (f"{100*x['win_rate']:.0f}% ({100*x['lo']:.0f}-{100*x['hi']:.0f}) n={x['pairs']}" + ("*" if x.get("confirmed") else "")) if x.get("win_rate") is not None else "-"
    print(f"{k:<40}{v['grade'] or '':<4}{v['status']:<12}" + (f"{100*v['win_rate']:5.1f}% {100*v['lo']:5.1f}-{100*v['hi']:<5.1f} {v['pairs']:>6} {v['accounts']:>4}" if v["win_rate"] is not None else " " * 31)
          + (f"  {lr['ratio']:5.2f} {lr['lo']:5.2f}-{lr['hi']:<5.2f}" if lr else " " * 19) + (f"  {ar['ratio']:5.2f} {ar['lo']:5.2f}-{ar['hi']:<5.2f}" if ar else " " * 19)
          + f" | {fp(L)} | {fp(pr['before_2026_04_23'])} / {fp(pr['from_2026_04_23'])}")
