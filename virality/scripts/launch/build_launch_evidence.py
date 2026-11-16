#!/usr/bin/env python3
"""Launch evidence table, in the shape of xviral/data/edit_evidence.json: one entry per kind of change, oriented so that
win_rate is the win rate of the version the entry describes.

    python scripts/launch/build_launch_evidence.py     # reads data/launch/*.json, writes xviral/data/launch_evidence.json

An entry is `supported` only when it survives every check, decided by a fixed rule:
  three tests per set of posts, each with a 95% interval from resampling accounts:
    pairs     win rate in same-account pairs of launch posts, posting order balanced   (interval excludes 50%)
    ratio     likes ratio with / without inside accounts that have both                (interval excludes 1)
    adjusted  the same ratio with every other attribute held fixed                     (interval excludes 1; where defined)
  two sets: MAIN = launch-type posts of company and founder accounts; STRICT = the subset whose opening has an
  announcement verb directly followed by a product name (removes posts the labelling rules got wrong).
  survives   every test passes in MAIN or in STRICT, at least one passes in the other, same direction in both
  main_only  every test passes in MAIN; same direction in STRICT but no test passes there
  wider_only every test passes only when the launch posts of every account are pooled (ALL)
  leaning    same direction everywhere, some test passes somewhere
  none       nothing passes, or the direction flips
  too_thin   under 200 pairs or under 25 accounts in MAIN
Each entry also carries a time split (launch posts of every account, before and from 2026-04-23, estimated separately)
and a grade: A survives and is confirmed in both periods; A- main_only and confirmed in both periods; B survives, not
separately confirmed in one period; B- main_only; C wider_only. Entries without a grade carry no advice.
"""
import json, os
E = json.load(open("data/launch/effects.json"))
MAIN, STRICT, ALL = "launch_cf", "launch_strict_cf", "launch_all"
ADJ = {"video_vs_none": "video", "image_vs_none": "image", "link_in_post_vs_none": "link_in_post", "points_to_reply_vs_none": "points_to_reply"}
FLIP = {  # measured attribute -> (key, description) of the version to prefer when the measured attribute loses
    "open_problem": ("announcement_not_problem_first", "the first line does not open with a problem statement or a question (it opens with the announcement or the product)"),
    "open_question": ("no_question_in_first_line", "the first line contains no question"),
    "is_quote": ("original_not_quote", "the announcement is an original post, not a quote post"),
    "link_in_post_vs_none": ("no_link_in_post", "no external link in the launch post, against a link in the post"),
    "tags": ("no_tags", "the post tags no other account"),
    "hashtag": ("no_hashtag", "the post has no hashtag"),
    "prev2h": ("two_hours_after_previous", "posted at least 2 hours after the account's previous post"),
    "h18_23": ("h13_17_not_h18_23", "posted 13:00 to 17:59 UTC, against 18:00 to 23:59 UTC"),
    "weekend": ("weekday", "posted Monday to Friday (UTC)"),
    "points_to_reply_vs_none": ("no_pointer_to_reply", "no 'link in reply / below' pointer, against one"),
    "long600": ("under_280_not_600_plus", "under 280 characters, against 600 or more"),
    "h00_12": ("h13_17_not_h00_12", "posted 13:00 to 17:59 UTC, against 00:00 to 12:59 UTC"),
    "thread_marker": ("no_thread_marker", "not marked as the start of a thread"),
    "open_result": ("no_result_in_first_line", "the first line states no result"),
    "long280": ("under_280", "the text is under 280 characters"),
    "tue_thu": ("mon_or_fri", "posted Monday or Friday (UTC), against Tuesday to Thursday"),
    "image_vs_none": ("no_media_not_image", "no media, against an image"),
    "media_vs_none": ("no_media", "no media, against an image or video"),
}
NOT_A_LEVER = {"next2h": "a launch that is doing well gets followed up; the follow-up does not cause the result", "near2h": "mixes the previous-post effect with follow-ups to launches that are already doing well"}

def tests(scope, k):
    e = E["scopes"][scope]["effects"][k]["likes"]; p, r = e["pairs"], e["within_account"]; a = E["scopes"][scope]["adjusted_likes"].get(ADJ.get(k, k))
    t = {"pairs": p if p and p.get("win_rate") is not None else None, "ratio": r, "adjusted": a}
    d = 0 if not t["pairs"] else (1 if t["pairs"]["win_rate"] > 0.5 else -1)
    ok = {}
    if t["pairs"]: ok["pairs"] = (t["pairs"]["lo"] > 0.5) if d > 0 else (t["pairs"]["hi"] < 0.5)
    if r: ok["ratio"] = (r["ratio_lo"] > 1) if d > 0 else (r["ratio_hi"] < 1)
    if a: ok["adjusted"] = (a["ratio_lo"] > 1) if d > 0 else (a["ratio_hi"] < 1)
    return t, d, ok

out = {}
for k, e in E["scopes"][MAIN]["effects"].items():
    tm, dm_, okm = tests(MAIN, k); ts, ds, oks = tests(STRICT, k); ta, da, oka = tests(ALL, k)
    allm, alls, alla = bool(okm) and all(okm.values()), bool(oks) and all(oks.values()), bool(oka) and all(oka.values())
    thin = not tm["pairs"] or tm["pairs"]["pairs"] < 200 or tm["pairs"]["accounts"] < 25
    same = dm_ == ds and dm_ != 0
    if k in NOT_A_LEVER: status = "not_a_lever"
    elif thin: status = "too_thin"
    elif same and ((allm and any(oks.values())) or (alls and any(okm.values()))): status = "survives"
    elif same and allm: status = "main_only"
    elif alla and da == dm_ and (ds == dm_ or not ts["pairs"]): status = "wider_only"
    elif same and da == dm_ and (any(okm.values()) or any(oks.values()) or any(oka.values())): status = "leaning"
    else: status = "none"
    flip = dm_ < 0 and k in FLIP
    key, desc = (FLIP[k] if flip else (k, e["description"]))
    def orient(t):
        o = {}
        if t["pairs"]:
            p = t["pairs"]; w, lo, hi = (1 - p["win_rate"], 1 - p["hi"], 1 - p["lo"]) if flip else (p["win_rate"], p["lo"], p["hi"])
            o.update(win_rate=round(w, 4), lo=round(lo, 4), hi=round(hi, 4), pairs=p["pairs"], accounts=p["accounts"])
        for nm, src in (("likes_ratio", t["ratio"]), ("adjusted_likes_ratio", t["adjusted"])):
            if src:
                r, lo, hi = (round(1 / src["ratio"], 3), round(1 / src["ratio_hi"], 3), round(1 / src["ratio_lo"], 3)) if flip else (src["ratio"], src["ratio_lo"], src["ratio_hi"])
                o[nm] = {"ratio": r, "lo": lo, "hi": hi} | ({"accounts_with_both": src["accounts_with_both"], "posts_with": src["posts_without" if flip else "posts_with"], "posts_without": src["posts_with" if flip else "posts_without"]} if "accounts_with_both" in src else {})
        return o
    m = orient(tm)
    entry = {"description": desc, **{x: m.get(x) for x in ("win_rate", "lo", "hi", "pairs", "accounts")}, "supported": status == "survives", "status": status, "outcome": "likes relative to the account's trailing median",
             "design": "same account, launch-type posts of company and founder accounts, posting order balanced", "likes_ratio": m.get("likes_ratio"), "adjusted_likes_ratio": m.get("adjusted_likes_ratio"),
             "tests_passed_main": okm, "strict_subset": orient(ts) | {"tests_passed": oks}, "all_accounts": orient(ta) | {"tests_passed": oka}, "measured_as": k, "flipped": flip}
    for sc, nm in (("launch_all_under_100k", "accounts_under_100k_followers"), ("launch_all_100k_plus", "accounts_100k_plus_followers")):
        if sc in E["scopes"]:
            tt, _, okk = tests(sc, k); entry[nm] = orient(tt) | {"tests_passed": okk}
    per = {}
    for sc, nm in (("launch_all_before_cutoff", "before_2026_04_23"), ("launch_all_after_cutoff", "from_2026_04_23")):
        tt, dd, okk = tests(sc, k); per[nm] = orient(tt) | {"tests_passed": okk, "same_direction": dd == dm_, "confirmed": dd == dm_ and any(okk.values())}
    entry["periods"] = per | {"design": "launch posts of every account, split at the model's test cutoff; each period estimated on its own"}
    entry["replicated_in_both_periods"] = bool(per["before_2026_04_23"]["confirmed"] and per["from_2026_04_23"]["confirmed"])
    entry["grade"] = ("A" if status == "survives" and entry["replicated_in_both_periods"] else "A-" if status == "main_only" and entry["replicated_in_both_periods"] else "B" if status == "survives" else
                      "B-" if status == "main_only" else "C" if status == "wider_only" else None)
    v = E["scopes"][ALL]["effects"][k].get("views")
    entry["views"] = "not measurable: view counts exist only in the 2026 view corpus, which stores text without line breaks and does not record video" if v is None else {
        "pairs": v["pairs"], "within_account": v["within_account"], "note": "all accounts; too few posts with view counts to state an effect"}
    if k in NOT_A_LEVER: entry["note"] = NOT_A_LEVER[k]
    out[key] = entry
# spacing entry in the shape advise.py reads
r = E["scopes"][ALL]["effects"]["prev2h"]["likes"]["within_account"]
out["spacing"] = {"description": "a launch post less than 2 hours after the account's previous post", "ratio_when_closer": r["ratio"], "ratio_lo": r["ratio_lo"], "ratio_hi": r["ratio_hi"], "accounts": r["accounts_with_both"],
                  "posts_closer": r["posts_with"], "posts_not": r["posts_without"], "outcome": "likes relative to the account's trailing median", "design": "within account, launch-type posts of every account", "supported": bool(r["ratio_hi"] < 1)}
out["_types"] = {"description": "how each type of own announcement does against the account's own trailing median (previous 30 posts); intervals from resampling accounts", **E["types"]}
for nm in ("second_account", "founder_vs_company", "same_topic", "bench_launch", "model_check", "counts"):
    if os.path.exists(f"data/launch/{nm}.json"): out["_" + nm] = json.load(open(f"data/launch/{nm}.json"))
out["_scopes"] = {k: {x: v[x] for x in ("description", "posts", "accounts", "posts_community_archive", "accounts_community_archive", "posts_with_views")} for k, v in E["scopes"].items()}
out["_meta"] = {"built": "2026-11-16", "corpus": "work/filtered/posts.parquet, 4.48M posts (Community Archive, likes only; 2026 view corpus, views)", "rule": __doc__.split("An entry is")[1].strip(),
                "baseline": "median likes of the account's previous 30 posts (at least 8); posts captured under 3 days after posting dropped", "min_like_ratio_pairs": 1.25, "max_gap_days_pairs": 365, "max_pairs_per_account": 50,
                "label_precision_read_by_hand": {"funding": "17 of 20", "hiring": "15 of 20", "launch": "19 of 25", "open_source": "about 14 of 17 after the final rule", "feature": "16 of 25", "milestone": "17 of 25", "demo": "about 8 of 20, not used for effects"},
                "account_precision_read_by_hand": {"company": "about 3 in 4 are product companies or labs", "founder": "about 6 in 10 are people who build and announce their own products; the rest are tech commentators"},
                "not_observable": ["replies (a link placed in a reply, the body of a thread): only the head post's wording is seen", "bios (author_bio is empty in the corpus)", "the audience's time zone: day and hour are UTC",
                                   "whether a second account belongs to the same team"]}
os.makedirs("xviral/data", exist_ok=True)
json.dump(out, open("xviral/data/launch_evidence.json", "w"), indent=1)
json.dump(out, open("data/launch/launch_evidence.json", "w"), indent=1)
print(f"{'entry':<32}{'grade':<4}{'status':<12}{'win':>6} {'interval':>12} {'pairs':>6} {'acc':>4}  {'ratio':>5} {'interval':>11}  {'adj':>5} {'interval':>11} | strict win, ratio")
for k, v in out.items():
    if k.startswith("_") or k == "spacing": continue
    lr, ar, s = v["likes_ratio"], v["adjusted_likes_ratio"], v["strict_subset"]
    pr = v["periods"]; fmtp = lambda x: (f"{100*x['win_rate']:.0f}% ({100*x['lo']:.0f}-{100*x['hi']:.0f}) n={x['pairs']}" + ("*" if x["confirmed"] else "")) if x.get("win_rate") is not None else "-"
    print(f"{k:<32}{v['grade'] or '':<4}{v['status']:<12}" + (f"{100*v['win_rate']:5.1f}% {100*v['lo']:5.1f}-{100*v['hi']:<5.1f} {v['pairs']:>6} {v['accounts']:>4}" if v["win_rate"] is not None else " " * 31)
          + (f"  {lr['ratio']:5.2f} {lr['lo']:5.2f}-{lr['hi']:<5.2f}" if lr else " " * 19) + (f"  {ar['ratio']:5.2f} {ar['lo']:5.2f}-{ar['hi']:<5.2f}" if ar else " " * 19)
          + (f" | {100*s['win_rate']:5.1f}% ({100*s['lo']:.0f}-{100*s['hi']:.0f}) n={s['pairs']}" if s.get("win_rate") is not None else " | -") + (f", {s['likes_ratio']['ratio']:.2f} ({s['likes_ratio']['lo']:.2f}-{s['likes_ratio']['hi']:.2f})" if s.get("likes_ratio") else "")
          + f" | before {fmtp(pr['before_2026_04_23'])} after {fmtp(pr['from_2026_04_23'])}")
