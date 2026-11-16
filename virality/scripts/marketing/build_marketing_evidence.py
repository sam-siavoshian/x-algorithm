#!/usr/bin/env python3
"""Apply the fixed rule to data/marketing/effects_mk.json and write the track's result files:

    data/marketing/marketing_evidence.json   same shape as the launch track's launch_evidence.json
    data/marketing/hypotheses.json           outside claim -> source grade -> our test result
    data/marketing/model_traits.json         traits the model should receive as inputs

    python scripts/marketing/build_marketing_evidence.py
The rule is the one in scripts/launch/build_launch_evidence.py (see effects_mk.py docstring), unchanged.
"""
import json, sys
sys.path.insert(0, "scripts/marketing")
from hypotheses_def import HYPOTHESES, MODEL_TRAITS
E = json.load(open("data/marketing/effects_mk.json"))
LE = json.load(open("data/tracks/launch/launch_evidence.json"))
ADJ_KEY = {"waitlist_vs_available_now": "cta_waitlist", "pt_weekend": "weekend"}
FLIP = {"offer_credits_discount": ("no_credits_or_discount_offer", "the launch post carries no credits, discount, deal or prize offer"),
        "launch_in_prev_week": ("no_launch_in_previous_week", "the account made no other launch-type post in the 7 days before"),
        "open_excited": ("no_stated_excitement_opener", "the first line does not open with stated excitement (excited / thrilled / proud to announce)"),
        "price_stated": ("no_price_stated", "the post states no price"), "product_hunt": ("no_product_hunt_mention", "the post does not mention Product Hunt / Hacker News"),
        "multi_image": ("one_image_not_several", "one image, against two or more"), "cta_engagement_bait": ("no_engagement_bait", "the post does not trade a reply / repost for access")}

def tests(scope, k, outcome="likes"):
    s = E["scopes"][scope]; e = s["effects"][k][outcome]; p, r = e["pairs"], e["within_account"]
    a = s["adjusted_likes"].get(ADJ_KEY.get(k, k)) if outcome == "likes" and E["kinds"][k] in ("text",) and k in (list(s["adjusted_likes"]) + list(ADJ_KEY)) else None
    if k in ("teaser_72h", "launch_in_prev_week") and outcome == "likes": a = s["adjusted_likes"].get(k)
    t = {"pairs": p if p and p.get("win_rate") is not None else None, "ratio": r, "adjusted": a}
    d = 0 if not t["pairs"] else (1 if t["pairs"]["win_rate"] > 0.5 else -1)
    ok = {}
    if t["pairs"]: ok["pairs"] = (t["pairs"]["lo"] > 0.5) if d > 0 else (t["pairs"]["hi"] < 0.5)
    if r: ok["ratio"] = (r["ratio_lo"] > 1) if d > 0 else (r["ratio_hi"] < 1)
    if a: ok["adjusted"] = (a["ratio_lo"] > 1) if d > 0 else (a["ratio_hi"] < 1)
    return t, d, ok

def thin(t): return not t["pairs"] or t["pairs"]["pairs"] < 200 or t["pairs"]["accounts"] < 25

def status_of(k):
    """(status, level, direction). level 'cf' = company and founder accounts, 'pooled' = every account."""
    for level, (MAIN, STRICT, ALL) in (("cf", ("launch_cf", "launch_strict_cf", "launch_all")), ("pooled", ("launch_all", "launch_strict_all", "launch_all"))):
        tm, dm, okm = tests(MAIN, k); ts, ds, oks = tests(STRICT, k); ta, da, oka = tests(ALL, k)
        if thin(tm):
            if level == "cf": continue
            return "too_thin", level, dm
        allm, alls, alla = bool(okm) and all(okm.values()), bool(oks) and all(oks.values()), bool(oka) and all(oka.values())
        same = dm == ds and dm != 0
        if same and ((allm and any(oks.values())) or (alls and any(okm.values()))): st = "survives"
        elif same and allm: st = "main_only"
        elif level == "cf" and alla and da == dm and (ds == dm or not ts["pairs"]): st = "wider_only"
        elif same and da == dm and (any(okm.values()) or any(oks.values()) or any(oka.values())): st = "leaning"
        else: st = "none"
        return st, level, dm

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

ev, res = {}, {}
for k in E["kinds"]:
    st, level, d = status_of(k)
    circ = E["kinds"][k] == "circular"
    MAIN, STRICT = ("launch_cf", "launch_strict_cf") if level == "cf" else ("launch_all", "launch_strict_all")
    flip = d < 0 and k in FLIP
    key, desc = FLIP[k] if flip else (k, E["descriptions"][k])
    tm, _, okm = tests(MAIN, k); m = orient(tm, flip)
    status = "not_a_lever" if circ else st
    grade = None
    if status in ("survives", "main_only", "wider_only"):
        pb, pa = (orient(tests(s, k)[0], flip) for s in ("launch_all_before_cutoff", "launch_all_after_cutoff"))
        both = all(x.get("lo") is not None and x["lo"] > 0.5 for x in (pb, pa))
        grade = "C" if (level == "pooled" or status == "wider_only") else ("A" if status == "survives" and both else "A-" if both else "B" if status == "survives" else "B-")
    entry = {"description": desc, **{x: m.get(x) for x in ("win_rate", "lo", "hi", "pairs", "accounts")}, "supported": status in ("survives",) and not circ, "status": status, "grade": grade,
             "level": "company and founder accounts" if level == "cf" else "every account pooled (too thin among company and founder accounts)",
             "outcome": "likes relative to the account's trailing median",
             "design": "same account, launch-type posts, posting order balanced" + ("; NOT a lever: the trait is partly a consequence of the outcome" if circ else ""),
             "likes_ratio": m.get("likes_ratio"), "adjusted_likes_ratio": m.get("adjusted_likes_ratio"), "tests_passed_main": okm,
             "strict_subset": orient(tests(STRICT, k)[0], flip) | {"tests_passed": tests(STRICT, k)[2]}, "all_accounts": orient(tests("launch_all", k)[0], flip),
             "reposts": orient(tests(MAIN, k, "reposts")[0], flip),
             "periods": {"before_2026_04_23": orient(tests("launch_all_before_cutoff", k)[0], flip), "from_2026_04_23": orient(tests("launch_all_after_cutoff", k)[0], flip)},
             "tiers": {"under_100k": orient(tests("launch_all_under_100k", k)[0], flip), "100k_plus": orient(tests("launch_all_100k_plus", k)[0], flip)},
             "model_releases_only": orient(tests("launch_ai_model_all", k)[0], flip),
             "posts_with_trait_main": E["scopes"][MAIN]["effects"][k]["posts_with"], "measured_as": k, "flipped": flip}
    ev[key] = entry; res[k] = (key, entry, d)
try: ev["_context_extra"] = json.load(open("data/marketing/context_extra.json"))
except FileNotFoundError: pass
ev["_meta"] = {"built": "2026-11-16", "corpus": "launch-type posts (launch, open-source release) in the Community Archive part of the 4.48M-post corpus; main set 1,980 posts of 530 company and founder accounts; pooled set 3,654 posts of 1,657 accounts",
               "rule": "identical to the launch track (scripts/launch/build_launch_evidence.py); when the main set has under 200 pairs or 25 accounts the rule is applied to every account pooled and the result is at most grade C",
               "baseline": "median likes of the account's previous 30 posts (at least 8)", "min_like_ratio_pairs": 1.25, "max_gap_days_pairs": 365, "max_pairs_per_account": 50,
               "traits_tested": len(E["kinds"]), "multiple_testing": "about 45 new comparisons on top of the launch track's 30; one or two passing by chance would not be surprising",
               "trait_precision_read_by_hand": "7 samples per text trait read once: clean for cta_try, offer_free, offer_open_source, benchmark_claim, compares_named, superlative_first, availability_now, cta_waitlist (research previews count); noisy (about half correct) for offer_credits_discount, names_customer, backstory, coming_soon, backed_by; teaser posts about half are teasers of the launch that followed",
               "not_observable": ["video length (duration_ms is empty for all 852 video launch posts)", "replies (link in reply, thread body, replies from well-known accounts, replying to commenters)", "reposts by other accounts", "paid promotion and clipping", "what an image shows (chart, face, screenshot) - see the media track", "team membership (no bios)"]}
json.dump(ev, open("data/marketing/marketing_evidence.json", "w"), indent=1)

def num(o):
    if not o or o.get("win_rate") is None: return None
    x = {"win_rate": o["win_rate"], "interval": [o["lo"], o["hi"]], "pairs": o["pairs"], "accounts": o["accounts"]}
    if o.get("likes_ratio"): x["likes_ratio"] = [o["likes_ratio"]["ratio"], o["likes_ratio"]["lo"], o["likes_ratio"]["hi"]]; x["posts_with"] = o["likes_ratio"].get("posts_with"); x["posts_without"] = o["likes_ratio"].get("posts_without")
    if o.get("adjusted_likes_ratio"): x["adjusted_likes_ratio"] = [o["adjusted_likes_ratio"]["ratio"], o["adjusted_likes_ratio"]["lo"], o["adjusted_likes_ratio"]["hi"]]
    return x

H = []
for h in HYPOTHESES:
    o = dict(h); k = h.get("trait"); src = h.get("from", "marketing")
    if k and src == "marketing":
        key, e, d = res[k]; claimed = h["direction"]
        # numbers are reported for the trait as measured (not flipped), so the direction of the claim is readable
        t = tests("launch_cf" if e["level"].startswith("company") else "launch_all", k)[0]
        o["result"] = num(orient(t, False)); o["status"] = e["status"]; o["grade_ours"] = e["grade"]; o["level"] = e["level"]; o["posts_with_trait"] = e["posts_with_trait_main"]
        o["strict_subset"] = num(orient(tests("launch_strict_cf" if e["level"].startswith("company") else "launch_strict_all", k)[0], False)); o["all_accounts"] = num(orient(tests("launch_all", k)[0], False))
        if e["status"] == "not_a_lever": v = "untestable"
        elif e["status"] == "too_thin": v = "untestable"
        elif e["status"] in ("survives", "main_only", "wider_only"): v = "confirmed" if d == claimed else "contradicted"
        else: v = "not_confirmed"
        o["verdict"] = h.get("verdict_override", v)
        o["evidence_key"] = key
    elif k and src == "launch":
        e = LE[k]; o["result"] = num(e); o["status"] = e.get("status"); o["grade_ours"] = e.get("grade"); o["level"] = "company and founder accounts (launch track)"; o["evidence_key"] = "launch_evidence.json:" + k
        o["verdict"] = h["verdict_override"]
    else:
        o["result"] = None; o["verdict"] = h.get("verdict_override", "untestable")
    H.append(o)
json.dump({"_how_to_read": "verdict is our corpus against the outside claim: confirmed, contradicted, not_confirmed (tested, nothing passes the rule), untestable. result = win rate of the version WITH the trait in same-account, order-balanced pairs of launch posts (50% = no effect), 95% interval from resampling accounts, likes ratio with/without inside accounts, adjusted ratio holding the other traits fixed. Outcome is likes relative to the account's own trailing median.",
           "hypotheses": H}, open("data/marketing/hypotheses.json", "w"), indent=1)
json.dump(MODEL_TRAITS, open("data/marketing/model_traits.json", "w"), indent=1)
from collections import Counter
print(Counter(h["verdict"] for h in H))
for h in H:
    r = h.get("result")
    print(f"{h['id']:<4}{h['verdict']:<14}{(h.get('status') or ''):<12}{(h.get('grade_ours') or '-'):<3}{h.get('trait') or '':<32}" + (f"{100*r['win_rate']:.1f}% ({100*r['interval'][0]:.1f}-{100*r['interval'][1]:.1f}) {r['pairs']}/{r['accounts']}" + (f" ratio {r['likes_ratio'][0]} ({r['likes_ratio'][1]}-{r['likes_ratio'][2]})" if r.get("likes_ratio") else "") if r else ""))
