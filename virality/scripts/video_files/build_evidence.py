#!/usr/bin/env python3
"""Evidence tables in the shape of xviral/data/launch_evidence.json from data/vf/effects.json:

    python scripts/video_files/build_evidence.py     # writes xviral/data/video_file_evidence.json and xviral/data/audio_evidence.json

One entry per trait, oriented so that win_rate belongs to the version the entry describes. The rule is the one in
research/tracks/video_files/PREREG.md, fixed before effects.py ran:
  survives    pairs, ratio and adjusted all exclude no effect, same direction, in LAUNCH (every account), LAUNCH_CF or
              ANNOUNCE with 200+ pairs and 25+ accounts, and pairs and ratio point the same way in BROAD
  broad_only  all three exclude no effect in BROAD (200+ pairs, 25+ accounts) but in no launch set
  leaning     two of three in a launch set or in BROAD, nothing significant the other way
  none / too_thin / not_measured (label failed validation by eye)
  audio traits must also hold with the visual kind held fixed (same-kind pairs exclude 50%, or the adjusted ratio with
  kind dummies excludes 1 when same-kind pairs are under 200); otherwise status "confounded_with_picture".
Grades: A survives and the pairs interval excludes 50% both before and from 2026-04-23; B survives; C broad_only.
"""
import json, os, sys
E = json.load(open("data/vf/effects.json")); VAL = json.load(open("data/vf/validation.json")) if os.path.exists("data/vf/validation.json") else {}
FAILED = set(VAL.get("failed_traits", []))
LSETS = ("launch_all", "launch_cf", "announce_cf"); BROAD = "broad_cf"


def tests(scope, k, flip):
    e = E["scopes"][scope]["effects"].get(k)
    if not e: return {"tests_passed": {}}, False, 0, False, False, False
    p, r, a = e.get("pairs"), e.get("within_account"), (e.get("adjusted_visual") if e["group"] == "audio" else None) or e.get("adjusted")
    def op(p):
        if not p or p["win_rate"] is None: return None
        return {**p, "win_rate": round(1 - p["win_rate"], 4), "lo": round(1 - p["hi"], 4), "hi": round(1 - p["lo"], 4)} if flip else p
    def orr(r):
        if not r: return None
        x = {"ratio": round(1 / r["ratio"], 3), "lo": round(1 / r["ratio_hi"], 3), "hi": round(1 / r["ratio_lo"], 3)} if flip else {"ratio": r["ratio"], "lo": r["ratio_lo"], "hi": r["ratio_hi"]}
        return x | {kk: r[kk] for kk in ("accounts_with_both", "posts_with", "posts_without", "posts", "accounts") if kk in r}
    p, r, a, sv = op(p), orr(r), orr(a), op(e.get("same_visual"))
    tp = {"pairs": bool(p and p["lo"] > 0.5), "ratio": bool(r and r["lo"] > 1), "adjusted": bool(a and a["lo"] > 1)}
    against = bool((p and p["hi"] < 0.5) or (r and r["hi"] < 1) or (a and a["hi"] < 1))
    same = bool(p and r and p["win_rate"] >= 0.5 and r["ratio"] >= 1)
    thin = not p or p["pairs"] < 200 or p["accounts"] < 25
    out = {} if not p else {"win_rate": p["win_rate"], "lo": p["lo"], "hi": p["hi"], "pairs": p["pairs"], "accounts": p["accounts"]}
    out |= {"likes_ratio": r, "adjusted_likes_ratio": a, "tests_passed": tp, "posts_with": e["posts_without"] if flip else e["posts_with"], "posts_without": e["posts_with"] if flip else e["posts_without"]}
    if e["group"] == "audio":
        out["same_visual_kind_pairs"] = sv; out["adjusted_with_visual_kind_and_length"] = orr(e.get("adjusted_visual_length"))
        vis_ok = bool(sv and sv["pairs"] >= 200 and sv["lo"] > 0.5) or bool((not sv or sv["pairs"] < 200) and a and a["lo"] > 1)
        out["holds_with_picture_fixed"] = vis_ok
    else:
        out["adjusted_with_kind_and_length"] = orr(e.get("adjusted_kind_length"))
    return out, all(tp.values()), sum(tp.values()), same, against, thin


def build(group):
    out = {}
    for k, e0 in E["scopes"]["launch_all"]["effects"].items():
        if e0["group"] != group: continue
        pb = E["scopes"][BROAD]["effects"][k].get("pairs")
        flip = bool(pb and pb["win_rate"] is not None and pb["win_rate"] < 0.5)
        R = {s: tests(s, k, flip) for s in LSETS + (BROAD,)}
        Bx, Ball, Bn, Bsame, Bag, Bthin = R[BROAD]
        launch_pass = [s for s in LSETS if R[s][1] and not R[s][5]]
        launch_two = [s for s in LSETS if R[s][2] >= 2 and not R[s][5]]
        if k in FAILED: status = "not_measured"
        elif all(R[s][5] for s in LSETS) and Bthin: status = "too_thin"
        elif launch_pass and Bsame: status = "survives"
        elif Ball and not Bthin and not any(R[s][4] for s in LSETS): status = "broad_only"
        elif (launch_two or (Bn >= 2 and not Bthin)) and not Bag and not any(R[s][4] for s in LSETS): status = "leaning"
        else: status = "none"
        if group == "audio" and status in ("survives", "broad_only"):
            ref = launch_pass[0] if status == "survives" else BROAD
            if not R[ref][0].get("holds_with_picture_fixed"): status = "confounded_with_picture"
        per = {nm: tests(sc, k, flip)[0] for nm, sc in (("before_2026_04_23", "launch_all_before_cutoff"), ("from_2026_04_23", "launch_all_after_cutoff"))}
        perb = {nm: tests(sc, k, flip)[0] for nm, sc in (("before_2026_04_23", "broad_cf_before_cutoff"), ("from_2026_04_23", "broad_cf_after_cutoff"))}
        conf = lambda d: all(v.get("lo") is not None and v["lo"] > 0.5 for v in d.values())
        grade = ("A" if conf(per) else "B") if status == "survives" else ("C" if status == "broad_only" else None)
        main = launch_pass[0] if launch_pass else "launch_all"; L = R[main][0]
        desc = e0["description"]
        ent = {"description": ("NOT: " + desc) if flip else desc} | {kk: L.get(kk) for kk in ("win_rate", "lo", "hi", "pairs", "accounts")} | {
            "supported": status == "survives", "status": status, "grade": grade, "outcome": "likes relative to the account's trailing median",
            "design": f"same account, video posts only, posting order balanced; headline numbers from the set `{main}`", "likes_ratio": L.get("likes_ratio"), "adjusted_likes_ratio": L.get("adjusted_likes_ratio"),
            "tests_passed_main": L.get("tests_passed"), "launch_all_accounts": R["launch_all"][0], "launch_company_founder": R["launch_cf"][0], "strict_subset": tests("launch_strict_cf", k, flip)[0],
            "own_announcements_company_founder": R["announce_cf"][0], "all_video_posts_company_founder": Bx, "measured_as": k, "flipped": flip,
            "all_video_posts_company_founder_under_100k": tests("broad_cf_under_100k", k, flip)[0], "all_video_posts_company_founder_100k_plus": tests("broad_cf_100k_plus", k, flip)[0],
            "accounts_under_100k_followers": tests("launch_all_under_100k", k, flip)[0], "accounts_100k_plus_followers": tests("launch_all_100k_plus", k, flip)[0],
            "periods": per | {"design": "launch-type video posts of every account, split at the model's test cutoff"}, "periods_all_video_posts_company_founder": perb,
            "replicated_in_both_periods": conf(per), "replicated_in_both_periods_all_video": conf(perb), "views": "not measurable: view counts exist only in the 2026 view corpus, which does not record video"}
        out[("not_" if flip else "") + k] = ent
        f = lambda x: f"{100*x['win_rate']:.1f}% ({100*x['lo']:.1f}-{100*x['hi']:.1f}) n={x['pairs']}/{x['accounts']}" if x.get("win_rate") is not None else "-"
        g = lambda x: f"{x['likes_ratio']['ratio']:.2f} ({x['likes_ratio']['lo']:.2f}-{x['likes_ratio']['hi']:.2f})" if x.get("likes_ratio") else "-"
        h = lambda x: f"{x['adjusted_likes_ratio']['ratio']:.2f} ({x['adjusted_likes_ratio']['lo']:.2f}-{x['adjusted_likes_ratio']['hi']:.2f})" if x.get("adjusted_likes_ratio") else "-"
        print(f"{('not_' if flip else '') + k:<34}{status:<24}{str(grade):<5} L {f(R['launch_all'][0]):<32}{g(R['launch_all'][0]):<17} A {f(R['announce_cf'][0]):<32}{g(R['announce_cf'][0]):<17} B {f(Bx):<34}{g(Bx):<17}{h(Bx)}")
    out["_levels"] = {s: {col: v for col, v in d.items() if (col == "sound_class") == (group == "audio")} for s, d in E["levels"].items()}
    out["_scopes"] = {k: {kk: v[kk] for kk in ("description", "posts", "accounts")} for k, v in E["scopes"].items()}
    out["_label_validation"] = VAL.get(group, {})
    mp = "data/vf/model_pairs.json"
    if os.path.exists(mp): out["_model"] = json.load(open(mp))
    out["_meta"] = {"built": "2026-11-17", "track": "video_files", "evidence": VAL.get("evidence_note", ""), "rule": __doc__.split("fixed before effects.py ran:")[1].strip(),
                    "traits_tested": len([k for k in out if not k.startswith("_")]), "traits_tested_both_files": E["n_traits"], "prereg": "research/tracks/video_files/PREREG.md"}
    print({s: [k for k, v in out.items() if not k.startswith("_") and v.get("status") == s] for s in ("survives", "broad_only", "leaning", "confounded_with_picture", "too_thin")})
    return out


if __name__ == "__main__":
    json.dump(build("video"), open("xviral/data/video_file_evidence.json", "w"), indent=1)
    if any(e["group"] == "audio" for e in E["scopes"]["launch_all"]["effects"].values()):
        json.dump(build("audio"), open("xviral/data/audio_evidence.json", "w"), indent=1)
