#!/usr/bin/env python3
"""Audio evidence table in the shape of xviral/data/launch_evidence.json: one entry per audio trait of a launch video.

    python scripts/audio/build_audio_evidence.py [effects.json]    # default data/audio/effects.json; writes xviral/data/audio_evidence.json

With measured effects (scripts/audio/effects.py) an entry is `supported` by the launch track's rule: pairs, likes ratio and
adjusted ratio (text traits and visual type held fixed) all exclude "no effect" in MAIN (launch-type video posts of company
and founder accounts) or in STRICT, at least one passes in the other, same direction; and, because videos autoplay muted,
the direction must not reverse among pairs with the same visual type. Under 200 pairs or 25 accounts in MAIN: too_thin.
Without measured effects (no video files could be fetched) every entry is written as `not_measured` with the sample that
exists and the smallest effect it could show (scripts/audio/power.py), and nothing is supported.
"""
import json, os, sys
sys.path.insert(0, "scripts/audio")
import effects as E

src = sys.argv[1] if len(sys.argv) > 1 else "data/audio/effects.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "xviral/data/audio_evidence.json"
P = json.load(open("data/audio/power.json")) if os.path.exists("data/audio/power.json") else None
V = json.load(open("data/audio/validation.json")) if os.path.exists("data/audio/validation.json") else None
MAIN, STRICT, ALL = "launch_cf", "launch_strict_cf", "announce_all"
out = {}


def tests(D, scope, k):
    e = D["scopes"][scope]["effects"][k]; p, r, a = e["pairs"], e["within_account"], e.get("adjusted_text_and_visual") or e.get("adjusted_text")
    p = p if p and p.get("win_rate") is not None else None
    d = 0 if not p else (1 if p["win_rate"] > 0.5 else -1); ok = {}
    if p: ok["pairs"] = p["lo"] > 0.5 if d > 0 else p["hi"] < 0.5
    if r: ok["ratio"] = r["ratio_lo"] > 1 if d > 0 else r["ratio_hi"] < 1
    if a: ok["adjusted"] = a["ratio_lo"] > 1 if d > 0 else a["ratio_hi"] < 1
    return {"pairs": p, "likes_ratio": r, "adjusted_likes_ratio": a, "same_visual_pairs": e.get("same_visual"), "tests_passed": ok}, d, ok


if os.path.exists(src):
    D = json.load(open(src))
    for k, (desc, _, _) in E.TRAITS.items():
        m, dm, okm = tests(D, MAIN, k); s, ds, oks = tests(D, STRICT, k); a, da, oka = tests(D, ALL, k)
        allm, alls = bool(okm) and all(okm.values()), bool(oks) and all(oks.values())
        thin = not m["pairs"] or m["pairs"]["pairs"] < 200 or m["pairs"]["accounts"] < 25
        sv = m["same_visual_pairs"]; sv_ok = not sv or sv.get("win_rate") is None or (sv["win_rate"] - 0.5) * dm >= 0
        same = dm == ds and dm != 0
        status = ("too_thin" if thin else "survives" if same and sv_ok and ((allm and any(oks.values())) or (alls and any(okm.values()))) else "main_only" if same and allm and sv_ok else
                  "leaning" if dm == da and (any(okm.values()) or any(oka.values())) else "none")
        p = m["pairs"] or {}
        entry = {"description": desc, "win_rate": p.get("win_rate"), "lo": p.get("lo"), "hi": p.get("hi"), "pairs": p.get("pairs"), "accounts": p.get("accounts"), "supported": status == "survives", "status": status,
                 "outcome": "likes relative to the account's trailing median", "design": "same account, launch-type video posts of company and founder accounts, posting order balanced",
                 "likes_ratio": m["likes_ratio"], "adjusted_likes_ratio": m["adjusted_likes_ratio"], "same_visual_type": sv, "tests_passed_main": okm, "strict_subset": s, "all_announcements_all_accounts": a, "measured_as": k}
        for sc, nm in (("announce_all_under_100k", "accounts_under_100k_followers"), ("announce_all_100k_plus", "accounts_100k_plus_followers")): entry[nm] = tests(D, sc, k)[0]
        per = {}
        for sc, nm in (("announce_all_before_cutoff", "before_2026_04_23"), ("announce_all_after_cutoff", "from_2026_04_23")):
            t, dd, okk = tests(D, sc, k); per[nm] = t | {"same_direction": dd == dm, "confirmed": dd == dm and any(okk.values())}
        entry["periods"] = per; entry["replicated_in_both_periods"] = bool(per["before_2026_04_23"]["confirmed"] and per["from_2026_04_23"]["confirmed"])
        entry["grade"] = ("A" if entry["replicated_in_both_periods"] else "B") if status == "survives" else "B-" if status == "main_only" else None
        out[k] = entry
    out["_scopes"] = {k: {x: v[x] for x in ("description", "video_posts", "accounts", "accounts_with_2plus_videos", "sound_classes")} for k, v in D["scopes"].items()}
else:
    for k, (desc, _, _) in E.TRAITS.items():
        out[k] = {"description": desc, "win_rate": None, "lo": None, "hi": None, "pairs": None, "accounts": None, "supported": False, "status": "not_measured", "grade": None,
                  "outcome": "likes relative to the account's trailing median", "design": "same account, launch-type video posts of company and founder accounts, posting order balanced",
                  "reason": "no video or audio file of the corpus posts was fetched in bulk, so the trait is unknown for every post", "measured_as": k}
out["_power"] = P
out["_validation_of_features"] = V
out["_meta"] = {"built": "2026-11-16", "measured": os.path.exists(src), "rule": __doc__.split("With measured effects")[1].strip(), "traits_fixed_in_advance": {k: v[0] for k, v in E.TRAITS.items()},
                "feature_code": "xviral/media/audio.py", "method_code": "scripts/audio/effects.py",
                "not_included": ["synthetic (text-to-speech) voice against a human voice: no detector could be validated by listening, so none was run"]}
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w"), indent=1)
print({k: v["status"] for k, v in out.items() if not k.startswith("_")})
