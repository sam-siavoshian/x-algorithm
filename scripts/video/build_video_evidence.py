#!/usr/bin/env python3
"""Video evidence table in the shape of xviral/data/launch_evidence.json: one entry per trait of a launch video's
first frame or file, oriented so that win_rate is the win rate of the version the entry describes.

    python scripts/video/build_video_evidence.py    # reads data/video/effects.json (+ validation, model results), writes xviral/data/video_evidence.json

Rule fixed before any effect was computed (docstring of video_effects.py). Three tests per set of posts, each with a
95% interval from resampling accounts: pairs (excludes 50%), ratio (excludes 1), adjusted (excludes 1; where defined).
  LAUNCH   = launch-type video posts of every account          ANNOUNCE = own-announcement video posts, company+founder
  BROAD    = every video post of company and founder accounts
  survives    every test passes in LAUNCH or in ANNOUNCE, and pairs and ratio point the same way in BROAD
  broad_only  every test passes in BROAD only (true of these accounts' videos in general; not shown for launch videos)
  leaning     some test passes somewhere and nothing points the other way
  none        nothing passes, or the direction flips between sets
  too_thin    under 200 pairs or under 25 accounts in both LAUNCH and ANNOUNCE
Grades: A survives and confirmed (all tests) separately before and from 2026-04-23 in LAUNCH; B survives; C broad_only.
Entries without a grade carry no advice.
"""
import json, os
E = json.load(open("data/video/effects.json"))
LAUNCH, ANNOUNCE, BROAD = "launch_all", "announce_cf", "allvideo_cf"
NOT = {"first_ui": "the first frame is not product UI", "first_face": "the first frame does not show a person", "first_title": "the first frame is not a title card or logo", "first_graphic": "the first frame is not motion graphics or a render",
       "first_real": "the first frame is not real-world footage", "first_chart": "the first frame is not a chart, diagram or slide", "has_face": "no face in the first frame", "big_face": "no face, against a large face, in the first frame",
       "empty_first": "the first frame is not empty (something is visible at once)", "dark_first": "the first frame is not dark", "text_on_frame": "no readable text in the first frame", "text_heavy": "little text in the first frame, against text-heavy",
       "big_headline": "no headline-size text in the first frame", "lower_text_nonui": "no caption-like text in the lower third of the first frame", "portrait": "landscape video, against portrait", "square": "landscape video, against square",
       "wide_16_9": "a shape other than 16:9", "hd1080": "under 720 px, against 1080 px or more", "hd720": "under 720 px on the short side", "gif": "a video, against a GIF", "studio_upload": "an ordinary upload, against Media Studio",
       "first_ui_vs_face": "first frame a person, against product UI", "first_ui_vs_title": "first frame a title card or logo, against product UI", "first_ui_vs_graphic": "first frame motion graphics, against product UI",
       "first_face_vs_title": "first frame a title card or logo, against a person", "first_real_vs_ui": "first frame product UI, against real-world footage or a person"}

def get(scope, k):
    S = E["scopes"][scope]; e = S["effects"][k]["likes"]; p, r = e["pairs"], e["within_account"]; a = S["adjusted_likes"].get(k)
    return p, r, a

def tests(scope, k, flip):
    p, r, a = get(scope, k)
    def orient_p(p):
        if not p or p["win_rate"] is None: return None
        return {**p, "win_rate": round(1 - p["win_rate"], 4), "lo": round(1 - p["hi"], 4), "hi": round(1 - p["lo"], 4)} if flip else p
    def orient_r(r):
        if not r: return None
        x = {"ratio": r["ratio"], "lo": r["ratio_lo"], "hi": r["ratio_hi"]}; r = {kk: vv for kk, vv in r.items() if kk != "posts"}
        if flip: x = {"ratio": round(1 / r["ratio"], 3), "lo": round(1 / r["ratio_hi"], 3), "hi": round(1 / r["ratio_lo"], 3)}
        return x | {kk: r[kk] for kk in ("accounts_with_both", "posts_with", "posts_without") if kk in r}
    p, r, a = orient_p(p), orient_r(r), orient_r(a)
    tp = {"pairs": bool(p and p["lo"] > 0.5), "ratio": bool(r and r["lo"] > 1)} | ({"adjusted": bool(a["lo"] > 1)} if a else {})
    against = bool((p and p["hi"] < 0.5) or (r and r["hi"] < 1))
    same = bool(p and r and p["win_rate"] >= 0.5 and r["ratio"] >= 1)
    out = {} if not p else {"win_rate": p["win_rate"], "lo": p["lo"], "hi": p["hi"], "pairs": p["pairs"], "accounts": p["accounts"]}
    return out | {"likes_ratio": r, "adjusted_likes_ratio": a, "tests_passed": tp}, all(tp.values()), any(tp.values()), same, against

out = {}
for k, e0 in E["scopes"][LAUNCH]["effects"].items():
    pb = get(BROAD, k)[0]; pl_ = get(LAUNCH, k)[0]
    ref = pl_ if pl_ and pl_["win_rate"] is not None and pl_["pairs"] >= 200 else pb
    flip = bool(ref and ref["win_rate"] is not None and ref["win_rate"] < 0.5)
    L, Lall, Lany, Lsame, Lag = tests(LAUNCH, k, flip); A, Aall, Aany, Asame, Aag = tests(ANNOUNCE, k, flip); B, Ball, Bany, Bsame, Bag = tests(BROAD, k, flip)
    thin = lambda x: x.get("pairs", 0) < 200 or x.get("accounts", 0) < 25
    if thin(L) and thin(A) and not Ball: status = "too_thin"
    elif ((Lall and not thin(L)) or (Aall and not thin(A))) and Bsame: status = "survives"
    elif Ball and not (Lag or Aag): status = "broad_only"
    elif (Lany or Aany or Bany) and not (Lag or Aag or Bag) and Lsame and Bsame: status = "leaning"
    else: status = "none"
    per = {}
    for nm, sc in (("before_2026_04_23", "launch_all_before_cutoff"), ("from_2026_04_23", "launch_all_after_cutoff")):
        X, Xall, _, Xsame, _ = tests(sc, k, flip); per[nm] = X | {"same_direction": Xsame, "confirmed": Xall}
    perb = {}
    for nm, sc in (("before_2026_04_23", "allvideo_cf_before_cutoff"), ("from_2026_04_23", "allvideo_cf_after_cutoff")):
        X, Xall, _, Xsame, _ = tests(sc, k, flip); perb[nm] = X | {"same_direction": Xsame, "confirmed": Xall}
    both = all(v["confirmed"] for v in per.values()); bothb = all(v["confirmed"] for v in perb.values())
    grade = ("A" if both else "B") if status == "survives" else ("C" if status == "broad_only" else None)
    ent = {"description": NOT[k] if flip else e0["description"]} | {kk: L.get(kk) for kk in ("win_rate", "lo", "hi", "pairs", "accounts")} | {
        "supported": status == "survives", "status": status, "outcome": "likes relative to the account's trailing median",
        "design": "same account, launch-type video posts of every account, posting order balanced; the trait is read from the poster frame or the media record",
        "likes_ratio": L["likes_ratio"], "adjusted_likes_ratio": L["adjusted_likes_ratio"], "tests_passed_main": L["tests_passed"],
        "own_announcements_company_founder": A, "all_video_posts_company_founder": B, "measured_as": k, "flipped": flip,
        "accounts_under_100k_followers": tests("launch_all_under_100k", k, flip)[0], "accounts_100k_plus_followers": tests("launch_all_100k_plus", k, flip)[0],
        "all_video_posts_company_founder_under_100k": tests("allvideo_cf_under_100k", k, flip)[0], "all_video_posts_company_founder_100k_plus": tests("allvideo_cf_100k_plus", k, flip)[0],
        "periods": per | {"design": "launch-type video posts of every account, split at the model's test cutoff; each period estimated on its own"},
        "periods_all_video_posts_company_founder": perb, "replicated_in_both_periods": both, "replicated_in_both_periods_all_video": bothb, "grade": grade,
        "views": "not measurable: view counts exist only in the 2026 view corpus, which does not record video"}
    out[k if not flip else "not_" + k] = ent
    f = lambda x: f"{100*x['win_rate']:.1f}% ({100*x['lo']:.1f}-{100*x['hi']:.1f}) n={x['pairs']}/{x['accounts']}" if x.get("win_rate") is not None else "-"
    g = lambda x: f"{x['likes_ratio']['ratio']:.2f} ({x['likes_ratio']['lo']:.2f}-{x['likes_ratio']['hi']:.2f})" if x.get("likes_ratio") else "-"
    print(f"{('not_' if flip else '') + k:<26}{status:<11}{str(grade):<5} L {f(L):<34}{g(L):<18} A {f(A):<34}{g(A):<18} B {f(B):<36}{g(B)}")
UNMEASURED = "not measured: the video files were not acquired (the public embed route returned them on a 6-post test; the bulk fetch was refused by the session's permission layer)"
for k, d in {"duration_bands": "short against long video (under 30 s, 30 to 90 s, over 90 s)", "pace_cuts_per_minute": "pace: cuts per minute", "motion": "amount of motion",
             "burned_in_captions": "captions burned into the video across its length", "audio": "speech, music or silence in the audio track", "share_of_frames_with_face": "share of frames with a face",
             "screen_recording_whole_video": "screen recording across the whole video (only the first frame was classified)", "bitrate": "encoded bitrate"}.items():
    out[k] = {"description": d, "supported": False, "status": "not_measured", "grade": None, "reason": UNMEASURED}
out["_class_lift"] = E["class_lift"]
out["_scopes"] = {k: {kk: v[kk] for kk in ("description", "posts", "accounts")} for k, v in E["scopes"].items()}
for nm, p in (("_label_validation", "data/video/validation.json"), ("_model", "data/video/model_pairs.json"), ("_acquisition", "data/video/acquisition.json"), ("_thumb_probe", "data/video/thumb_probe.json")):
    if os.path.exists(p): out[nm] = json.load(open(p))
out["_meta"] = {"built": "2026-11-16", "track": "video", "evidence": "poster frames (first frame) only; no video files, no audio",
                "rule": "survives = pairs, ratio and adjusted intervals all exclude no effect among launch-type video posts (every account) or among own-announcement video posts of company and founder accounts, same direction among all video posts of company and founder accounts",
                "traits_tested": len([k for k in out if not k.startswith("_") and out[k].get("status") != "not_measured"])}
json.dump(out, open("xviral/data/video_evidence.json", "w"), indent=1)
print({s: [k for k, v in out.items() if not k.startswith("_") and v.get("status") == s] for s in ("survives", "broad_only", "leaning")})
