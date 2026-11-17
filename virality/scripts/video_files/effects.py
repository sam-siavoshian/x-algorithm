#!/usr/bin/env python3
"""What kind of video, and what kind of sound, does better inside accounts: whole-video and audio traits.

    python scripts/video_files/effects.py [trait-prefix]
        reads  data/launch/table.parquet (scripts/launch/build_table.py), data/vf/video_features.parquet (label_videos.py)
        writes data/vf/effects.json

Method of scripts/launch/effects.py; the estimators are imported from scripts/audio/effects.py (pairs with posting
order balanced and an optional "same visual kind" restriction, within-account ratio, account fixed effects), each with
an interval from resampling accounts. Rule, sets of posts, traits and cut-offs: research/tracks/video_files/PREREG.md,
written before this script first ran. One correction after the first run on video traits and before anything was
written up: the adjusted estimate now also holds the post type fixed (dummies for launch, feature, demo and so on), as
the launch track's regression does; without it, in the set of all video posts, a trait that is more common in launch
videos inherits the launch lift. Every comparison is between video posts of one account; GIFs are excluded.
"""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, "scripts/audio")
from effects import pairs_est, ratio_est, adjusted_one, STRICT  # noqa: E402
CUT = 1776902400000; LAUNCH = ["launch", "open_source"]; ANN = LAUNCH + ["feature", "funding", "milestone", "hiring"]
c = pl.col
COV = ["names_product_first_line", "lines3", "long280", "link_in_post", "is_quote", "open_problem", "portrait", "sub720"]


def load():
    T = pl.read_parquet("data/launch/table.parquet").filter((c("source") == "community_archive") & c("has_video") & c("lift_likes").is_not_null())
    T = T.with_columns(c("text").fill_null("").str.replace_all(r"https?://\S+", "").str.slice(0, 300).str.contains(STRICT).alias("strict"), (c("chars") >= 280).alias("long280"), (c("lines") >= 3).alias("lines3"),
                       (c("ext_link") & ~c("points_to_reply")).alias("link_in_post")).drop("open_number", "open_result", "open_announce", "open_question")
    V = pl.read_parquet("data/vf/video_features.parquet").rename({"open_problem": "sp_open_problem", "kind": "vkind"}, strict=False)
    T = T.join(V, on="post_id", how="inner").filter(~c("is_gif"))
    med = T.filter(c("speech_rate_wpm").is_not_null())["speech_rate_wpm"].median() if "speech_rate_wpm" in T.columns else None
    if med is not None:
        T = T.with_columns(c("sound_class").is_in(["speech_only", "speech_and_music"]).fill_null(False).alias("has_speech"), c("sound_class").is_in(["music_only", "speech_and_music"]).fill_null(False).alias("has_music"),
                           c("audible").fill_null(False), c("sound_class").is_not_null().alias("sound_known"), pl.lit(med).alias("wpm_median"))
    return T


def traits(T):
    D = c("dur_band"); K = c("vkind")
    V = {f"dur_{b}": (f"the video is {d}, against every other length", D == b, None) for b, d in {"u15": "under 15 seconds", "15_30": "15 to 30 seconds", "30_60": "30 to 60 seconds", "60_120": "60 to 120 seconds", "o120": "over 120 seconds"}.items()}
    V |= {"dur_15_30_vs_u15": ("15 to 30 seconds, against under 15", D == "15_30", D.is_in(["15_30", "u15"])), "dur_30_60_vs_15_30": ("30 to 60 seconds, against 15 to 30", D == "30_60", D.is_in(["30_60", "15_30"])),
          "dur_60_120_vs_30_60": ("60 to 120 seconds, against 30 to 60", D == "60_120", D.is_in(["60_120", "30_60"])), "dur_o120_vs_60_120": ("over 120 seconds, against 60 to 120", D == "o120", D.is_in(["o120", "60_120"])),
          "dur_lt60": ("under 60 seconds, against 60 seconds or more", c("duration") < 60, None)}
    V |= {f"kind_{k}": (f"{d}, against every other kind of video", K == k, None) for k, d in {"screen": "a screen recording of the product over most of its length", "person": "a person on camera over most of its length (talking head, interview)",
          "film": "motion graphics, title cards or filmed footage over most of its length, with no product UI (a brand film)", "slideshow": "a slideshow of still images", "mixed": "a mix (no kind covers 60% of the video)"}.items()}
    V |= {"kind_screen_vs_person": ("a screen recording, against a person on camera", K == "screen", K.is_in(["screen", "person"])), "kind_screen_vs_film": ("a screen recording, against a brand film", K == "screen", K.is_in(["screen", "film"])),
          "kind_person_vs_film": ("a person on camera, against a brand film", K == "person", K.is_in(["person", "film"])), "kind_mixed_vs_screen": ("a mix of kinds, against a screen recording", K == "mixed", K.is_in(["screen", "mixed"])),
          "face_any": ("a face is on screen at some point, against never", c("face_share") > 0, None), "face_half": ("a face is on screen in half the frames or more, against less", c("face_share") >= 0.5, None),
          "ui_any": ("product UI is on screen at some point (any of 5 sampled frames), against never", c("ui_share") > 0, None), "ui_half": ("product UI in 3 or more of 5 sampled frames, against fewer", c("ui_share") >= 0.5, None),
          "any_cut": ("the picture changes scene at least once, against one continuous shot", c("cuts") > 0, None),
          "fast_cuts": ("more than 12 scene changes a minute (one every 5 seconds), against fewer; videos with at least one", c("cuts_per_min") > 12, c("cuts") > 0),
          "very_fast_cuts": ("more than 20 scene changes a minute, against 20 or fewer", c("cuts_per_min") > 20, None),
          "captions": ("captions burned into the picture (changing text in the lower third on most sampled frames)", c("captions"), None),
          "captions_nonscreen": ("burned-in captions, against none; videos that are not screen recordings", c("captions"), K != "screen"),
          "product_first3s": ("product UI is on screen in the first 3 seconds (the frame at 0 s or at 2.5 s)", c("product_first3s"), c("cls_0").is_not_null()),
          "person_first3s": ("a person is on screen in the first 3 seconds", c("person_first3s"), c("cls_0").is_not_null()),
          "title_first3s": ("a title card or logo is on screen in the first 3 seconds", c("title_first3s"), c("cls_0").is_not_null()),
          "cut_first3s": ("the scene changes inside the first 3 seconds", c("cut_first3s"), c("duration") >= 3)}
    V = {k: v + ("video",) for k, v in V.items()}
    if "has_speech" not in T.columns: return V
    SP, MU, AUD, KN = c("has_speech"), c("has_music"), c("audible"), c("sound_known") & (c("duration") >= 1)
    EN = SP & (c("language") == "en") & c("opening_line").str.len_chars().gt(0)
    A = {"sound_vs_silent": ("the video has audible sound, against a silent video (no audio track, or an empty one)", AUD, KN),
         "voice_vs_none": ("someone speaks (speech in 15% or more of the analysed sound), against sound without speech", SP, KN & AUD),
         "voice_vs_silent": ("someone speaks, against a silent video", SP, KN & (SP | ~AUD)),
         "music_vs_none": ("there is music (30% or more of the analysed sound), against sound without music", MU, KN & AUD),
         "voice_and_music_vs_voice_only": ("speech over music, against speech alone", SP & MU, KN & SP), "voice_and_music_vs_music_only": ("speech over music, against music alone", SP & MU, KN & MU),
         "voice_only_vs_music_only": ("speech alone, against music alone", SP & ~MU, KN & ((SP & ~MU) | (MU & ~SP))),
         "speech_first_2s": ("the first word comes within two seconds, against later", c("speech_first_2s"), KN & SP),
         "musical_intro": ("music opens the video and speech comes after two seconds, against speech from the start", c("musical_intro"), KN & SP & (c("musical_intro") | c("speech_first_2s"))),
         "fast_speech": ("speech faster than the median video with speech, against slower", c("speech_rate_wpm") >= c("wpm_median"), KN & SP & c("speech_rate_wpm").is_not_null()),
         "loud": ("integrated loudness of -16 LUFS or louder, against quieter", c("lufs") >= -16, KN & AUD), "quiet": ("integrated loudness under -24 LUFS, against louder", c("lufs") < -24, KN & AUD),
         "wide_dynamics": ("loudness range of 8 LU or more, against less", c("lra_lu") >= 8, KN & AUD & c("lra_lu").is_not_null()),
         "fast_music": ("music at 120 beats a minute or more, against slower", c("tempo_bpm") >= 120, KN & MU & c("tempo_bpm").is_not_null()),
         "busy_music": ("three or more note onsets a second in the music, against fewer", c("onset_rate") >= 3, KN & MU & c("onset_rate").is_not_null()),
         "spoken_open_names_product": ("the spoken opening line names the product (same rule as the post's first line)", c("open_names_product"), KN & EN),
         "spoken_open_says_name": ("the spoken opening line says a name the post or the account carries", c("says_name"), KN & EN & c("says_name").is_not_null()),
         "spoken_open_problem": ("the spoken opening line states a problem or asks a question", c("sp_open_problem"), KN & EN),
         "spoken_open_number": ("the spoken opening line contains a number", c("open_number"), KN & EN), "spoken_open_result": ("the spoken opening line states a result", c("open_result"), KN & EN),
         "spoken_open_announce": ("the spoken opening line carries an announcement verb", c("open_announce"), KN & EN)}
    return V | {k: v + ("audio",) for k, v in A.items()}


CF = c("akind") != "other"; U100 = c("tier").is_in(["<1k", "1k-10k", "10k-100k"]); O100 = c("tier") == "100k+"; LA = c("ptype").is_in(LAUNCH); AN = c("ptype").is_in(ANN)
SCOPES = {
    "launch_all": ("launch-type video posts (launch, open-source release) of every account", LA),
    "launch_cf": ("launch-type video posts of company and founder accounts", LA & CF),
    "launch_strict_cf": ("the same, opening with an announcement verb directly followed by a product name", LA & CF & c("strict")),
    "announce_cf": ("video posts that are own announcements (launch, open source, feature, funding, milestone, hiring) of company and founder accounts", AN & CF),
    "broad_cf": ("every video post of company and founder accounts", CF),
    "broad_cf_under_100k": ("every video post of company and founder accounts under 100,000 followers", CF & U100), "broad_cf_100k_plus": ("every video post of company and founder accounts with 100,000 followers or more", CF & O100),
    "broad_cf_before_cutoff": ("every video post of company and founder accounts, before 2026-04-23", CF & (c("created_at_ms") < CUT)), "broad_cf_after_cutoff": ("every video post of company and founder accounts, from 2026-04-23", CF & (c("created_at_ms") >= CUT)),
    "launch_all_under_100k": ("launch-type video posts, accounts under 100,000 followers", LA & U100), "launch_all_100k_plus": ("launch-type video posts, accounts with 100,000 followers or more", LA & O100),
    "launch_all_before_cutoff": ("launch-type video posts, every account, before 2026-04-23", LA & (c("created_at_ms") < CUT)), "launch_all_after_cutoff": ("launch-type video posts, every account, from 2026-04-23", LA & (c("created_at_ms") >= CUT)),
    "announce_cf_before_cutoff": ("own-announcement video posts of company and founder accounts, before 2026-04-23", AN & CF & (c("created_at_ms") < CUT)), "announce_cf_after_cutoff": ("the same, from 2026-04-23", AN & CF & (c("created_at_ms") >= CUT)),
}
MAIN = ("launch_all", "launch_cf", "announce_cf", "broad_cf")


def levels(S, col):
    """Mean likes over the account's usual per value of a column (account-weighted, bootstrap over accounts)."""
    rng = np.random.default_rng(0); out = {}
    for v in sorted(x for x in S[col].unique().to_list() if x is not None):
        z = S.filter(c(col) == v)
        if z.height < 15: continue
        per = z.group_by("author_id").agg(c("lift_likes").mean().alias("m"), pl.len().alias("n")).sort("author_id"); m, n = per["m"].to_numpy(), per["n"].to_numpy().astype(float)
        idx = rng.integers(0, len(m), (2000, len(m))); b = (m[idx] * n[idx]).sum(1) / n[idx].sum(1)
        out[str(v)] = {"posts": z.height, "accounts": per.height, "share": round(z.height / S.height, 4), "geo_mean_ratio": round(float(np.exp((m * n).sum() / n.sum())), 3), "lo": round(float(np.exp(np.quantile(b, .025))), 3), "hi": round(float(np.exp(np.quantile(b, .975))), 3)}
    return out


def main():
    T = load(); TR = traits(T); pre = sys.argv[1] if len(sys.argv) > 1 else None
    out = {"scopes": {}, "levels": {}, "n_traits": len(TR)}
    for sk, (sdesc, sf) in SCOPES.items():
        S = T.filter(sf); res = {"description": sdesc, "posts": S.height, "accounts": S["author_id"].n_unique(), "effects": {}}
        full = sk in MAIN
        for k, (desc, fa, restrict, grp) in TR.items():
            if pre and not k.startswith(pre): continue
            q = (S.filter(restrict.fill_null(False)) if restrict is not None else S).with_columns(fa.fill_null(False).alias("a"))
            e = {"description": desc, "group": grp, "posts_with": int(q["a"].sum()), "posts_without": int((~q["a"]).sum()), "pairs": pairs_est(q), "within_account": ratio_est(q)}
            nb = 400 if full else 200
            if e["posts_with"] >= 15 and e["posts_without"] >= 15:
                e["adjusted"] = adjusted_one(q, controls=COV, cats=["ptype"], B=nb)
                if grp == "audio":
                    e["same_visual"] = pairs_est(q, same="vkind"); e["adjusted_visual"] = adjusted_one(q, controls=COV, cats=["ptype", "vkind"], B=nb)
                    if full: e["adjusted_visual_length"] = adjusted_one(q, controls=COV, cats=["ptype", "vkind", "dur_band"], B=nb)
                elif full:
                    cats = [x for x in ("vkind", "dur_band") if not k.startswith(("kind", "ui_", "face_") if x == "vkind" else "dur")]
                    e["adjusted_kind_length"] = adjusted_one(q, controls=COV, cats=["ptype"] + cats, B=nb)
            res["effects"][k] = e
        out["scopes"][sk] = res
        print(f"\n== {sk}: {S.height} posts, {res['accounts']} accounts", flush=True)
        for k, e in res["effects"].items():
            p, r, a, sv = e["pairs"], e["within_account"], e.get("adjusted_visual") or e.get("adjusted"), e.get("same_visual")
            f1 = f"{100*p['win_rate']:5.1f}% {100*p['lo']:5.1f}-{100*p['hi']:<5.1f} {p['pairs']:>6} {p['accounts']:>4}" if p and p["win_rate"] is not None else " " * 31
            f2 = f"{r['ratio']:5.2f} {r['ratio_lo']:5.2f}-{r['ratio_hi']:<5.2f} {r['accounts_with_both']:>4}" if r else " " * 22
            f3 = f"{a['ratio']:5.2f} {a['ratio_lo']:5.2f}-{a['ratio_hi']:<5.2f}" if a else " " * 17
            f4 = f"{100*sv['win_rate']:5.1f}% {100*sv['lo']:5.1f}-{100*sv['hi']:<5.1f} {sv['pairs']:>5}" if sv and sv["win_rate"] is not None else ""
            print(f"{k:<30}{e['posts_with']:>5}{e['posts_without']:>5} | {f1} | {f2} | {f3} | {f4}")
    for sk in MAIN:
        S = T.filter(SCOPES[sk][1]); out["levels"][sk] = {col: levels(S, col) for col in ("dur_band", "vkind", "pace_band", "sound_class") if col in S.columns}
    json.dump(out, open("data/vf/effects.json" if not pre else f"data/vf/effects_{pre}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
