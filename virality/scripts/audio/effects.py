#!/usr/bin/env python3
"""Does the sound of a launch video travel? Audio traits among video posts, inside accounts, by the launch track's method.

    python scripts/audio/effects.py [features.parquet] [visual.parquet]
        reads  data/tracks/launch/launch_posts.parquet    (posts, lift against the account's trailing median, text traits)
               data/audio/features.parquet                (one row per video: scripts/audio/extract.py)
               data/media/videos/thumb_features.parquet   (optional: the video track's first-frame class per post)
        writes data/audio/effects.json

Same three estimates as scripts/launch/effects.py, each with a 95% interval from resampling accounts:
  pairs     same-account pairs of video posts, at most 365 days apart, exactly one with the trait, lifts differing by 1.25x
            or more; win rate for "with it first" and "with it second" averaged; at most 50 pairs per account
  ratio     accounts with both versions: mean lift with minus without, precision-weighted, as a ratio of likes
  adjusted  account fixed effects, every text trait of the launch analysis, and the visual type of the video held fixed
Plus `same_visual`: the pairs estimate restricted to pairs whose two videos have the same visual type, which is the
direct answer to "does sound matter once the picture is held fixed" (videos autoplay muted in the feed).

Traits and their cut-offs were fixed before any video was analysed (TRAITS below); nothing here is tuned on outcomes.
"""
import json, os, sys, numpy as np, polars as pl

rng = np.random.default_rng(0)
CUT = 1776902400000      # 2026-04-23
LAUNCH = ["launch", "open_source"]; ANN = LAUNCH + ["feature", "funding", "milestone", "hiring"]
STRICT = r"(?:[Ii]ntroducing|[Aa]nnouncing|[Uu]nveiling|[Mm]eet|[Ll]aunching|[Ll]aunch(?:ed)?|[Rr]eleasing|[Rr]eleased?|[Oo]pen[- ]?sourc(?:ed|ing)|INTRODUCING|ANNOUNCING)[:\s]+(?:the |our |my |a |an )?(?:new |all-new )?[@A-Z0-9][\w.@-]+"
c = pl.col
SCOPES = {
    "launch_cf": ("launch-type video posts of company and founder accounts", c("ptype").is_in(LAUNCH) & (c("akind") != "other")),
    "launch_strict_cf": ("the same, opening with an announcement verb directly followed by a product name", c("ptype").is_in(LAUNCH) & (c("akind") != "other") & c("strict")),
    "launch_all": ("launch-type video posts of every account", c("ptype").is_in(LAUNCH)),
    "announce_cf": ("video posts that are any own announcement, company and founder accounts", c("ptype").is_in(ANN) & (c("akind") != "other")),
    "announce_all": ("video posts that are any own announcement, every account", c("ptype").is_in(ANN)),
    "announce_all_under_100k": ("own announcements with video, accounts with a known follower count under 100,000", c("ptype").is_in(ANN) & c("tier").is_in(["<1k", "1k-10k", "10k-100k"])),
    "announce_all_100k_plus": ("own announcements with video, accounts with 100,000 followers or more", c("ptype").is_in(ANN) & (c("tier") == "100k+")),
    "announce_all_before_cutoff": ("own announcements with video, every account, before 2026-04-23", c("ptype").is_in(ANN) & (c("created_at_ms") < CUT)),
    "announce_all_after_cutoff": ("own announcements with video, every account, from 2026-04-23", c("ptype").is_in(ANN) & (c("created_at_ms") >= CUT)),
}
SP = c("has_speech"); MU = c("has_music"); AUD = c("audible"); EN = SP & (c("language") == "en") & c("opening_line").str.len_chars().gt(0)
# name: (description, with, restrict to)
TRAITS = {
    "sound_vs_silent": ("the video has audible sound, against a silent video (no track or an empty one; GIFs excluded)", AUD, None),
    "voice_vs_none": ("someone speaks (speech in 15% or more of the video), against sound without speech", SP, AUD),
    "music_vs_none": ("there is music (30% or more of the video), against sound without music", MU, AUD),
    "voice_and_music_vs_voice_only": ("voiceover over music, against voiceover alone", SP & MU, SP),
    "voice_and_music_vs_music_only": ("voiceover over music, against music alone", SP & MU, MU),
    "voice_only_vs_music_only": ("voiceover alone, against music alone", SP & ~MU, (SP & ~MU) | (MU & ~SP)),
    "speech_first_2s": ("speech starts in the first two seconds, against later", c("speech_first_2s"), SP),
    "musical_intro": ("music opens the video and speech comes after two seconds, against speech from the start", c("musical_intro"), SP & (c("musical_intro") | c("speech_first_2s"))),
    "fast_speech": ("170 words a minute or more, against slower", c("speech_rate_wpm") >= 170, SP & c("speech_rate_wpm").is_not_null()),
    "slow_speech": ("under 130 words a minute, against faster", c("speech_rate_wpm") < 130, SP & c("speech_rate_wpm").is_not_null()),
    "fast_music": ("music at 120 beats a minute or more, against slower", c("tempo_bpm") >= 120, MU & c("tempo_bpm").is_not_null()),
    "busy_music": ("three or more note onsets a second in the music, against fewer", c("onset_rate") >= 3, MU & c("onset_rate").is_not_null()),
    "percussive_music": ("a quarter or more of the music's energy is percussive, against less", c("percussive_share") >= 0.25, MU & c("percussive_share").is_not_null()),
    "loud": ("integrated loudness of -16 LUFS or louder, against quieter", c("lufs") >= -16, AUD),
    "quiet": ("integrated loudness under -24 LUFS, against louder", c("lufs") < -24, AUD),
    "wide_dynamics": ("loudness range of 8 LU or more, against less", c("lra_lu") >= 8, AUD & c("lra_lu").is_not_null()),
    "spoken_open_names_product": ("the spoken opening line names the product (same rule as the post's first line)", c("open_names_product"), EN),
    "spoken_open_says_name": ("the spoken opening line says a name the post or the account carries", c("says_name"), EN & c("says_name").is_not_null()),
    "spoken_open_problem": ("the spoken opening line states a problem or asks a question", c("open_problem"), EN),
    "spoken_open_number": ("the spoken opening line contains a number", c("open_number"), EN),
    "spoken_open_result": ("the spoken opening line states a result", c("open_result"), EN),
    "spoken_open_announce": ("the spoken opening line carries an announcement verb", c("open_announce"), EN),
}
TEXT_ADJ = ["link_in_post", "thread_marker", "open_announce_text", "open_problem_text", "open_number_text", "names_product_first_line", "long280", "lines3", "tags", "is_quote", "prev2h"]


def boot_ci(stat, n, B=2000):
    idx = rng.integers(0, n, (B, n))
    b = np.array([stat(i) for i in idx]); b = b[np.isfinite(b)]
    return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) > B / 2 else (float("nan"), float("nan"))


def pairs_est(q, y="lift_likes", same=None, B=2000):
    """q has a boolean column `a`. `same`: a column whose value must be equal on both sides (the visual type)."""
    keep = ["author_id", "created_at_ms", "post_id", "a", pl.col(y).alias("y")] + ([same] if same else [])
    q = q.filter(pl.col(y).is_not_null()).select(keep)
    p = q.filter(pl.col("a")).join(q.filter(~pl.col("a")), on="author_id", suffix="_n").filter(((pl.col("created_at_ms") - pl.col("created_at_ms_n")).abs() <= 365 * 86_400_000) & ((pl.col("y") - pl.col("y_n")).abs() >= np.log(1.25)))
    if same: p = p.filter(pl.col(same).is_not_null() & (pl.col(same) == pl.col(same + "_n")))
    if p.height == 0: return None
    p = p.with_columns((pl.col("y") > pl.col("y_n")).alias("win"), (pl.col("created_at_ms") < pl.col("created_at_ms_n")).alias("wf"))
    p = p.with_columns(pl.int_range(pl.len()).shuffle(seed=1).over("author_id").alias("r")).filter(pl.col("r") < 50)
    per = p.group_by("author_id").agg((pl.col("win") & pl.col("wf")).sum().alias("w1"), pl.col("wf").sum().alias("n1"), (pl.col("win") & ~pl.col("wf")).sum().alias("w2"), (~pl.col("wf")).sum().alias("n2")).sort("author_id")
    w1, m1, w2, m2 = (per[k].to_numpy().astype(float) for k in ("w1", "n1", "w2", "n2"))
    def bal(i):
        a, b = m1[i].sum(), m2[i].sum()
        if a < 1 and b < 1: return np.nan
        if a < 1 or b < 1: return (w1[i].sum() + w2[i].sum()) / (a + b)      # one order only: no balancing possible
        return 0.5 * (w1[i].sum() / a + w2[i].sum() / b)
    rate = bal(np.arange(len(w1))); lo, hi = boot_ci(bal, len(w1), B)
    return {"win_rate": None if not np.isfinite(rate) else round(float(rate), 4), "lo": round(lo, 4), "hi": round(hi, 4), "pairs": int(p.height), "accounts": int(per.height),
            "pairs_with_first": int(m1.sum()), "pairs_with_second": int(m2.sum())}


def ratio_est(q, y="lift_likes", B=2000):
    q = q.filter(pl.col(y).is_not_null())
    per = q.group_by("author_id").agg(pl.col(y).filter(pl.col("a")).mean().alias("m1"), pl.col("a").sum().alias("n1"), pl.col(y).filter(~pl.col("a")).mean().alias("m0"), (~pl.col("a")).sum().alias("n0")).filter((pl.col("n1") > 0) & (pl.col("n0") > 0)).sort("author_id")
    if per.height < 5: return None
    d = (per["m1"] - per["m0"]).to_numpy(); w = (per["n1"] * per["n0"] / (per["n1"] + per["n0"])).to_numpy().astype(float)
    st = lambda i: float(np.sum(w[i] * d[i]) / np.sum(w[i]))
    lo, hi = boot_ci(st, len(w), B)
    return {"ratio": round(float(np.exp(st(np.arange(len(w))))), 3), "ratio_lo": round(float(np.exp(lo)), 3), "ratio_hi": round(float(np.exp(hi)), 3), "accounts_with_both": int(per.height), "posts_with": int(per["n1"].sum()), "posts_without": int(per["n0"].sum())}


def adjusted_one(q, y="lift_likes", controls=(), cats=(), B=500):
    """Within-account coefficient of `a` with the controls (0/1 columns) and the categorical columns as dummies."""
    q = q.filter(pl.col(y).is_not_null()).filter(pl.len().over("author_id") >= 2)
    if q.height < 30 or q["a"].n_unique() < 2: return None
    cols = [q["a"].cast(pl.Float64).to_numpy()] + [q[k].cast(pl.Float64).fill_null(0).to_numpy() for k in controls]
    for k in cats:
        for v in sorted(set(q[k].drop_nulls().to_list()))[1:]: cols.append((q[k] == v).fill_null(False).cast(pl.Float64).to_numpy())
    X = np.column_stack(cols); Y = q[y].to_numpy(); u, inv = np.unique(q["author_id"].to_numpy(), return_inverse=True)
    def dm(M):
        s = np.zeros((len(u),) + M.shape[1:]); np.add.at(s, inv, M); n = np.bincount(inv).astype(float)
        return M - (s / n.reshape(-1, *([1] * (M.ndim - 1))))[inv]
    Xd, Yd = dm(X), dm(Y)
    beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0][0]
    rows = [np.where(inv == k)[0] for k in range(len(u))]; bs = []
    for _ in range(B):
        ii = np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))]); bs.append(np.linalg.lstsq(Xd[ii], Yd[ii], rcond=None)[0][0])
    return {"ratio": round(float(np.exp(beta)), 3), "ratio_lo": round(float(np.exp(np.quantile(bs, 0.025))), 3), "ratio_hi": round(float(np.exp(np.quantile(bs, 0.975))), 3), "posts": int(len(Y)), "accounts": int(len(u))}


def load_table(features="data/audio/features.parquet", visual="data/media/videos/thumb_features.parquet"):
    T = pl.read_parquet("data/tracks/launch/launch_posts.parquet").filter(c("has_video") & (c("source") == "community_archive") & c("lift_likes").is_not_null())
    T = T.with_columns(c("text").fill_null("").str.replace_all(r"https?://\S+", "").str.slice(0, 300).str.contains(STRICT).alias("strict"), (c("chars") >= 280).alias("long280"), (c("lines") >= 3).alias("lines3"),
                       (c("gap_prev_h") < 2).fill_null(False).alias("prev2h"), (c("ext_link") & ~c("points_to_reply")).alias("link_in_post"),
                       c("open_announce").alias("open_announce_text"), c("open_problem").alias("open_problem_text"), c("open_number").alias("open_number_text")).drop("open_announce", "open_problem", "open_number", "open_result", "open_question")
    F = pl.read_parquet(features)
    T = T.join(F, on="post_id", how="inner").filter(c("is_gif").not_())
    T = T.with_columns(c("audible").fill_null(False), c("sound_class").is_in(["speech_only", "speech_and_music"]).fill_null(False).alias("has_speech"), c("sound_class").is_in(["music_only", "speech_and_music"]).fill_null(False).alias("has_music"))
    if visual and os.path.exists(visual):
        V = pl.read_parquet(visual)
        vcol = next((k for k in ("first_frame_class", "visual_type", "frame_class", "zs_class") if k in V.columns), None)
        T = T.join(V.select("post_id", pl.col(vcol).alias("visual_type")).unique("post_id"), on="post_id", how="left") if vcol else T.with_columns(pl.lit(None, pl.String).alias("visual_type"))
    elif "visual_type" not in T.columns:
        T = T.with_columns(pl.lit(None, pl.String).alias("visual_type"))
    return T


def run(T, B=2000, scopes=None, traits=None):
    out = {"scopes": {}}
    for sk, (sdesc, sf) in SCOPES.items():
        if scopes and sk not in scopes: continue
        S = T.filter(sf)
        res = {"description": sdesc, "video_posts": S.height, "accounts": S["author_id"].n_unique(), "accounts_with_2plus_videos": int((S.group_by("author_id").len()["len"] >= 2).sum()),
               "sound_classes": dict(S.group_by("sound_class").len().iter_rows()), "effects": {}}
        has_visual = S["visual_type"].null_count() < S.height
        for k, (desc, fa, restrict) in TRAITS.items():
            if traits and k not in traits: continue
            q = (S.filter(restrict.fill_null(False)) if restrict is not None else S).with_columns(fa.fill_null(False).alias("a"))
            e = {"description": desc, "posts_with": int(q["a"].sum()), "posts_without": int((~q["a"]).sum()), "pairs": pairs_est(q, B=B), "within_account": ratio_est(q, B=B),
                 "adjusted_text": adjusted_one(q, controls=TEXT_ADJ, B=max(B // 4, 50))}
            if has_visual:
                e["same_visual"] = pairs_est(q, same="visual_type", B=B)
                e["adjusted_text_and_visual"] = adjusted_one(q, controls=TEXT_ADJ, cats=["visual_type"], B=max(B // 4, 50))
            res["effects"][k] = e
        out["scopes"][sk] = res
    return out


def show(out):
    for sk, res in out["scopes"].items():
        print(f"\n== {sk}: {res['video_posts']} video posts, {res['accounts']} accounts ({res['accounts_with_2plus_videos']} with two or more)  {res['sound_classes']}")
        print(f"{'trait':<32}{'n+':>5}{'n-':>5} | {'win':>6} {'interval':>12} {'pairs':>6} {'acc':>4} | {'ratio':>5} {'interval':>11} {'acc':>4} | {'adj':>5} {'interval':>11} | same visual")
        for k, e in res["effects"].items():
            p, r, a, sv = e["pairs"], e["within_account"], e.get("adjusted_text_and_visual") or e["adjusted_text"], e.get("same_visual")
            f1 = f"{100*p['win_rate']:5.1f}% {100*p['lo']:5.1f}-{100*p['hi']:<5.1f} {p['pairs']:>6} {p['accounts']:>4}" if p and p["win_rate"] is not None else " " * 31
            f2 = f"{r['ratio']:5.2f} {r['ratio_lo']:5.2f}-{r['ratio_hi']:<5.2f} {r['accounts_with_both']:>4}" if r else " " * 22
            f3 = f"{a['ratio']:5.2f} {a['ratio_lo']:5.2f}-{a['ratio_hi']:<5.2f}" if a else " " * 17
            f4 = f"{100*sv['win_rate']:5.1f}% {100*sv['lo']:5.1f}-{100*sv['hi']:<5.1f} {sv['pairs']:>5}" if sv and sv["win_rate"] is not None else ""
            print(f"{k:<32}{e['posts_with']:>5}{e['posts_without']:>5} | {f1} | {f2} | {f3} | {f4}")


if __name__ == "__main__":
    T = load_table(*(sys.argv[1:3]))
    out = run(T)
    show(out)
    json.dump(out, open("data/audio/effects.json", "w"), indent=1)
