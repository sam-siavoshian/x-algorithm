#!/usr/bin/env python3
"""Per-video labels and attributes from the frame features, the fetch metadata and the audio features.

    python scripts/video_files/label_videos.py      # writes data/vf/video_features.parquet (+ video_emb.npy, row-aligned)

Every threshold below was set on contact sheets without outcomes (cut_sheet.py, kind_sheet.py) before effects.py ran.

Scene change (the previews hold 2 frames a second): a half-second step counts as a change when the grey difference of
the two frames is D_CUT or more AND their colour histograms differ by H_CUT or more. What this resolves: no change,
occasional change, constant change. It counts a fast animation or a cross-fade as changes, misses zooms and page
changes inside a white-background screen recording, and cannot separate two cuts under half a second apart, so
the measure tops out at 120 a minute. Read by eye on 9 sheets (48 frames each): about 4 in 5 flagged steps are a real
cut or an abrupt change of scene.

Frame class: the video track's first-frame rule (label_thumbs.py) on each sampled frame.
Kind of video, from the 5 body frames (mid-points of 5 equal parts of the preview):
  slideshow   under 60% UI frames, at least 2 scene changes, and 85%+ of the other steps show no motion (d < STILL_D)
  screen      60%+ of body frames are product UI
  person      60%+ are a person (talking head, interview)
  film        60%+ are title, logo, motion graphics, render or real footage, and no body frame is UI
  mixed       everything else
"""
import glob, json, re, numpy as np, polars as pl
D_CUT, H_CUT, STILL_D = 14.0, 0.35, 1.0
G = ["ui", "face", "title", "logo", "graphic", "real", "chart", "game", "blank"]
MERGE = {"logo": "title", "game": "graphic", "blank": "other"}
CUT = 1776902400000


def is_cut(d, hd):
    return (np.asarray(d) >= D_CUT) & (np.asarray(hd) >= H_CUT)


def frame_class(z, face_h, n_faces):
    """z: scores of the 9 prompt groups for one frame."""
    top = G[int(np.argmax(z))]; c = MERGE.get(top, top)
    if c == "graphic" and face_h >= 0.20: c = "face"
    if c == "real" and face_h >= 0.10: c = "face"
    if c == "face" and n_faces == 0 and z[0] >= z[1] - 1.0: c = "ui"
    return c


def caption_frame(items):
    """One body frame's lower-40% text boxes: a caption-like line is centred, in the lower third, one line high, and the
    band is not full of text (a UI or a slide has many boxes there)."""
    cand = [it for it in items if 0.62 <= it["cy"] <= 0.97 and abs(it["cx"] - 0.5) <= 0.2 and it["h"] <= 0.12 and 0.04 <= it["w"] <= 0.92]
    return (cand[0]["t"] if cand and len(items) <= 3 else None)


def main():
    F = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/vf/frames/part_*.parquet"))], how="diagonal_relaxed")
    E = np.concatenate([np.load(f.replace("part_", "emb_").replace(".parquet", ".npy")) for f in sorted(glob.glob("data/vf/frames/part_*.parquet"))])
    keep = (F["n_frames"] > 0).to_numpy(); F, E = F.filter(pl.Series(keep)), E[keep]
    rows = []
    for r in F.iter_rows(named=True):
        n = r["n_frames"]; d, hd = np.array(r["d"] or []), np.array(r["hd"] or []); cut = is_cut(d, hd) if n > 1 else np.zeros(0, bool)
        z = np.array(r["z"]).reshape(-1, len(G)); samp = r["samp_idx"]; fc, fh = r["faces"], r["face_h"]
        cls = {i: frame_class(z[k], fh[i], fc[i]) for k, i in enumerate(samp)}
        body = [cls[i] for i in r["body_idx"]]; nb = len(body); sh = lambda *k: sum(b in k for b in body) / nb
        prev_s = n / 2.0
        noncut = d[~cut] if n > 1 else np.zeros(0); still = float((noncut < STILL_D).mean()) if len(noncut) else 1.0
        if sh("ui") < 0.6 and cut.sum() >= 2 and still >= 0.85: kind = "slideshow"
        elif sh("ui") >= 0.6: kind = "screen"
        elif sh("face") >= 0.6: kind = "person"
        elif sh("title", "graphic", "real", "chart") >= 0.6 and sh("ui") == 0: kind = "film"
        else: kind = "mixed"
        low = json.loads(r["lower_text"]); caps = [caption_frame(it) for it in low]; ct = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in caps if c]
        rows.append({"post_id": r["post_id"], "n_frames": n, "preview_s": prev_s, "cuts": int(cut.sum()), "cuts_per_min": float(60 * cut.sum() / max(prev_s - 0.5, 0.5)) if n > 1 else 0.0,
                     "cut_first3s": bool(cut[:6].any()), "still_share": still, "motion_med": float(np.median(noncut)) if len(noncut) else 0.0,
                     "face_share": float(np.mean(np.array(fc) > 0)), "bigface_share": float(np.mean(np.array(fh) >= 0.25)),
                     "ui_share": sh("ui"), "person_share": sh("face"), "title_share": sh("title"), "graphic_share": sh("graphic"), "real_share": sh("real"), "chart_share": sh("chart"),
                     "kind": kind, "cls_3s": cls[r["first_idx"][0]], "body_classes": ",".join(body), "n_body_classes": len(set(body)),
                     "caption_frames": len(ct), "caption_distinct": len(set(ct)), "body_frames": nb,
                     "captions": bool(len(ct) >= 3 and len(set(ct)) >= max(2, len(ct) - 1)), "caption_sample": " | ".join(c for c in caps if c)[:200],
                     "ui_score_mean": float(z[:, 0].mean()), "face_score_mean": float(z[:, 1].mean())})
    V = pl.DataFrame(rows, infer_schema_length=None)
    M = pl.read_ndjson("data/media/videos/files/meta.jsonl", infer_schema_length=None).filter(pl.col("status") == "ok").select("post_id", "media_type", "duration", pl.col("width").alias("vid_w"), pl.col("height").alias("vid_h"), "max_bitrate", "has_audio", "likes_now")
    TH = pl.read_parquet("data/media/videos/thumb_features.parquet").select("post_id", pl.col("cls").alias("cls_0"), pl.col("has_face").alias("face_0"), pl.col("big_headline").alias("headline_0"), pl.col("empty_first").alias("empty_0"))
    V = V.join(M, on="post_id", how="left").join(TH, on="post_id", how="left")
    c = pl.col
    V = V.with_columns(pl.when(c("duration") < 15).then(pl.lit("u15")).when(c("duration") < 30).then(pl.lit("15_30")).when(c("duration") < 60).then(pl.lit("30_60")).when(c("duration") < 120).then(pl.lit("60_120")).otherwise(pl.lit("o120")).alias("dur_band"),
                       (c("media_type") == "animated_gif").alias("is_gif"), (c("vid_w") < c("vid_h") * 0.8).alias("portrait"), (pl.min_horizontal("vid_w", "vid_h") < 720).alias("sub720"),
                       ((c("cls_0") == "ui") | (c("cls_3s") == "ui")).alias("product_first3s"), ((c("cls_0") == "face") | (c("cls_3s") == "face")).alias("person_first3s"),
                       ((c("cls_0") == "title") | (c("cls_3s") == "title")).alias("title_first3s"),
                       pl.when(c("cuts") == 0).then(pl.lit("0")).when(c("cuts_per_min") <= 6).then(pl.lit("le6")).when(c("cuts_per_min") <= 20).then(pl.lit("6_20")).otherwise(pl.lit("gt20")).alias("pace_band"))
    au = sorted(glob.glob("data/vf/audio/part_*.parquet"))
    if au:
        A = pl.concat([pl.read_parquet(f) for f in au], how="diagonal_relaxed").unique("post_id")
        A = A.drop([k for k in ("error", "windows") if k in A.columns]).with_columns(pl.lit(True).alias("audio_analysed"))
        V = V.join(A, on="post_id", how="left")
        V = V.with_columns(pl.when(~c("has_audio")).then(pl.lit("silent")).otherwise(c("sound_class")).alias("sound_class"), pl.when(~c("has_audio")).then(False).otherwise(c("audible")).alias("audible"))
    V.write_parquet("data/vf/video_features.parquet"); np.save("data/vf/video_emb.npy", E)
    print(V.height, "videos"); print(V.group_by("kind").len().sort("len", descending=True)); print(V.group_by("dur_band").len()); print(V.group_by("pace_band").len())
    print({k: round(float(V[k].mean()), 3) for k in ("captions", "product_first3s", "person_first3s", "title_first3s", "cut_first3s")}, "face_share>=.5:", float((V["face_share"] >= .5).mean()))
    if au: print(V.group_by("sound_class").len())


if __name__ == "__main__":
    main()
