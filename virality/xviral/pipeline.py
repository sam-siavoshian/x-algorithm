"""Stage orchestration. Each stage reads the previous stage's Parquet under
paths.work and writes its own, so any stage can be re-run on its own.

Phase 0 (CPU):  ingest -> unify -> filter -> authors -> hygiene -> dedup ->
                relevance -> features -> dataset -> report -> baseline
Phase 1 (GPU):  media-select -> fetch-images / fetch-videos -> asr ->
                embed-mm -> sid-train -> sid-assign -> embed-siglip -> neighbors
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from xviral import audit, authors, baseline, dedup, filters, labels, relevance, splits
from xviral.authors import follower_tier_expr
from xviral.config import Config
from xviral.constants import FOLLOWER_TIER_NAMES
from xviral.features import audience, content, derive, suppression, text_structure
from xviral.util.core import scan_dir, stable_hash64, write_parquet

log = logging.getLogger(__name__)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.is_file() else scan_dir(path).collect()


def _maybe(path: Path) -> pl.DataFrame | None:
    if path.is_file() or (path.is_dir() and any(path.glob("*.parquet"))):
        return _read(path)
    return None


# --- Phase 0 ------------------------------------------------------------------


def ingest(cfg: Config, sources: list[str] | None = None, limit: int | None = None) -> None:
    from xviral.sources import analytics, community_archive, nitter_sample, twitter100m, usc, user_scrape

    sources = sources or ["usc", "community_archive", "user_scrape", "analytics", "nitter_2026", "twitter100m"]
    root = cfg.paths.stage("sources")
    if "usc" in sources and cfg.paths.usc_root:
        usc.ingest(cfg.paths.usc_root, root / "usc", workers=cfg.workers, keep_langs=tuple(cfg.filters.keep_langs), limit=limit)
    if "community_archive" in sources and cfg.paths.community_archive:
        res = community_archive.to_unified(cfg.paths.community_archive)
        for k, v in res.items():
            write_parquet(v, root / "community_archive" / k / "part-00000.parquet")
    if "user_scrape" in sources and cfg.paths.user_scrape:
        user_scrape.ingest(cfg.paths.user_scrape, root / "user_scrape")
    if "analytics" in sources and cfg.paths.analytics:
        frames = [analytics.read_export(p) for p in cfg.paths.analytics]
        write_parquet(pl.concat(frames), root / "analytics" / "posts" / "part-00000.parquet")
    seeds = cfg.paths.seed_accounts.read_text().split() if cfg.paths.seed_accounts else None
    if "nitter_2026" in sources and cfg.paths.nitter_2026:
        nitter_sample.ingest(cfg.paths.nitter_2026, root / "nitter_2026", general_authors=cfg.select.general_accounts)
    if "twitter100m" in sources and cfg.paths.twitter100m_tweets and cfg.paths.twitter100m_users:
        twitter100m.ingest(cfg.paths.twitter100m_tweets, cfg.paths.twitter100m_users, root / "twitter100m", seed_handles=seeds,
                           min_created=cfg.filters.min_created_by_source.get("twitter100m", cfg.filters.min_created))


def unify(cfg: Config) -> None:
    from xviral import unify as unify_mod

    root = cfg.paths.work / "sources"
    dirs = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}
    unify_mod.run(dirs, cfg.paths.stage("unified"))
    donors = [p / "donors" for p in dirs.values() if (p / "donors").exists()]
    if donors:
        write_parquet(pl.concat([_read(d) for d in donors]).unique(), cfg.paths.stage("unified") / "donors.parquet")


def filter_rows(cfg: Config) -> None:
    u = cfg.paths.work / "unified"
    posts = _read(u / "posts.parquet")
    flog = filters.FilterLog()
    posts = flog.apply("validity", posts, lambda d: filters.validity(d, cfg.filters), exempt_analytics=False)

    # Author activity (reply volume) needs every kind, so it is computed before the kind filter
    # and merged with the USC side table, which counted rows before the USC prefilter.
    # Sources that prefilter at ingest (USC, Community Archive) bring their own activity counts, taken
    # before the prefilter; for the rest it is counted here.
    act_side = _maybe(u / "activity.parquet")
    counted = set("+".join(act_side["source"].unique().to_list()).split("+")) if act_side is not None and "source" in act_side.columns else set()
    act_posts = posts.filter(~pl.col("source").is_in(list(counted))).group_by("author_id").agg(
        pl.len().alias("rows"), (pl.col("kind") == "reply").sum().alias("replies"),
        (pl.col("kind") == "repost").sum().alias("reposts"), pl.col("kind").is_in(["original", "quote"]).sum().alias("originals"),
        pl.col("created_at_ms").min().alias("first_ms"), pl.col("created_at_ms").max().alias("last_ms"))
    parts = [act_posts] + ([act_side.drop("source")] if act_side is not None and "source" in act_side.columns else [act_side] if act_side is not None else [])
    act = pl.concat(parts, how="vertical_relaxed").group_by("author_id").agg(
        pl.col("rows", "replies", "reposts", "originals").sum(), pl.col("first_ms").min(), pl.col("last_ms").max())
    write_parquet(act, cfg.paths.stage("filtered") / "activity.parquet")

    posts = flog.apply("kind", posts, lambda d: filters.kind(d, cfg.filters))
    usc_audit = _maybe(cfg.paths.work / "sources" / "usc" / "audit")
    posts = flog.apply("maturity", posts, lambda d: filters.maturity(d, cfg.filters, usc_audit), exempt_analytics=False)
    posts = flog.apply("recency", posts, lambda d: filters.recency(d, cfg.filters))
    posts = flog.apply("language", posts, lambda d: filters.language(d, cfg.filters))
    out = cfg.paths.stage("filtered")
    write_parquet(posts, out / "posts.parquet")
    write_parquet(flog.frame(), out / "filter_log.parquet")


def author_stage(cfg: Config) -> None:
    f = cfg.paths.work / "filtered"
    u = cfg.paths.work / "unified"
    posts = _read(f / "posts.parquet")
    accounts = _read(u / "accounts.parquet")
    posts = authors.followers_at_post(posts, accounts)
    trail = authors.trailing_features(posts, window=cfg.labels.trailing_window)
    cp = dedup.author_copypasta(posts)
    edges = _maybe(u / "edges.parquet")
    donors = _maybe(u / "donors.parquet")
    atab = authors.author_table(posts, accounts, _maybe(f / "activity.parquet"), cp, edges, donors)
    if edges is not None and edges.height:
        atab = atab.join(suppression.pagerank(edges), on="author_id", how="left")
    zones = audience.estimate_audience_zone(posts)
    out = cfg.paths.stage("authors")
    write_parquet(posts.select("post_id", "followers_at_post"), out / "followers_at_post.parquet")
    write_parquet(trail, out / "trailing.parquet")
    write_parquet(cp, out / "copypasta.parquet")
    write_parquet(atab, out / "authors.parquet")
    write_parquet(zones, out / "audience_zones.parquet")


def hygiene_stage(cfg: Config) -> None:
    posts = _read(cfg.paths.work / "filtered" / "posts.parquet")
    atab = _read(cfg.paths.work / "authors" / "authors.parquet")
    kept = filters.select_accounts(filters.hygiene(posts, atab, cfg.hygiene), cfg.select)
    write_parquet(kept, cfg.paths.stage("hygiene") / "posts.parquet")


def dedup_stage(cfg: Config) -> None:
    posts = _read(cfg.paths.work / "hygiene" / "posts.parquet")
    nd = dedup.near_duplicate_clusters(posts)
    write_parquet(nd, cfg.paths.stage("dedup") / "near_dup.parquet")


def relevance_stage(cfg: Config) -> None:
    posts = _read(cfg.paths.work / "hygiene" / "posts.parquet")
    if not cfg.relevance.enabled:
        work = cfg.paths.stage("relevance")
        write_parquet(posts, work / "posts.parquet")
        (work / "report.json").write_text(json.dumps({"enabled": False, "rows": posts.height}))
        return
    posts = relevance.political_prethin(posts, cfg.relevance)
    nd = _maybe(cfg.paths.work / "dedup" / "near_dup.parquet")
    if nd is not None:
        posts = posts.join(nd, on="post_id", how="left")
    seeds = cfg.paths.seed_accounts.read_text().split() if cfg.paths.seed_accounts else None
    work = cfg.paths.stage("relevance")
    scored, report = relevance.score(posts, cfg.relevance, work, seeds, device=cfg.embed.device, seed=cfg.splits.seed)
    thr = report["threshold"]
    audit_path = work / "audit_labeled.csv"
    if audit_path.exists():
        t = relevance.threshold_from_audit(pl.read_csv(audit_path), cfg.relevance.target_precision)
        if t is not None:
            thr, report["threshold"], report["threshold_source"] = t, t, "audit"
    kept = relevance.decide(scored, cfg.relevance, thr)
    write_parquet(kept.drop([c for c in ("near_dup_cluster_id", "near_dup_rank", "near_dup_cluster_size") if c in kept.columns]), work / "posts.parquet")
    relevance.audit_sample(scored, kept["post_id"], cfg.relevance.audit_sample, thr).write_csv(work / "audit_to_label.csv")
    (work / "report.json").write_text(json.dumps(report, indent=2))


def feature_stage(cfg: Config, blocklist_dir: str | None = None) -> None:
    w = cfg.paths.work
    posts = _read(w / "relevance" / "posts.parquet")
    a = w / "authors"
    df = (
        posts.join(_read(a / "followers_at_post.parquet"), on="post_id", how="left")
        .join(_read(a / "trailing.parquet"), on="post_id", how="left")
        .join(_read(a / "copypasta.parquet").select("post_id", "is_self_repeat"), on="post_id", how="left")
        .join(_read(a / "authors.parquet").select(
            "author_id", "verified", "mutual_share_est", "reply_share", "copypasta_events",
            *(["graph_pagerank"] if "graph_pagerank" in _read(a / "authors.parquet").columns else [])), on="author_id", how="left")
    )
    nd = _maybe(w / "dedup" / "near_dup.parquet")
    if nd is not None:
        df = df.join(nd, on="post_id", how="left")
    df = derive.author_columns(df)
    df = df.join(content.content_features(df), on="post_id", how="left")
    df = df.join(text_structure.admitted(df), on="post_id", how="left")
    zones = _read(a / "audience_zones.parquet")
    df = df.join(audience.audience_time_features(df, zones), on="post_id", how="left")
    df = df.join(suppression.url_features(df, suppression.load_blocklists(blocklist_dir)), on="post_id", how="left")
    write_parquet(df, cfg.paths.stage("features") / "posts.parquet")


def dataset_stage(cfg: Config) -> None:
    df = _read(cfg.paths.work / "features" / "posts.parquet")
    df = labels.build(df, cfg.labels)
    df = splits.assign(df, cfg.splits)
    p = labels.pairs(df, cfg.labels, seed=cfg.splits.seed)
    out = cfg.paths.stage("dataset")
    write_parquet(df, out / "rows.parquet")
    write_parquet(p, out / "pairs.parquet")
    write_parquet(splits.summary(df), out / "splits.parquet")


def report_stage(cfg: Config) -> dict:
    out = cfg.paths.stage("report")
    df = _read(cfg.paths.work / "dataset" / "rows.parquet")
    pairs = _read(cfg.paths.work / "dataset" / "pairs.parquet")
    write_parquet(audit.count_table(df), out / "counts.parquet")
    audit.count_table(df).write_csv(out / "counts.csv")
    labels.label_summary(df).write_csv(out / "labels_by_tier.csv")
    flog = _maybe(cfg.paths.work / "filtered" / "filter_log.parquet")
    if flog is not None:
        flog.write_csv(out / "filter_log.csv")
    head = audit.headline(df)
    head["pairs"] = pairs.height
    head["pairs_by_split"] = dict(pairs.group_by("split").len().iter_rows()) if pairs.height else {}
    (out / "headline.json").write_text(json.dumps(head, indent=2, default=str))
    return head


def baseline_stage(cfg: Config) -> dict:
    df = _read(cfg.paths.work / "dataset" / "rows.parquet")
    pairs = _read(cfg.paths.work / "dataset" / "pairs.parquet")
    res = baseline.run(df, pairs, seed=cfg.splits.seed)
    (cfg.paths.stage("report") / "baseline.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def phase0(cfg: Config, blocklist_dir: str | None = None, limit: int | None = None) -> dict:
    ingest(cfg, limit=limit)
    unify(cfg)
    filter_rows(cfg)
    author_stage(cfg)
    hygiene_stage(cfg)
    dedup_stage(cfg)
    relevance_stage(cfg)
    feature_stage(cfg, blocklist_dir)
    dataset_stage(cfg)
    head = report_stage(cfg)
    head["baseline"] = baseline_stage(cfg)
    return head


# --- Phase 1 (GPU) ---------------------------------------------------------------


def media_select(cfg: Config) -> None:
    from xviral.media.fetch import select_media

    df = _read(cfg.paths.work / "dataset" / "rows.parquet")
    imgs, vids = select_media(df, cfg.media)
    out = cfg.paths.stage("media")
    write_parquet(imgs, out / "images_selected.parquet")
    write_parquet(vids, out / "videos_selected.parquet")


def fetch_images(cfg: Config) -> None:
    from xviral.media.fetch import fetch_images as fi

    sel = _read(cfg.paths.work / "media" / "images_selected.parquet")
    fi(sel, cfg.paths.stage("media") / "images", cfg.media)


def fetch_videos(cfg: Config, shard_rows: int = 200) -> None:
    from xviral.media import video
    from xviral.media.fetch import RateLimiter

    sel = _read(cfg.paths.work / "media" / "videos_selected.parquet")
    out = cfg.paths.stage("media") / "videos"
    out.mkdir(parents=True, exist_ok=True)
    done = set(scan_dir(out).select("post_id").collect()["post_id"].to_list()) if any(out.glob("*.parquet")) else set()
    todo = [r for r in sel.select("post_id", "media_idx", "video_url").iter_rows() if r[0] not in done]
    limiter = RateLimiter(cfg.media.requests_per_second)
    shard = len(list(out.glob("*.parquet")))
    rows: list[dict] = []
    for pid, idx, url in todo:
        limiter.wait()
        rec = {"post_id": pid, "media_idx": idx, "status": "ok"}
        try:
            path = video.download(url, timeout=cfg.media.timeout_s * 3)
            try:
                s = video.sample(path, side=cfg.media.vlm_side, embed_n=cfg.media.embed_frames, temporal_n=cfg.media.temporal_frames,
                                 max_audio_s=cfg.media.max_video_seconds_for_asr)
            finally:
                path.unlink(missing_ok=True)
            rec.update(duration_s=s.probe.duration_s, width=s.probe.width, height=s.probe.height, fps=s.probe.fps,
                       has_audio=s.probe.has_audio, resolution_class=s.probe.resolution_class, embed_frames=s.embed,
                       temporal_times=[t for t, _ in s.temporal], temporal_frames=[b for _, b in s.temporal],
                       thumb=s.thumb, audio_flac=s.audio_flac)
        except Exception as e:  # noqa: BLE001 - recorded per video
            rec["status"] = f"error: {type(e).__name__}"
        rows.append(rec)
        if len(rows) >= shard_rows:
            write_parquet(pl.DataFrame(rows, infer_schema_length=None), out / f"videos-{shard:05d}.parquet")
            shard += 1
            rows = []
    if rows:
        write_parquet(pl.DataFrame(rows, infer_schema_length=None), out / f"videos-{shard:05d}.parquet")


def asr_stage(cfg: Config) -> None:
    from xviral.media.asr import transcribe

    vids = scan_dir(cfg.paths.work / "media" / "videos").filter(pl.col("audio_flac").is_not_null()).select("post_id", "audio_flac").collect()
    recs = []
    for pid, audio in vids.iter_rows():
        t = transcribe(audio, name=cfg.embed.whisper_model, device=cfg.embed.device)
        recs.append({"post_id": pid, "transcript": t.text, "asr_lang": t.language, "speech_ratio": t.speech_ratio,
                     "speech_in_first_3s": t.speech_in_first_3s, "transcript_len": len(t.text)})
    write_parquet(pl.DataFrame(recs), cfg.paths.stage("media") / "transcripts.parquet")


def _media_lookup(cfg: Config) -> tuple[dict, dict, dict]:
    m = cfg.paths.work / "media"
    images: dict[int, list[bytes]] = {}
    if (m / "images").exists() and any((m / "images").glob("*.parquet")):
        for pid, idx, jpg in scan_dir(m / "images").filter(pl.col("status") == 200).select("post_id", "media_idx", "jpeg").sort("post_id", "media_idx").collect().iter_rows():
            images.setdefault(pid, []).append(jpg)
    frames: dict[int, list[list[bytes]]] = {}
    if (m / "videos").exists() and any((m / "videos").glob("*.parquet")):
        for pid, ef in scan_dir(m / "videos").filter(pl.col("status") == "ok").select("post_id", "embed_frames").collect().iter_rows():
            frames.setdefault(pid, []).append(ef or [])
    trans: dict[int, str] = {}
    if (m / "transcripts.parquet").exists():
        trans = dict(_read(m / "transcripts.parquet").select("post_id", "transcript").iter_rows())
    return images, frames, trans


def embed_mm(cfg: Config, batch: int = 512) -> None:
    from xviral.embed import mm
    from xviral.embed.store import EmbeddingStore

    df = _read(cfg.paths.work / "dataset" / "rows.parquet")
    store = EmbeddingStore(cfg.paths.stage("emb") / "mm", dim=1024)
    todo = set(store.missing(df["post_id"].to_list()))
    df = df.filter(pl.col("post_id").is_in(list(todo)))
    images, frames, trans = _media_lookup(cfg)
    texts = dict(_read(cfg.paths.work / "dataset" / "rows.parquet").select("post_id", "text").iter_rows())
    enc = mm.MMEncoder(device=cfg.embed.device, batch_size=cfg.embed.batch_size)
    rows = df.select("post_id", "text", "text_note", "author_handle", "author_display_name", "quoted_post_id").to_dicts()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        rendered = []
        for r in chunk:
            q = {"text": texts.get(r["quoted_post_id"])} if r.get("quoted_post_id") in texts else None
            post = mm.build_post(r, images.get(r["post_id"]), frames.get(r["post_id"]), q)
            rendered.append(mm.render(post, transcript=trans.get(r["post_id"]), max_images=cfg.embed.mm_max_images))
        vecs = enc.encode(rendered)
        store.write(np.array([r["post_id"] for r in chunk], dtype=np.int64), vecs)
        log.info("embed-mm %d/%d", min(i + batch, len(rows)), len(rows))


def sid_train(cfg: Config) -> None:
    from xviral.embed import sid
    from xviral.embed.store import EmbeddingStore

    ids, vecs = EmbeddingStore(cfg.paths.work / "emb" / "mm").all()
    npy = cfg.paths.stage("emb") / "mm_train.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    sid.train(npy, cfg.paths.work / "emb" / "sid_codebook.npz", cfg.embed.sid_levels, cfg.embed.sid_centroids,
              cfg.embed.sid_iters, cfg.embed.sid_sample)


def sid_assign(cfg: Config) -> None:
    from xviral.embed import sid
    from xviral.embed.store import EmbeddingStore

    ids, vecs = EmbeddingStore(cfg.paths.work / "emb" / "mm").all()
    codes = sid.phoenix_codes(sid.assign(np.asarray(vecs, dtype=np.float32), cfg.paths.work / "emb" / "sid_codebook.npz"))
    df = pl.DataFrame({"post_id": ids, **{f"sid_{i}": codes[:, i] for i in range(codes.shape[1])}})
    write_parquet(df, cfg.paths.stage("emb") / "sid.parquet")


def embed_siglip(cfg: Config) -> None:
    from xviral.embed.siglip import SiglipEncoder
    from xviral.embed.store import EmbeddingStore
    from xviral.media.video import frame_motion

    m = cfg.paths.work / "media"
    enc = SiglipEncoder(cfg.embed.siglip_model, device=cfg.embed.device)
    if (m / "images").exists():
        im = scan_dir(m / "images").filter(pl.col("status") == 200).select("post_id", "media_idx", "jpeg").collect()
        st = EmbeddingStore(cfg.paths.stage("emb") / "siglip_images")
        # Keyed by a stable 63-bit hash of (post_id, media_idx); post_id * k would overflow int64.
        im = im.with_columns(pl.struct("post_id", "media_idx").map_elements(
            lambda r: stable_hash64("img", r["post_id"], r["media_idx"]), return_dtype=pl.Int64).alias("key"))
        write_parquet(im.select("key", "post_id", "media_idx"), cfg.paths.stage("emb") / "siglip_images_keys.parquet")
        for i in range(0, im.height, 4096):
            c = im.slice(i, 4096)
            st.write(c["key"].to_numpy(), enc.encode(c["jpeg"].to_list()))
    if (m / "videos").exists():
        vids = scan_dir(m / "videos").filter(pl.col("status") == "ok").select("post_id", "temporal_frames", "thumb").collect()
        st_f = EmbeddingStore(cfg.paths.stage("emb") / "siglip_thumbs")
        recs = []
        for pid, tf, thumb in vids.iter_rows():
            fv = enc.encode(list(tf or []))
            recs.append({"post_id": pid, "frame_motion": frame_motion(fv), "frames_n": len(fv),
                         "frame_emb": fv.astype(np.float16).tolist()})
            if thumb:
                st_f.write(np.array([pid]), enc.encode([thumb]))
        write_parquet(pl.DataFrame(recs), cfg.paths.stage("emb") / "video_frames.parquet")


def neighbors_stage(cfg: Config) -> None:
    from xviral.embed.store import EmbeddingStore
    from xviral.features.neighbors import author_centroid_and_selfsim, global_neighbors

    df = _read(cfg.paths.work / "dataset" / "rows.parquet")
    st = EmbeddingStore(cfg.paths.work / "emb" / "mm")
    vecs, found = st.get(df["post_id"].to_list(), fill_nan=False)
    cols = ["post_id", "author_id", "created_at_ms", "log_reach_ratio", "log_like_ratio", "rate_fav", "rate_reply", "follower_tier_idx"]
    meta = df.filter(pl.Series(found)).select([c for c in cols if c in df.columns])
    vecs = vecs[found]
    nb = global_neighbors(meta, vecs)
    cent, ss = author_centroid_and_selfsim(meta, vecs, window=cfg.labels.trailing_window)
    out = cfg.paths.stage("emb")
    write_parquet(nb.join(ss, on="post_id"), out / "neighbors.parquet")
    EmbeddingStore(out / "author_centroid", dim=vecs.shape[1]).write(meta["post_id"].to_numpy(), np.nan_to_num(cent))


# --- Model (MODEL.md) ---------------------------------------------------------------


def model_prepare(cfg: Config) -> dict:
    """Build the model's feature table from the dataset stage and whatever embedding stages have run."""
    from xviral.embed.store import EmbeddingStore
    from xviral.model import features

    w = cfg.paths.work
    rows = _read(w / "dataset" / "rows.parquet")
    emb = EmbeddingStore(w / "emb" / "mm") if (w / "emb" / "mm").exists() and any((w / "emb" / "mm").glob("shard-*.npy")) else None
    sid = _maybe(w / "emb" / "sid.parquet")
    nbrs = _maybe(w / "emb" / "neighbors.parquet")
    extra = None
    for p, cols in ((w / "media" / "transcripts.parquet", ["post_id", "speech_ratio", "speech_in_first_3s", "transcript_len"]),
                    (w / "emb" / "video_frames.parquet", ["post_id", "frame_motion"])):
        f = _maybe(p)
        if f is not None:
            f = f.select(cols).unique("post_id")
            extra = f if extra is None else extra.join(f, on="post_id", how="full", coalesce=True)
    return features.build(rows, cfg.paths.stage("model") / "table", emb=emb, sid=sid, neighbors=nbrs, extra=extra)


def model_train(cfg: Config, preset: str = "base", steps: int | None = None, device: str | None = None, name: str = "run",
                live_model: str | None = None, init: str | None = None, batch_size: int | None = None, seed: int = 0) -> dict:
    """Stage A by default. With `live_model`, stage B: that encoder with LoRA adapters trains through the
    model's loss, normally warm-started from a stage-A checkpoint via `init`."""
    from xviral.model import train as tr
    from xviral.model.config import TrainCfg
    from xviral.model.features import FeatureTable

    dev = device or cfg.embed.device
    table = FeatureTable.load(cfg.paths.work / "model" / "table", dev)
    kw = {"seed": seed}
    if steps:
        kw["steps"] = steps
    if batch_size:
        kw["batch_size"] = batch_size
    overrides = {}
    if live_model:
        overrides["live"] = {"model_id": live_model, "device": dev, "grad_checkpointing": dev != "cpu",
                             "dtype": "bfloat16" if dev != "cpu" else None}
    res = tr.train(table, cfg.paths.stage("model") / name, tr.model_cfg_for(table, preset, **overrides), TrainCfg(**kw), init_from=init)
    return {"best": res["best"], "params": res["params"]}


def ca_download(cfg: Config, mode: str = "donors", shards: int = 4, max_rows_per_account: int = 20_000) -> dict:
    """Community Archive download.

    donors  every donor account's posts since community_archive.min_created, through the account index
    full    every post since min_created in `shards` parallel time ranges, plus account, media, URL and
            quote tables: the stream-captured rows of hundreds of thousands of accounts
    tables  the tables listed in community_archive.tables, each paged whole
    """
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, timezone

    from xviral.sources.community_archive import RestDownloader

    out = cfg.paths.community_archive
    if out is None:
        raise ValueError("set paths.community_archive in the config")
    dl = RestDownloader(cfg.community_archive, out)
    if mode == "tables":
        return {t: dl.download(t) for t in cfg.community_archive.tables}
    donors = dl.donor_accounts()["account_id"].to_list()
    if mode == "donors":
        return {"donors": len(donors), "accounts": dl.download_accounts_meta(donors),
                "tweets": dl.download_account_tweets(donors, max_rows_per_account=max_rows_per_account)}
    lo = datetime.fromisoformat(cfg.community_archive.min_created).replace(tzinfo=timezone.utc)
    hi = datetime.now(timezone.utc) + timedelta(days=1)
    edges = [lo + (hi - lo) * i / shards for i in range(shards + 1)]

    def one(i: int) -> int:
        return RestDownloader(cfg.community_archive, out).download(
            "tweets", lo=edges[i].isoformat(), hi=edges[i + 1].isoformat(), tag=f"s{i:02d}")

    with ThreadPoolExecutor(max_workers=shards) as ex:
        rows = sum(ex.map(one, range(shards)))
    res = {"tweets": rows}
    for t in ("all_account", "tweet_media", "tweet_urls", "quote_tweets"):
        res[t] = dl.download(t)
    return res


def model_eval(cfg: Config, device: str | None = None, name: str = "run") -> dict:
    """Test-split metrics for the best checkpoint, after fitting temperatures on validation rows."""
    from xviral.model import evaluate
    from xviral.model import train as tr
    from xviral.model.features import FeatureTable

    dev = device or cfg.embed.device
    table = FeatureTable.load(cfg.paths.work / "model" / "table", dev)
    run = cfg.paths.work / "model" / name
    model, tcfg, ck = tr.load(run / "best.pt", dev)
    temps = evaluate.fit_temperature(model, table, table.rows_of_split("val", audience_only=True), tcfg)
    tr.save(run / "best.pt", model, model.cfg, tcfg, ck["table_meta"], ck["metrics"], temps)
    test = table.rows_of_split("test", audience_only=True)
    res = {"temperature": temps,
           "test": evaluate.metrics(evaluate.predict_rows(model, table, test, tcfg, temperature=temps), table, tcfg)}
    (run / "eval.json").write_text(json.dumps(res, indent=2, default=float))
    return res
