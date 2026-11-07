"""Command line entry point: `xviral <stage> --config configs/pipeline.yaml`."""

from __future__ import annotations

import argparse
import json
import logging

from xviral import pipeline
from xviral.config import Config
from xviral.util.core import setup_logging

STAGES = {
    "ingest": pipeline.ingest,
    "unify": pipeline.unify,
    "filter": pipeline.filter_rows,
    "authors": pipeline.author_stage,
    "hygiene": pipeline.hygiene_stage,
    "dedup": pipeline.dedup_stage,
    "relevance": pipeline.relevance_stage,
    "features": pipeline.feature_stage,
    "dataset": pipeline.dataset_stage,
    "report": pipeline.report_stage,
    "baseline": pipeline.baseline_stage,
    "phase0": pipeline.phase0,
    "media-select": pipeline.media_select,
    "fetch-images": pipeline.fetch_images,
    "fetch-videos": pipeline.fetch_videos,
    "asr": pipeline.asr_stage,
    "embed-mm": pipeline.embed_mm,
    "sid-train": pipeline.sid_train,
    "sid-assign": pipeline.sid_assign,
    "embed-siglip": pipeline.embed_siglip,
    "neighbors": pipeline.neighbors_stage,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="xviral")
    ap.add_argument("stage", choices=[*STAGES, "ca-download"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="ingest: only the first N USC chunks (smoke runs)")
    ap.add_argument("--blocklists", default=None, help="features/phase0: directory of domain blocklist .txt files")
    ap.add_argument("--tables", nargs="*", default=None, help="ca-download: subset of tables")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    cfg = Config.load(args.config)

    if args.stage == "ca-download":
        from xviral.sources.community_archive import RestDownloader

        if cfg.paths.community_archive is None:
            ap.error("set paths.community_archive in the config")
        dl = RestDownloader(cfg.community_archive, cfg.paths.community_archive)
        for t in args.tables or cfg.community_archive.tables:
            dl.download(t)
        return
    fn = STAGES[args.stage]
    kwargs = {}
    if args.stage in ("ingest", "phase0"):
        kwargs["limit"] = args.limit
    if args.stage in ("features", "phase0"):
        kwargs["blocklist_dir"] = args.blocklists
    res = fn(cfg, **kwargs)
    if res is not None:
        logging.getLogger("xviral").info("result:\n%s", json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
