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
    "embed-text": pipeline.embed_text,
    "sid-train": pipeline.sid_train,
    "sid-assign": pipeline.sid_assign,
    "embed-siglip": pipeline.embed_siglip,
    "neighbors": pipeline.neighbors_stage,
    "model-prepare": pipeline.model_prepare,
    "model-train": pipeline.model_train,
    "model-eval": pipeline.model_eval,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="xviral")
    ap.add_argument("stage", choices=[*STAGES, "ca-download"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="ingest: only the first N USC chunks (smoke runs)")
    ap.add_argument("--blocklists", default=None, help="features/phase0: directory of domain blocklist .txt files")
    ap.add_argument("--mode", default="donors", choices=["donors", "full", "tables"], help="ca-download: what to pull")
    ap.add_argument("--shards", type=int, default=4, help="ca-download --mode full: parallel time shards")
    ap.add_argument("--live-model", default=None, help="model-train: stage B encoder id (LoRA-tuned through the model's loss)")
    ap.add_argument("--init", default=None, help="model-train: checkpoint to warm-start from")
    ap.add_argument("--batch-size", type=int, default=None, help="model-train: episodes per step")
    ap.add_argument("--seed", type=int, default=0, help="model-train: random seed (vary it to build an ensemble)")
    ap.add_argument("--preset", default="base", help="model-train: nano | small | base | large")
    ap.add_argument("--steps", type=int, default=None, help="model-train: optimisation steps")
    ap.add_argument("--device", default=None, help="model stages: cpu | cuda (default: embed.device)")
    ap.add_argument("--run", default="run", help="model stages: run directory name under work/model/")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    cfg = Config.load(args.config)

    if args.stage == "ca-download":
        res = pipeline.ca_download(cfg, mode=args.mode, shards=args.shards)
        logging.getLogger("xviral").info("result: %s", res)
        return
    fn = STAGES[args.stage]
    kwargs = {}
    if args.stage in ("ingest", "phase0"):
        kwargs["limit"] = args.limit
    if args.stage in ("features", "phase0"):
        kwargs["blocklist_dir"] = args.blocklists
    if args.stage == "model-train":
        kwargs.update(preset=args.preset, steps=args.steps, device=args.device, name=args.run, live_model=args.live_model,
                      init=args.init, batch_size=args.batch_size, seed=args.seed)
    if args.stage == "model-eval":
        kwargs.update(device=args.device, name=args.run)
    res = fn(cfg, **kwargs)
    if res is not None:
        logging.getLogger("xviral").info("result:\n%s", json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
