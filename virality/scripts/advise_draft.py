#!/usr/bin/env python3
"""The advice report for one draft: rules from X's code, the account's own record, and scored single changes.

    python scripts/advise_draft.py --checkpoint a/best.pt --checkpoint b/best.pt --scales pair_confidence.json \
        --posts <work>/hygiene/posts.parquet --handle acme --draft "the draft" [--alt "another version"] [--json out.json]

Give at least two checkpoints so the agreement gate has members to compare. --scales comes from
scripts/fit_pair_confidence.py. A change is recommended only when every gate in xviral/advise.py passes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--posts", required=True)
    ap.add_argument("--handle", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--alt", action="append", default=[], help="another version of the draft to compare, repeatable")
    ap.add_argument("--followers", type=int, default=None)
    ap.add_argument("--when", default=None, help="posting time, ISO 8601 UTC; default: one day after the account's last post in --posts")
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sid-codebook", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from xviral import advise
    from xviral.embed.text_small import SentenceEmbedder
    from xviral.serve import DraftScorer

    handle = args.handle.lstrip("@").lower()
    history = pl.scan_parquet(args.posts).filter(pl.col("author_handle").str.to_lowercase() == handle).collect().sort("created_at_ms")
    followers = args.followers or (history["author_followers"].drop_nulls().tail(1).item() if history.height and history["author_followers"].drop_nulls().len() else None)
    when = int(datetime.fromisoformat(args.when).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.when else (
        int(history["created_at_ms"].max()) + 86_400_000 if history.height else None)
    enc = SentenceEmbedder(args.encoder, device=args.device)
    scorer = DraftScorer(args.checkpoint if len(args.checkpoint) > 1 else args.checkpoint[0], lambda texts: enc.encode(texts), device=args.device,
                         sid_codebook=args.sid_codebook)
    scales = {k: v for k, v in json.load(open(args.scales)).items() if k in ("views", "likes")}
    account = {"handle": args.handle.lstrip("@"), "followers": followers}
    if not history.height:
        account["author_id"] = -1
    out = advise.Advisor(scorer, scales).advise(history, {"text": args.draft}, account, when,
                                                alternatives=[{"text": t, "label": f"your version {i + 2}"} for i, t in enumerate(args.alt)])
    print(f"@{account['handle']}: {history.height} posts of history, {followers or 0:,} followers\n")
    print(advise.render(out))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
