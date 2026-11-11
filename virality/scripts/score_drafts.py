#!/usr/bin/env python3
"""Score drafts for an account with a trained model.

    python scripts/score_drafts.py --checkpoint data/work/model/run/best.pt --posts data/work/hygiene/posts.parquet \
        --handle acme --draft "first version" --draft "second version" [--timing] [--json out.json]

--posts is any Parquet file in the unified schema that holds the account's past posts (the hygiene
stage's output works). The follower count defaults to the latest one in that file. --encoder must
be the encoder the model's embeddings came from (embed.text_model for a development run).
Pass --checkpoint several times to score with an ensemble.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--posts", required=True)
    ap.add_argument("--handle", required=True)
    ap.add_argument("--draft", action="append", required=True)
    ap.add_argument("--followers", type=int, default=None)
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sid-codebook", default=None, help="<work>/emb/sid_codebook.npz when the model was trained with semantic IDs")
    ap.add_argument("--when", default=None, help="posting time, ISO 8601 UTC; default: one day after the account's last post in --posts")
    ap.add_argument("--timing", action="store_true", help="also sweep posting times over the following 48 hours for the first draft")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from xviral.embed.text_small import SentenceEmbedder
    from xviral.serve import DraftScorer

    handle = args.handle.lstrip("@").lower()
    history = pl.scan_parquet(args.posts).filter(pl.col("author_handle").str.to_lowercase() == handle).collect().sort("created_at_ms")
    if history.height == 0:
        print(f"no posts by @{handle} in {args.posts}: scoring without history")
    followers = args.followers or (history["author_followers"].drop_nulls().tail(1).item() if history["author_followers"].drop_nulls().len() else None)
    when = int(datetime.fromisoformat(args.when).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.when else (
        int(history["created_at_ms"].max()) + 86_400_000 if history.height else None)
    enc = SentenceEmbedder(args.encoder, device=args.device)
    scorer = DraftScorer(args.checkpoint if len(args.checkpoint) > 1 else args.checkpoint[0], lambda texts: enc.encode(texts), device=args.device, sid_codebook=args.sid_codebook)
    account = {"handle": args.handle.lstrip("@"), "followers": followers}
    if not history.height:
        account["author_id"] = -1
    drafts = [{"text": t} for t in args.draft]
    reports = scorer.score(history, drafts, account, when)

    num = lambda v: "-" if v is None else f"{v:,.0f}"       # noqa: E731
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    print(f"\n@{account['handle']}: {history.height} posts of history, {num(followers)} followers")
    print(f"{'draft':<52} {'views q10':>10} {'median':>10} {'q90':>10} {'likes med':>10} {'own top 10%':>12} {'vs own median':>14}")
    for r in reports:
        lift = r["lift_vs_own_median"]["views"] or r["lift_vs_own_median"]["likes"]
        top = r["p_own_top_10pct"]["views"] if r["p_own_top_10pct"]["views"] is not None else r["p_own_top_10pct"]["likes"]
        text = " ".join((r["text"] or "").split())
        print(f"{text[:50]:<52} {num(r['views_24h']['q10']):>10} {num(r['views_24h']['q50']):>10} {num(r['views_24h']['q90']):>10} "
              f"{num(r['likes']['q50']):>10} {pct(top):>12} {'-' if lift is None else f'{lift:.2f}x':>14}")
    flagged = [g for g in reports[0]["gates"] if g["status"] in ("warn", "fail")]
    for g in flagged:
        print(f"  [{g['status']}] {g['gate']}: {g['detail']}")
    out = {"account": account, "history_posts": history.height, "reports": reports}
    if args.timing:
        sweep = scorer.timing(history, drafts[0], account, start_ms=when)
        out["timing"] = sweep
        print("\nposting time for the first draft, best first:")
        for r in sweep[:5]:
            t = datetime.fromtimestamp(r["when_ms"] / 1000, tz=timezone.utc).strftime("%a %H:%M UTC")
            print(f"  {t}  views median {num(r['views_q50'])}  likes median {num(r['likes_q50'])}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
