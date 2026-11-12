#!/usr/bin/env python3
"""The product's central claim, tested the way the product is used.

    python scripts/evidence/tac_serve_eval.py --posts <work>/filtered/posts.parquet --checkpoint <run>/best.pt \
        --after 2026-04-23 --max-pairs 1500 --out tac_serve.json

Takes pairs of posts by one account about the same topic (same quoted post or same link, different
wording, at most 48 hours apart), created after --after, from the whole filtered corpus. For each
pair the two posts are handed to DraftScorer as alternative drafts at the time of the earlier one,
with only the account's earlier posts as history: neither side's outcome is visible, and neither
side sees the other. The question is whether the model prefers the version that really did better.

Pairs before --after from accounts the model was not trained on calibrate the confidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import polars as pl

from xviral.features import text_structure as ts
from xviral.model import evaluate

log = logging.getLogger("tac")
COLS = ["post_id", "author_id", "author_handle", "author_display_name", "created_at_ms", "snapshot_at_ms", "text", "kind", "media", "urls",
        "quoted_post_id", "quoted_text", "likes", "replies", "reposts", "quotes", "views", "author_followers", "author_following", "source"]


def find_pairs(d: pl.DataFrame, max_gap_h: float, max_uses: int = 4) -> pl.DataFrame:
    u = d.select("post_id", pl.col("urls").list.eval(pl.element().struct.field("expanded")).alias("u")).explode("u").filter(pl.col("u").is_not_null())
    u = u.with_columns(pl.col("u").str.to_lowercase().str.replace(r"^https?://(www\.)?", "").str.strip_chars_end("/").alias("key"))
    u = u.filter(~pl.col("key").str.contains(r"^(x\.com|twitter\.com|t\.co)") & (pl.col("key").str.len_chars() > 12) & pl.col("key").str.contains("/"))
    u = u.unique(["post_id", "key"]).select("post_id", pl.concat_str([pl.lit("url:"), pl.col("key")]).alias("key"))
    q = d.filter(pl.col("quoted_post_id").is_not_null()).select("post_id", pl.concat_str([pl.lit("quote:"), pl.col("quoted_post_id").cast(pl.String)]).alias("key"))
    norm = pl.col("text").fill_null("").str.to_lowercase().str.replace_all(r"https?://\S+", "").str.replace_all(r"\s+", " ").str.strip_chars()
    k = pl.concat([u, q]).join(d.select("post_id", "author_id", "created_at_ms", "likes", "views", norm.alias("nt")), on="post_id")
    uses = k.group_by("author_id", "key").len().filter(pl.col("len").is_between(2, max_uses))
    k = k.join(uses.select("author_id", "key"), on=["author_id", "key"], how="semi")
    p = k.join(k, on=["author_id", "key"], suffix="_b").filter((pl.col("created_at_ms") < pl.col("created_at_ms_b")) & (pl.col("nt") != pl.col("nt_b")))
    return p.filter((pl.col("created_at_ms_b") - pl.col("created_at_ms")) <= max_gap_h * 3_600_000).unique(["post_id", "post_id_b"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--after", required=True, help="test cutoff date, the one the model's split used")
    ap.add_argument("--trained-accounts", default=None, help="dataset rows.parquet: its accounts are excluded from the calibration pairs")
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--max-calibration-pairs", type=int, default=600)
    ap.add_argument("--max-gap-hours", type=float, default=48.0)
    ap.add_argument("--history", type=int, default=80)
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sid-codebook", default=None, help="<work>/emb/sid_codebook.npz when the model was trained with semantic IDs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from xviral.embed.text_small import SentenceEmbedder
    from xviral.serve import DraftScorer
    from xviral.util.core import date_to_ms

    d = pl.scan_parquet(args.posts).select(COLS).collect()
    pairs = find_pairs(d, args.max_gap_hours)
    cut = date_to_ms(args.after)
    trained = set(pl.read_parquet(args.trained_accounts, columns=["author_id"])["author_id"].unique().to_list()) if args.trained_accounts else set()
    test = pairs.filter(pl.col("created_at_ms") >= cut)
    calib = pairs.filter((pl.col("created_at_ms_b") < cut) & ~pl.col("author_id").is_in(list(trained)))
    log.info("pairs: %d in the test period, %d for calibration (accounts not trained on)", test.height, calib.height)
    test = test.sample(min(args.max_pairs, test.height), seed=0).with_columns(pl.lit("test").alias("part"))
    calib = calib.sample(min(args.max_calibration_pairs, calib.height), seed=0).with_columns(pl.lit("calibration").alias("part"))
    todo = pl.concat([test, calib])
    by_author = {a: g.sort("created_at_ms") for (a,), g in d.filter(pl.col("author_id").is_in(todo["author_id"].unique().implode())).group_by("author_id")}
    by_id = {r["post_id"]: r for r in d.filter(pl.col("post_id").is_in(pl.concat([todo["post_id"], todo["post_id_b"]]).unique().implode())).iter_rows(named=True)}

    enc = SentenceEmbedder(args.encoder, device=args.device)
    scorer = DraftScorer(args.checkpoint if len(args.checkpoint) > 1 else args.checkpoint[0], lambda t: enc.encode(t), device=args.device, sid_codebook=args.sid_codebook)
    # Every scored pair is appended to a line file as it is produced, and a rerun picks up where it stopped.
    lines = args.out.replace(".json", ".scored.jsonl")
    recs = [json.loads(x) for x in open(lines)] if os.path.exists(lines) else []
    done = {(r["post_id"], r["post_id_b"]) for r in recs}
    sink = open(lines, "a")
    for i, r in enumerate(todo.iter_rows(named=True)):
        if (r["post_id"], r["post_id_b"]) in done:
            continue
        a, b = by_id[r["post_id"]], by_id[r["post_id_b"]]
        hist = by_author[r["author_id"]].filter(pl.col("created_at_ms") < a["created_at_ms"]).tail(args.history)
        account = {"author_id": r["author_id"], "handle": a["author_handle"], "display_name": a["author_display_name"],
                   "followers": a["author_followers"], "following": a["author_following"]}
        drafts = [{"text": x["text"], "kind": x["kind"], "media": x["media"], "urls": x["urls"], "quoted_text": x["quoted_text"]} for x in (a, b)]
        try:
            ra, rb = scorer.score(hist, drafts, account, a["created_at_ms"])
        except Exception as e:  # noqa: BLE001
            log.warning("pair %d skipped: %s", i, e)
            continue
        recs.append({"part": r["part"], "post_id": r["post_id"], "post_id_b": r["post_id_b"], "author_id": r["author_id"], "history": hist.height,
                     "likes": a["likes"], "likes_b": b["likes"], "views": a["views"], "views_b": b["views"],
                     "s_likes": float(np.log10(ra["likes"]["q50"] + 1) - np.log10(rb["likes"]["q50"] + 1)),
                     "s_views": float(np.log10(ra["views_24h"]["q50"] + 1) - np.log10(rb["views_24h"]["q50"] + 1))})
        sink.write(json.dumps(recs[-1]) + "\n")
        sink.flush()
        if (i + 1) % 100 == 0:
            log.info("scored %d/%d pairs", i + 1, todo.height)
    sink.close()
    R = pl.DataFrame(recs, schema={"part": pl.String, "post_id": pl.Int64, "post_id_b": pl.Int64, "author_id": pl.Int64, "history": pl.Int64,
                                   "likes": pl.Int64, "likes_b": pl.Int64, "views": pl.Int64, "views_b": pl.Int64, "s_likes": pl.Float64, "s_views": pl.Float64})
    R.write_parquet(args.out.replace(".json", ".parquet"))

    # What differs between the two versions.
    posts = pl.DataFrame([by_id[x] for x in pl.concat([R["post_id"], R["post_id_b"]]).unique().to_list()], schema={c: d.schema[c] for c in COLS})
    f = ts.text_structure(posts).select("post_id", "ts_lines", "ts_words", "ts_url_first_line").join(posts.select("post_id", (pl.col("media").list.len() > 0).alias("m")), on="post_id")
    R = R.join(f, on="post_id").join(f.rename({c: c + "_b" for c in f.columns}), on="post_id_b")
    res: dict = {"scored_pairs": R.height, "accounts": R["author_id"].n_unique()}
    for target in ("likes", "views"):
        q = R.filter(pl.col(target).is_not_null() & pl.col(target + "_b").is_not_null())
        dd = ((q[target].cast(pl.Float64) + 1).log() - (q[target + "_b"].cast(pl.Float64) + 1).log()).to_numpy()
        keep = np.abs(dd) >= np.log(1.25)
        q, dd = q.filter(pl.Series(keep)), dd[keep]
        if q.height < 50:
            continue
        s = q["s_" + target].to_numpy()
        test_m, cal_m = (q["part"] == "test").to_numpy(), (q["part"] == "calibration").to_numpy()
        correct = (s * dd) > 0
        first_wins = dd > 0                                   # A is always the earlier post
        out = {"pairs": int(test_m.sum()), "accuracy": float(correct[test_m].mean()),
               "accuracy_order_balanced": float(np.mean([correct[test_m & first_wins].mean(), correct[test_m & ~first_wins].mean()])),
               "later_version_wins": float((~first_wins[test_m]).mean()), "model_prefers_later": float((s[test_m] < 0).mean()),
               "with_history_20_plus": float(correct[test_m & (q["history"].to_numpy() >= 20)].mean())}
        if cal_m.sum() >= 100:
            sig = lambda k, v: 1 / (1 + np.exp(-k * v))       # noqa: E731
            k = evaluate.fit_pair_scale(s[cal_m], first_wins[cal_m])
            out["confidence_scale"] = k
            conf = np.maximum(sig(k, s), 1 - sig(k, s))
            out["by_confidence"] = []
            for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
                m = test_m & (conf >= lo) & (conf < hi)
                out["by_confidence"].append({"stated": f"{lo:.2f} to {min(hi, 1):.2f}", "pairs": int(m.sum()), "accuracy": float(correct[m].mean()) if m.sum() else None})
            m = test_m & (conf >= 0.6)
            out["when_confident_0.6_plus"] = {"share_of_pairs": float(m.sum() / max(test_m.sum(), 1)), "accuracy": float(correct[m].mean()) if m.sum() else None}
        per = {}
        for name, fa, fb in (("image or video", q["m"].to_numpy(), q["m_b"].to_numpy()),
                             ("three or more lines", (q["ts_lines"] >= 3).to_numpy(), (q["ts_lines_b"] >= 3).to_numpy()),
                             ("1.5x more words", (q["ts_words"] >= 1.5 * q["ts_words_b"].clip(lower_bound=5)).to_numpy(), (q["ts_words_b"] >= 1.5 * q["ts_words"].clip(lower_bound=5)).to_numpy()),
                             ("link in first line", (q["ts_url_first_line"] > 0).to_numpy(), (q["ts_url_first_line_b"] > 0).to_numpy())):
            m = test_m & (fa != fb)
            if m.sum() >= 30:
                per[name] = {"pairs": int(m.sum()), "model_accuracy": float(correct[m].mean()), "version_with_it_wins": float((first_wins == fa)[m].mean()),
                             "model_prefers_version_with_it": float(((s > 0) == fa)[m].mean())}
        out["per_change"] = per
        res[target] = out
        print(target, json.dumps(out, indent=1, default=float))
    with open(args.out, "w") as fjson:
        json.dump(res, fjson, indent=2, default=float)


if __name__ == "__main__":
    main()
