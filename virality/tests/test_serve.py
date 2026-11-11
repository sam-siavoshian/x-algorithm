"""Draft scoring: raw history + drafts -> features -> stored transforms -> report."""

import numpy as np
import polars as pl
import pytest
import torch

from tests.conftest import frame, make_post
from xviral import serve
from xviral.embed.store import EmbeddingStore
from xviral.model import features, train
from xviral.model.config import TrainCfg
from xviral.model.features import FeatureTable

T0 = 1_760_000_000_000
HOUR = 3_600_000


def embed(texts: list[str]) -> np.ndarray:
    """A stand-in encoder: one direction for launch posts, another for the rest, plus a little text-dependent noise."""
    out = np.zeros((len(texts), 16), np.float32)
    for i, t in enumerate(texts):
        out[i, 0 if "launch" in t else 1] = 1.0
        out[i, 2 + len(t) % 14] = 0.2
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def corpus(rng) -> pl.DataFrame:
    rows, pid = [], 1
    for a in range(1, 25):
        base = float(rng.uniform(1.0, 2.2))                       # the account's own level, log10 views
        t = T0 + int(rng.integers(0, 24)) * HOUR
        for _ in range(40):
            t += int(rng.integers(5, 30)) * HOUR
            good = rng.random() < 0.4
            views = int(10 ** (base + (0.8 if good else 0.0) + rng.normal(0, 0.15)) * 20)
            text = ("launch day: we shipped the new agent" if good else "some thoughts on the week") + f" {pid}"
            rows.append(make_post(pid, a, t, text, views=views, likes=max(views // 40, 0), replies=views // 400, reposts=views // 300,
                                  author_followers=int(2000 * a), source="community_archive"))
            pid += 1
    return frame(rows)


def test_draft_scoring_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    posts = corpus(rng)
    df = serve.feature_frame(posts)
    df = df.with_columns(pl.when(pl.col("author_id") > 21).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split"))
    store = EmbeddingStore(tmp_path / "emb", dim=16)
    store.write(df["post_id"].to_numpy(), embed(features.render_texts(df)))
    features.build(df, tmp_path / "table", emb=store)
    table = FeatureTable.load(tmp_path / "table")
    mcfg = train.model_cfg_for(table, "nano", dropout=0.0)
    train.train(table, tmp_path / "run", mcfg, TrainCfg(steps=300, batch_size=32, lr=2e-3, warmup_steps=30, eval_every=150, ema_decay=0.98), log_every=10_000)

    scorer = serve.DraftScorer(tmp_path / "run" / "best.pt", embed)
    history = posts.filter(pl.col("author_id") == 23)
    when = int(history["created_at_ms"].max()) + 30 * HOUR
    account = {"handle": "user23", "followers": 46_000, "following": 300}
    drafts = [{"text": "some thoughts on the week ahead"}, {"text": "launch day: we shipped the new agent today"}]

    # Stored transforms: history rows come out of serving exactly as they sit in the training table.
    t2, rows2 = scorer.featurize(history, drafts, account, when)
    assert t2.n == history.height + 2 and rows2["source"].to_list()[-2:] == ["draft", "draft"]
    # Each draft is the account's next post: neither sees the other as a previous post.
    assert rows2["history_n"].to_list()[-2:] == [history.height] * 2 and rows2["hours_since_prev_original"].to_list()[-2:] == pytest.approx([30.0, 30.0])
    ids = t2.t["post_id"][:-2]
    pos = {int(p): i for i, p in enumerate(table.t["post_id"].tolist())}
    at = torch.tensor([pos[int(p)] for p in ids])
    assert torch.allclose(t2.t["num"][:-2], table.t["num"][at], atol=1e-5)
    assert torch.equal(t2.t["cat"][:-2], table.t["cat"][at]) and torch.equal(t2.t["counts"][:-2], table.t["counts"][at])
    assert (t2.t["counts"][-2:] == -1).all()                       # drafts carry no outcome

    reports = scorer.score(history, drafts, account, when)
    plain, launch = reports
    assert plain["text"] == drafts[0]["text"] and plain["history_posts_used"] == min(40, mcfg.max_history)
    assert launch["views_24h"]["q50"] > 1.5 * plain["views_24h"]["q50"], (plain["views_24h"], launch["views_24h"])
    assert launch["p_own_top_10pct"]["views"] > plain["p_own_top_10pct"]["views"]
    assert plain["views_24h"]["q10"] <= plain["views_24h"]["q50"] <= plain["views_24h"]["q90"]
    # Alternatives are isolated: a draft's score does not depend on which other drafts it is compared with.
    alone = scorer.score(history, drafts[1:], account, when)[0]
    assert abs(alone["views_24h"]["q50"] - launch["views_24h"]["q50"]) < 1e-3 * launch["views_24h"]["q50"]
    assert scorer.score(history, drafts[::-1], account, when)[0]["views_24h"]["q50"] == launch["views_24h"]["q50"]   # order-free
    again = scorer.score(history, drafts, account, when)
    assert again[1]["views_24h"] == launch["views_24h"]
    # Posting-time sweep and link variants reuse the cached history embeddings.
    calls = []
    scorer._embed_raw = lambda texts: (calls.append(len(texts)), embed(texts))[1]
    sweep = scorer.timing(history, drafts[1], account, start_ms=when, hours_ahead=12, step_hours=4)
    assert len(sweep) == 4 and sorted(r["hours_from_start"] for r in sweep) == [0, 4, 8, 12]
    assert sweep[0]["views_q50"] == max(r["views_q50"] for r in sweep) and sum(calls) == 0       # nothing new to embed
    link = {"text": "launch day: the new agent is live https://foo.dev read more below"}
    lv = scorer.link_variants(history, link, account, when)
    assert [r["variant"] for r in lv] == ["as written", "link moved to the end", "link removed (put it in a reply)"]
    assert lv[1]["text"].endswith("https://foo.dev") and "http" not in lv[2]["text"] and scorer.link_variants(history, drafts[0], account, when) == []
    # An account with no history still gets a report; the lift fields are empty.
    cold = scorer.score(history.head(0), drafts[:1], {"handle": "newco", "followers": 150, "author_id": 999}, when)[0]
    assert cold["lift_vs_own_median"]["views"] is None and cold["views_24h"]["q50"] >= 0
