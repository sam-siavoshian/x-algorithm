"""The surrogate: invariants (isolation, padding, leakage), head maths, and learning a planted signal."""

import math

import numpy as np
import polars as pl
import pytest
import torch

from xviral.embed.store import EmbeddingStore
from xviral.model import episodes, evaluate, features, train
from xviral.model.config import HistHead, TrainCfg
from xviral.model.features import FeatureTable
from xviral.model.layers import isolation_mask, right_anchored_positions
from xviral.model.model import Surrogate, hist_mean, hist_quantile, hist_tail, hist_targets, post_age_bucket, time_gap_bucket

DAY = 86_400_000
T0 = 1_735_689_600_000


def synthetic_world(tmp_path, n_authors=60, per=50, d=32, seed=0, views=True):
    """Outcome = author baseline + a global content direction + an author-specific taste + noise."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=d)
    g /= np.linalg.norm(g)
    rows, embs = [], []
    pid = 1
    for a in range(n_authors):
        base = rng.normal()
        followers = int(np.exp(7 + base))
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        t = T0 + int(rng.integers(0, 5 * DAY))
        for _ in range(per):
            t += int(rng.integers(2, 12)) * 3_600_000
            e = rng.normal(size=d)
            e /= np.linalg.norm(e)
            z = 1.0 * (g @ e) * math.sqrt(d) + 0.8 * (u @ e) * math.sqrt(d) + 0.5 * rng.normal()
            v = max(int(followers * math.exp(-1.0 + z)), 1)
            likes = int(rng.binomial(v, min(0.03 * math.exp(0.2 * rng.normal()), 0.9)))
            rows.append({"post_id": pid, "author_id": a + 1, "created_at_ms": t, "followers_at_post": followers,
                         "likes": likes, "views": v if views else None, "rate_fav": likes / v if views else None,
                         "weighted_text_len": int(rng.integers(10, 280)), "has_media": bool(rng.integers(0, 2)),
                         "local_hour_idx": int(rng.integers(1, 25)), "kind": "original", "w_row": 1.0})
            embs.append(e)
            pid += 1
    df = pl.DataFrame(rows)
    cut = df["created_at_ms"].quantile(0.85)
    val_authors = list(range(1, n_authors + 1, 6))
    df = df.with_columns(pl.when(pl.col("created_at_ms") >= cut).then(pl.lit("test"))
                         .when(pl.col("author_id").is_in(val_authors)).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split"))
    store = EmbeddingStore(tmp_path / "emb", dim=d)
    store.write(df["post_id"].to_numpy(), np.stack(embs))
    features.build(df, tmp_path / "table", emb=store)
    return FeatureTable.load(tmp_path / "table")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return synthetic_world(tmp_path_factory.mktemp("w"))


def make_model(table, **kw):
    torch.manual_seed(0)
    mcfg = train.model_cfg_for(table, "nano", dropout=0.0, **kw)
    return Surrogate(mcfg, table.meta["action_index"]).eval(), mcfg


def test_phoenix_bucket_ports():
    assert post_age_bucket(torch.tensor([0, 3599, 3600, 80 * 3600, 10**7, -5])).tolist() == [1, 1, 2, 81, 81, 0]
    # edges 60 * 2^k: gap 59 s -> 1, 60 s -> 2, 120 s -> 3; negative -> 0 (unknown); cardinality 19
    assert time_gap_bucket(torch.tensor([59, 60, 119, 120, -1, 10**9])).tolist() == [1, 2, 2, 3, 0, 18]


def test_mask_and_positions():
    valid = torch.tensor([[True, False, True, True, True]])
    m = isolation_mask(valid, n_shared=3)[0]
    assert m[3].tolist() == [True, False, True, True, False]   # candidate sees context, valid history, itself
    assert m[4].tolist() == [True, False, True, False, True]   # never the other candidate
    assert m[0].tolist() == [True, False, True, False, False]  # context never sees candidates
    assert right_anchored_positions(1, 3, 2, "cpu").tolist() == [0, 1, 2, 3, 4, 4]


def test_hist_head_maths():
    spec = HistHead(lo=-4, hi=4, bins=80)
    y = torch.tensor([-3.2, 0.0, 0.37, 2.5])
    tgt = hist_targets(y, spec, 0.75)
    assert torch.allclose(tgt.sum(-1), torch.ones(4), atol=1e-5)
    lg = tgt.clamp_min(1e-12).log()
    assert torch.allclose(hist_mean(lg, spec), y, atol=0.02)
    assert torch.allclose(hist_quantile(lg, spec, 0.5), y, atol=0.06)
    tails = torch.stack([hist_tail(lg, spec, t) for t in (-4.0, -1.0, 0.0, 1.0, 4.0)])
    assert (tails[:-1] >= tails[1:] - 1e-6).all() and torch.allclose(tails[0], torch.ones(4), atol=1e-5)
    assert hist_tail(lg, spec, 1.0)[3] > 0.99 and hist_tail(lg, spec, 1.0)[1] < 0.01


def test_candidate_isolation(world):
    model, mcfg = make_model(world)
    tc = TrainCfg()
    rows = world.rows_of_split("train")[40:48]
    one = model(world, episodes.build(world, rows, mcfg, tc, train=False))
    ep4 = episodes.build(world, rows, mcfg, tc, train=False, n_candidates=4)
    four = model(world, ep4)
    assert ep4.cand_mask[:, 1:].any()
    for k in one["hist"]:
        assert torch.allclose(one["hist"][k][:, 0], four["hist"][k][:, 0], atol=1e-5)
    assert torch.allclose(one["actions"][:, 0], four["actions"][:, 0], atol=1e-5)


def test_padding_content_is_ignored(world):
    model, mcfg = make_model(world)
    tc = TrainCfg()
    starts = torch.unique(world.t["a_start"])[:6] + 3      # authors' 4th posts: 3 history rows, rest padding
    ep = episodes.build(world, starts, mcfg, tc, train=False)
    assert (~ep.hist_mask).any() and ep.hist_mask.sum(1).tolist() == [3] * 6
    a = model(world, ep)
    ep.hist_rows = torch.where(ep.hist_mask, ep.hist_rows, torch.randint(0, world.n, ep.hist_rows.shape))
    b = model(world, ep)
    assert torch.allclose(a["hist"]["like"], b["hist"]["like"], atol=1e-5)


def test_no_label_leak_and_outcome_masking(world):
    model, mcfg = make_model(world)
    tc = TrainCfg(outcome_mask_hours=24.0)
    # One post per author (each author's 31st), so no altered row is another altered row's history.
    rows = (torch.unique(world.t["a_start"]) + 30)[:16]
    ep = episodes.build(world, rows, mcfg, tc, train=False)
    base = model(world, ep)["hist"]["like"].clone()
    saved = {k: world.t[k].clone() for k in ("counts", "y_like", "y_reach", "rates")}
    try:
        # 1. The candidate's own outcome must not reach its prediction.
        world.t["counts"][rows] = 123456
        world.t["y_like"][rows] = 1.5
        world.t["y_reach"][rows] = 1.5
        world.t["rates"][rows] = 0.5
        assert torch.allclose(model(world, episodes.build(world, rows, mcfg, tc, train=False))["hist"]["like"], base, atol=1e-5)
        for k, v in saved.items():
            world.t[k] = v.clone()
        # 2. History posts younger than the mask window: outcomes hidden.
        young = ep.hist_mask & ~ep.hist_visible
        assert young.any()
        world.t["counts"][ep.hist_rows[young]] = 999999
        assert torch.allclose(model(world, episodes.build(world, rows, mcfg, tc, train=False))["hist"]["like"], base, atol=1e-5)
        for k, v in saved.items():
            world.t[k] = v.clone()
        # 3. Visible history outcomes do matter.
        world.t["counts"][ep.hist_rows[ep.hist_visible]] = 999999
        assert not torch.allclose(model(world, episodes.build(world, rows, mcfg, tc, train=False))["hist"]["like"], base, atol=1e-4)
    finally:
        for k, v in saved.items():
            world.t[k] = v


def test_history_is_strictly_earlier_and_splits_do_not_mix(world):
    _, mcfg = make_model(world)
    tc = TrainCfg()
    rows = world.rows_of_split("train")
    ep = episodes.build(world, rows[:256], mcfg, tc, train=True, generator=torch.Generator().manual_seed(0))
    t = world.t
    assert (t["author_idx"][ep.hist_rows][ep.hist_mask] == t["author_idx"][ep.anchor][:, None].expand_as(ep.hist_rows)[ep.hist_mask]).all()
    assert (ep.hist_rows < ep.anchor[:, None])[ep.hist_mask].all()
    assert (t["split"][ep.cand_rows][ep.cand_mask] == 0).all()
    assert ((t["created_s"][ep.cand_rows] - ep.cutoff_s[:, None])[ep.cand_mask] <= 7 * 86400).all()
    assert (ep.cand_obs_counts == -1).all()   # cold start unless p_live > 0


def test_thin_counts():
    tc = TrainCfg(p_live=1.0)
    final = torch.tensor([[[100, -1, 5, 0, 10_000]]]).expand(64, 4, 5).contiguous()
    obs, age = episodes.thin_counts(final, tc, torch.Generator().manual_seed(0))
    assert (obs[..., 1] == -1).all() and (obs[..., [0, 2, 3, 4]] >= 0).all()
    assert (obs <= final).all() and (age >= 300).all() and (age <= 86400).all()
    assert obs[..., 4].float().mean() < 10_000 * 0.9


def test_learns_planted_signal_and_roundtrips(world, tmp_path):
    mcfg = train.model_cfg_for(world, "nano", dropout=0.0)
    tc = TrainCfg(steps=500, batch_size=48, lr=2e-3, warmup_steps=50, eval_every=250, ema_decay=0.98)
    res = train.train(world, tmp_path / "run", mcfg, tc, log_every=10_000)
    model, tc2, ck = train.load(tmp_path / "run" / "best.pt")
    test_rows = world.rows_of_split("test")
    pred = evaluate.predict_rows(model, world, test_rows, tc2)
    m = evaluate.metrics(pred, world, tc2)
    assert m["pairwise_views"] > 0.66 and m["pairwise_likes"] > 0.62, m      # author-median baseline is 0.50
    assert m["within_rho_views"] > 0.35, m
    assert m["auc_reach_1x"] > 0.8, m
    base = evaluate.baseline_author_median(world, test_rows, mcfg, tc2)
    mb = evaluate.metrics({**pred, **base}, world, tc2)
    assert mb["pairwise_views"] < 0.55                                       # the baseline cannot rank within an author
    # Removing content collapses pairwise accuracy: the skill is in the content.
    model.cfg.use_content = False
    m0 = evaluate.metrics(evaluate.predict_rows(model, world, test_rows, tc2), world, tc2)
    model.cfg.use_content = True
    assert m0["pairwise_views"] < m["pairwise_views"] - 0.1, (m0, m)
    # Calibration and reload.
    temps = evaluate.fit_temperature(model, world, world.rows_of_split("val"), tc2)
    assert set(temps) == set(mcfg.hist_heads) and all(0.4 < v < 3.1 for v in temps.values())
    again = evaluate.predict_rows(train.load(tmp_path / "run" / "best.pt")[0], world, test_rows[:64], tc2)
    assert np.allclose(again["views_mean"], pred["views_mean"][:64], atol=1e-5)
    assert res["best"]["step"] > 0


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cached("BAAI/bge-small-en-v1.5"), reason="small encoder not in the local cache")
def test_stage_b_live_encoder_trains_and_reloads(tmp_path):
    pytest.importorskip("peft")
    rng = np.random.default_rng(0)
    words = ["launch", "agent", "model", "startup", "gpu", "open", "source", "demo", "raise", "ship", "eval", "paper"]
    rows, embs, pid = [], [], 1
    for a in range(8):
        t = T0
        for _ in range(20):
            t += 6 * 3_600_000
            good = rng.random() < 0.5
            text = ("we just shipped a big launch " if good else "random thoughts today ") + " ".join(rng.choice(words, 4))
            likes = int(rng.poisson(40 if good else 8))
            rows.append({"post_id": pid, "author_id": a + 1, "author_handle": f"u{a}", "created_at_ms": t, "followers_at_post": 1000,
                         "likes": likes, "views": None, "text": text, "kind": "original", "w_row": 1.0,
                         "split": "val" if a == 7 else "train"})
            embs.append(rng.normal(size=16))
            pid += 1
    df = pl.DataFrame(rows)
    store = EmbeddingStore(tmp_path / "emb", dim=16)
    store.write(df["post_id"].to_numpy(), np.stack(embs))
    features.build(df, tmp_path / "table", emb=store)
    table = FeatureTable.load(tmp_path / "table")
    assert table.texts is not None and table.texts[0].startswith("@u0") and "shipped" in " ".join(table.texts)
    mcfg = train.model_cfg_for(table, "nano", dropout=0.0, live={"model_id": "BAAI/bge-small-en-v1.5", "r": 4, "max_tokens": 24})
    tc = TrainCfg(steps=6, batch_size=8, lr=1e-3, live_lr=1e-3, warmup_steps=1, eval_every=3)
    res = train.train(table, tmp_path / "run", mcfg, tc, log_every=10_000)
    ck = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    live_keys = [k for k in ck["state"] if k.startswith("live_enc.")]
    assert live_keys and all("lora_" in k for k in live_keys)            # adapters only, never the frozen encoder
    assert any(k.startswith("live_proj.") for k in ck["state"])
    model, tc2, _ = train.load(tmp_path / "run" / "best.pt")
    rows_t = table.rows_of_split("val")
    a = evaluate.predict_rows(model, table, rows_t, tc2)
    b = evaluate.predict_rows(train.load(tmp_path / "run" / "best.pt")[0], table, rows_t, tc2)
    assert np.allclose(a["likes_mean"], b["likes_mean"], atol=1e-4)
    # The adapters receive gradient: at least one LoRA B matrix has moved from its zero initialisation.
    assert any(v.abs().sum() > 0 for k, v in ck["state"].items() if "lora_B" in k)
    assert res["params"] < 5_000_000
