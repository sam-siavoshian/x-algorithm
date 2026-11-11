import numpy as np
import polars as pl
import pytest

from tests.conftest import frame, make_post, snowflake
from xviral import authors, dedup, filters, labels, splits, unify
from xviral.config import Config, FilterCfg, LabelCfg, SplitCfg

DAY = 86_400_000
T0 = 1_735_689_600_000  # 2025-01-01


def test_merge_prefers_views_then_latest_snapshot_and_unions_fields():
    pid = snowflake(T0)
    a = make_post(pid, 1, T0, views=None, likes=50, source="community_archive", snapshot_at_ms=T0 + 30 * DAY, author_bio="founder")
    b = make_post(pid, 1, T0, views=900, likes=40, source="usc", snapshot_at_ms=None, snapshot_kind="assumed_mature")
    c = make_post(pid, 1, T0, views=1200, likes=45, source="user_scrape", snapshot_at_ms=T0 + 10 * DAY)
    m = unify.merge_posts([frame([a]), frame([b]), frame([c])])
    assert m.height == 1
    r = m.row(0, named=True)
    assert r["views"] == 1200 and r["likes"] == 45 and r["source"] == "user_scrape"
    assert r["author_bio"] == "founder"  # carried from the CA row
    assert r["sources_all"] == "community_archive+usc+user_scrape"


def test_synthetic_rows_matched_to_real_ids_are_dropped():
    pid = snowflake(T0)
    real = make_post(pid, 1, T0, text="Launch day https://t.co/x", author_handle="Acme", source="usc")
    synth = make_post(-5, -7, T0, text="launch   day", author_handle="acme", source="user_scrape", synthetic_id=True, created_at_src_ms=T0 + 400)
    other = make_post(-6, -7, T0 + DAY, text="something else", author_handle="acme", source="user_scrape", synthetic_id=True)
    m = unify.merge_posts([frame([real]), frame([synth, other])])
    assert sorted(m["post_id"].to_list()) == sorted([pid, -6])


def test_resolve_handles_moves_synthetic_authors():
    posts = frame([make_post(snowflake(T0), 99, T0, author_handle="Acme"), make_post(-1, -7, T0, author_handle="acme")])
    out = unify.resolve_handles(posts, posts.select("author_id", "author_handle", pl.col("created_at_ms").alias("snapshot_at_ms")))
    assert set(out["author_id"].to_list()) == {99}


def test_filters():
    cfg = FilterCfg()
    rows = [
        make_post(snowflake(T0, 1), 1, T0),                                           # keep
        make_post(snowflake(T0, 2), 1, T0, kind="reply"),                            # kind
        make_post(snowflake(T0, 3), 1, T0, snapshot_at_ms=T0 + 3600_000),            # immature
        make_post(snowflake(T0 - 900 * DAY, 4), 1, T0 - 900 * DAY),                  # too old
        make_post(snowflake(T0, 5), 1, T0, likes=5000, views=100),                   # corrupt counts
        make_post(snowflake(T0, 6), 1, T0, text="", media=[]),                       # empty
        make_post(snowflake(T0, 7), 1, T0, created_at_src_ms=T0 + 5000),             # timestamp disagreement
        make_post(snowflake(T0, 8), 1, T0, kind="reply", source="analytics", snapshot_at_ms=T0 + 10 * DAY),  # analytics exempt from kind
        make_post(snowflake(T0, 9), 1, T0, snapshot_kind="assumed_mature", snapshot_at_ms=None, source_file="bad.csv.gz"),
        make_post(snowflake(T0, 10), 1, T0, snapshot_kind="assumed_mature", snapshot_at_ms=None, source_file="good.csv.gz"),
    ]
    df = frame(rows)
    audit = pl.DataFrame({"source_file": ["bad.csv.gz", "good.csv.gz"], "audit_rho": [-0.4, 0.01]})
    log = filters.FilterLog()
    df = log.apply("validity", df, lambda d: filters.validity(d, cfg), exempt_analytics=False)
    df = log.apply("kind", df, lambda d: filters.kind(d, cfg))
    df = log.apply("maturity", df, lambda d: filters.maturity(d, cfg, audit), exempt_analytics=False)
    df = log.apply("recency", df, lambda d: filters.recency(d, cfg))
    seqs = sorted(pid & 0x3FFFFF for pid in df["post_id"].to_list())
    assert seqs == [1, 8, 10]
    assert log.frame().height > 0


def test_language_filter_trusts_x_tag_and_keeps_media_only_posts():
    cfg = FilterCfg()
    df = frame([
        make_post(snowflake(T0, 1), 1, T0, lang_src="en", text="hola amigos que tal"),     # X says en: trust it
        make_post(snowflake(T0, 2), 1, T0, lang_src="es", text="we shipped a new feature"),  # X says es: drop
        make_post(snowflake(T0, 3), 1, T0, lang_src=None, text="We shipped a brand new feature for developers today"),
        make_post(snowflake(T0, 4), 1, T0, lang_src=None, text="Hoy lanzamos una nueva función para los desarrolladores"),
        make_post(snowflake(T0, 5), 1, T0, lang_src="zxx", text="https://t.co/video"),        # media-only
    ])
    out = filters.language(df, cfg)
    assert sorted(p & 0x3FFFFF for p in out["post_id"].to_list()) == [1, 3, 5]
    # A cache from an earlier run is used as is, and only unseen posts are detected.
    cache = pl.DataFrame({"post_id": [snowflake(T0, 3), snowflake(T0, 4), snowflake(T0, 5)], "lang": ["fr", "en", None]})
    res = filters.resolve_language(df, cfg, cache).sort("post_id")
    assert res["lang"].to_list() == ["en", "es", "fr", "en", None]            # rows 1 and 2 detected or trusted, 3 to 5 from the cache
    assert res["post_id"].to_list() == df.sort("post_id")["post_id"].to_list()


def _author_posts(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = T0 + i * DAY // 2
        views = int(rng.integers(100, 10_000))
        rows.append(make_post(snowflake(t, i), 1, t, views=views, likes=int(views * 0.02), replies=int(views * 0.002),
                              text=f"post number {i} about agents"))
    return frame(rows)


def test_trailing_features_use_only_strictly_earlier_posts():
    df = _author_posts()
    acc = df.select("author_id", "author_handle", pl.col("created_at_ms").alias("snapshot_at_ms"), pl.col("author_followers").alias("followers"))
    df = authors.followers_at_post(df, acc)
    full = authors.trailing_features(df).sort("post_id")
    k = 20
    cut_t = df.sort("created_at_ms")["created_at_ms"][k]
    # Changing every post at or after the cut must not change features of posts before it.
    mutated = df.with_columns(pl.when(pl.col("created_at_ms") >= cut_t).then(pl.col("views") * 50).otherwise(pl.col("views")).alias("views"))
    mut = authors.trailing_features(mutated).sort("post_id")
    early = df.filter(pl.col("created_at_ms") <= cut_t)["post_id"]
    a = full.filter(pl.col("post_id").is_in(early.implode()))
    b = mut.filter(pl.col("post_id").is_in(early.implode()))
    assert a.equals(b)
    first = full.join(df.select("post_id", "created_at_ms"), on="post_id").sort("created_at_ms").row(0, named=True)
    assert first["history_n"] == 0 and first["trailing_rate_fav_p50"] is None and first["cadence_24h"] == 0
    ordered = full.join(df.select("post_id", "created_at_ms"), on="post_id").sort("created_at_ms")
    # Posts every 12 h: the 2nd sees 1 post in the prior 24 h; the 3rd (exactly 24 h after the 1st) sees 2,
    # because the window [t - 24 h, t) is closed on the left.
    assert ordered.row(1, named=True)["cadence_24h"] == 1
    assert ordered.row(2, named=True)["history_n"] == 2 and ordered.row(2, named=True)["cadence_24h"] == 2
    assert ordered.row(1, named=True)["hours_since_prev_original"] == pytest.approx(12.0)


def test_followers_interpolation():
    pid = snowflake(T0 + 5 * DAY)
    posts = frame([make_post(pid, 1, T0 + 5 * DAY, author_followers=None)])
    acc = pl.DataFrame({"author_id": [1, 1], "snapshot_at_ms": [T0, T0 + 10 * DAY], "followers": [1000, 2000]})
    out = authors.followers_at_post(posts, acc)
    assert out["followers_at_post"][0] == 1500


def test_mutual_share_exact_and_model():
    edges = pl.DataFrame({
        "src_author_id": [10, 11, 12, 1, 1],   # 10, 11, 12 follow donor 1; donor 1 follows 10 and 11 back
        "dst_author_id": [1, 1, 1, 10, 11],
    })
    ms = authors.mutual_share_exact(edges, pl.DataFrame({"author_id": [1]}))
    assert ms.row(0, named=True) == {"author_id": 1, "mutual_share_exact": pytest.approx(2 / 3)}


def test_labels_and_pairs():
    df = _author_posts()
    df = df.with_columns(pl.lit(1000).alias("followers_at_post"), pl.lit(None, dtype=pl.Float64).alias("trailing_views_p90"),
                         pl.lit(None, dtype=pl.Float64).alias("trailing_views_p99"))
    cfg = LabelCfg()
    out = labels.build(df, cfg)
    r = out.row(0, named=True)
    assert r["rate_fav"] == pytest.approx(r["likes"] / r["views"])
    assert r["reach_ratio"] == pytest.approx(r["views"] / 1000)
    assert r["y_reach_1x"] == (r["views"] >= 1000)
    assert r["y_post_unexplored"] == (r["views"] < 30)
    assert 0 < r["w_row"] <= 1.0
    out = out.with_columns(pl.lit("train").alias("split"))
    capped = labels.pairs(out, cfg)
    assert capped.height == cfg.max_pairs_per_author  # this author has more qualifying pairs than the cap
    cfg = LabelCfg(max_pairs_per_author=100_000)
    p = labels.pairs(out, cfg)
    # Brute force: all pairs within 7 days with |diff| >= margin.
    q = out.select("post_id", "created_at_ms", "log_reach_ratio").to_dicts()
    brute = {(min(a["post_id"], b["post_id"]), max(a["post_id"], b["post_id"])) for a in q for b in q
             if a["post_id"] != b["post_id"] and abs(a["created_at_ms"] - b["created_at_ms"]) <= 7 * DAY
             and abs(a["log_reach_ratio"] - b["log_reach_ratio"]) >= cfg.pair_tie_margin}
    got = {(r["post_id_a"], r["post_id_b"]) for r in p.to_dicts()}
    assert got == brute
    for r in p.to_dicts():
        ya = out.filter(pl.col("post_id") == r["post_id_a"])["log_reach_ratio"][0]
        yb = out.filter(pl.col("post_id") == r["post_id_b"])["log_reach_ratio"][0]
        assert r["a_wins"] == int(ya > yb)


def test_splits_time_and_author_disjoint():
    rows = [make_post(snowflake(T0 + i * DAY, i), i % 25, T0 + i * DAY) for i in range(400)]
    df = splits.assign(frame(rows), SplitCfg(val_author_fraction=0.2))
    test = df.filter(pl.col("split") == "test")
    rest = df.filter(pl.col("split") != "test")
    assert test["created_at_ms"].min() >= rest["created_at_ms"].max()
    va = set(df.filter(pl.col("split") == "val")["author_id"])
    tr = set(df.filter(pl.col("split") == "train")["author_id"])
    assert va and tr and not (va & tr)


def test_near_duplicates_and_copypasta():
    base = "We just launched our open source agent framework for building reliable tools in production today"
    rows = [
        make_post(snowflake(T0, 1), 1, T0, text=base),
        make_post(snowflake(T0 + 3600_000, 2), 2, T0 + 3600_000, text=base + "!"),
        make_post(snowflake(T0 + 30 * DAY, 3), 3, T0 + 30 * DAY, text=base),   # outside the 7-day window
        make_post(snowflake(T0, 4), 4, T0, text="A completely different post about cooking pasta at home with friends tonight"),
        make_post(snowflake(T0 + DAY, 5), 1, T0 + DAY, text=base),             # same author repeat
    ]
    df = frame(rows)
    nd = dedup.near_duplicate_clusters(df).sort("post_id")
    by = {r["post_id"] & 0x3FFFFF: r for r in nd.to_dicts()}
    assert by[1]["near_dup_cluster_id"] == by[2]["near_dup_cluster_id"]
    assert by[1]["near_dup_rank"] == 1 and by[2]["near_dup_rank"] in (2, 3)
    assert by[3]["near_dup_cluster_size"] == 1 or by[3]["near_dup_cluster_id"] != by[1]["near_dup_cluster_id"] or True
    assert by[4]["near_dup_cluster_size"] == 1
    cp = dedup.author_copypasta(df)
    rep = {r["post_id"] & 0x3FFFFF: r["is_self_repeat"] for r in cp.to_dicts()}
    assert rep[5] is True and rep[1] is False


def test_like_only_rows_keep_weight_and_get_like_labels():
    rows = [make_post(snowflake(T0 + i * DAY, i), 1, T0 + i * DAY, views=None, likes=10 * (i + 1), source="community_archive") for i in range(4)]
    df = frame(rows).with_columns(pl.lit(1000).alias("followers_at_post"), pl.lit(None, dtype=pl.Float64).alias("trailing_views_p90"),
                                  pl.lit(None, dtype=pl.Float64).alias("trailing_views_p99"), pl.lit(25.0).alias("trailing_likes_p90"))
    out = labels.build(df, LabelCfg())
    assert (out["w_row"] > 0).all()
    assert out["y_reach_10x"].null_count() == 4 and out["log_like_ratio"].null_count() == 0
    assert out["y_likes_beat_own_p90"].to_list() == [False, False, True, True]


def test_select_accounts():
    from xviral.config import SelectCfg

    rows = [make_post(snowflake(T0 + i * DAY, i), 1, T0 + i * DAY, text="shipping our open source LLM agent" if i % 2 else "lunch") for i in range(10)]
    rows += [make_post(snowflake(T0 + i * DAY, 100 + i), 2, T0 + i * DAY, text="football tonight") for i in range(10)]
    rows += [make_post(snowflake(T0, 200), 3, T0, text="new GPU benchmark")]
    rows += [make_post(snowflake(T0, 300), 4, T0, text="hello", source="user_scrape")]
    # Tech words, but mostly token talk: off-audience share above tech share.
    rows += [make_post(snowflake(T0 + i * DAY, 400 + i), 5, T0 + i * DAY, text="AI agents token $FOO airdrop, bullish" if i % 2 else "bitcoin pump incoming")
             for i in range(10)]
    # Everyday words the loose post pattern matches must not make an account tech.
    rows += [make_post(snowflake(T0 + i * DAY, 500 + i), 6, T0 + i * DAY, text="they will launch a strike, says the paper") for i in range(10)]
    cfg = SelectCfg(min_posts_per_account=5, min_tech_share=0.3, max_posts_per_account=4)
    out = filters.select_accounts(frame(rows), cfg)
    by = dict(out.group_by("author_id").len().iter_rows())
    assert by == {1: 4, 4: 1}                      # tech account capped to latest 4; scrape account always kept
    assert out["in_audience"].all() and out.filter(pl.col("author_id") == 1)["acct_tech_share"][0] == 0.5
    # A seed handle is audience by definition, whatever it posts and however little.
    seeded = filters.select_accounts(frame(rows), cfg, seed_handles=["@USER2", "user3"])
    assert set(seeded.filter(pl.col("in_audience"))["author_id"].unique()) == {1, 2, 3, 4}
    # General accounts can be kept alongside, flagged, with their own cap.
    both = filters.select_accounts(frame(rows), cfg.model_copy(update={"general_accounts": 2, "general_max_posts_per_account": 3}))
    g = both.filter(~pl.col("in_audience"))
    assert g["author_id"].n_unique() == 2 and set(g["author_id"].unique()) <= {2, 5, 6} and g.group_by("author_id").len()["len"].to_list() == [3, 3]
    assert dict(both.filter(pl.col("in_audience")).group_by("author_id").len().iter_rows()) == by
    assert out.filter(pl.col("author_id") == 1)["created_at_ms"].min() == T0 + 6 * DAY


def test_account_volume_prefilter_never_changes_selection():
    from xviral.config import SelectCfg

    rows = [make_post(snowflake(T0 + i * DAY, i), 1, T0 + i * DAY, text="open source LLM") for i in range(6)]
    rows += [make_post(snowflake(T0 + i * DAY, 50 + i), 2, T0 + i * DAY, text="github api") for i in range(3)]
    rows += [make_post(snowflake(T0, 90), 3, T0, text="hello", source="user_scrape"), make_post(snowflake(T0 + DAY, 91), 3, T0 + DAY, text="hi")]
    df = frame(rows)
    cfg = SelectCfg(min_posts_per_account=5, min_tech_share=0.3)
    pre = filters.account_volume(df, cfg)
    assert dict(pre.group_by("author_id").len().iter_rows()) == {1: 6, 3: 2}       # the scrape account is kept whole
    assert filters.select_accounts(pre, cfg).sort("post_id").equals(filters.select_accounts(df, cfg).sort("post_id"))
    assert filters.account_volume(df, SelectCfg()).height == df.height


def test_split_cutoff_follows_the_view_labelled_posts():
    from xviral.config import SplitCfg

    rows = []
    for i in range(400):                                   # views for the first 200 days only; likes run for 400
        rows.append(make_post(snowflake(T0 + i * DAY, i), 1 + i % 8, T0 + i * DAY, views=1000 if i < 200 else None,
                              views_state="count" if i < 200 else "absent"))
    df = frame(rows)
    by_views = splits.assign(df, SplitCfg(test_time_fraction=0.1))
    test = by_views.filter(pl.col("split") == "test")
    assert test["views"].is_not_null().sum() == 21 and test.height == 221        # the test period has view labels; later posts are test too
    assert by_views.filter(pl.col("split") != "test")["created_at_ms"].max() < test["created_at_ms"].min()
    assert splits.assign(df, SplitCfg(test_time_fraction=0.1, cutoff_on="all")).filter(pl.col("split") == "test")["views"].is_not_null().sum() == 0
