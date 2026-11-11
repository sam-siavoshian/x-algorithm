"""Row filters, in the order of DATA_PIPELINE.md section 4.

Each filter is a pure function DataFrame -> DataFrame and is wrapped by
FilterLog, which records rows in and out per source. Analytics rows are
exempt from every filter except validity and maturity: their labels are exact
and rare (section 6.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from xviral.config import FilterCfg
from xviral.text import lang as langmod
from xviral.util.core import date_to_ms

log = logging.getLogger(__name__)

_IS_ANALYTICS = pl.col("source") == "analytics"


@dataclass
class FilterLog:
    rows: list[dict] = field(default_factory=list)

    def apply(self, name: str, df: pl.DataFrame, fn: Callable[[pl.DataFrame], pl.DataFrame], exempt_analytics: bool = True) -> pl.DataFrame:
        before = df.group_by("source").len().rename({"len": "in"})
        if exempt_analytics and "source" in df.columns:
            keep = pl.concat([fn(df.filter(~_IS_ANALYTICS)), df.filter(_IS_ANALYTICS)], how="diagonal_relaxed")
        else:
            keep = fn(df)
        after = keep.group_by("source").len().rename({"len": "out"})
        for r in before.join(after, on="source", how="left").fill_null(0).iter_rows(named=True):
            self.rows.append({"filter": name, **r})
        log.info("filter %-12s %d -> %d", name, df.height, keep.height)
        return keep

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, schema={"filter": pl.String, "source": pl.String, "in": pl.Int64, "out": pl.Int64})


def validity(df: pl.DataFrame, cfg: FilterCfg) -> pl.DataFrame:
    has_text = pl.col("text").fill_null("").str.strip_chars().str.len_chars() > 0
    has_media = pl.col("media").list.len() > 0
    created_ok = pl.col("created_at_ms").is_not_null()
    # Snowflake time and the source timestamp must agree to the source's resolution: a second for most,
    # a minute for sources that record minutes (config timestamp_tolerance_ms_by_source).
    tol = pl.lit(cfg.timestamp_tolerance_ms)
    for src, ms in cfg.timestamp_tolerance_ms_by_source.items():
        tol = pl.when(pl.col("source") == src).then(pl.lit(ms)).otherwise(tol)
    ts_ok = pl.col("created_at_src_ms").is_null() | pl.col("synthetic_id") | ((pl.col("created_at_ms") - pl.col("created_at_src_ms")).abs() <= tol)
    # A source that loses many rows here has a timestamp format this check does not understand; say so loudly
    # instead of dropping its rows in silence.
    bad = df.group_by("source").agg((~ts_ok).mean().alias("share"), pl.len().alias("n")).filter((pl.col("share") > 0.2) & (pl.col("n") >= 50))
    for src, share, n in bad.iter_rows():
        log.warning("validity: %.0f%% of %d %s rows fail the timestamp check; set filters.timestamp_tolerance_ms_by_source "
                    "if this source records coarse timestamps", 100 * share, n, src)
    views_ok = pl.col("views").is_null() | (pl.col("views") >= 0)
    counts_ok = pl.col("views").is_null() | (pl.col("views") < 1) | (pl.col("likes").fill_null(0) <= cfg.max_counts_over_views * pl.col("views"))
    return df.filter(pl.col("post_id").is_not_null() & pl.col("author_id").is_not_null() & created_ok & ts_ok & (has_text | has_media) & views_ok & counts_ok)


def kind(df: pl.DataFrame, cfg: FilterCfg) -> pl.DataFrame:
    return df.filter(pl.col("kind").is_in(cfg.keep_kinds))


def maturity(df: pl.DataFrame, cfg: FilterCfg, usc_audit: pl.DataFrame | None = None, max_abs_rho: float = 0.1) -> pl.DataFrame:
    """Counts must be read at least `maturity_hours` after creation.

    USC rows carry no snapshot time; they pass when their chunk's maturity audit
    shows no within-chunk age trend (|rho| <= max_abs_rho), see sources/usc.py.
    """
    min_age_ms = int(cfg.maturity_hours * 3_600_000)
    timed = (pl.col("snapshot_at_ms") - pl.col("created_at_ms")) >= min_age_ms
    assumed = pl.col("snapshot_kind") == "assumed_mature"
    if usc_audit is not None and usc_audit.height:
        bad = usc_audit.filter(pl.col("audit_rho").is_not_null() & (pl.col("audit_rho").abs() > max_abs_rho)).select("source_file")
        if bad.height:
            log.warning("maturity: %d USC chunks fail the audit and are dropped", bad.height)
        df = df.with_columns(pl.col("source_file").is_in(bad["source_file"].implode()).alias("_bad_chunk"))
    else:
        df = df.with_columns(pl.lit(False).alias("_bad_chunk"))
    return df.filter((assumed & ~pl.col("_bad_chunk")) | (~assumed & timed)).drop("_bad_chunk")


def recency(df: pl.DataFrame, cfg: FilterCfg) -> pl.DataFrame:
    floor = pl.lit(date_to_ms(cfg.min_created))
    for src, d in cfg.min_created_by_source.items():
        floor = pl.when(pl.col("source") == src).then(pl.lit(date_to_ms(d))).otherwise(floor)
    return df.filter(pl.col("created_at_ms") >= floor)


def resolve_language(df: pl.DataFrame, cfg: FilterCfg, cache: pl.DataFrame | None = None) -> pl.DataFrame:
    """Adds `lang`. Trust X's tag; detect only where it is missing or non-linguistic.

    cache: post_id, lang from an earlier run of this stage. Detection is the slowest step of phase 0
    (about 5,000 posts a second), so a rerun only detects posts it has not seen.
    """
    und = set(cfg.undetermined_langs)
    src = df.select("post_id", "lang_src", "text")
    if cache is not None and cache.height:
        src = src.join(cache.select("post_id", pl.col("lang").alias("_cached"), pl.lit(True).alias("_hit")).unique("post_id"),
                       on="post_id", how="left", maintain_order="left")
    else:
        src = src.with_columns(pl.lit(None, dtype=pl.String).alias("_cached"), pl.lit(None, dtype=pl.Boolean).alias("_hit"))
    hit = src["_hit"].fill_null(False)
    todo = src.filter(~hit)
    resolved: list[str | None] = []
    step = 250_000                      # texts are materialised as Python strings one slice at a time
    for i in range(0, todo.height, step):
        part = todo.slice(i, step)
        resolved += langmod.resolve(part["lang_src"].to_list(), part["text"].to_list(), und)
    fresh = todo.select("post_id").with_columns(pl.Series("_new", resolved, dtype=pl.String))
    out = src.select("post_id", "_cached", "_hit").join(fresh, on="post_id", how="left", maintain_order="left")
    lang = pl.when(pl.col("_hit").fill_null(False)).then(pl.col("_cached")).otherwise(pl.col("_new"))
    log.info("language: %d posts, %d from the cache, %d detected", df.height, int(hit.sum()), todo.height)
    return df.with_columns(out.select(lang.alias("lang"))["lang"])


def keep_languages(df: pl.DataFrame, cfg: FilterCfg) -> pl.DataFrame:
    """Keep the configured languages and text-free media posts."""
    return df.filter(pl.col("lang").is_in(cfg.keep_langs) | pl.col("lang").is_in(["zxx"]) | pl.col("lang").is_null())


def language(df: pl.DataFrame, cfg: FilterCfg) -> pl.DataFrame:
    return keep_languages(resolve_language(df, cfg), cfg)


def hygiene_bad_accounts(authors: pl.DataFrame, cfg) -> pl.Series:
    """Accounts that look automated. Thresholds sit far beyond the flagging thresholds in features."""
    bad = authors.filter(
        (pl.col("max_originals_24h") > cfg.max_originals_per_24h)
        | ((pl.col("follow_ratio") < cfg.min_follow_ratio) & (pl.col("following_latest") > cfg.min_following_for_ratio_rule))
        | ((pl.col("top_domain_share") > cfg.max_same_domain_share) & (pl.col("originals_n") >= cfg.min_posts_for_domain_rule))
        | (pl.col("copypasta_events") > cfg.max_copypasta_events)
    )["author_id"]
    log.info("hygiene: %d accounts dropped", len(bad))
    return bad


def hygiene(df: pl.DataFrame, authors: pl.DataFrame, cfg) -> pl.DataFrame:
    return df.filter(~pl.col("author_id").is_in(hygiene_bad_accounts(authors, cfg).implode()))


def account_volume(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Drop accounts that cannot reach `select.min_posts_per_account`, before the expensive language pass.

    Every later filter only removes rows, so an account below the minimum here is below it at selection
    too: the final dataset is unchanged, and on a general corpus most rows never reach language detection.
    """
    if cfg.min_posts_per_account <= 1:
        return df
    keep = (pl.len().over("author_id") >= cfg.min_posts_per_account) | pl.col("source").is_in(cfg.always_keep_sources).any().over("author_id")
    return df.filter(keep)


SELECT_COLUMNS = ("post_id", "author_id", "author_handle", "created_at_ms", "source", "text")


def account_selection(narrow: pl.DataFrame, cfg, seed_handles: list[str] | None = None) -> pl.DataFrame:
    """Which posts survive account selection (config.SelectCfg), decided on SELECT_COLUMNS alone.

    Returns post_id, acct_tech_share, acct_off_share, in_audience for the kept posts. Audience accounts:
    enough posts, tech-leaning, not dominated by off-audience vocabulary. Optionally a fixed number of
    general accounts with enough posts is kept too, flagged `in_audience` = false. Working on five narrow
    columns keeps a corpus of millions of posts from being copied while the decision is made.

    seed_handles: accounts that are the audience by definition (config paths.seed_accounts). They are kept
    whatever their vocabulary or post count, the same way the always-keep sources are.
    """
    from xviral.text.tech import is_off_audience_post, is_tech_post

    keep_src = pl.col("source").is_in(cfg.always_keep_sources)
    if seed_handles:
        seeds = sorted({h.lower().lstrip("@") for h in seed_handles if h.strip()})
        keep_src = keep_src | pl.col("author_handle").str.to_lowercase().is_in(seeds)
    stats = narrow.group_by("author_id").agg(
        pl.len().alias("_n"), is_tech_post(strong=True).mean().alias("acct_tech_share"),
        is_off_audience_post().mean().alias("acct_off_share"), keep_src.any().alias("_keep"))
    on_topic = pl.col("acct_tech_share") >= cfg.min_tech_share
    if cfg.min_tech_share > 0 and cfg.max_off_audience_ratio is not None:
        on_topic = on_topic & (pl.col("acct_off_share") <= cfg.max_off_audience_ratio * pl.col("acct_tech_share"))
    enough = pl.col("_n") >= cfg.min_posts_per_account
    stats = stats.with_columns((pl.col("_keep") | (enough & on_topic)).alias("in_audience"))
    general = stats.filter(~pl.col("in_audience") & enough)
    if cfg.general_accounts and general.height:
        general = general.sort(pl.col("author_id").hash(seed=11)).head(cfg.general_accounts)
    else:
        general = general.head(0)
    ok = pl.concat([stats.filter(pl.col("in_audience")), general]).select("author_id", "acct_tech_share", "acct_off_share", "in_audience")
    keys = narrow.select("post_id", "author_id", "created_at_ms").join(ok, on="author_id", how="inner")
    cap_a, cap_g = cfg.max_posts_per_account, cfg.general_max_posts_per_account or cfg.max_posts_per_account
    if cap_a or cap_g:
        big = 1 << 60
        cap = pl.when(pl.col("in_audience")).then(pl.lit(cap_a or big)).otherwise(pl.lit(cap_g or big))
        keys = keys.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("_r")).filter(pl.col("_r") <= cap)
    log.info("select: %d -> %d posts; %d accounts -> %d audience + %d general", narrow.height, keys.height, stats.height,
             int(ok["in_audience"].sum()), int((~ok["in_audience"]).sum()))
    return keys.select("post_id", "acct_tech_share", "acct_off_share", "in_audience")


def select_accounts(df: pl.DataFrame, cfg, seed_handles: list[str] | None = None) -> pl.DataFrame:
    """Account-level selection on an in-memory frame; adds `acct_tech_share`, `acct_off_share` and `in_audience`."""
    return df.join(account_selection(df.select(SELECT_COLUMNS), cfg, seed_handles), on="post_id", how="inner")
