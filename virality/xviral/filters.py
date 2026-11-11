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


def language(df: pl.DataFrame, cfg: FilterCfg) -> pl.DataFrame:
    """Trust X's tag; detect only where it is missing or non-linguistic. Keep text-free media posts."""
    und = set(cfg.undetermined_langs)
    resolved = langmod.resolve(df["lang_src"].to_list(), df["text"].to_list(), und)
    df = df.with_columns(pl.Series("lang", resolved, dtype=pl.String))
    keep = pl.col("lang").is_in(cfg.keep_langs) | pl.col("lang").is_in(["zxx"]) | pl.col("lang").is_null()
    return df.filter(keep)


def hygiene(df: pl.DataFrame, authors: pl.DataFrame, cfg) -> pl.DataFrame:
    """Drop accounts that look automated. Thresholds sit far beyond the flagging thresholds in features."""
    bad = authors.filter(
        (pl.col("max_originals_24h") > cfg.max_originals_per_24h)
        | ((pl.col("follow_ratio") < cfg.min_follow_ratio) & (pl.col("following_latest") > cfg.min_following_for_ratio_rule))
        | ((pl.col("top_domain_share") > cfg.max_same_domain_share) & (pl.col("originals_n") >= cfg.min_posts_for_domain_rule))
        | (pl.col("copypasta_events") > cfg.max_copypasta_events)
    ).select("author_id")
    log.info("hygiene: %d accounts dropped", bad.height)
    return df.join(bad, on="author_id", how="anti")


def account_volume(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Drop accounts that cannot reach `select.min_posts_per_account`, before the expensive language pass.

    Every later filter only removes rows, so an account below the minimum here is below it at selection
    too: the final dataset is unchanged, and on a general corpus most rows never reach language detection.
    """
    if cfg.min_posts_per_account <= 1:
        return df
    keep = (pl.len().over("author_id") >= cfg.min_posts_per_account) | pl.col("source").is_in(cfg.always_keep_sources).any().over("author_id")
    return df.filter(keep)


def select_accounts(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Account-level selection (config.SelectCfg). Adds `acct_tech_share`, `acct_off_share` and `in_audience`.

    Audience accounts: enough posts, tech-leaning, not dominated by off-audience vocabulary. Optionally a
    fixed number of general accounts with enough posts is kept too, flagged `in_audience` = false.
    """
    from xviral.text.tech import is_off_audience_post, is_tech_post

    keep_src = pl.col("source").is_in(cfg.always_keep_sources)
    stats = df.group_by("author_id").agg(
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
    out = df.join(ok, on="author_id", how="inner")
    cap_a, cap_g = cfg.max_posts_per_account, cfg.general_max_posts_per_account or cfg.max_posts_per_account
    if cap_a or cap_g:
        big = 1 << 60
        cap = pl.when(pl.col("in_audience")).then(pl.lit(cap_a or big)).otherwise(pl.lit(cap_g or big))
        out = out.with_columns(pl.col("created_at_ms").rank("ordinal", descending=True).over("author_id").alias("_r")).filter(
            pl.col("_r") <= cap).drop("_r")
    log.info("select: %d -> %d posts; %d accounts -> %d audience + %d general", df.height, out.height, stats.height,
             int(ok["in_audience"].sum()), int((~ok["in_audience"]).sum()))
    return out
