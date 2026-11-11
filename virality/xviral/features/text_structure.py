"""Candidate features on how a post is written, grouped so each group can be vetted on held-out data.

None of these is read by X's ranker directly: Phoenix sees the post through its embedding. They are
candidates because a frozen embedding encodes topic far better than layout, and layout is something
a writer can change. A group enters the model's feature list (model/features.NUM_SPEC) only after
scripts/vet_features.py shows it improves same-author ranking on the time-forward test split.

Everything here is computed from the post's own text and the author's strictly earlier posts.
"""

from __future__ import annotations

import polars as pl

URL = r"https?://\S+"
EMOJI = r"[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}\x{2B00}-\x{2BFF}]"
ANNOUNCE = r"(?i)^\W*(introducing|announcing|launching|we(?:'re| are) (?:excited|thrilled|launching|releasing)|today[, ]|new:|just (?:shipped|launched|released)|meet )"

GROUPS: dict[str, list[str]] = {
    "opening": ["ts_first_line_chars", "ts_first_line_words", "ts_first_line_share", "ts_open_question", "ts_open_colon",
                "ts_open_number", "ts_open_first_person", "ts_open_announce", "ts_open_mention"],
    "shape": ["ts_words", "ts_lines", "ts_blank_lines", "ts_bullet_lines", "ts_mean_line_chars", "ts_max_line_chars",
              "ts_sentences", "ts_words_per_sentence"],
    "typography": ["ts_caps_words", "ts_exclaims", "ts_questions", "ts_emoji", "ts_number_tokens", "ts_money", "ts_percent",
                   "ts_ellipsis", "ts_quotes", "ts_parens", "ts_arrows", "ts_code"],
    "link_position": ["ts_url_first_line", "ts_url_at_end", "ts_chars_before_url_share"],
    "voice": ["ts_i", "ts_we", "ts_you", "ts_negation", "ts_superlative", "ts_comparison", "ts_how_why", "ts_thread_marker",
              "ts_call_to_action"],
    "self_relative": ["ts_len_vs_own", "ts_media_vs_own", "ts_link_vs_own", "ts_gap_vs_own", "ts_lines_vs_own"],
}
ALL = [c for cols in GROUPS.values() for c in cols]

# Groups that passed vetting and are model inputs (research/10_text_structure_vetting.json). On 152k
# like-labelled posts they add 0.8 points of same-author pairwise accuracy to trees that already have
# the embedding; voice and typography add nothing on top, and self_relative adds nothing at all.
ADMITTED_GROUPS = ("opening", "shape", "link_position")
ADMITTED = [c for g in ADMITTED_GROUPS for c in GROUPS[g]]


def _count(col: pl.Expr, pattern: str) -> pl.Expr:
    return col.str.count_matches(pattern).cast(pl.Float32)


def text_structure(df: pl.DataFrame) -> pl.DataFrame:
    """post_id plus every candidate column. Needs text, created_at_ms, author_id; uses has_media/has_url when present."""
    raw = pl.col("text_note").fill_null(pl.col("text")).fill_null("") if "text_note" in df.columns else pl.col("text").fill_null("")
    d = df.with_columns(raw.alias("_raw")).with_columns(
        pl.col("_raw").str.replace_all(URL, " ").str.strip_chars().alias("_t"),
        pl.col("_raw").str.split("\n").list.eval(pl.element().str.strip_chars()).alias("_lines_all"),
    ).with_columns(
        pl.col("_lines_all").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).alias("_lines"),
        pl.col("_t").str.to_lowercase().alias("_lt"),
    ).with_columns(
        pl.col("_lines").list.first().fill_null("").alias("_first"),
        pl.col("_t").str.len_chars().cast(pl.Float32).alias("_chars"),
    )
    t, lt, first = pl.col("_t"), pl.col("_lt"), pl.col("_first")
    first_nourl = first.str.replace_all(URL, " ").str.strip_chars()
    url_pos = pl.col("_raw").str.find(URL).cast(pl.Float32)
    raw_len = pl.col("_raw").str.len_chars().cast(pl.Float32)
    d = d.with_columns(
        # opening
        first_nourl.str.len_chars().cast(pl.Float32).alias("ts_first_line_chars"),
        _count(first_nourl, r"\S+").alias("ts_first_line_words"),
        (first_nourl.str.len_chars() / pl.col("_chars").clip(lower_bound=1)).cast(pl.Float32).alias("ts_first_line_share"),
        first_nourl.str.contains(r"\?").cast(pl.Float32).alias("ts_open_question"),
        first_nourl.str.contains(r":\s*$").cast(pl.Float32).alias("ts_open_colon"),
        first_nourl.str.contains(r"^\W*\$?\d").cast(pl.Float32).alias("ts_open_number"),
        first_nourl.str.contains(r"(?i)^\W*(i|i'm|i've|we|we're|we've|my|our)\b").cast(pl.Float32).alias("ts_open_first_person"),
        first_nourl.str.contains(ANNOUNCE).cast(pl.Float32).alias("ts_open_announce"),
        first_nourl.str.contains(r"^\s*@\w").cast(pl.Float32).alias("ts_open_mention"),
        # shape
        _count(t, r"\S+").alias("ts_words"),
        pl.col("_lines").list.len().cast(pl.Float32).alias("ts_lines"),
        (pl.col("_lines_all").list.len() - pl.col("_lines").list.len()).cast(pl.Float32).alias("ts_blank_lines"),
        pl.col("_lines").list.eval(pl.element().str.contains(r"^(?:[-•*·▪→✅☑✔]|\d{1,2}[.)/])\s*")).list.sum().cast(pl.Float32).alias("ts_bullet_lines"),
        pl.col("_lines").list.eval(pl.element().str.len_chars()).list.mean().cast(pl.Float32).alias("ts_mean_line_chars"),
        pl.col("_lines").list.eval(pl.element().str.len_chars()).list.max().cast(pl.Float32).alias("ts_max_line_chars"),
        _count(t, r"[.!?]+(?:\s|$)").clip(lower_bound=1).alias("ts_sentences"),
        # typography
        _count(t, r"\b[A-Z]{3,}\b").alias("ts_caps_words"),
        _count(t, r"!").alias("ts_exclaims"),
        _count(t, r"\?").alias("ts_questions"),
        _count(t, EMOJI).alias("ts_emoji"),
        _count(t, r"\b\d[\d,.]*[kKmMbBx%]?\b").alias("ts_number_tokens"),
        t.str.contains(r"\$\s?\d").cast(pl.Float32).alias("ts_money"),
        t.str.contains(r"\d\s?%").cast(pl.Float32).alias("ts_percent"),
        t.str.contains(r"\.\.\.|…").cast(pl.Float32).alias("ts_ellipsis"),
        _count(t, "[\"“”]").alias("ts_quotes"),
        _count(t, r"\(").alias("ts_parens"),
        _count(t, r"->|→|=>|➡").alias("ts_arrows"),
        t.str.contains(r"`|\b(?:pip|npm|npx|uv|brew|cargo|git) (?:install|add|clone|run)\b").cast(pl.Float32).alias("ts_code"),
        # link position
        first.str.contains(URL).cast(pl.Float32).alias("ts_url_first_line"),
        pl.col("_raw").str.contains(URL + r"\s*$").cast(pl.Float32).alias("ts_url_at_end"),
        (url_pos / raw_len.clip(lower_bound=1)).alias("ts_chars_before_url_share"),
        # voice
        _count(lt, r"\b(?:i|i'm|i've|i'd|my|me)\b").alias("ts_i"),
        _count(lt, r"\b(?:we|we're|we've|our|us)\b").alias("ts_we"),
        _count(lt, r"\b(?:you|your|you're)\b").alias("ts_you"),
        _count(lt, r"\b(?:not|no|never|don't|doesn't|isn't|can't|won't|nobody|nothing)\b").alias("ts_negation"),
        _count(lt, r"\b(?:best|worst|fastest|biggest|first|largest|most|insane|crazy|wild|huge|massive)\b").alias("ts_superlative"),
        _count(lt, r"\b(?:vs\.?|versus|than|compared to|better|worse|faster|cheaper)\b").alias("ts_comparison"),
        lt.str.contains(r"\b(?:how to|how i|how we|why|here's (?:how|why|what)|what i learned|lessons)\b").cast(pl.Float32).alias("ts_how_why"),
        t.str.contains(r"🧵|\b1/\d*\b|(?i)\bthread\b|👇").cast(pl.Float32).alias("ts_thread_marker"),
        lt.str.contains(r"\b(?:try it|check it out|sign up|join us|apply|waitlist|link below|link in|dm me|rt if|retweet|comment|reply with|follow (?:me|us))\b")
        .cast(pl.Float32).alias("ts_call_to_action"),
    ).with_columns((pl.col("ts_words") / pl.col("ts_sentences")).alias("ts_words_per_sentence"))

    # Relative to the author's own strictly earlier posts (shifted rolling statistics over the last 50).
    flag = lambda c: (pl.col(c).cast(pl.Float32).fill_null(0.0) if c in d.columns else pl.lit(0.0, dtype=pl.Float32))  # noqa: E731
    d = d.sort("author_id", "created_at_ms", "post_id").with_columns(
        flag("has_media").alias("_m"), flag("has_url").alias("_u"),
        (pl.col("created_at_ms").diff().over("author_id") / 3_600_000).cast(pl.Float32).alias("_gap"),
    )
    prev = lambda e, fn: getattr(e.shift(1), fn)(window_size=50, min_samples=5).over("author_id")  # noqa: E731
    d = d.with_columns(
        ((pl.col("_chars") + 1).log() - (prev(pl.col("_chars"), "rolling_median") + 1).log()).alias("ts_len_vs_own"),
        (pl.col("_m") - prev(pl.col("_m"), "rolling_mean")).alias("ts_media_vs_own"),
        (pl.col("_u") - prev(pl.col("_u"), "rolling_mean")).alias("ts_link_vs_own"),
        ((pl.col("_gap") + 1).log() - (prev(pl.col("_gap"), "rolling_median") + 1).log()).alias("ts_gap_vs_own"),
        ((pl.col("ts_lines") + 1).log() - (prev(pl.col("ts_lines"), "rolling_median") + 1).log()).alias("ts_lines_vs_own"),
    )
    return d.select("post_id", *ALL)


def admitted(df: pl.DataFrame) -> pl.DataFrame:
    """post_id plus the admitted columns: what the feature stage and serving join in."""
    return text_structure(df).select("post_id", *ADMITTED)
