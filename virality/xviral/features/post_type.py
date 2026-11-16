"""Model inputs that tell the model what kind of company post a post is: the post type with the own-announcement
test, the launch traits that have a measured effect, and the account's trailing record by type.

Everything here is computed from one frame of posts (an account's history plus the post being featurised), so the
feature stage and serving (xviral/serve.py feature_frame) produce the same values. Types use the rules of
xviral/text/launch_labels.py (add_types, own_expr) and traits are xviral/text/post_type.py traits(), called as they are.

The own-announcement test needs to know whether the account speaks as a company. The launch track labelled accounts
from their whole timeline, which serving cannot do and which would let later posts leak into earlier rows. Here the
account kind is read from the account's previous posts only (COMPANY_* below): an organisation voice is under 10% of
posts using I/my and either 12%+ using we/our or 15%+ typed tech posts, over at least MIN_VOICE previous posts.

Leakage rule: every trailing value of a post is computed from posts of the same account created strictly earlier
(the asof joins below do not allow equal timestamps), over at most WINDOW of them, and drafts (source == "draft")
never count as anyone's history. The outcome of the post itself is never read for its own row.
"""

from __future__ import annotations

import polars as pl

from xviral.text import launch_labels as L
from xviral.text import post_type as PT

TYPES = list(L.ORDER)                    # funding, hiring, launch, open_source, feature, milestone, demo
LAUNCH_TYPES = ("launch", "open_source")
WINDOW = 50                              # the same window as authors.trailing_features
MIN_VOICE = 8
COMPANY_I_MAX, COMPANY_WE_MIN, COMPANY_TYPED_MIN = 0.10, 0.12, 0.15
TRAITS = ("names_product_first_line", "lines3", "video", "image", "open_problem", "open_number", "ext_link", "is_quote")

TYPE_COLS = [f"pt_{k}" for k in TYPES]
FLAG_COLS = ["pt_raw_typed", "pt_fp_head", "pt_opener", "pt_techy", "pt_company_voice"]
TRAIT_COLS = [f"pt_{k}" for k in TRAITS]
# The traits again, switched on only for the account's own launch-type posts: the measured effects are effects
# among launch posts, and the same trait on an ordinary post is a different thing.
LAUNCH_TRAIT_COLS = [f"pt_launch_{k}" for k in TRAITS]
RECORD_COLS = ["pt_prev_launch_n", "pt_prev_launch_lift", "pt_prev_ann_n", "pt_prev_ann_lift", "pt_prev_same_n", "pt_prev_same_lift"]
COLUMNS = TYPE_COLS + FLAG_COLS + TRAIT_COLS + LAUNCH_TRAIT_COLS + RECORD_COLS
NUM_SPEC = ([(c, "bool") for c in TYPE_COLS + FLAG_COLS + TRAIT_COLS + LAUNCH_TRAIT_COLS]
            + [(c, "log1p" if c.endswith("_n") else "id") for c in RECORD_COLS])

_I = r"\b(i|i'm|i've|my|me)\b"
_WE = r"\b(we|we're|we've|our|us)\b"


def _video(df: pl.DataFrame) -> pl.Expr:
    """Video or GIF attached. From the content features when the frame has them, otherwise from the media list."""
    if "has_video" in df.columns:
        v = pl.col("has_video").cast(pl.Boolean).fill_null(False)
        return (v | pl.col("has_gif").cast(pl.Boolean).fill_null(False)) if "has_gif" in df.columns else v
    return pl.col("media").list.eval(pl.element().struct.field("type").is_in(["video", "gif", "animated_gif"])).list.any().fill_null(False)


def _any_media(df: pl.DataFrame) -> pl.Expr:
    if "has_media" in df.columns:
        return pl.col("has_media").cast(pl.Boolean).fill_null(False)
    return (pl.col("media").list.len() > 0).fill_null(False)


def typed(df: pl.DataFrame) -> pl.DataFrame:
    """Per-post rule matches (launch_labels.add_types) plus the columns the trailing state needs. No outcome is used
    except `_l`, the post's own log likes, which only ever feeds later posts."""
    d = df.with_columns(_video(df).alias("_video"), _any_media(df).alias("_media"))
    d = L.add_types(d.select("post_id", "author_id", "created_at_ms", "text", "kind", "likes", "_video", "_media",
                             pl.col("source") if "source" in df.columns else pl.lit("post").alias("source")))
    low = pl.col("text").fill_null("").str.to_lowercase().str.replace_all("’", "'")
    # launch_labels.own_expr without the account clause: tech-worded, not a reprint, not addressed to a handle, and
    # typed (a demo counts only with a video).
    is_typed = pl.when(pl.col("ptype_raw") == "demo").then(pl.col("_video")).otherwise(pl.col("ptype_raw") != "other")
    base = pl.col("techy") & ~pl.col("reprint") & ~pl.col("at_start") & is_typed
    return d.with_columns(low.str.contains(_I).alias("_i"), low.str.contains(_WE).alias("_we"), base.alias("_base"),
                          (base & pl.col("fp_head")).alias("_own_fp"), (base & ~pl.col("fp_head") & pl.col("opener")).alias("_own_op"),
                          (pl.col("likes").cast(pl.Float64) + 1).log().alias("_l")).drop("_lc", "_head")


def _state(t: pl.DataFrame) -> pl.DataFrame:
    """After each real post with a like count: sums over the account's latest WINDOW such posts, that post included."""
    s = t.filter((pl.col("source") != "draft") & pl.col("likes").is_not_null()).sort("author_id", "created_at_ms", "post_id")
    roll = lambda e: e.cast(pl.Float64).rolling_sum(window_size=WINDOW, min_samples=1).over("author_id")   # noqa: E731
    cols = [roll(pl.col("likes").is_not_null()).alias("s_n"), roll(pl.col("_i")).alias("s_i"), roll(pl.col("_we")).alias("s_we"), roll(pl.col("_base")).alias("s_typed"),
            pl.col("_l").rolling_median(window_size=WINDOW, min_samples=1).over("author_id").alias("s_med")]
    for k in TYPES:
        for v in ("fp", "op"):
            m = pl.col(f"_own_{v}") & (pl.col("ptype_raw") == k)
            cols += [roll(m).alias(f"s_n_{v}_{k}"), roll(pl.when(m).then(pl.col("_l")).otherwise(0.0)).alias(f"s_l_{v}_{k}")]
    s = s.select("author_id", "created_at_ms", *cols)
    return s.group_by("author_id", "created_at_ms", maintain_order=True).agg(pl.all().last()).sort("created_at_ms")


def post_type_features(df: pl.DataFrame, only: pl.Series | None = None) -> pl.DataFrame:
    """post_id plus COLUMNS for every row of `df` (or the rows whose post_id is in `only`).

    df: posts of one or more accounts with post_id, author_id, created_at_ms, text, kind, likes, and either the
    content features (has_video, has_gif, has_media) or the media list; `source` marks drafts.
    """
    t = typed(df)
    state = _state(t)
    target = t if only is None else t.filter(pl.col("post_id").is_in(only.implode()))
    x = target.sort("created_at_ms").join_asof(state, on="created_at_ms", by="author_id", strategy="backward", allow_exact_matches=False, check_sortedness=False)
    n = pl.col("s_n").fill_null(0.0)
    company = ((n >= MIN_VOICE) & (pl.col("s_i") / n < COMPANY_I_MAX)
               & ((pl.col("s_we") / n >= COMPANY_WE_MIN) | (pl.col("s_typed") / n >= COMPANY_TYPED_MIN))).fill_null(False)
    x = x.with_columns(company.alias("pt_company_voice"))
    # launch_labels.own_expr, with the account kind read from the previous posts.
    x = x.with_columns(pl.col("pt_company_voice").replace_strict({True: "company", False: "other"}, return_dtype=pl.String).alias("akind"), pl.col("_video").alias("has_video"))
    x = x.with_columns(L.own_expr().alias("_own")).with_columns(pl.when(pl.col("_own")).then(pl.col("ptype_raw")).otherwise(pl.lit("other")).alias("pt_type"))

    # Trailing record. A previous post counts as the account's own announcement under the same test, with the
    # account kind as it stands now, so one definition holds for the whole window.
    c = pl.col("pt_company_voice").cast(pl.Float64)
    cnt = lambda k: pl.col(f"s_n_fp_{k}").fill_null(0.0) + c * pl.col(f"s_n_op_{k}").fill_null(0.0)      # noqa: E731
    tot = lambda k: pl.col(f"s_l_fp_{k}").fill_null(0.0) + c * pl.col(f"s_l_op_{k}").fill_null(0.0)      # noqa: E731

    def record(kinds, name: str) -> list[pl.Expr]:
        nn, ss = pl.sum_horizontal([cnt(k) for k in kinds]), pl.sum_horizontal([tot(k) for k in kinds])
        return [nn.alias(f"pt_prev_{name}_n"), pl.when(nn > 0).then(ss / nn - pl.col("s_med")).otherwise(None).alias(f"pt_prev_{name}_lift")]

    x = x.with_columns(record(LAUNCH_TYPES, "launch") + record(TYPES, "ann"))
    same_n, same_s = pl.lit(0.0), pl.lit(0.0)
    for k in TYPES:
        same_n = pl.when(pl.col("pt_type") == k).then(cnt(k)).otherwise(same_n)
        same_s = pl.when(pl.col("pt_type") == k).then(tot(k)).otherwise(same_s)
    x = x.with_columns(same_n.alias("pt_prev_same_n"), pl.when(same_n > 0).then(same_s / same_n - pl.col("s_med")).otherwise(None).alias("pt_prev_same_lift"))

    # Traits: the package function, one post at a time.
    tr = [PT.traits(text, [{"type": "video"}] if v else ([{"type": "photo"}] if m else []), kind or "original")
          for text, v, m, kind in zip(x["text"].to_list(), x["_video"].to_list(), x["_media"].to_list(), x["kind"].to_list())]
    x = x.with_columns([pl.Series(f"pt_{k}", [r[k] for r in tr], dtype=pl.Boolean) for k in TRAITS])
    is_launch = pl.col("pt_type").is_in(list(LAUNCH_TYPES))
    x = x.with_columns([(pl.col("pt_type") == k).alias(f"pt_{k}") for k in TYPES]
                       + [(pl.col("ptype_raw") != "other").alias("pt_raw_typed"), pl.col("fp_head").alias("pt_fp_head"), pl.col("opener").alias("pt_opener"), pl.col("techy").alias("pt_techy")])
    x = x.with_columns([(is_launch & pl.col(f"pt_{k}")).alias(f"pt_launch_{k}") for k in TRAITS])
    return x.select("post_id", "pt_type", *COLUMNS)
