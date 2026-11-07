"""Small shared helpers: snowflake time, stable hashing, safe literal parsing, Parquet IO."""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from xviral.constants import TWITTER_EPOCH_MS

log = logging.getLogger("xviral")

# Snowflake IDs began on 2010-11-04; anything below this is a pre-snowflake sequential ID.
_MIN_SNOWFLAKE = 1 << 32


def snowflake_ms(post_id: int | None) -> int | None:
    """Creation time in ms from a snowflake ID. Same decode as recsys_batch.py:996-997."""
    if post_id is None or post_id < _MIN_SNOWFLAKE:
        return None
    return (post_id >> 22) + TWITTER_EPOCH_MS


def snowflake_ms_expr(col: str | pl.Expr) -> pl.Expr:
    c = pl.col(col) if isinstance(col, str) else col
    return pl.when(c >= _MIN_SNOWFLAKE).then((c // (1 << 22)) + TWITTER_EPOCH_MS).otherwise(None)


def stable_hash64(*parts: Any) -> int:
    """Deterministic signed 63-bit hash, stable across processes and Python versions."""
    h = hashlib.blake2b("\x1f".join("" if p is None else str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") & ((1 << 63) - 1)


def synthetic_post_id(handle: str | None, created_ms: int | None, text: str | None) -> int:
    """Negative ID for rows whose source carries no post ID. Never collides with real IDs."""
    return -stable_hash64("post", (handle or "").lower(), created_ms, text)


def synthetic_author_id(handle: str) -> int:
    return -stable_hash64("author", handle.lower())


# --- Safe parsing of Python-repr fields (USC stores nested objects this way) ---

_DT_RE = re.compile(r"datetime\.datetime\(([^()]*?)(?:,\s*tzinfo=[^()]*(?:\([^()]*\))?)?\)")


def _dt_repl(m: re.Match) -> str:
    nums = [int(x) for x in m.group(1).split(",") if x.strip()]
    nums += [0] * (7 - len(nums))
    dt = datetime(*nums[:7], tzinfo=timezone.utc)
    return repr(dt.isoformat())


def parse_pyrepr(s: str | None) -> Any:
    """Parse a Python-literal repr string without eval.

    datetime.datetime(...) calls are rewritten to ISO strings first, then
    ast.literal_eval does the rest. Returns None on anything unparseable.
    """
    if s is None:
        return None
    s = s.strip()
    if not s or s in ("nan", "NaN", "None"):
        return None
    if "datetime.datetime(" in s:
        s = _DT_RE.sub(_dt_repl, s)
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def to_int(x: Any) -> int | None:
    """Parse ints that may arrive as '12', '12.0', 12.0, '1.8e+18' (precision-lost) or NaN."""
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(f)


def iso_to_ms(s: str | None) -> int | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def date_to_ms(s: str) -> int:
    return iso_to_ms(s) or 0


# --- Parquet IO ---------------------------------------------------------------


def write_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    """Atomic zstd Parquet write: write to a temp file then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True)
    os.replace(tmp, path)
    return path


def scan_dir(path: str | Path, pattern: str = "*.parquet") -> pl.LazyFrame:
    path = Path(path)
    files = sorted(path.glob(pattern)) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return pl.scan_parquet([str(f) for f in files])


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
