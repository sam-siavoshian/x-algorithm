"""Read production defaults straight from the repository's Rust parameter files.

home-mixer/params/param.rs is synced from the production configuration system
by a cron job (README, "Experiments and Configuration"), so parsing it at load
time means a weight change upstream reaches this package with a `git pull` and
no code edit. vm-ranker/params.rs uses the same `param!` macro for the
parameters that only the reranking service reads (author diversity, the
out-of-network factor, DPP).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HOME_MIXER_PARAMS = REPO_ROOT / "home-mixer" / "params" / "param.rs"
VM_RANKER_PARAMS = REPO_ROOT / "vm-ranker" / "params.rs"

_PARAM_RE = re.compile(r'param!\(\s*(\w+)\s*,\s*([\w<>:&\' ]+?)\s*,\s*"([^"]*)"\s*,\s*(.+?)\s*,?\s*\)\s*;', re.S)


def _value(raw: str, ty: str):
    raw = raw.strip()
    if raw in ("true", "false"):
        return raw == "true"
    if raw.startswith('"'):
        return raw.strip('"')
    try:
        num = raw.replace("_", "")
        return float(num) if ("f" in ty or "." in num or "e" in num.lower()) else int(num)
    except ValueError:
        return raw  # an expression; kept verbatim


@lru_cache(maxsize=None)
def load(path: str | Path) -> dict[str, object]:
    text = Path(path).read_text()
    text = re.sub(r"//[^\n]*", "", text)
    out: dict[str, object] = {}
    for name, ty, _key, raw in _PARAM_RE.findall(text):
        out[name] = _value(raw, ty)
    if not out:
        raise ValueError(f"no param! entries found in {path}")
    return out


def home_mixer() -> dict[str, object]:
    return load(HOME_MIXER_PARAMS)


def vm_ranker() -> dict[str, object]:
    return load(VM_RANKER_PARAMS)
