"""Shared helpers of the judge_product track: tables with every judge's log-odds joined on, text similarity, metrics."""
import difflib, glob, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_judge"))
from lib import ci, load_judge  # noqa: E402,F401

D = "data/llm/"
L125, L2 = np.log(1.25), np.log(2)
# 2026-04-23 (test cutoff), 2026-04-30 and 2026-05-18 (knowledge cutoffs on OpenAI's model pages), 2026-02-16 (the day the
# Qwen3.5-397B weights were published on Hugging Face: no knowledge cutoff is stated, and it cannot be later than that)
CUT = {"test": 1776902400000, "gpt-6.1-sol": 1777507200000, "gpt-6-luna": 1779062400000, "Qwen/Qwen3.5-397B-A17B": 1771200000000}
SHORT = {"gpt-6-luna": "luna", "gpt-6.1-sol": "sol", "openai--gpt-oss-120b": "oss", "deepseek-ai--DeepSeek-V4.1-Flash": "dsflash", "Qwen--Qwen3.5-397B-A17B": "qwen"}


def textsim(name: str) -> pl.DataFrame:
    p = f"{D}jp_textsim_{name}.parquet"
    if not os.path.exists(p):
        rows = [(x["post_id"], x["post_id_b"], difflib.SequenceMatcher(None, x["a"]["text"], x["b"]["text"]).ratio()) for x in map(json.loads, open(f"{D}items_{name}.jsonl"))]
        pl.DataFrame(rows, schema=["post_id", "post_id_b", "sim"], orient="row").write_parquet(p)
    return pl.read_parquet(p)


def table(name: str, clear: bool = True) -> pl.DataFrame:
    """One row per pair: d (log like ratio, post_id over post_id_b), sg (surrogate gap, log10), sim, and one column per judge
    run named <model>[_<config>] (n30 is the default config and has no suffix), holding order-averaged log-odds or null."""
    P = pl.read_parquet(f"{D}jp_pairs_{name}.parquet").join(textsim(name), on=["post_id", "post_id_b"], how="left")
    for f in sorted(glob.glob(f"{D}out/{name}__*.jsonl")):
        _, cfg, model, mode = os.path.basename(f)[:-6].split("__")
        col = SHORT.get(model, model) + ("" if cfg == "n30" else "_" + cfg) + ("" if mode in ("logprob", "json-low", "json-default") or (model, mode) == ("gpt-6-luna", "logprob") else "_" + mode)
        if model == "gpt-6-luna" and mode != "logprob":
            col = "luna_" + cfg + "_" + mode
        try:
            Jd = load_judge(f).select("post_id", "post_id_b", pl.col("j").alias(col), pl.col("pos").alias(col + "__pos"))
        except Exception:  # noqa: BLE001  a run with no complete pair yet
            continue
        P = P.join(Jd, on=["post_id", "post_id_b"], how="left")
    if clear:
        P = P.filter(pl.col("d").abs() >= L125)
    return P.sort("post_id", "post_id_b")


def correct(s, d):
    """1 right, 0 wrong, 0.5 when the score is exactly undecided."""
    return np.where(s == 0, 0.5, (s * d) > 0).astype(float)


def rate(c, m, auth, n=1000):
    """Mean of c over mask m with an account-resampled 95% interval: (pairs, mean, lo, hi)."""
    if m.sum() == 0:
        return (0, None, None, None)
    lo, hi = ci(auth, lambda i: c[i][m[i]].mean() if m[i].sum() else np.nan, n=n)
    return (int(m.sum()), float(c[m].mean()), lo, hi)


def fmt_rate(r, pct=True):
    if r[1] is None:
        return "-"
    return f"{100 * r[1]:.1f}% ({100 * r[2]:.1f} to {100 * r[3]:.1f})"


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
