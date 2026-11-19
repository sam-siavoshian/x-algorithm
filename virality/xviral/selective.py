"""When to speak: the probability that the ensemble's pick between two versions is right, and the speak / stay-silent
decision (research/tracks/selective/REPORT.md).

The rule is a small JSON file (xviral/data/selective_rule.json) fitted on pairs from before the test cutoff. Per kind
of comparison ("same_topic": two versions for one slot; "launch": launch posts) it holds one model that maps what is
known before posting to P(the pick is right). Nothing here needs more than numpy.

    rule = SelectiveRule.load()
    out = rule.decide(version_a, version_b, kind="same_topic")
    # {"pick": "a" | "b", "p_right": 0.66, "p_a_better": 0.66, "speak": True, "threshold": 0.6, "method": "logistic", ...}

A version is a dict: {"members": [{"likes_q50": log10(median likes + 1), optional "likes_q10", "likes_q90",
"rel_like_q50", "p_own_top10", "likes_hist"}], optional "features" (the version's row of the feature frame),
"history" (HISTORY_KEYS), "novelty", "embedding"}. `version_from_report` builds it from a DraftScorer report.
Every input beyond the members' medians is optional: a rule that needs a missing input falls back to its stored
logistic scale on the size of the gap, which is the rule the product used before.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RULE_PATH = Path(__file__).parent / "data" / "selective_rule.json"
HISTORY_KEYS = ("h_n", "h_like_med", "h_like_sd", "h_like_iqr", "h_like_p90", "h_like_max", "h_span_days", "h_media_share", "h_words_med",
                "followers_at_post", "follower_tier_idx", "cadence_7d", "trailing_media_share", "trailing_link_share")
_DIFFS = {"media": "has_media", "video": "has_video", "photo": "has_photo", "url": "has_url", "url_first": "ts_url_first_line", "quote": "is_quote",
          "lines": "ts_lines", "hashtags": "hashtags_n", "mentions": "mentions_n", "emoji": "ts_emoji", "question": "ts_open_question",
          "number": "ts_open_number", "first_person": "ts_open_first_person", "announce": "ts_open_announce", "cta": "ts_call_to_action",
          "bullets": "ts_bullet_lines", "pt_launch": "pt_launch", "pt_any": "pt_any"}
NAN = float("nan")


def _f(x) -> float:
    return NAN if x is None else float(x)


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else NAN


def hist_p_greater(pa: np.ndarray, pb: np.ndarray) -> float:
    """P(a draw from histogram a lands in a higher bin than a draw from b), ties counted half."""
    cb = np.cumsum(pb)
    return float(np.sum(pa * (cb - 0.5 * pb)))


def pair_features(a: dict, b: dict, kind: str = "same_topic", cos_ab: float | None = None, other: tuple | None = None) -> dict:
    """Deferral inputs for the comparison of versions a and b. All are oriented by the call (pick minus other) or are
    symmetric in the two versions, so swapping a and b changes nothing. `other`: the same two versions scored by a
    second ensemble (e.g. untouched members next to fine-tuned ones), optional."""
    qa = np.array([m["likes_q50"] for m in a["members"]], float)
    qb = np.array([m["likes_q50"] for m in b["members"]], float)
    dm = qa - qb
    s = float(dm.mean())
    sign = 1.0 if s >= 0 else -1.0
    pick, oth = (a, b) if sign > 0 else (b, a)
    od = sign * dm
    n = len(dm)
    sd = float(od.std()) if n > 1 else NAN
    f = {"gap": abs(s), "gap_sd": sd, "gap_t": min(abs(s) / (sd / math.sqrt(n) + 1e-3), 50.0) if n > 1 else NAN, "agree": float((od > 0).mean()),
         "gap_min": float(od.min()), "gap_med": float(np.median(od)), "level": float((qa.mean() + qb.mean()) / 2), "kind_launch": float(kind == "launch")}

    def mm(v, key):
        return _mean([m.get(key) for m in v["members"]])

    wp = _mean([m["likes_q90"] - m["likes_q10"] for m in pick["members"] if m.get("likes_q90") is not None and m.get("likes_q10") is not None])
    wo = _mean([m["likes_q90"] - m["likes_q10"] for m in oth["members"] if m.get("likes_q90") is not None and m.get("likes_q10") is not None])
    f["width_pick"], f["width_other"] = wp, wo
    f["z_ovl"] = abs(s) / math.sqrt((wp / 2.563) ** 2 + (wo / 2.563) ** 2 + 1e-6) if not (math.isnan(wp) or math.isnan(wo)) else NAN
    if all(m.get("likes_hist") is not None for m in pick["members"] + oth["members"]):
        hp = np.stack([np.asarray(m["likes_hist"], float) for m in pick["members"]])
        ho = np.stack([np.asarray(m["likes_hist"], float) for m in oth["members"]])
        f["ovl_hist"] = float(np.mean([hist_p_greater(x, y) for x, y in zip(hp, ho)]))
        f["ovl_mix"] = hist_p_greater(hp.mean(0), ho.mean(0))
        ent = lambda h: float(-(h * np.log(np.clip(h, 1e-12, 1))).sum(-1).mean())      # noqa: E731
        f["entropy"] = (ent(hp) + ent(ho)) / 2
    else:
        f["ovl_hist"] = f["ovl_mix"] = f["entropy"] = NAN
    f["gap_rel"] = mm(pick, "rel_like_q50") - mm(oth, "rel_like_q50")
    f["lift_pick"], f["lift_other"] = mm(pick, "rel_like_q50"), mm(oth, "rel_like_q50")
    # The part of the gap that is the account's own level moving between the two posting times (zero for two versions of
    # one slot, large for two posts months apart): a gap made of it is the account growing, not the content differing.
    f["gap_level"] = abs(s) - f["gap_rel"]
    lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4))) if not math.isnan(p) else NAN      # noqa: E731
    f["gap_top10"] = lg(mm(pick, "p_own_top10")) - lg(mm(oth, "p_own_top10"))
    if other is not None and other[0] is not None and other[1] is not None:
        o = np.array([m["likes_q50"] for m in other[0]["members"]], float) - np.array([m["likes_q50"] for m in other[1]["members"]], float)
        f["o_gap"], f["o_sd"], f["o_agree"] = float(sign * o.mean()), float(o.std()), float((sign * o > 0).mean())
    else:
        f["o_gap"] = f["o_sd"] = f["o_agree"] = NAN

    ha, hb = a.get("history") or {}, b.get("history") or {}
    for k in HISTORY_KEYS:
        f[k] = _mean([ha.get(k), hb.get(k)])
    fol = f.pop("followers_at_post")
    f["log_followers"] = math.log10(fol + 1) if not math.isnan(fol) else NAN
    f["h_p90_gap"] = f["h_like_p90"] - f["h_like_med"]
    f["gap_over_hsd"] = abs(s) / (f["h_like_sd"] + 0.1) if not math.isnan(f["h_like_sd"]) else NAN
    f["level_vs_hist"] = f["level"] - f["h_like_med"]

    np_, no_ = pick.get("novelty") or {}, oth.get("novelty") or {}
    for k in ("sim_top3", "sim_centroid"):
        f[f"{k}_pick"], f[f"{k}_other"] = _f(np_.get(k)), _f(no_.get(k))
    f["sim_centroid_diff"] = f["sim_centroid_pick"] - f["sim_centroid_other"]
    f["cohesion"] = _mean([np_.get("cohesion"), no_.get("cohesion")])
    if cos_ab is None and a.get("embedding") is not None and b.get("embedding") is not None:
        ea, eb = np.asarray(a["embedding"], float), np.asarray(b["embedding"], float)
        cos_ab = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
    f["cos_ab"] = _f(cos_ab)

    fp, fo = pick.get("features") or {}, oth.get("features") or {}
    ta, tb = fp.get("created_at_ms"), fo.get("created_at_ms")
    f["days_apart"] = abs(ta - tb) / 86_400_000 if ta is not None and tb is not None else NAN
    for name, col in _DIFFS.items():
        x, y = fp.get(col), fo.get(col)
        f[f"d_{name}"] = float(x) - float(y) if x is not None and y is not None else NAN
    wpk, wot = fp.get("ts_words"), fo.get("ts_words")
    f["d_logwords"] = math.log((wpk + 1) / (wot + 1)) if wpk is not None and wot is not None else NAN
    f["abs_logwords"] = abs(f["d_logwords"])
    f["words_mean"] = _mean([wpk, wot])
    f["any_media"] = _f(max(fp.get("has_media") or 0, fo.get("has_media") or 0)) if fp and fo else NAN
    f["same_kind"] = float(fp.get("kind") == fo.get("kind")) if fp and fo else NAN
    for col in ("ts_len_vs_own", "ts_media_vs_own", "ts_link_vs_own"):
        f[f"{col}_pick"], f[f"{col}_other"] = _f(fp.get(col)), _f(fo.get(col))
    return f


def post_type_flags(text: str | None) -> dict:
    """pt_launch: the opening reads as a launch or open-source release; pt_any: as any typed company post (text rules of
    xviral/text/launch_labels.py, before the own-announcement test)."""
    import polars as pl

    from xviral.text import launch_labels as L

    t = L.add_types(pl.DataFrame({"text": [text or ""]}))["ptype_raw"][0]
    return {"pt_launch": float(t in ("launch", "open_source")), "pt_any": float(t != "other")}


def history_stats(hist_rows, when_ms: int) -> dict:
    """Summary of the account's visible history (feature-frame rows before the draft), as scripts/selective/sel_score.py computes it."""
    n = hist_rows.height
    hl = np.log10(hist_rows["likes"].drop_nulls().to_numpy().astype(float) + 1) if n else np.array([])
    return {"h_n": n, "h_like_med": float(np.median(hl)) if len(hl) else NAN, "h_like_sd": float(hl.std()) if len(hl) > 1 else NAN,
            "h_like_iqr": float(np.subtract(*np.percentile(hl, [75, 25]))) if len(hl) > 3 else NAN,
            "h_like_p90": float(np.percentile(hl, 90)) if len(hl) else NAN, "h_like_max": float(hl.max()) if len(hl) else NAN,
            "h_span_days": float((when_ms - hist_rows["created_at_ms"].min()) / 86_400_000) if n else NAN,
            "h_media_share": float(hist_rows["has_media"].mean()) if n else NAN, "h_words_med": float(hist_rows["ts_words"].median()) if n else NAN}


def novelty(hist_emb: np.ndarray, draft_emb: np.ndarray) -> dict:
    """How close a version sits to the account's last 48 embedded posts: mean cosine, mean of the three nearest, cosine to
    their centroid, and how tight the history itself is (cohesion)."""
    nrm = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)      # noqa: E731
    hv = np.asarray(hist_emb, np.float32)
    hv = hv[np.abs(hv).sum(1) > 0][-48:] if len(hv) else hv
    if not len(hv):
        return {"sim_mean": NAN, "sim_top3": NAN, "sim_centroid": NAN, "cohesion": NAN}
    hn, v = nrm(hv), nrm(np.asarray(draft_emb, np.float32))
    cen = nrm(hn.mean(0))
    sims = hn @ v
    return {"sim_mean": float(sims.mean()), "sim_top3": float(np.sort(sims)[-3:].mean()), "sim_centroid": float(cen @ v), "cohesion": float((hn @ cen).mean())}


def version_from_report(report: dict) -> dict:
    """A DraftScorer report as a version. Members' medians always exist for an ensemble; the richer fields are read from
    the members and report["selective"] when DraftScorer ran with `selective_inputs = True`."""
    lg = lambda v: math.log10(v + 1)      # noqa: E731
    members = [{"likes_q50": lg(m["likes_q50"]), **{k: lg(m[k]) for k in ("likes_q10", "likes_q90") if m.get(k) is not None},
                **{k: m[k] for k in ("rel_like_q50", "p_own_top10", "likes_hist") if m.get(k) is not None}} for m in report.get("members") or []]
    if not members:
        members = [{"likes_q50": lg(report["likes"]["q50"]), "likes_q10": lg(report["likes"]["q10"]), "likes_q90": lg(report["likes"]["q90"])}]
    extra = report.get("selective") or {}
    return {"members": members, "features": extra.get("features"), "history": extra.get("history"), "novelty": extra.get("novelty"), "embedding": extra.get("embedding")}


# --- models: evaluated from plain JSON ------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _prep(spec: dict, f: dict) -> np.ndarray | None:
    x = np.array([f.get(k, NAN) for k in spec["features"]], float)
    miss = np.isnan(x)
    if miss.any():
        if spec.get("impute") is None or float(miss.mean()) > spec.get("max_missing", 0.34):
            return None
        x = np.where(miss, np.asarray(spec["impute"], float), x)
    return x


def _tree_value(node: dict, x: np.ndarray) -> float:
    while "leaf_value" not in node:
        v = x[node["split_feature"]]
        left = (v <= node["threshold"]) if not math.isnan(v) else node.get("default_left", True)
        node = node["left_child"] if left else node["right_child"]
    return node["leaf_value"]


def predict(spec: dict, f: dict) -> float | None:
    """P(the pick is right) from one model spec, or None when its inputs are missing."""
    t = spec["type"]
    if t == "logistic_gap":
        return float(_sigmoid(spec["scale"] * f["gap"] + spec.get("bias", 0.0)))
    if t == "gbm":                                  # trees route missing values themselves; only a mostly empty input falls back
        x = np.array([f.get(k, NAN) for k in spec["features"]], float)
        if float(np.isnan(x).mean()) > spec.get("max_missing", 0.6):
            return None
    else:
        x = _prep(spec, f)
        if x is None:
            return None
    if t == "linear":
        z = float(((x - np.asarray(spec["mean"])) / np.asarray(spec["std"])) @ np.asarray(spec["coef"]) + spec["intercept"])
    elif t == "mlp":
        h0 = (x - np.asarray(spec["mean"])) / np.asarray(spec["std"])
        zs = []
        for net in spec["nets"]:
            h = np.tanh(np.asarray(net["w1"]) @ h0 + np.asarray(net["b1"]))
            zs.append(float(np.asarray(net["w2"]) @ h + net["b2"]))
        z = float(np.mean(zs))
    elif t == "gbm":
        z = spec.get("init", 0.0) + sum(_tree_value(tr, x) for tr in spec["trees"])
    else:
        raise ValueError(f"unknown model type {t}")
    cal = spec.get("calibration") or {"a": 1.0, "b": 0.0}
    return float(_sigmoid(cal["a"] * z + cal["b"]))


class SelectiveRule:
    def __init__(self, rule: dict, ensemble: str | None = None):
        """ensemble: which members make the call ("base3", "start10", "ft10" in the shipped file); the confidence scale
        depends on it (fine-tuned members spread their scores about twice as wide). None = the file's default."""
        if ensemble is not None or "kinds" not in rule:
            name = ensemble or rule["default_ensemble"]
            if name not in rule["ensembles"]:
                raise KeyError(f"no rule for ensemble {name!r}; the file has {sorted(rule['ensembles'])}")
            rule = {**rule, "kinds": rule["ensembles"][name]["kinds"], "ensemble": name}
        self.rule = rule
        self.speak = float(rule.get("speak", 0.60))

    @classmethod
    def load(cls, path: str | Path | None = None, ensemble: str | None = None) -> "SelectiveRule":
        return cls(json.loads(Path(path or RULE_PATH).read_text()), ensemble)

    def p_right(self, f: dict, kind: str = "same_topic") -> tuple[float, str]:
        spec = self.rule["kinds"].get(kind) or self.rule["kinds"]["same_topic"]
        p = predict(spec["model"], f)
        if p is None:
            return predict(spec["fallback"], f), "fallback:" + spec["fallback"]["type"]
        return p, spec["model"]["type"]

    def decide(self, a: dict, b: dict, kind: str = "same_topic", cos_ab: float | None = None, other: tuple | None = None, speak: float | None = None) -> dict:
        """a, b: two scored versions (see module docstring). Returns the pick, P(pick is right), P(a does better) and whether to speak."""
        f = pair_features(a, b, kind=kind, cos_ab=cos_ab, other=other)
        p, how = self.p_right(f, kind)
        s = float(np.mean([m["likes_q50"] for m in a["members"]]) - np.mean([m["likes_q50"] for m in b["members"]]))
        pick = "a" if s >= 0 else "b"
        thr = self.speak if speak is None else speak
        thr = float((self.rule["kinds"].get(kind) or {}).get("speak", thr)) if speak is None else thr
        return {"pick": pick, "p_right": p, "p_a_better": p if pick == "a" else 1.0 - p, "speak": bool(p >= thr), "threshold": thr,
                "method": how, "kind": kind, "gap_log10": s, "members": len(a["members"])}

    def compare_reports(self, base: dict, alt: dict, kind: str = "same_topic") -> dict:
        """The advisor's signature: two DraftScorer reports from one scoring pass. p_better is P(alt does better than base)."""
        out = self.decide(version_from_report(alt), version_from_report(base), kind=kind)
        return {"p_better": out["p_a_better"], "speak": out["speak"], "p_right": out["p_right"], "method": out["method"], "threshold": out["threshold"]}
