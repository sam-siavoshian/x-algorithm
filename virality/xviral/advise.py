"""Reasons and fixes for a draft (research/11_reasons_and_fixes.md).

The report makes three kinds of statement and nothing else:

- rules      what X's code decides for this draft, deterministically (xalgo/gates.py);
- record     facts about the account's own past posts that the user can check;
- changes    single, defined edits of the draft, each scored against the original in one pass.

A change is *recommended* only when every gate passes: that kind of change has a measured effect in
same-account, same-topic pairs (data/edit_evidence.json); the calibrated probability that the edited
version does better is at least `speak`; every member of the ensemble agrees on the direction; and,
for a rewrite, the meaning is preserved. Otherwise the report says there is no confident difference.

Nothing here writes advice in free text. The reason for a recommendation is the name of the one thing
the operator changed. A language model, when supplied as `rewrite_fn`, only carries out an operator's
instruction on the text; what is better is decided by the scorer and the gates.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

URL = re.compile(r"https?://\S+")
EVIDENCE_PATH = Path(__file__).parent / "data" / "edit_evidence.json"
PHOTO = {"type": "photo", "url": None, "video_url": None, "width": None, "height": None, "duration_ms": None, "alt": None}


def load_evidence(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or EVIDENCE_PATH).read_text())


# --- Operators: one defined change each ---------------------------------------------------------------


@dataclass(frozen=True)
class Operator:
    name: str
    change: str                 # what differs, in the words the report uses
    evidence: str | None        # key into the evidence table; None = this kind of change has not been measured
    apply: Callable[[dict, dict], dict | None]
    needs_rewriter: bool = False


def _body_and_links(text: str) -> tuple[str, list[str]]:
    links = URL.findall(text)
    body = "\n".join(" ".join(line.split()) for line in URL.sub(" ", text).split("\n"))
    return re.sub(r"\n{3,}", "\n\n", body).strip(), links


def _link_out_of_first_line(d: dict, ctx: dict) -> dict | None:
    text = d.get("text") or ""
    body, links = _body_and_links(text)
    if not links or not body or links[0] not in text.split("\n")[0]:
        return None
    return {**d, "text": body + "\n\n" + links[0]}


def _one_point_per_line(d: dict, ctx: dict) -> dict | None:
    text = d.get("text") or ""
    body, links = _body_and_links(text)
    if len([line for line in body.split("\n") if line.strip()]) >= 3 or (links and links[0] in text.split("\n")[0]):
        return None                     # already laid out, or the link leads: moving it too would be two changes at once
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", " ".join(body.split())) if x.strip()]
    if len(sentences) < 3:
        return None
    return {**d, "text": "\n\n".join(sentences) + ("\n\n" + " ".join(links) if links else "")}


def _attach_image(d: dict, ctx: dict) -> dict | None:
    if d.get("media"):
        return None
    return {**d, "media": [dict(PHOTO)]}


def _remove_hashtags(d: dict, ctx: dict) -> dict | None:
    text = d.get("text") or ""
    new = re.sub(r"[ \t]{2,}", " ", re.sub(r"(?<![\w&])#\w+", "", text)).strip()
    return {**d, "text": new} if new != text.strip() and new else None


def _add_detail(d: dict, ctx: dict) -> dict | None:
    text = d.get("text") or ""
    body, links = _body_and_links(text)
    new = ctx["rewrite_fn"](text, "Rewrite this post so that it says the same thing with more concrete detail: what exactly, how much, for whom. "
                                  "Keep the author's voice, keep every fact and any link, invent nothing. At least one and a half times as long.")
    nbody, _ = _body_and_links(new or "")
    return {**d, "text": new} if new and len(nbody) >= 1.5 * max(len(body), 20) else None


OPERATORS: list[Operator] = [
    Operator("attach_image", "an image is attached", "media", _attach_image),
    Operator("link_out_of_first_line", "the link moves from the first line to the end", "link_not_first", _link_out_of_first_line),
    Operator("one_point_per_line", "each sentence gets its own line", "lines", _one_point_per_line),
    Operator("add_detail", "the same message with more concrete detail, at least 1.5 times longer", "longer", _add_detail, needs_rewriter=True),
    Operator("remove_hashtags", "the hashtags are removed", "no_hashtag", _remove_hashtags),
]


# --- The account's own record ----------------------------------------------------------------------------


def own_record(history: pl.DataFrame, min_each: int = 8, seed: int = 0) -> list[dict]:
    """Differences inside the account's own history that are large enough to state: median outcome of
    posts with an attribute against posts without, kept only when the interval excludes no difference."""
    if history.height < 2 * min_each:
        return []
    h = history.sort("created_at_ms").with_columns(
        (pl.col("media").list.len() > 0).alias("an image or video"),
        ((pl.col("urls").list.len() > 0) | (pl.col("text").fill_null("").str.count_matches(r"https?://")
                                           > pl.col("text").fill_null("").str.count_matches(r"https?://t\.co/"))).alias("an external link"),
        (pl.col("text").fill_null("").str.split("\n").list.eval(pl.element().str.strip_chars().str.len_chars() > 0).list.sum() >= 3).alias("three or more lines"),
        ((pl.col("created_at_ms").diff() / 3_600_000) < 2).fill_null(False).alias("less than 2 hours after the previous post"),
    )
    outcome = "views" if h["views"].is_not_null().sum() >= 2 * min_each else "likes"
    h = h.filter(pl.col(outcome).is_not_null())
    rng = np.random.default_rng(seed)
    facts = []
    for name in ("an image or video", "an external link", "three or more lines", "less than 2 hours after the previous post"):
        a = np.log10(h.filter(pl.col(name))[outcome].to_numpy().astype(float) + 1)
        b = np.log10(h.filter(~pl.col(name))[outcome].to_numpy().astype(float) + 1)
        if len(a) < min_each or len(b) < min_each:
            continue
        boots = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(1000)]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        if lo > 0 or hi < 0:
            facts.append({"attribute": name, "outcome": outcome, "median_with": float(10 ** np.median(a) - 1), "n_with": int(len(a)),
                          "median_without": float(10 ** np.median(b) - 1), "n_without": int(len(b)),
                          "ratio": float(10 ** (np.median(a) - np.median(b))), "ratio_lo": float(10 ** lo), "ratio_hi": float(10 ** hi)})
    return facts


# --- The advisor -----------------------------------------------------------------------------------------


class Advisor:
    def __init__(self, scorer, scales: dict, evidence: dict | None = None, speak: float = 0.60, strong: float = 0.70,
                 min_meaning_cos: float = 0.75):
        """scorer: serve.DraftScorer (an ensemble gives the agreement gate something to check).
        scales: {"views": k, "likes": k} from held-out pairs (evaluate.fit_pair_scale); a missing target cannot speak.
        speak / strong: calibrated probability at which a change is recommended / called likely."""
        self.scorer, self.scales, self.evidence = scorer, scales, evidence or load_evidence()
        self.speak, self.strong, self.min_meaning_cos = speak, strong, min_meaning_cos

    def _target(self) -> str:
        return "views" if self.scorer.meta.get("has_views") and "views" in self.scales else "likes"

    @staticmethod
    def _score(report: dict, target: str) -> float:
        q = report["views_24h"]["q50"] if target == "views" else report["likes"]["q50"]
        return math.log10(q + 1)

    def _compare(self, base: dict, alt: dict, target: str) -> dict:
        k = self.scales.get(target)
        diff = self._score(alt, target) - self._score(base, target)
        p = None if k is None else 1.0 / (1.0 + math.exp(-k * diff))
        key = "views_q50" if target == "views" else "likes_q50"
        signs = [np.sign(math.log10(a[key] + 1) - math.log10(b[key] + 1)) for a, b in zip(alt.get("members", []), base.get("members", []))]
        agree = all(s == np.sign(diff) for s in signs) if signs else None
        return {"p_better": p, "predicted_ratio": float(10 ** diff), "members_agree": agree, "members": len(signs)}

    def _status(self, cmp: dict, ev: dict | None, meaning_ok: bool = True) -> tuple[str, str]:
        p = cmp["p_better"]
        if p is None:
            return "uncalibrated", "no confidence scale for this outcome"
        if cmp["members_agree"] is False:
            return "no_confident_difference", "the models in the ensemble disagree on the direction"
        if p >= self.speak:
            if not meaning_ok:
                return "no_confident_difference", "the rewrite drifts from what the draft says"
            if ev is None or not ev.get("supported"):
                return "untested", "the model prefers it, but this kind of change has no measured effect in controlled pairs"
            return ("recommended_likely" if p >= self.strong else "recommended"), "all gates passed"
        if p <= 1 - self.speak:
            return "predicted_worse", "the model expects this version to do worse"
        return "no_confident_difference", "the two versions are too close to call"

    def advise(self, history: pl.DataFrame, draft: dict, account: dict, when_ms: int | None = None,
               rewrite_fn: Callable[[str, str], str] | None = None, alternatives: list[dict] | None = None) -> dict:
        """alternatives: the user's own other versions ({"text": ..., "label": ...}); they are compared with the
        draft like any change, with no reason attached, since more than one thing differs."""
        ctx = {"rewrite_fn": rewrite_fn}
        variants: list[tuple[Operator | None, str, dict]] = []
        for op in OPERATORS:
            if op.needs_rewriter and rewrite_fn is None:
                continue
            new = op.apply(draft, ctx)
            if new is not None:
                variants.append((op, op.change, new))
        for i, alt in enumerate(alternatives or []):
            variants.append((None, alt.get("label") or f"your version {i + 2}", {**draft, **{k: v for k, v in alt.items() if k != "label"}}))
        reports = self.scorer.score(history, [draft] + [v for _, _, v in variants], account, when_ms)
        base, target = reports[0], self._target()

        changes = []
        for (op, label, new), rep in zip(variants, reports[1:]):
            cmp = self._compare(base, rep, target)
            ev = self.evidence.get(op.evidence) if op and op.evidence else None
            meaning_ok = True
            if op is not None and op.needs_rewriter:
                a, b = self.scorer.embed_fn([draft.get("text") or "", new.get("text") or ""])
                meaning_ok = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) >= self.min_meaning_cos
            if op is None:                       # a user's alternative: compared, never explained
                status, why = self._status(cmp, {"supported": True}, True)
            else:
                status, why = self._status(cmp, ev, meaning_ok)
            changes.append({"change": label, "operator": op.name if op else None, "status": status, "why": why, **cmp, "outcome": target,
                            "text": new.get("text"), "adds_media": bool(new.get("media")) and not draft.get("media"),
                            "evidence": None if ev is None else {k: ev[k] for k in ("win_rate", "lo", "hi", "pairs", "design", "supported") if k in ev},
                            "views_q50": rep["views_24h"]["q50"], "likes_q50": rep["likes"]["q50"]})

        # Spacing: a post close behind the previous one competes with it; compared at a later time, same draft.
        gap = next((g for g in base["gates"] if g["gate"] == "author_diversity"), None)
        if gap and gap["hours_since_previous"] < 2 and when_ms is not None:
            later = when_ms + int((2.0 - gap["hours_since_previous"]) * 3_600_000) + 60_000
            rep = self.scorer.score(history, [draft], account, later)[0]
            cmp = self._compare(base, rep, target)
            status, why = self._status(cmp, self.evidence.get("spacing"))
            changes.append({"change": "the post goes out 2 hours after the previous one instead of %.1f" % gap["hours_since_previous"], "operator": "wait_for_gap",
                            "status": status, "why": why, **cmp, "outcome": target, "text": draft.get("text"), "adds_media": False, "when_ms": later,
                            "evidence": {k: v for k, v in self.evidence.get("spacing", {}).items() if k != "description"},
                            "views_q50": rep["views_24h"]["q50"], "likes_q50": rep["likes"]["q50"]})

        order = {"recommended_likely": 0, "recommended": 1, "untested": 2, "no_confident_difference": 3, "predicted_worse": 4, "uncalibrated": 5}
        changes.sort(key=lambda c: (order[c["status"]], -(c["p_better"] or 0)))
        return {
            "draft": {k: base[k] for k in ("text", "views_24h", "likes", "p_reach_1x", "p_reach_10x", "p_reach_100x", "p_own_top_10pct",
                                           "lift_vs_own_median", "own_median", "history_posts_used")},
            "rules": [g for g in base["gates"] if g["status"] in ("warn", "fail")],
            "record": own_record(history),
            "changes": changes,
            "outcome_compared": target,
            "thresholds": {"speak": self.speak, "likely": self.strong},
        }


def render(advice: dict) -> str:
    """The report as plain text. Every sentence is a rule, a fact from the account's history, or a scored change."""
    num = lambda v: "n/a" if v is None else f"{v:,.0f}"            # noqa: E731
    d = advice["draft"]
    top = d["p_own_top_10pct"]["views"] if d["p_own_top_10pct"]["views"] is not None else d["p_own_top_10pct"]["likes"]
    out = [f"Your draft: views {num(d['views_24h']['q10'])} to {num(d['views_24h']['q90'])}, median {num(d['views_24h']['q50'])}; likes median {num(d['likes']['q50'])}."
           + (f" Odds of landing in your own top 10%: {100 * top:.0f}%." if top is not None else "")
           + f" Odds of reaching 10x your followers: {100 * d['p_reach_10x']:.1f}%."]
    if advice["rules"]:
        out.append("\nRules in X's code that apply:")
        out += [f"  [{g['status']}] {g['detail']} ({g['source']})" for g in advice["rules"]]
    if advice["record"]:
        out.append("\nFrom your own posts:")
        for f in advice["record"]:
            out.append(f"  With {f['attribute']}: median {num(f['median_with'])} {f['outcome']} over {f['n_with']} posts. Without: {num(f['median_without'])} over "
                       f"{f['n_without']}. Ratio {f['ratio']:.2f} ({f['ratio_lo']:.2f} to {f['ratio_hi']:.2f}).")
    rec = [c for c in advice["changes"] if c["status"].startswith("recommended")]
    out.append("\nChanges we stand behind:" if rec else "\nNo single change we can stand behind for this draft.")
    for c in rec:
        ev = c["evidence"] or {}
        proof = (f" In {ev['pairs']:,} controlled pairs the version with this won {100 * ev['win_rate']:.0f}% of the time." if "win_rate" in ev else
                 f" Posts closer than 2 hours to the previous one got {ev['ratio_when_closer']:.2f}x the views across {ev['accounts']} accounts." if "ratio_when_closer" in ev else "")
        out.append(f"  {'Likely better' if c['status'] == 'recommended_likely' else 'Probably better'} ({100 * c['p_better']:.0f}%): {c['change']}."
                   f" Predicted {c['outcome']} {c['predicted_ratio']:.2f}x.{proof}")
    rest = [c for c in advice["changes"] if not c["status"].startswith("recommended")]
    if rest:
        out.append("\nAlso checked:")
        label = {"untested": "not recommended, untested kind of change", "no_confident_difference": "no confident difference",
                 "predicted_worse": "predicted worse", "uncalibrated": "not calibrated"}
        out += [f"  {c['change']}: {label[c['status']]}" + (f" ({100 * c['p_better']:.0f}% that it is better)" if c["p_better"] is not None else "") for c in rest]
    return "\n".join(out)
