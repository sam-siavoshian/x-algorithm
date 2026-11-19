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
                 min_meaning_cos: float = 0.75, selective=None, agreement_gate: bool = True, judge=None, judge_rule=None,
                 judge_deadline: float = 40.0):
        """scorer: serve.DraftScorer (an ensemble gives the agreement gate something to check).
        scales: {"views": k, "likes": k} from held-out pairs (evaluate.fit_pair_scale); a missing target cannot speak.
        speak / strong: calibrated probability at which a change is recommended / called likely."""
        self.scorer, self.scales, self.evidence = scorer, scales, evidence or load_evidence()
        self.speak, self.strong, self.min_meaning_cos = speak, strong, min_meaning_cos
        # selective: xviral.selective.SelectiveRule (or None). With it, the like comparison's probability comes from the
        # fitted speak / stay-silent rule for the kind of comparison instead of the single scale in `scales`
        # (research/tracks/selective/REPORT.md). agreement_gate=False drops the "every member agrees" requirement, which
        # with ten members removes more than half of the calls the probability alone would make.
        self.selective, self.agreement_gate = selective, agreement_gate
        if selective is not None:
            scorer.selective_inputs = True
        # judge: xviral.judge.Judge (or None). With it, a comparison of two versions whose wording differs (the user's
        # versions against each other, a rewrite against the draft) is decided by the forecasting model and a language model
        # that read both versions, blended by the rule fitted in research/tracks/judge_product (xviral/data/judge_rule.json).
        # Any failure of the judge leaves the forecasting model's own comparison in place.
        self.judge, self.judge_rule, self.judge_deadline = judge, judge_rule, judge_deadline
        if judge is not None and judge_rule is None:
            from xviral.judge import JudgeRule

            self.judge_rule = JudgeRule.load()

    # --- the second opinion ----------------------------------------------------------------------------------

    def judge_ready(self) -> bool:
        return self.judge is not None and self.judge_rule is not None and bool(self.judge_rule.rule.get("models", {}).get(self.judge.model))

    def start_judging(self, history: pl.DataFrame | None, account: dict, pairs: list[tuple[dict, dict]], share_history: bool = True,
                      when_ms: int | None = None):
        """Start the judge on pairs of drafts (first, second) in the background; returns a xviral.judge.JudgeBatch whose
        results are the log-odds that the first does better, or None when no judge is configured. What is sent: the two
        drafts of each pair, the account line, and, only with `share_history`, the account's most recent posts with likes
        (those from before `when_ms`, the planned posting time, when it is given: the judge was measured with earlier posts only)."""
        if not self.judge_ready() or not pairs or self._target() != "likes":      # the rule is fitted on likes only
            return None
        from xviral import judge as J

        try:
            if share_history and when_ms is not None and history is not None and history.height:
                history = history.filter(pl.col("created_at_ms") < when_ms)
            hist = J.history_posts(history, self.judge.history_n) if share_history else []
            acct = {"name": account.get("name"), "handle": account.get("handle"), "followers": account.get("followers")}
            r = lambda d: J.render_post(d.get("text"), d.get("media"))      # noqa: E731
            return J.JudgeBatch(self.judge, acct, hist, [(r(a), r(b)) for a, b in pairs], deadline=self.judge_deadline)
        except Exception:      # noqa: BLE001  the second opinion must never fail a request: the forecasting model decides alone
            return None

    @staticmethod
    def _kind(base: dict, alt: dict) -> str:
        """launch when both versions read as a launch (the text rules of xviral/text/launch_labels.py), else same_topic."""
        from xviral import selective as sel

        def launch(rep: dict) -> bool:
            if rep.get("reads_as_launch") is not None:
                return bool(rep["reads_as_launch"])
            f = (rep.get("selective") or {}).get("features") or {}
            if f.get("pt_launch") is not None:
                return f["pt_launch"] > 0
            return rep.get("text") is not None and sel.post_type_flags(rep["text"])["pt_launch"] > 0
        return "launch" if launch(base) and launch(alt) else "same_topic"

    def _target(self) -> str:
        # Likes first: the pair benchmark that validates the stated confidence (2,852 same-account, same-topic pairs)
        # has like counts on both sides for every pair and view counts for 13, so only the like comparison is tested.
        if "likes" in self.scales:
            return "likes"
        return "views" if self.scorer.meta.get("has_views") and "views" in self.scales else "likes"

    @staticmethod
    def _score(report: dict, target: str) -> float:
        q = report["views_24h"]["q50"] if target == "views" else report["likes"]["q50"]
        return math.log10(q + 1)

    def _compare(self, base: dict, alt: dict, target: str, judged: dict | None = None) -> dict:
        """judged: the judge's result for (alt, base), log-odds that alt does better; None = the forecasting model alone."""
        k = self.scales.get(target)
        diff = self._score(alt, target) - self._score(base, target)
        p = None if k is None else 1.0 / (1.0 + math.exp(-k * diff))
        key = "views_q50" if target == "views" else "likes_q50"
        # Which members decide this comparison: the scorer's own, or, when the model set names other members for this kind
        # of comparison (serve.DraftScorer.kind_scorers; the reports then carry `kind_reports`), those. The forecast and the
        # predicted ratio stay the scorer's own either way.
        kind_members = None
        if target == "likes" and (alt.get("kind_reports") or base.get("kind_reports")):
            kd = self._kind(base, alt)
            ka, kb = (alt.get("kind_reports") or {}).get(kd), (base.get("kind_reports") or {}).get(kd)
            if ka and kb and ka.get("members") and kb.get("members"):
                kind_members = (ka["members"], kb["members"], kd)
        m_alt, m_base = (kind_members[0], kind_members[1]) if kind_members else (alt.get("members", []), base.get("members", []))
        signs = [np.sign(math.log10(a[key] + 1) - math.log10(b[key] + 1)) for a, b in zip(m_alt, m_base)]
        side = float(np.mean([math.log10(a[key] + 1) - math.log10(b[key] + 1) for a, b in zip(m_alt, m_base)])) if kind_members else diff
        agree = all(s == np.sign(side) for s in signs) if signs else None
        out = {"p_better": p, "predicted_ratio": float(10 ** diff), "members_agree": agree, "members": len(signs)}
        if kind_members:
            out["decided_by_members"] = kind_members[2]
        if self.selective is not None and target == "likes":
            from xviral import selective as sel

            va, vb = sel.version_from_report({**alt, "members": m_alt} if kind_members else alt), sel.version_from_report({**base, "members": m_base} if kind_members else base)
            if kind_members or (base.get("reads_as_launch") is not None and alt.get("reads_as_launch") is not None):
                launch = (kind_members[2] if kind_members else self._kind(base, alt)) == "launch"
            else:
                launch = all(((v.get("features") or {}).get("pt_launch") or 0) > 0 for v in (va, vb))
            d = self.selective.decide(va, vb, kind="launch" if launch else "same_topic", speak=self.speak)
            out.update({"p_better": d["p_a_better"], "p_scale_only": p, "confidence_method": d["method"], "comparison_kind": d["kind"]})
        if judged is not None and judged.get("ok") and target == "likes" and self.judge_ready():
            ma, mb = m_alt or [], m_base or []
            gap = float(np.mean([math.log10(a["likes_q50"] + 1) - math.log10(b["likes_q50"] + 1) for a, b in zip(ma, mb)])) if ma and mb else diff
            kind = self._kind(base, alt)
            d = self.judge_rule.decide(gap, judged["logodds"], kind, self.judge.model, speak=self.speak)
            if d is not None:
                side = lambda first: "neither" if first is None else "this version" if first else "the other version"      # noqa: E731
                out.update({"p_better": d["p_first_better"], "p_forecast_model_only": out["p_better"], "confidence_method": "forecast model + language model",
                            "comparison_kind": kind, "speak": d["speak"], "why_silent": d["why_silent"],
                            "opinions": {"forecast_model": {"prefers": side(d["surrogate"]["pick_first"]), "p_better": d["surrogate"]["p_first_better"]},
                                         "language_model": {"prefers": side(d["judge"]["pick_first"]), "p_better": d["judge"]["p_first_better"]},
                                         "agree": d["agree"]}})
        return out

    def compare(self, base: dict, alt: dict, judged: dict | None = None) -> dict:
        """One version against another on the validated outcome, both taken from the same scoring pass. The pair is
        treated like any user's alternative: compared, never explained. judged: the judge's result for (alt, base)."""
        cmp = self._compare(base, alt, self._target(), judged)
        status, why = self._status(cmp, {"supported": True}, True)
        return {**cmp, "status": status, "why": why}

    def _status(self, cmp: dict, ev: dict | None, meaning_ok: bool = True) -> tuple[str, str]:
        p = cmp["p_better"]
        if p is None:
            return "uncalibrated", "no confidence scale for this outcome"
        if "opinions" in cmp:               # forecasting model + language model: the fitted rule decides whether to speak
            if not cmp["speak"]:
                return "no_confident_difference", cmp["why_silent"]
            if p < 0.5:
                return "predicted_worse", "the forecasting model and the language model together expect this version to do worse"
            if not meaning_ok:
                return "no_confident_difference", "the rewrite drifts from what the draft says"
            if ev is None or not ev.get("supported"):
                return "untested", "the models prefer it, but this kind of change has no measured effect in controlled pairs"
            return ("recommended_likely" if p >= self.strong else "recommended"), "all gates passed"
        if cmp["members_agree"] is False and self.agreement_gate:
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
               rewrite_fn: Callable[[str, str], str] | None = None, alternatives: list[dict] | None = None,
               record: list[dict] | None = None, judged: list | None = None, use_judge: bool = True, share_history: bool = True) -> dict:
        """alternatives: the user's own other versions ({"text": ..., "label": ...}); they are compared with the
        draft like any change, with no reason attached, since more than one thing differs.
        record: own_record(history) when the caller already holds it (the product caches it per account).
        judged: the judge's results for (alternative, draft), one per alternative (None where it has none), when the caller
        already ran it. Otherwise, with a judge configured and `use_judge`, the alternatives are judged here, in the
        background while the forecasting model scores; `share_history` says whether the account's posts may be sent.
        Single defined edits (the operators) and the spacing check are never sent: they are near-identical texts, which is
        not what the judge was validated on."""
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
        batch = None
        n_alt = len(alternatives or [])
        if judged is None and use_judge and n_alt:
            batch = self.start_judging(history, account, [(v, draft) for op, _, v in variants if op is None], share_history, when_ms)
        reports = self.scorer.score(history, [draft] + [v for _, _, v in variants], account, when_ms)
        base, target = reports[0], self._target()
        if batch is not None:
            judged = batch.results()
        judged_of = dict(zip([i for i, (op, _, _) in enumerate(variants) if op is None], judged or []))

        changes = []
        for i, ((op, label, new), rep) in enumerate(zip(variants, reports[1:])):
            cmp = self._compare(base, rep, target, judged_of.get(i))
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
                            "views_q50": rep["views_24h"]["q50"], "likes_q50": rep["likes"]["q50"],
                            **({"report": {**{k: rep.get(k) for k in ("views_24h", "likes", "members")}, "reads_as_launch": self._kind(rep, rep) == "launch",
                                           **({"kind_reports": rep["kind_reports"]} if rep.get("kind_reports") else {})}} if op is None else {})})

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
                                           "lift_vs_own_median", "own_median", "history_posts_used")} | {"members": base.get("members"), "reads_as_launch": self._kind(base, base) == "launch"}
                     | ({"kind_reports": base["kind_reports"]} if base.get("kind_reports") else {}),
            "rules": [g for g in base["gates"] if g["status"] in ("warn", "fail")],
            "record": own_record(history) if record is None else record,
            "changes": changes,
            "outcome_compared": target,
            "thresholds": {"speak": self.speak, "likely": self.strong},
            "judge": None if batch is None else batch.summary(),
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
