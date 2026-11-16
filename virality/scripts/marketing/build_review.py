#!/usr/bin/env python3
"""Assemble the graded review.

    python scripts/marketing/build_review.py <dir with research_peer.json, research_platform.json, research_playbooks.json>

Writes research/tracks/marketing/research_records.json (every outside claim, one record each, ids prefixed R: peer-reviewed
review, X: platform code / statements / vendor studies, PB playbooks) and expands research/tracks/marketing/
MARKETING_RESEARCH.template.md into MARKETING_RESEARCH.md:
    {{ID}}                 -> linked short source name
    @@rows ID ID ...       -> table rows  claim | source | sample | account control | grade
    @@hyp verdict[,verdict] [ai|general] -> rows of data/marketing/hypotheses.json
    @@appendix             -> every record
"""
import json, re, sys
src = sys.argv[1]; OUT = "research/tracks/marketing/"
R = {}
for f, pre in (("research_peer", "R:"), ("research_platform", "X:"), ("research_playbooks", "")):
    for r in json.load(open(f"{src}/{f}.json")):
        r = dict(r); r["id"] = pre + r["id"]; r["set"] = {"R:": "peer-reviewed and working papers", "X:": "platform code, platform statements, vendor studies", "": "launch playbooks"}[pre]
        if r["id"].startswith("X:L"):   # this project's own earlier corpus results, added by the reviewer as a cross-check: observational, not a controlled study
            r["grade"] = "own corpus (within account, observational)"
        r["note"] = r.pop("verification_note", None) or r.pop("notes", None) or r.get("note")
        R[r["id"]] = r
json.dump(list(R.values()), open(OUT + "research_records.json", "w"), indent=1)
H = json.load(open("data/marketing/hypotheses.json"))["hypotheses"]
esc = lambda s: str(s if s is not None else "").replace("|", "/").replace("\n", " ").strip()
def short(r):
    s = esc(r["source"]); s = s if len(s) <= 70 else s[:67] + "..."
    return f"[{s}]({r['url']})" if r.get("url") else s
def row(i):
    r = R[i]; v = "" if r.get("verified") else " (figure from memory, not re-checked)"
    return f"| {i} | {esc(r['claim'])}{v} | {short(r)} | {esc(r['sample_size'])[:120]} | {esc(r['controlled_for_account'])[:80]} | {esc(r['grade'])} |"
HEAD = "| Id | Claim | Source | Sample | Controlled for the account | Grade |\n|---|---|---|---|---|---|"
def hrow(h):
    r = h.get("result"); n = ""
    if r:
        n = f"{100*r['win_rate']:.1f}% ({100*r['interval'][0]:.1f} to {100*r['interval'][1]:.1f})"
        lr = f"{r['likes_ratio'][0]:.2f}x ({r['likes_ratio'][1]:.2f} to {r['likes_ratio'][2]:.2f})" if r.get("likes_ratio") else "n/a"
        ad = f"{r['adjusted_likes_ratio'][0]:.2f}x ({r['adjusted_likes_ratio'][1]:.2f} to {r['adjusted_likes_ratio'][2]:.2f})" if r.get("adjusted_likes_ratio") else "n/a"
        n = f"{n} | {lr} | {ad} | {r['pairs']:,} / {r['accounts']}"
    else: n = "n/a | n/a | n/a | n/a"
    return f"| {h['id']} | {esc(h['claim'])} | {h['source_grade']} ({', '.join(h['sources'][:4])}) | {'AI' if h['ai_startup_specific'] else 'general'} | {n} | {h.get('grade_ours') or '-'} | {esc(h.get('note') or '')} |"
HH = "| Id | Outside claim | Best outside grade (sources) | Scope | Win rate with the trait | Likes ratio | Adjusted | Pairs / accounts | Our grade | Note |\n|---|---|---|---|---|---|---|---|---|---|"
out = []
for line in open(OUT + "MARKETING_RESEARCH.template.md"):
    line = line.rstrip("\n")
    if line.startswith("@@rows"):
        out.append(HEAD); out += [row(i) for i in line.split()[1:]]
    elif line.startswith("@@hyp"):
        a = line.split(); vs = a[1].split(","); sc = a[2] if len(a) > 2 else None
        out.append(HH); out += [hrow(h) for h in H if h["verdict"] in vs and (sc is None or (sc == "ai") == h["ai_startup_specific"])]
    elif line.startswith("@@appendix"):
        for st in ("peer-reviewed and working papers", "platform code, platform statements, vendor studies", "launch playbooks"):
            out += ["", f"### {st[0].upper() + st[1:]}", "", HEAD] + [row(i) for i, r in R.items() if r["set"] == st]
    else:
        out.append(re.sub(r"\{\{([A-Z:]+\d+)\}\}", lambda m: short(R[m.group(1)]), line))
open(OUT + "MARKETING_RESEARCH.md", "w").write("\n".join(out) + "\n")
from collections import Counter
print(len(R), "records;", Counter(r["grade"] for r in R.values()))
print("AI-specific:", sum(bool(r.get("ai_startup_specific")) for r in R.values()), "unverified:", sum(not r.get("verified") for r in R.values()))
