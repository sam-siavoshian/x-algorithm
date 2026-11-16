#!/usr/bin/env python3
"""Put research/tracks/expand/REPORT.md together from the written sections (_part_*.md) and the generated result tables."""
from pathlib import Path

R = Path("research/tracks/expand")
E = Path("data/expand/eval")


def part(name: str) -> str:
    return (R / f"_part_{name}.md").read_text().rstrip() + "\n"


def bump(md: str) -> str:
    return md.replace("\n### ", "\n#### ").replace("### ", "#### ", 1) if md.startswith("### ") else md.replace("\n### ", "\n#### ")


tail = part("results_tail")
for key, name in (("EXPLONG_PARAGRAPH", "explong"), ("FT_COMMENT", "ft_comment")):
    p = R / f"_part_{name}.md"
    tail = tail.replace(key, p.read_text().rstrip() if p.exists() else "(not run)")
ft = (E / "results_ft.md").read_text() if (E / "results_ft.md").exists() else "(not run)"
tail = tail.replace("FT_TABLES", bump(ft).rstrip())
doc = "\n".join([part("head"), part("data"), part("levels"), part("results_head") + "\n" + bump((E / "results.md").read_text()).rstrip() + "\n", tail, part("cmds"), part("files")])
(R / "REPORT.md").write_text(doc)
print(len(doc.splitlines()), "lines")
