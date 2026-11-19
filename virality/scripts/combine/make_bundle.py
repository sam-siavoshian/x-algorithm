#!/usr/bin/env python3
"""Assemble the production bundle: one folder, one config (`bundle.json`), loadable with
xviral.app.service.load_bundle(folder) or XVIRAL_BUNDLE=<folder>.

    python scripts/combine/make_bundle.py --name te3l_broad2 --checkpoint te3l_broad_s1=.../best.pt --checkpoint te3l_broad_s2=.../best.pt \
        --selective data/combine/selective_rule.json --type-correction data/combine/type_correction.json --judge-rule data/combine/judge_rule.json \
        --validation data/combine/validation.json --out data/combine/bundle

The fallback model set (local bge-small encoder, the three models in production before this track) is copied in with
the files it was validated with, so the folder is self-contained.
"""
import argparse, json, os, shutil
import torch
from xviral.embed import api

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--checkpoint", action="append", required=True); ap.add_argument("--sid", default="data/combine/serving_te3l/sid_codebook.npz")
ap.add_argument("--selective", required=True); ap.add_argument("--type-correction", default=None); ap.add_argument("--judge-rule", default=None)
ap.add_argument("--validation", default=None, help="JSON with statement and launch_statement for the report's method note")
ap.add_argument("--launch-checkpoint", action="append", default=[], help="name=path: members that decide launch comparisons (same encoder)")
ap.add_argument("--out", default="data/combine/bundle"); ap.add_argument("--no-fallback", action="store_true")
a = ap.parse_args()
os.makedirs(a.out + "/checkpoints", exist_ok=True)
cks, metas = [], []
for spec in a.checkpoint:
    nm, path = spec.split("=", 1)
    shutil.copy(path, f"{a.out}/checkpoints/{nm}.pt"); cks.append(f"checkpoints/{nm}.pt")
    metas.append(torch.load(path, map_location="cpu", weights_only=False)["table_meta"])
enc = api.common_encoder(metas)
shutil.copy(a.sid, a.out + "/sid_codebook.npz")
sel = json.load(open(a.selective))
shutil.copy(a.selective, a.out + "/selective_rule.json")
ens = sel["ensembles"][a.name]
cfg = {"name": a.name, "bundle_version": 1, "made": "2026-11-19", "checkpoints": cks, "sid_codebook": "sid_codebook.npz",
       "encoder": {k: enc[k] for k in ("name", "provider", "model", "dimensions", "max_chars", "price_per_mtok") if k in enc},
       "scales": {"likes": ens["kinds"]["same_topic"]["model"]["scale"]}, "selective_rule": "selective_rule.json", "selective_ensemble": a.name}
if a.launch_checkpoint:
    lc = []
    for spec in a.launch_checkpoint:
        nm, path = spec.split("=", 1)
        shutil.copy(path, f"{a.out}/checkpoints/{nm}.pt"); lc.append(f"checkpoints/{nm}.pt")
        assert api.encoder_spec(torch.load(path, map_location="cpu", weights_only=False)["table_meta"])["name"] == enc["name"]
    cfg["comparison_members"] = {"launch": {"checkpoints": lc, "note": "These members decide comparisons of two launch posts (their gap feeds the launch confidence scale and the launch judge rule). "
                                                                       "The forecast, same-topic comparisons and everything else come from `checkpoints`."}}
if a.type_correction:
    shutil.copy(a.type_correction, a.out + "/type_correction.json"); cfg["type_correction"] = "type_correction.json"
if a.judge_rule:
    shutil.copy(a.judge_rule, a.out + "/judge_rule.json"); cfg["judge_rule"] = "judge_rule.json"
if a.validation:
    cfg["validation"] = json.load(open(a.validation))
if not a.no_fallback:
    os.makedirs(a.out + "/fallback", exist_ok=True)
    fb = []
    for nm, path in (("ts46_s1", "data/runs/ts46_s1/best.pt"), ("ts46_s2", "data/runs/ts46_s2/best.pt"), ("compare_v2_full", "data/work/model/compare_v2/full/best.pt")):
        shutil.copy(path, f"{a.out}/fallback/{nm}.pt"); fb.append(f"fallback/{nm}.pt")
    shutil.copy("data/work/emb/sid_codebook.npz", a.out + "/fallback/sid_codebook.npz")
    shutil.copy("research/tracks/posttype/type_correction.json", a.out + "/fallback/type_correction.json")
    shutil.copy("xviral/data/judge_rule.json", a.out + "/fallback/judge_rule.json")
    shutil.copy("xviral/data/selective_rule.json", a.out + "/fallback/selective_rule.json")
    scales = {k: v for k, v in json.load(open("research/15_pair_confidence.json")).items() if k in ("likes", "views")}
    cfg["fallback"] = {"name": "base3 (bge-small, local encoder)", "checkpoints": fb, "sid_codebook": "fallback/sid_codebook.npz", "encoder": {"name": api.BGE_SMALL},
                       "scales": scales, "selective_rule": "fallback/selective_rule.json", "selective_ensemble": "base3", "type_correction": "fallback/type_correction.json",
                       "judge_rule": "fallback/judge_rule.json",
                       "note": "Answers when the embedding provider fails or a request does not allow its text to be sent. Its validated numbers are those of research/tracks/judge_product and selective."}
json.dump(cfg, open(a.out + "/bundle.json", "w"), indent=1)
env = [f"# Environment for the service with this bundle (docs/launch_advisor_api.md). Paths are examples.",
       f"XVIRAL_BUNDLE=/models/{os.path.basename(a.out.rstrip('/'))}          # this folder: checkpoints, encoder, codebook, rules, fallback set",
       "OPENAI_API_KEY=...                         # the embedding (text-embedding-3-large) and an OpenAI judge; omit behind a key-injecting proxy",
       "XVIRAL_JUDGE_MODEL=gpt-6.1-sol             # the second opinion on version comparisons; unset = forecasting model alone",
       "XVIRAL_POSTS=/data/work/hygiene/posts.parquet   # optional corpus of public posts for accounts looked up by handle",
       "XVIRAL_PRELOAD=1", "XVIRAL_TORCH_THREADS=4",
       "# optional, shown with their defaults",
       "XVIRAL_EMBED_TIMEOUT=20", "XVIRAL_EMBED_RETRIES=2", "XVIRAL_EMBED_BATCH=256", "XVIRAL_EMBED_CONCURRENCY=4", "XVIRAL_JUDGE_DEADLINE=40",
       "# XVIRAL_EMBED_STUB=1                      # dry run without any provider (numbers are meaningless)"]
open(a.out + "/service.env", "w").write("\n".join(env) + "\n")
print(json.dumps(cfg, indent=1)[:1500]); print(sorted(os.listdir(a.out)))
