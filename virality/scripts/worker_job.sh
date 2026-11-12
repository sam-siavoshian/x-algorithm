#!/bin/bash
# One independent job on a fresh machine: set up, fetch the shared work data from the private Hugging Face
# dataset, train one model variant, score it on the same-topic pair benchmark, upload the results.
#
#   bash scripts/worker_job.sh <name> <seed> <features: ts0|ts20|ts46> [preset] [steps] [extra train_one args...]
#
# Run from virality/. The Hugging Face credential is injected by the session's network proxy; the
# placeholder below only satisfies the client library.
set -euo pipefail
NAME=$1; SEED=$2; FEATS=$3; PRESET=${4:-small}; STEPS=${5:-4000}; shift $(( $# < 5 ? $# : 5 ))
REPO=${XVIRAL_HF_REPO:-sam-siavoshian/xviral-work}
export HF_TOKEN=${HF_TOKEN:-hf_proxyinjected}
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip -q install -U pip
  .venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu      # the CPU build: these machines have no GPU
  .venv/bin/pip -q install -e . sentence-transformers peft jax huggingface_hub
fi
PY=.venv/bin/python
$PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("$REPO", repo_type="dataset", local_dir="data", allow_patterns=["work/model/table/*", "work/emb/*", "work/filtered/posts.parquet", "work/dataset/rows.parquet", "bench/*"])
PYEOF
OUT=data/runs/$NAME
$PY scripts/train_one.py --table data/work/model/table --out $OUT --seed $SEED --features $FEATS --preset $PRESET --steps $STEPS "$@" 2>&1 | tee $OUT.train.log | grep -E "val @|pairwise" || true
$PY scripts/evidence/tac_serve_eval.py --posts data/work/filtered/posts.parquet --checkpoint $OUT/best.pt --after 2026-04-23 \
   --trained-accounts data/work/dataset/rows.parquet --max-pairs 700 --max-calibration-pairs 300 --history 56 \
   --sid-codebook data/work/emb/sid_codebook.npz --feature-cache data/bench/tac_cache.pkl --out $OUT/tac.json > $OUT.tac.log 2>&1
$PY - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="$OUT", path_in_repo="runs/$NAME", repo_id="$REPO", repo_type="dataset", commit_message="run $NAME",
                      allow_patterns=["best.pt", "metrics.json", "tac.json", "tac.parquet"])
print("uploaded runs/$NAME")
PYEOF
