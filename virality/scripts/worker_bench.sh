#!/bin/bash
# One slice of the same-topic pair benchmark on a fresh machine: fetch the shared data and every trained
# checkpoint from the private Hugging Face dataset, score each model on this machine's slice of the pairs,
# upload one small table per model. All machines score the same pairs, so results can be joined and
# member scores averaged into ensembles afterwards (scripts/evidence/compare_runs.py).
#
#   bash scripts/worker_bench.sh <slice index> <slices> [max test pairs] [max calibration pairs]
#
# Run from virality/. The Hugging Face credential is injected by the session's network proxy.
set -euo pipefail
I=$1; N=$2; MAXT=${3:-2100}; MAXC=${4:-1500}
REPO=${XVIRAL_HF_REPO:-sam-siavoshian/xviral-work}
export HF_TOKEN=${HF_TOKEN:-hf_proxyinjected}
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip -q install -U pip
  .venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip -q install -e . sentence-transformers peft jax huggingface_hub
fi
PY=.venv/bin/python
$PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("$REPO", repo_type="dataset", local_dir="data", allow_patterns=["work/emb/*", "work/filtered/posts.parquet", "work/dataset/rows.parquet", "work/model/compare*/*/best.pt", "runs/*/best.pt"])
PYEOF
OUT=data/bench2; mkdir -p $OUT
for CK in data/runs/*/best.pt data/work/model/compare*/*/best.pt; do
  M=$(echo "$CK" | sed -e 's#data/runs/##' -e 's#data/work/model/##' -e 's#/best.pt##' -e 's#/#_#g')
  mkdir -p $OUT/$M
  $PY scripts/evidence/tac_serve_eval.py --posts data/work/filtered/posts.parquet --checkpoint $CK --after 2026-04-23 \
     --trained-accounts data/work/dataset/rows.parquet --max-pairs $MAXT --max-calibration-pairs $MAXC --history 56 --shard $I/$N \
     --sid-codebook data/work/emb/sid_codebook.npz --feature-cache $OUT/cache_$I.pkl --out $OUT/$M/shard_$I.json > $OUT/$M.shard_$I.log 2>&1 \
     || { echo "FAILED $M"; tail -5 $OUT/$M.shard_$I.log; continue; }
  $PY - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="$OUT/$M/shard_$I.parquet", path_in_repo="bench2/$M/shard_$I.parquet", repo_id="$REPO", repo_type="dataset", commit_message="bench2 $M shard $I")
print("uploaded bench2/$M/shard_$I")
PYEOF
done
echo BENCH_SHARD_DONE $I
