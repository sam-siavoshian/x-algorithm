#!/bin/bash
# One slice of the general pair benchmark for any number of checkpoints, with the flags of scripts/worker_bench.sh
# (2,100 test pairs, 1,500 calibration pairs, 56 posts of history, 6 slices), single-threaded so slices run side by side.
#   bash scripts/posttype/bench_shard.sh <slice 0..5> <name=checkpoint> [<name=checkpoint> ...]
# POSTS=<parquet> may point at the posts of the accounts that have pairs (same pairs, far less memory).
# The slice's model-independent feature cache is shared by every model, so only the first model pays for features.
set -uo pipefail
I=$1; shift
PY=.venv/bin/python; OUT=data/bench2; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 TOKENIZERS_PARALLELISM=false
[ -s $OUT/cache_$I.pkl ] || { [ -s data/bench/tac_cache.pkl ] && cp data/bench/tac_cache.pkl $OUT/cache_$I.pkl; }
for SPEC in "$@"; do
  M=${SPEC%%=*}; CK=${SPEC#*=}; mkdir -p $OUT/$M
  [ -s $OUT/$M/shard_$I.parquet ] && [ "${FORCE:-0}" = 0 ] && continue
  $PY scripts/evidence/tac_serve_eval.py --posts ${POSTS:-data/work/filtered/posts.parquet} --checkpoint $CK --after 2026-04-23 \
     --trained-accounts data/work/dataset/rows.parquet --max-pairs 2100 --max-calibration-pairs 1500 --history 56 --shard $I/6 \
     --sid-codebook data/work/emb/sid_codebook.npz --feature-cache $OUT/cache_$I.pkl --emb-table data/work/model/table \
     --out $OUT/$M/shard_$I.json > $OUT/$M.shard_$I.log 2>&1 || { echo "FAILED $M shard $I"; tail -3 $OUT/$M.shard_$I.log; }
done
echo BENCH_SHARD_DONE $I
