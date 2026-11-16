#!/bin/bash
# Score one checkpoint on the fixed 3,530 benchmark pairs using the per-slice feature caches of the baseline run.
#   bash scripts/bench_pairs.sh <name> <checkpoint> [parallel slices, default 4]
# Needs data/pairs/benchmark_pairs.parquet and data/bench3/cache_{0..3}.pkl (tracks/pairs/bench/ in the shared dataset).
set -uo pipefail
NAME=$1; CK=$2; PAR=${3:-4}
OUT=data/bench3/$NAME; mkdir -p $OUT logs
run() {
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/evidence/tac_serve_eval.py --posts data/work/filtered/posts.parquet --checkpoint $CK --after 2026-04-23 \
    --pairs data/pairs/benchmark_pairs.parquet --history 56 --shard $1/4 --sid-codebook data/work/emb/sid_codebook.npz \
    --feature-cache data/bench3/cache_$1.pkl --out $OUT/shard_$1.json > logs/bench_${NAME}_$1.log 2>&1
}
for i in 0 1 2 3; do
  run $i &
  if (( (i + 1) % PAR == 0 )); then wait; fi
done
wait
ls $OUT/shard_*.parquet | wc -l
