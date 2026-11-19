#!/bin/bash
# Score checkpoints once on the two test sets: the 3,530 benchmark pairs (serving path, per-slice caches) and the launch posts.
#   bash scripts/sota/eval_ckpt.sh name=ckpt.pt [name=ckpt.pt ...]
mkdir -p data/sota/launch
[ -s data/sota/launch/feature_cache.pkl ] || cp data/tracks/replicate/launch_scores/feature_cache.pkl data/sota/launch/
for spec in "$@"; do
  N=${spec%%=*}; CK=${spec#*=}
  [ "$(ls data/bench3/$N/shard_*.parquet 2>/dev/null | wc -l)" = 4 ] || bash scripts/bench_pairs.sh $N $CK 4
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/rep_score_launch.py --out data/sota/launch "$@" > logs/sota_launch.log 2>&1
echo EVAL_DONE
