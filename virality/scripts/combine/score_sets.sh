#!/bin/bash
# Serving-path scores of checkpoints on the four pair sets (scripts/selective/sel_score.py), with the feature caches of
# their own encoder:   bash scripts/combine/score_sets.sh <te3l|bge> <tag> "<sets>" name=ckpt [name=ckpt ...]
#   sets: any of dev launch_pre bench launch_test (the test sets are scored only when named)
set -uo pipefail
K=$1; TAG=$2; SETS=$3; shift 3
U=data/tracks/selective; O=data/combine/sel/$K; mkdir -p data/combine/logs data/combine/tmp
if [ $K = te3l ]; then CB=data/combine/serving_te3l/sid_codebook.npz; else CB=data/work/emb/sid_codebook.npz; fi
cache() {  # set shard
  if [ $K = te3l ]; then echo data/combine/caches/te3l/$1_$2.pkl; else
    case $1 in bench) echo data/tracks/pairs/bench/cache_$2.pkl;; launch_test) echo data/tracks/replicate/launch_scores/feature_cache.pkl;; *) echo data/tracks/selective/caches/$1_cache_$2.pkl;; esac; fi; }
for S in $SETS; do
  mkdir -p $O/$S
  if [ $S = launch_test ]; then N=1; else N=4; fi
  for i in $(seq 0 $((N-1))); do
    [ -s $O/$S/shard_${i}__$TAG.npz ] && continue
    C=$(cache $S $i); [ -s $C ] || { echo "missing cache $C"; exit 1; }
    cp $C data/combine/tmp/sel_cache_$$_$i.pkl      # sel_score writes a cache back when it grew; a copy keeps the shared one untouched
    ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/selective/sel_score.py --units $U/units_$S.parquet --out $O/$S/shard_${i}__$TAG --shard $i/$N \
        --feature-cache data/combine/tmp/sel_cache_$$_$i.pkl --sid-codebook $CB "$@" > data/combine/logs/sel_${K}_${S}_${i}_$TAG.log 2>&1; rm -f data/combine/tmp/sel_cache_$$_$i.pkl ) &
    [ $(( (i + 1) % ${PAR:-4} )) = 0 ] && wait      # PAR: shards scored at once (each holds its cache and the accounts' posts in memory)
  done
  wait
  grep -l "encoder loaded (cache miss)\|skipped:" data/combine/logs/sel_${K}_${S}_*_$TAG.log 2>/dev/null | sed 's/^/WARNING cache miss or skipped unit in /'
  echo "SCORED $K $S $TAG $(ls $O/$S/shard_*__$TAG.npz 2>/dev/null | wc -l) shards $(date +%H:%M)"
done
