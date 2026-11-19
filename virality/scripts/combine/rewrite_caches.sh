#!/bin/bash
# The serving-path feature caches of the four pair sets, rewritten with the OpenAI embedding and its semantic IDs
# (research/tracks/embed_api/bench_api.py; texts already in the store cost nothing).
#   bash scripts/combine/rewrite_caches.sh <absolute cap on the store's ledger in dollars>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=data/combine/caches/te3l; mkdir -p $O; C=""; W=""
add() { [ -s $2 ] || { C="$C --cache $1"; W="$W --out $2"; }; }
for i in 0 1 2 3; do
  add data/tracks/pairs/bench/cache_$i.pkl $O/bench_$i.pkl
  add data/tracks/selective/caches/dev_cache_$i.pkl $O/dev_$i.pkl
  add data/tracks/selective/caches/launch_pre_cache_$i.pkl $O/launch_pre_$i.pkl
done
add data/tracks/replicate/launch_scores/feature_cache.pkl $O/launch_test_0.pkl
[ -z "$C" ] || .venv/bin/python research/tracks/embed_api/bench_api.py --max-dollars ${1:-19} --table data/combine/serving_te3l --store te3l=data/tracks/embed_api/store/te3large_3072 $C $W
echo REWRITE_DONE
