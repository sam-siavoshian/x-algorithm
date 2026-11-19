#!/bin/bash
# rebench.sh <table name>: rewrite the four benchmark caches and the launch cache for data/api/table_<name>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
V=$1; B=data/api/bench/$V
C=""; O=""
for i in 0 1 2 3; do [ -s $B/cache_$i.pkl ] || { C="$C --cache data/tracks/pairs/bench/cache_$i.pkl"; O="$O --out $B/cache_$i.pkl"; }; done
[ -s $B/launch_cache.pkl ] || { C="$C --cache data/tracks/replicate/launch_scores/feature_cache.pkl"; O="$O --out $B/launch_cache.pkl"; }
[ -z "$C" ] || .venv/bin/python research/tracks/embed_api/bench_api.py --max-dollars 12 --table data/api/table_$V --store te3l=data/api/store/te3large_3072 --store te3s=data/api/store/te3small_1536 $C $O
echo REBENCH_DONE $V
