#!/bin/bash
# score.sh <run name> <table variant>: train/val/test metrics, the 3,530 benchmark pairs (4 slices) and the launch posts, with the run's own embedding
N=$1; V=$2; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CK=data/api/runs/$N/best.pt; T=data/api/table_$V; B=data/api/bench/$V; O=data/api/bench_out/$N; mkdir -p $O data/api/launch
CB=$T/sid_codebook.npz; [ $V = bge ] && { T=data/work/model/table; CB=data/work/emb/sid_codebook.npz; }
[ -s data/api/runs/$N/splits.json ] || .venv/bin/python research/tracks/embed_api/eval_splits.py --table $T --run data/api/runs/$N > data/api/logs/splits_$N.log 2>&1 &
for i in 0 1 2 3; do
  C=$B/cache_$i.pkl; [ $V = bge ] && C=data/tracks/pairs/bench/cache_$i.pkl
  [ -s $O/shard_$i.parquet ] || .venv/bin/python scripts/evidence/tac_serve_eval.py --posts data/work/filtered/posts.parquet --checkpoint $CK --after 2026-04-23 \
    --pairs data/tracks/pairs/benchmark_pairs.parquet --history 56 --shard $i/4 --sid-codebook $CB --feature-cache $C --out $O/shard_$i.json > data/api/logs/bench_${N}_$i.log 2>&1 &
  [ $i = 1 ] && wait
done
wait
LC=$B/launch_cache.pkl; [ $V = bge ] && LC=data/tracks/replicate/launch_scores/feature_cache.pkl
.venv/bin/python research/tracks/embed_api/score_launch_api.py --cache $LC --sid-codebook $CB --out data/api/launch $N=$CK > data/api/logs/launch_$N.log 2>&1
echo "SCORED $N $(ls $O/shard_*.parquet | wc -l) shards $(date +%H:%M)"; cat data/api/logs/splits_$N.log | tail -1
