#!/bin/bash
# Selective track: rescore every set with the ten untouched and ten fine-tuned checkpoints (full member outputs).
#   bash scripts/selective/sel_run.sh bench|launch_test|dev|launch_pre
set -uo pipefail
R=data/tracks/replicate/runs; P=data/tracks/pairs/runs
M="start_ts46_s1=data/runs/ts46_s1/best.pt start_ts46_s2=data/runs/ts46_s2/best.pt start_compare_v2_full=data/work/model/compare_v2/full/best.pt start_ts0_s0=data/runs/ts0_s0/best.pt start_ts0_s1=data/runs/ts0_s1/best.pt start_ts20_s1=data/runs/ts20_s1/best.pt start_ts20_s2=data/runs/ts20_s2/best.pt start_cmp_full=data/work/model/compare/full/best.pt start_cmp_aud=data/work/model/compare/audience_only_training/best.pt start_cmpv2_aud=data/work/model/compare_v2/audience_only_training/best.pt"
M="$M pairs_ft_ts46_s1=$P/ft_w10/best.pt pairs_ft_ts46_s2=$P/ft_w10_s2/best.pt pairs_ft_compare_v2_full=$P/ft_w10_v2full/best.pt"
for m in ts0_s0 ts0_s1 ts20_s1 ts20_s2 cmp_full cmp_aud cmpv2_aud; do M="$M ft_$m=$R/rep_w10_$m/best.pt"; done
SET=$1; mkdir -p data/sel/$SET logs
case $SET in
  bench) for i in 0 1 2 3; do OMP_NUM_THREADS=1 .venv/bin/python scripts/selective/sel_score.py --units data/sel/units_bench.parquet --out data/sel/bench/shard_$i --shard $i/4 --feature-cache data/bench3/cache_$i.pkl $M > logs/sel_bench_$i.log 2>&1 & done; wait;;
  launch_test) OMP_NUM_THREADS=2 .venv/bin/python scripts/selective/sel_score.py --units data/sel/units_launch_test.parquet --out data/sel/launch_test/shard_0 --feature-cache data/tracks/replicate/launch_scores/feature_cache.pkl $M > logs/sel_launch_test.log 2>&1;;
  dev|launch_pre) for i in 0 1 2 3; do OMP_NUM_THREADS=1 .venv/bin/python scripts/selective/sel_score.py --units data/sel/units_$SET.parquet --out data/sel/$SET/shard_$i --shard $i/4 --feature-cache data/sel/$SET/cache_$i.pkl --fast \
      --emb-seed data/sel/emb_seed_$SET.pkl $M > logs/sel_${SET}_$i.log 2>&1 & done; wait;;
esac
echo SEL_DONE $SET
