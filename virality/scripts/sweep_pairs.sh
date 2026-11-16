#!/bin/bash
# Fine-tune variants one after another, each followed by the benchmark.   bash scripts/sweep_pairs.sh <pair table> <threads> name:"train_pairs args" ...
PT=$1; TH=$2; shift 2
for spec in "$@"; do
  NAME=${spec%%:*}; ARGS=${spec#*:}
  OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH .venv/bin/python scripts/train_pairs.py --table data/work/model/table --pair-table $PT --pair-info data/pairs/train_pairs.parquet \
     --out data/runs/$NAME --features ts46 $ARGS > logs/$NAME.log 2>&1
  bash scripts/bench_pairs.sh $NAME data/runs/$NAME/best.pt 2 > logs/bench_$NAME.log 2>&1
  echo "SWEEP_DONE $NAME $(date)"
done
