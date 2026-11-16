#!/bin/bash
# Benchmark (3,530 pairs, cached features) and launch posts for one checkpoint:  after_train.sh <name> <checkpoint>
set -uo pipefail
NAME=$1; CK=$2
bash scripts/bench_pairs.sh $NAME $CK 4
.venv/bin/python scripts/expand/score_launch.py --checkpoint $CK --out data/expand/launch/$NAME.parquet --cache data/expand/launch/cache.pkl
