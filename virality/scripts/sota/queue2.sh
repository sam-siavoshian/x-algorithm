#!/bin/bash
# Second part of a seed's queue (after the time budget was cut): waits for the running ridge run, then history dropout and masked history.
SEED=$1; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
while pgrep -f "runs/ridge_s$SEED " > /dev/null; do sleep 10; done
run() { N=$1; shift; [ -s data/sota/runs/${N}_s$SEED/metrics.json ] && return
  .venv/bin/python scripts/sota/train_sota.py --out data/sota/runs/${N}_s$SEED --seed $SEED --curve-rows 6000 "$@" > logs/sota_${N}_s$SEED.log 2>&1; }
run histdrop --set hist_outcome_drop=0.15 --set hist_drop_all=0.1
run mask --set mask_hist_p=0.15
echo QUEUE2_DONE $SEED
