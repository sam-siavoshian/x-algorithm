#!/bin/bash
# One seed's queue of the sota track: the control and every candidate under identical conditions, one after another.
#   bash scripts/sota/queue.sh <seed> [threads]
SEED=$1; export OMP_NUM_THREADS=${2:-2} MKL_NUM_THREADS=${2:-2}
run() { N=$1; shift; [ -s data/sota/runs/${N}_s$SEED/metrics.json ] && return
  .venv/bin/python scripts/sota/train_sota.py --out data/sota/runs/${N}_s$SEED --seed $SEED --curve-rows 6000 "$@" > logs/sota_${N}_s$SEED.log 2>&1; }
run ctl
run thurstone --set pair_loss=thurstone
run soft --set pair_loss=soft
run ridge --mset use_hist_ridge=true
run histdrop --set hist_outcome_drop=0.15 --set hist_drop_all=0.1
run mask --set mask_hist_p=0.15
run snap --set lr_cycles=3
run listnet --set pair_loss=listnet
echo QUEUE_DONE $SEED
