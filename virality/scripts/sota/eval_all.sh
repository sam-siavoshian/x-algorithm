#!/bin/bash
# Everything the report needs for the named runs' chosen checkpoints (or name=path specs).
SPECS=(); for x in "$@"; do case $x in *=*) SPECS+=("$x");; *) SPECS+=("$x=data/sota/runs/$x/best.pt");; esac; done
bash scripts/sota/eval_ckpt.sh "${SPECS[@]}"
OMP_NUM_THREADS=2 .venv/bin/python scripts/sota/dev_dist.py --out data/sota/dev "${SPECS[@]}" 2>&1 | grep -v Warn
OMP_NUM_THREADS=2 .venv/bin/python scripts/sota/test_rows.py --out data/sota/test "${SPECS[@]}" 2>&1 | grep -v Warn
