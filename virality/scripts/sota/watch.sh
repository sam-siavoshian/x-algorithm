#!/bin/bash
# Scores each candidate on the test sets once both of its seeds have finished training.
for c in soft ridge histdrop mask snap listnet; do
  until [ -s data/sota/runs/${c}_s1/metrics.json ] && [ -s data/sota/runs/${c}_s2/metrics.json ]; do sleep 30; [ -e data/sota/STOP_$c ] && continue 2; done
  bash scripts/sota/eval_all.sh ${c}_s1 ${c}_s2 > logs/eval_$c.log 2>&1
  echo "$(date) evaluated $c"
done
echo WATCH_DONE
