#!/bin/bash
# After the six runs: table slices for every model, the sampling-input probe, the image analysis.
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q QUEUE_DONE data/scrape_track/queue.log; do sleep 20; done
PY=.venv/bin/python; R=data/scrape_track/runs
echo "$(date +%T) eval"
RUNS=""; for n in ctl_s1 ctl_s2 hist_s1 hist_s2 histsrc_s1 histsrc_s2 allsrc_s1 allsrc_s2; do RUNS="$RUNS --run $n=$R/$n/best.pt"; done
$PY scripts/scrape/eval_table.py $RUNS --out data/scrape_track/eval/slices.json > data/scrape_track/eval/eval.log 2>&1
echo "$(date +%T) probe"
$PY scripts/scrape/regime_probe.py --run histsrc_s1=$R/histsrc_s1/best.pt --run histsrc_s2=$R/histsrc_s2/best.pt --run allsrc_s1=$R/allsrc_s1/best.pt --run allsrc_s2=$R/allsrc_s2/best.pt --out data/scrape_track/eval/regime_probe.json > data/scrape_track/eval/probe.log 2>&1
echo "$(date +%T) images"
$PY scripts/scrape/images.py > data/scrape_track/images.log 2>&1
echo "$(date +%T) POST_DONE"
