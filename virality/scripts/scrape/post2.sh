#!/bin/bash
# After queue2.sh: the 12-hour variant on the table slices, merged into the existing result; then every table again.
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q QUEUE2_DONE data/scrape_track/queue2.log; do sleep 20; done
PY=.venv/bin/python; R=data/scrape_track/runs
$PY scripts/scrape/eval_table.py --run h12src_s1=$R/h12src_s1/best.pt --run h12src_s2=$R/h12src_s2/best.pt --out data/scrape_track/eval/slices.json --merge > data/scrape_track/eval/eval2.log 2>&1
$PY scripts/scrape/summarize.py --families ctl,hist,histsrc,allsrc,h12src --out data/scrape_track/eval/tables > /dev/null 2>&1
XVIRAL_TRACK_DIR=data/scrape_track $PY scripts/expand/summarize.py --families ctl,hist,histsrc,allsrc,h12src --labels "Control|+ accounts 20+ (no input)|+ accounts 20+, sampling input|+ all posts, sampling input|+ posts captured 12h+, sampling input" --slices data/scrape_track/eval/none.json --out data/scrape_track/eval/results > /dev/null 2>&1
$PY scripts/scrape/regime_probe.py --run h12src_s1=$R/h12src_s1/best.pt --run h12src_s2=$R/h12src_s2/best.pt --out data/scrape_track/eval/regime_probe_h12.json > /dev/null 2>&1
echo "$(date +%T) POST2_DONE"
