#!/bin/bash
# The six training runs of the scrape track, one at a time, each followed by the benchmark and the launch posts.
#   hist     existing rows + scrape accounts with 20 or more posts, no sampling input
#   histsrc  the same rows, with the sampling and capture-age input (src_regime)
#   allsrc   existing rows + every scrape post (accounts down to one post), with the input
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; T=data/scrape_track/table; R=data/scrape_track/runs; C="--features ts46 --preset small --steps 4000 --eval-rows data/scrape_track/orig_test_ids.npy"
run() {  # name seed extra...
  local NAME=$1 SEED=$2; shift 2
  if [ ! -s $R/$NAME/metrics.json ]; then
    echo "$(date +%T) train $NAME"; $PY scripts/train_one.py --table $T --out $R/$NAME --seed $SEED $C "$@" > $R/$NAME.train.log 2>&1
  fi
  if [ ! -s data/bench3/$NAME/shard_3.parquet ]; then echo "$(date +%T) bench $NAME"; bash scripts/bench_pairs.sh $NAME $R/$NAME/best.pt 4 > /dev/null 2>&1; fi
  if [ ! -s data/scrape_track/launch/$NAME.parquet ]; then
    echo "$(date +%T) launch $NAME"; $PY scripts/expand/score_launch.py --checkpoint $R/$NAME/best.pt --out data/scrape_track/launch/$NAME.parquet --cache data/scrape_track/launch/cache.pkl > $R/$NAME.launch.log 2>&1
  fi
  echo "$(date +%T) done $NAME"
}
mkdir -p $R
for S in 1 2; do
  run histsrc_s$S $S --train-exclude data/scrape_track/under20_ids.npy
  run allsrc_s$S $S
  run hist_s$S $S --train-exclude data/scrape_track/under20_ids.npy --drop-cat src_regime
done
echo QUEUE_DONE
