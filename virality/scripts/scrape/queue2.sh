#!/bin/bash
# After queue.sh: one more variant, two seeds.
#   h12src   existing rows + every scrape post captured 12 hours or more after posting (younger posts stay as history only), with the sampling input
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q POST_DONE data/scrape_track/post.log; do sleep 30; done
PY=.venv/bin/python; T=data/scrape_track/table; R=data/scrape_track/runs; C="--features ts46 --preset small --steps 4000 --eval-rows data/scrape_track/orig_test_ids.npy"
for S in 1 2; do
  NAME=h12src_s$S
  [ -s $R/$NAME/metrics.json ] || { echo "$(date +%T) train $NAME"; $PY scripts/train_one.py --table $T --out $R/$NAME --seed $S $C --train-exclude data/scrape_track/under12h_ids.npy > $R/$NAME.train.log 2>&1; }
  [ -s data/bench3/$NAME/shard_3.parquet ] || bash scripts/bench_pairs.sh $NAME $R/$NAME/best.pt 4 > /dev/null 2>&1
  [ -s data/scrape_track/launch/$NAME.parquet ] || $PY scripts/expand/score_launch.py --checkpoint $R/$NAME/best.pt --out data/scrape_track/launch/$NAME.parquet --cache data/scrape_track/launch/cache.pkl > $R/$NAME.launch.log 2>&1
  echo "$(date +%T) done $NAME"
done
echo QUEUE2_DONE
