#!/bin/bash
# The general track's runs, one at a time (two heavy jobs together run at a fifth of the speed on these machines).
cd "$(dirname "$0")/../.."; PY=.venv/bin/python; R=data/general/runs; mkdir -p $R data/logs
until grep -q PRED_DONE data/logs/pred_base.log; do sleep 20; done
run() { name=$1; shift; [ -f $R/$name/DONE ] || { $PY scripts/general/train_general.py --out $R/$name "$@" > data/logs/train_$name.log 2>&1 && touch $R/$name/DONE; }; }
for S in 1 2; do run broad_s$S --seed $S --roles orig,expand,train --steps 6000; done
for S in 1 2; do run broadft_s$S --seed $S --roles orig --steps 1200 --lr 2e-4 --warmup 100 --evals 6 --init $R/broad_s$S/best.pt --select val_aud; done
run ctlg_s1 --seed 1 --roles orig --steps 4000 --drop-cat src_name,src_regime --select val_aud
run nosrc_s1 --seed 1 --roles orig,expand,train --steps 6000 --drop-cat src_name,src_regime
echo QUEUE_DONE
