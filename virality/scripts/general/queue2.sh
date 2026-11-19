#!/bin/bash
# Remaining runs after broad_s1: fine-tunes, predictions, benchmark and launch scoring, then the control under this trainer.
cd "$(dirname "$0")/../.."; PY=.venv/bin/python; R=data/general/runs
until grep -q TRAIN_DONE data/logs/train_broad_s2.log; do sleep 20; done; touch $R/broad_s2/DONE
run() { name=$1; shift; [ -f $R/$name/DONE ] || { $PY scripts/general/train_general.py --out $R/$name "$@" > data/logs/train_$name.log 2>&1 && touch $R/$name/DONE; }; }
for S in 1 2; do run broadft_s$S --seed $S --roles orig --steps 1200 --lr 2e-4 --warmup 100 --evals 6 --init $R/broad_s$S/best.pt --select val_aud; done
$PY scripts/general/predict_rows.py --customer --run broad_s1=$R/broad_s1/best.pt --run broad_s2=$R/broad_s2/best.pt --run broadft_s1=$R/broadft_s1/best.pt --run broadft_s2=$R/broadft_s2/best.pt > data/logs/pred_new.log 2>&1
for n in broad_s1 broad_s2 broadft_s1 broadft_s2; do bash scripts/expand/after_train.sh g_$n $R/$n/best.pt > data/logs/after_$n.log 2>&1; done
echo BENCH_DONE
run ctlg_s1 --seed 1 --roles orig --steps 4000 --drop-cat src_name,src_regime --select val_aud
$PY scripts/general/predict_rows.py --run ctlg_s1=$R/ctlg_s1/best.pt > data/logs/pred_ctlg.log 2>&1
echo QUEUE_DONE
