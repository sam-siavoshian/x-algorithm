#!/bin/bash
# after the running trainings end: train arm C while scoring arm D, then arm S while scoring C, then score S
D=data/api
tr() { for s in 1 2; do OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/train_one.py --table $D/table_$1 --out $D/runs/$1_s$s --seed $s --features ts46 --preset small --steps 4000 > $D/runs/$1_s$s.log 2>&1 & done; }
sc() { $D/score.sh $1_s1 $1; $D/score.sh $1_s2 $1; }
while pgrep -f train_one > /dev/null; do sleep 15; done
tr te3l1024_ad; sc bge_ad; wait
tr te3s1024; sc te3l1024_ad; wait
sc te3s1024
echo CHAIN_DONE $(date +%H:%M)
