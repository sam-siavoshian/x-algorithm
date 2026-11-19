#!/bin/bash
# Step 2: the replicate track's pair-loss fine-tune recipe on each seed of the chosen pair, weight 10 (treatment) next to
# weight 0 (control), same steps, learning rate, pair table and seed.   bash scripts/combine/finetune.sh <threads> name=ckpt [name=ckpt ...]
set -uo pipefail
TH=$1; shift
for spec in "$@"; do
  N=${spec%%=*}; CK=${spec#*=}
  for W in 10 0; do
    O=data/combine/runs/ft_w${W}_$N
    [ -s $O/best.pt ] && continue
    OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH .venv/bin/python scripts/train_pairs.py --table data/combine/table_te3l_broad --pair-table data/combine/pair_table_te3l \
      --pair-info data/tracks/pairs/train_pairs.parquet --out $O --features ts46 --init $CK --steps 600 --lr 2e-4 --warmup 50 --seed 0 --w-ctl $W > data/combine/logs/ft_w${W}_$N.log 2>&1
    echo "FT_DONE w$W $N $(date +%H:%M) $(tail -1 data/combine/logs/ft_w${W}_$N.log | cut -c1-200)"
  done
done
echo FINETUNE_ALL_DONE
