#!/usr/bin/env bash
# Full-scale training on one GPU machine (48 GB or 80 GB card), from an empty directory to an
# evaluated ensemble. Run from virality/. Every step is resumable; rerun the script after a stop.
#
#   CA_ANON_KEY=<key from community-archive docs/api-doc.md> scripts/gpu_run.sh
#
# Environment:
#   CFG          config file (default configs/gpu.yaml)
#   PRESET       trunk size: base | large (default large)
#   LIVE_MODEL   stage-B encoder (default Qwen/Qwen3-VL-Embedding-8B)
#   SEEDS        seeds for the ensemble (default "0 1 2")
#   SKIP_MEDIA   set to 1 to skip image and video fetching (text-only run)
set -euo pipefail

CFG=${CFG:-configs/gpu.yaml}
PRESET=${PRESET:-large}
LIVE_MODEL=${LIVE_MODEL:-Qwen/Qwen3-VL-Embedding-8B}
SEEDS=${SEEDS:-"0 1 2"}
X=".venv/bin/xviral"
WORK=$(.venv/bin/python -c "from xviral.config import Config; print(Config.load('$CFG').paths.work)" 2>/dev/null || echo data/work)

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -e ".[encoders,codebook,ann,dev]" peft
fi
step() { echo; echo "=== $* ==="; }

step "1. Community Archive: donor accounts, then every post since 2023"
$X ca-download --config "$CFG" --mode donors
$X ca-download --config "$CFG" --mode full --shards 8

step "2. Phase 0: unify, filter to the audience, author history, labels, baseline"
$X phase0 --config "$CFG"

if [ "${SKIP_MEDIA:-0}" != "1" ]; then
  step "3. Media: images, video frames, speech"
  $X media-select --config "$CFG"
  $X fetch-images --config "$CFG"
  $X fetch-videos --config "$CFG"
  $X asr --config "$CFG"
fi

step "4. X's post embedding, semantic IDs, image vectors, neighbours"
$X embed-mm --config "$CFG"
$X sid-train --config "$CFG"
$X sid-assign --config "$CFG"
[ "${SKIP_MEDIA:-0}" = "1" ] || $X embed-siglip --config "$CFG"
$X neighbors --config "$CFG"
$X model-prepare --config "$CFG"

step "5. Stage A: frozen embeddings, one model per seed"
for s in $SEEDS; do
  $X model-train --config "$CFG" --preset "$PRESET" --device cuda --steps 60000 --batch-size 256 --seed "$s" --run "a_seed$s"
  $X model-eval --config "$CFG" --device cuda --run "a_seed$s"
done

step "6. Stage B: the encoder with LoRA adapters, trained through the model's loss"
for s in $SEEDS; do
  $X model-train --config "$CFG" --preset "$PRESET" --device cuda --steps 20000 --batch-size 32 --seed "$s" \
     --live-model "$LIVE_MODEL" --init "$WORK/model/a_seed$s/best.pt" --run "b_seed$s"
  $X model-eval --config "$CFG" --device cuda --run "b_seed$s"
done

step "7. Comparison with ablations and baselines"
.venv/bin/python scripts/compare_models.py --table "$WORK/model/table" --out "$WORK/model/compare" --preset "$PRESET" \
  --steps 30000 --batch-size 256 --device cuda

echo; echo "Done. Per-run metrics: $WORK/model/*/eval.json   Comparison: $WORK/model/compare/results.md"
