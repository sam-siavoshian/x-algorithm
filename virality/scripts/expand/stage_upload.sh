#!/bin/bash
# Assemble what the track delivers (besides rows/ and table/, uploaded on their own) and upload it to tracks/expand/.
set -uo pipefail
U=data/expand/upload; rm -rf $U; mkdir -p $U/runs $U/bench $U/launch $U/eval $U/queue $U/pair_table_src
for d in data/expand/runs/*/; do n=$(basename $d); [ -s $d/best.pt ] && [ -s $d/metrics.json ] || continue; mkdir -p $U/runs/$n; cp $d/best.pt $d/metrics.json $U/runs/$n/; [ -s $d/history.json ] && cp $d/history.json $U/runs/$n/; done
for d in data/bench3/*/; do n=$(basename $d); ls $d/shard_*.parquet > /dev/null 2>&1 || continue; mkdir -p $U/bench/$n; cp $d/shard_*.parquet $U/bench/$n/; done
cp data/expand/launch/*.parquet $U/launch/ 2>/dev/null
cp data/expand/eval/results*.md data/expand/eval/results*.json data/expand/eval/table_slices.json data/expand/eval/table_slices.npz $U/eval/ 2>/dev/null
cp data/expand/queue/*.sh data/expand/queue/log.txt $U/queue/ 2>/dev/null
cp data/expand/pair_table_src/cat.npy data/expand/pair_table_src/meta.json $U/pair_table_src/ 2>/dev/null
cp data/expand/emb/check.txt $U/embedding_check.txt 2>/dev/null
cp data/expand/orig_test_ids.npy $U/ 2>/dev/null
[ -s research/tracks/expand/REPORT.md ] && cp research/tracks/expand/REPORT.md $U/
{ git diff; for f in $(git ls-files --others --exclude-standard scripts/expand research/tracks/expand | grep -v "_part_"); do git diff --no-index /dev/null $f; done; } > $U/patch.diff
du -sh $U | cut -f1
.venv/bin/python - <<'P'
import os
from huggingface_hub import HfApi
os.environ.setdefault("HF_TOKEN", "hf_proxyinjected")
HfApi().upload_folder(folder_path="data/expand/upload", path_in_repo="tracks/expand", delete_patterns=["eval/*"], repo_id="sam-siavoshian/xviral-work", repo_type="dataset", commit_message="tracks/expand: results")
print("uploaded tracks/expand")
P
