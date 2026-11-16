#!/usr/bin/env python3
"""Upload files to the shared dataset under tracks/<track>/.   python scripts/upload_track.py <track> <local path>[=<name in repo>] ..."""
import os
import sys

from huggingface_hub import HfApi

os.environ.setdefault("HF_TOKEN", "hf_proxyinjected")
api, track = HfApi(), sys.argv[1]
repo = os.environ.get("XVIRAL_HF_REPO", "sam-siavoshian/xviral-work")
for spec in sys.argv[2:]:
    src, _, dst = spec.partition("=")
    dst = f"tracks/{track}/{dst or os.path.basename(src.rstrip('/'))}"
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=repo, repo_type="dataset", commit_message=f"tracks/{track}: {os.path.basename(dst)}")
    else:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="dataset", commit_message=f"tracks/{track}: {os.path.basename(dst)}")
    print("uploaded", dst)
