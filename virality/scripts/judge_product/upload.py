#!/usr/bin/env python3
"""Collect the track's deliverables and upload them to the dataset under tracks/judge_product/.

    HF_TOKEN=hf_proxyinjected python scripts/judge_product/upload.py [--dry]

patch.diff is `git diff` against the branch head with every new file included (nothing is committed)."""
import argparse, glob, os, shutil, subprocess

U = "data/upload/judge_product"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    os.makedirs(U + "/out", exist_ok=True)
    subprocess.run("git add -N . && git diff > " + U + "/patch.diff && git reset -q", shell=True, check=True)
    shutil.copy("research/tracks/judge_product/REPORT.md", U)
    shutil.copy("xviral/data/judge_rule.json", U)
    for f in ("jp_fit.md", "jp_fit.json", "jp_test.md", "jp_test.json", "jp_extras.md", "jp_extras.json", "judge_rule_fitted_frozen_luna_sol.json", "judge_rule_fitted_frozen_with_open_models.json",
              "batches.json", "spend.jsonl"):
        if os.path.exists("data/llm/" + f):
            shutil.copy("data/llm/" + f, U)
    for f in glob.glob("data/llm/jp_pairs_*.parquet") + glob.glob("data/llm/jp_textsim_*.parquet") + glob.glob("data/llm/sample_*.json"):
        shutil.copy(f, U)
    for f in glob.glob("data/llm/out/*.jsonl"):          # the request logs this track wrote or added to
        old = "data/tracks/llm_judge/out/" + os.path.basename(f)
        if not os.path.exists(old) or os.path.getsize(old) != os.path.getsize(f):
            shutil.copy(f, U + "/out")
    for d in ("e2e", "e2e_luna"):
        if os.path.isdir("data/llm/" + d):
            shutil.copytree("data/llm/" + d, f"{U}/{d}", dirs_exist_ok=True)
    if os.path.exists("data/llm/page.png"):
        shutil.copy("data/llm/page.png", U)
    n = sum(len(fs) for _, _, fs in os.walk(U)); size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(U) for f in fs)
    print(f"{n} files, {size / 1e6:.1f} MB in {U}")
    if a.dry:
        return
    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=U, path_in_repo="tracks/judge_product", repo_id="sam-siavoshian/xviral-work", repo_type="dataset", commit_message="judge_product: report, rule, code, results")
    print("uploaded to tracks/judge_product/")


if __name__ == "__main__":
    main()
