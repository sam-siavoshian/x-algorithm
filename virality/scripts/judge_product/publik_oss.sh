#!/bin/bash
# gpt-oss-120b (weights published 2025-08-04) on 150 + 150 development validation pairs, slowly, beside the other run.
R=".venv/bin/python scripts/judge_product/run.py"; M=openai/gpt-oss-120b; RPM=${1:-8}
$R sync --set dev --config n30 --keys data/llm/sample_dev_val150.json --model $M --concurrency 12 --rpm $RPM | tail -1
$R sync --set launchpre --config n30 --keys data/llm/sample_launchpre_val150.json --model $M --concurrency 12 --rpm $RPM | tail -1
