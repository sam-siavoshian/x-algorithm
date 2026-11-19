#!/bin/bash
# Qwen3.5-397B (weights published 2026-02-16, before every test pair) on the test sets: all 850 launch pairs and a
# random 700 of the 1,660 benchmark pairs with both posts on or after 2026-04-23. The key allows 60 requests a minute in total.
R=".venv/bin/python scripts/judge_product/run.py"; M=Qwen/Qwen3.5-397B-A17B; RPM=${1:-45}
for pass in 1 2; do          # the second pass repeats requests that failed on the rate limit
  $R sync --set launch --config n30 --model $M --concurrency 16 --timeout 40 --rpm $RPM | tail -1
  $R sync --set bench --config n30 --keys data/llm/sample_bench_post700.json --model $M --concurrency 16 --timeout 40 --rpm $RPM | tail -1
done
