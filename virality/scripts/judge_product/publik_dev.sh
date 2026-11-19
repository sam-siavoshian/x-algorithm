#!/bin/bash
# Development validation samples (pairs the stronger OpenAI model also scored: 300 same-topic, 300 launch, later period and
# held-out accounts) through the Publik models, one after another: the key allows 60 requests a minute in total.
R=".venv/bin/python scripts/judge_product/run.py"
for M in deepseek-ai/DeepSeek-V4.1-Flash Qwen/Qwen3.5-397B-A17B; do
  $R sync --set dev --config n30 --keys data/llm/sample_dev_val_sol.json --model $M --concurrency 16 --rpm 55 | tail -1
  $R sync --set launchpre --config n30 --keys data/llm/sample_launchpre_val.json --model $M --concurrency 16 --rpm 55 | tail -1
done
M=openai/gpt-oss-120b
$R sync --set dev --config n30 --keys data/llm/sample_dev_val150.json --model $M --concurrency 60 --rpm 55 | tail -1
$R sync --set launchpre --config n30 --keys data/llm/sample_launchpre_val150.json --model $M --concurrency 60 --rpm 55 | tail -1
