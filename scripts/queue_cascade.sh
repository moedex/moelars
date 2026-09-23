#!/bin/zsh
# Serial GPU queue for the cascade simulation: both suites again with per-row dumps, then
# evals/cascade.py with the adapted 4B as primary and the 30B-A3B as fallback.
# Each step logs to logs/<step>.log; a failing step stops the queue.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-~/Library/Python/3.9/bin/uv}
mkdir -p logs
M4=checkpoints/lora-4b-fused-bf16
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
step() { echo "=== $(date '+%H:%M:%S') $1"; }

step "rows-4b-lora: fused bf16 adapter, per-row dumps"
$UV run python evals/run_suite.py --backend mlx --model $M4 --tag rows --dump-rows > logs/rows-4b-lora.log 2>&1
step "rows-30b: per-row dumps"
$UV run python evals/run_suite.py --backend mlx --model $M30 --tag rows --dump-rows > logs/rows-30b.log 2>&1
step "cascade"
$UV run python evals/cascade.py evals/results/rows/lora-4b-fused-bf16-rows \
  evals/results/rows/qwen3-30b-a3b-instruct-2507-4bit-rows > logs/cascade.log 2>&1
step "ALL DONE"
