#!/usr/bin/env bash
set -euo pipefail

# Terminal-progress GRPO baseline with a larger PPO mini-batch.
# This keeps the regular baseline settings and only changes the mini-batch size.
SCRIPT_DIR="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-96}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilegym-qwen3-vl-4b-train-GRPO-MINIBSZ96}"

export PPO_MINI_BATCH_SIZE EXPERIMENT_NAME

exec bash "$SCRIPT_DIR/train_qwen3_vl_4b_verl.sh" "$@"
