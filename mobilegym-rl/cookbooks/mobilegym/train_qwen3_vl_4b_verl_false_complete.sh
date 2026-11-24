#!/usr/bin/env bash
set -euo pipefail

# False-complete reward control experiment.
# This reuses the regular GRPO baseline and changes only the evaluator mode:
# a trajectory that declares COMPLETE without full success gets reward=0.
SCRIPT_DIR="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"

SAMPLE_N="${SAMPLE_N:-1}"
SAMPLE_TEMPLATES="${SAMPLE_TEMPLATES:-true}"
TASK_SEED="${TASK_SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
ROLLOUT_N="${ROLLOUT_N:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
ENV_POOL_SIZE="${ENV_POOL_SIZE:-$((GEN_BATCH_SIZE * ROLLOUT_N))}"
ENV_NUM_BROWSERS="${ENV_NUM_BROWSERS:-8}"
FALSE_COMPLETE_MODE="${FALSE_COMPLETE_MODE:-zero}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilegym-qwen3-vl-4b-train-FALSE-COMPLETE-ZERO}"

export MOBILEGYM_FALSE_COMPLETE_MODE="$FALSE_COMPLETE_MODE"
export DYNAMIC_SAMPLING=false
export REJECTION_ENABLE=false
export REJECTION_MULTIPLIER=1
export SAMPLE_N SAMPLE_TEMPLATES TASK_SEED
export TRAIN_BATCH_SIZE ROLLOUT_N GEN_BATCH_SIZE ENV_POOL_SIZE ENV_NUM_BROWSERS
export EXPERIMENT_NAME

exec bash "$SCRIPT_DIR/train_qwen3_vl_4b_verl.sh" "$@"
