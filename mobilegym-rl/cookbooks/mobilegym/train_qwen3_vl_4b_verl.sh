#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"
RL_ROOT="$(realpath -- "$SCRIPT_DIR/../..")"
cd "$RL_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${#visible_gpus[@]}}"
# HF repo id by default (auto-downloaded); override with a local path:
#   MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash cookbooks/mobilegym/train_qwen3_vl_4b_verl.sh
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
SPLIT="${SPLIT:-train}"

# Optional interaction cap and terminal reward coefficients.
MAX_STEPS_CAP="${MAX_STEPS_CAP:-0}"
REWARD_CLEAN_PENALTY="${REWARD_CLEAN_PENALTY:-0.8}"
REWARD_ABORT_PENALTY="${REWARD_ABORT_PENALTY:-0.5}"
REWARD_OVERDUE_PENALTY="${REWARD_OVERDUE_PENALTY:-0.5}"
REWARD_PREMATURE_PENALTY="${REWARD_PREMATURE_PENALTY:-0.8}"
REWARD_FORMAT_ERROR="${REWARD_FORMAT_ERROR:-0.0}"
REWARD_REPETITIVE_LOOP="${REWARD_REPETITIVE_LOOP:-1.0}"
REWARD_MAX_STEPS="${REWARD_MAX_STEPS:-1.0}"
REWARD_INFRA_ERROR="${REWARD_INFRA_ERROR:-0.0}"
REWARD_REPEATED_ACTION="${REWARD_REPEATED_ACTION:-0.95}"
REWARD_NO_STATE_CHANGE="${REWARD_NO_STATE_CHANGE:-0.95}"
REWARD_MAX_BEHAVIOR_PENALTIES="${REWARD_MAX_BEHAVIOR_PENALTIES:-8}"

SAMPLE_N="${SAMPLE_N:-1}"
SAMPLE_TEMPLATES="${SAMPLE_TEMPLATES:-true}"
TASK_SEED="${TASK_SEED:-42}"
DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING:-false}"
DYNAMIC_TARGET_GROUPS="${DYNAMIC_TARGET_GROUPS:-12}"
DYNAMIC_MAX_ROUNDS="${DYNAMIC_MAX_ROUNDS:-2}"
DYNAMIC_MIN_REWARD_RANGE="${DYNAMIC_MIN_REWARD_RANGE:-1e-6}"
REJECTION_MULTIPLIER="${REJECTION_MULTIPLIER:-1}"
REJECTION_ENABLE="${REJECTION_ENABLE:-false}"
MIN_PARTIAL_SOLVE_TASKS="${MIN_PARTIAL_SOLVE_TASKS:-1}"
MIN_TRAJS_PER_GROUP="${MIN_TRAJS_PER_GROUP:-2}"
FILTER_UNIFORM_GROUPS="${FILTER_UNIFORM_GROUPS:-false}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
ROLLOUT_N="${ROLLOUT_N:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * REJECTION_MULTIPLIER))}"
ENV_POOL_SIZE="${ENV_POOL_SIZE:-$((GEN_BATCH_SIZE * ROLLOUT_N))}"
ENV_NUM_BROWSERS="${ENV_NUM_BROWSERS:-8}"
ENV_INIT_BATCH_SIZE="${ENV_INIT_BATCH_SIZE:-8}"
ENV_INIT_WARMUP="${ENV_INIT_WARMUP:-true}"
USE_KL_LOSS="${USE_KL_LOSS:-true}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-3}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
GATEWAY_DB_PATH="${GATEWAY_DB_PATH:-$RL_ROOT/.rllm_local/gateway_traces.db}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilegym-qwen3-vl-4b-train}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/mobilegym}"
LOG_DIR="$LOG_ROOT/$EXPERIMENT_NAME/$RUN_NAME"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname -- "$GATEWAY_DB_PATH")"

export RLLM_GATEWAY_LOG_LEVEL=warning

export CUDA_VISIBLE_DEVICES
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

exec python cookbooks/mobilegym/train.py \
  rllm/backend=verl \
  trainer.project_name=mobilegym \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='[console,tensorboard]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.default_local_dir="checkpoints/mobilegym/$EXPERIMENT_NAME/$RUN_NAME" \
  trainer.save_freq=3 \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_only=false \
  trainer.resume_mode="$RESUME_MODE" \
  ${RESUME_FROM_PATH:+trainer.resume_from_path="$RESUME_FROM_PATH"} \
  trainer.use_legacy_worker_impl=disable \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +model.name="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.disable_log_stats=false \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34000 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.gen_batch_size="$GEN_BATCH_SIZE" \
  data.max_prompt_length=32768 \
  data.max_response_length=1024 \
  data.return_multi_modal_inputs=true \
  data.dataloader_num_workers=0 \
  rllm.workflow.n_parallel_tasks="$ENV_POOL_SIZE" \
  rllm.workflow.retry_limit=3 \
  rllm.workflow.raise_on_error=false \
  rllm.gateway.db_path="$GATEWAY_DB_PATH" \
  rllm.compact_filtering.enable=true \
  rllm.compact_filtering.mask_error=true \
  rllm.rejection_sample.enable="$REJECTION_ENABLE" \
  rllm.rejection_sample.multiplier="$REJECTION_MULTIPLIER" \
  rllm.rejection_sample.min_partial_solve_tasks="$MIN_PARTIAL_SOLVE_TASKS" \
  rllm.rejection_sample.min_trajs_per_group="$MIN_TRAJS_PER_GROUP" \
  rllm.rejection_sample.filter_uniform_groups="$FILTER_UNIFORM_GROUPS" \
  rllm.rejection_sample.dynamic_sampling="$DYNAMIC_SAMPLING" \
  rllm.rejection_sample.dynamic_target_groups="$DYNAMIC_TARGET_GROUPS" \
  rllm.rejection_sample.dynamic_max_rounds="$DYNAMIC_MAX_ROUNDS" \
  rllm.rejection_sample.dynamic_min_reward_range="$DYNAMIC_MIN_REWARD_RANGE" \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.use_kl_in_reward="$USE_KL_IN_REWARD" \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  algorithm.kl_ctrl.kl_coef=0.01 \
  rllm.algorithm.adv_estimator=grpo \
  +suite=null \
  +runs_dir="$LOG_ROOT/$EXPERIMENT_NAME" \
  +run_name="$RUN_NAME" \
  +split="$SPLIT" \
  +sample_n="$SAMPLE_N" \
  +sample_templates="$SAMPLE_TEMPLATES" \
  +task_seed="$TASK_SEED" \
  +max_val_tasks=256 \
  +shared_val=false \
  +val_split=test \
  +agent_temperature=0.7 \
  +agent_val_temperature=0.1 \
  +max_steps_cap="$MAX_STEPS_CAP" \
  +reward_clean_penalty="$REWARD_CLEAN_PENALTY" \
  +reward_abort_penalty="$REWARD_ABORT_PENALTY" \
  +reward_overdue_penalty="$REWARD_OVERDUE_PENALTY" \
  +reward_premature_penalty="$REWARD_PREMATURE_PENALTY" \
  +reward_format_error="$REWARD_FORMAT_ERROR" \
  +reward_repetitive_loop="$REWARD_REPETITIVE_LOOP" \
  +reward_max_steps="$REWARD_MAX_STEPS" \
  +reward_infra_error="$REWARD_INFRA_ERROR" \
  +reward_repeated_action="$REWARD_REPEATED_ACTION" \
  +reward_no_state_change="$REWARD_NO_STATE_CHANGE" \
  +reward_max_behavior_penalties="$REWARD_MAX_BEHAVIOR_PENALTIES" \
  +monitor=true \
  +env_pool_size="$ENV_POOL_SIZE" \
  +env_num_browsers="$ENV_NUM_BROWSERS" \
  +env_init_batch_size="$ENV_INIT_BATCH_SIZE" \
  +env_init_warmup="$ENV_INIT_WARMUP" \
  ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps="$TOTAL_TRAINING_STEPS"} \
  "$@" 2>&1 | tee -a "$LOG_DIR/terminal.log"
