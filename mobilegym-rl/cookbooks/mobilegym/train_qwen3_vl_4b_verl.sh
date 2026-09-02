#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# HF repo id by default (auto-downloaded); override with a local path:
#   MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash cookbooks/mobilegym/train_qwen3_vl_4b_verl.sh
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
SPLIT="${SPLIT:-train}"
# Optional global cap for MobileGym interaction steps. 0 keeps the
# task/difficulty-specific limits unchanged.
MAX_STEPS_CAP="${MAX_STEPS_CAP:-0}"

# P1 reward ablation knobs (multiplicative coefficients). Defaults preserve
# the original reward behavior; set e.g. REWARD_OVERDUE_PENALTY=0.25.
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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilegym-qwen3-vl-4b-train}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/mobilegym}"
LOG_DIR="$LOG_ROOT/$EXPERIMENT_NAME/$RUN_NAME"
mkdir -p "$LOG_DIR"

# export VERL_LOGGING_LEVEL=DEBUG
export RLLM_GATEWAY_LOG_LEVEL=warning
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES
# export NCCL_CUMEM_ENABLE=0
# export NCCL_CUMEM_HOST_ENABLE=0
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=WARN

exec python cookbooks/mobilegym/train.py \
  rllm/backend=verl \
  trainer.project_name=mobilegym \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='[console, tensorboard]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.default_local_dir="checkpoints/mobilegym/$EXPERIMENT_NAME/$RUN_NAME" \
  trainer.save_freq=4 \
  trainer.test_freq=2 \
  trainer.total_epochs=5 \
  trainer.val_before_train=True \
  trainer.val_only=false \
  trainer.resume_mode=disable \
  trainer.use_legacy_worker_impl=disable \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +model.name="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_model_len=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.disable_log_stats=false \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=48 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
  data.train_batch_size=12 \
  data.max_prompt_length=32768 \
  data.max_response_length=1024 \
  data.return_multi_modal_inputs=true \
  data.dataloader_num_workers=0 \
  rllm.workflow.n_parallel_tasks=96 \
  rllm.workflow.retry_limit=3 \
  rllm.workflow.raise_on_error=false \
  rllm.compact_filtering.enable=true \
  rllm.compact_filtering.mask_error=true \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.kl_ctrl.kl_coef=0.01 \
  rllm.algorithm.adv_estimator=grpo \
  +suite=null \
  +run_name="$RUN_NAME" \
  +split="$SPLIT" \
  +sample_n=1 \
  +task_seed=42 \
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
  +env_pool_size=96 \
  +env_num_browsers=32 \
  2>&1 | tee -a "$LOG_DIR/terminal.log"
