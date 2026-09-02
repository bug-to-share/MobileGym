#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
# HF repo id by default (auto-downloaded); override with a local path:
#   MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash cookbooks/mobilegym/train_qwen3_vl_4b_verl.sh
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
SPLIT="${SPLIT:-train}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilegym-qwen3-vl-4b-train}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/mobilegym/$EXPERIMENT_NAME/$RUN_NAME"
mkdir -p "$LOG_DIR"

export VERL_LOGGING_LEVEL=DEBUG
export RLLM_GATEWAY_LOG_LEVEL=warning

export CUDA_VISIBLE_DEVICES
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# -----------------------------------------------------------------------------
# Memory guard
#
# The default values are deliberately conservative for an 80-GiB GPU host.  The
# guard only observes resources and terminates this training run if a limit is
# crossed; it never touches unrelated processes.  Override these variables when
# running on a different machine, for example:
#   MEMORY_GUARD_GPU_PCT=99 MEMORY_GUARD_HOST_AVAILABLE_GIB=16 bash ...
# -----------------------------------------------------------------------------
MEMORY_GUARD_ENABLE="${MEMORY_GUARD_ENABLE:-1}"
MEMORY_GUARD_INTERVAL="${MEMORY_GUARD_INTERVAL:-10}"
MEMORY_GUARD_GPU_PCT="${MEMORY_GUARD_GPU_PCT:-99}"
MEMORY_GUARD_HOST_AVAILABLE_GIB="${MEMORY_GUARD_HOST_AVAILABLE_GIB:-80}"
MEMORY_GUARD_CGROUP_PCT="${MEMORY_GUARD_CGROUP_PCT:-95}"
MEMORY_GUARD_LOG="$LOG_DIR/memory_guard.log"

memory_guard() {
  local train_pid="$1"
  local visible_gpu_ids="${CUDA_VISIBLE_DEVICES//,/ }"
  local host_limit_kib=$((MEMORY_GUARD_HOST_AVAILABLE_GIB * 1024 * 1024))

  {
    echo "# memory_guard_start=$(date --iso-8601=seconds) pid=$train_pid"
    echo "# gpu_pct_limit=$MEMORY_GUARD_GPU_PCT host_available_limit_gib=$MEMORY_GUARD_HOST_AVAILABLE_GIB cgroup_pct_limit=$MEMORY_GUARD_CGROUP_PCT interval=${MEMORY_GUARD_INTERVAL}s"
  } >> "$MEMORY_GUARD_LOG"

  while kill -0 "$train_pid" 2>/dev/null; do
    local now gpu_row gpu_id used_mib total_mib gpu_pct available_kib
    now=$(date --iso-8601=seconds)

    # nvidia-smi reports physical GPU indices, while CUDA_VISIBLE_DEVICES may
    # be a comma-separated list such as 2,3.
    while IFS=',' read -r gpu_id used_mib total_mib utilization; do
      gpu_id="${gpu_id// /}"
      used_mib="${used_mib// /}"
      total_mib="${total_mib// /}"
      utilization="${utilization// /}"
      case " $visible_gpu_ids " in
        *" $gpu_id "*)
          if [[ "$total_mib" =~ ^[0-9]+$ && "$used_mib" =~ ^[0-9]+$ && "$total_mib" -gt 0 ]]; then
            gpu_pct=$((used_mib * 100 / total_mib))
            echo "[$now] gpu=$gpu_id used_mib=$used_mib total_mib=$total_mib used_pct=$gpu_pct utilization_pct=$utilization" >> "$MEMORY_GUARD_LOG"
            if (( gpu_pct >= MEMORY_GUARD_GPU_PCT )); then
              echo "[$now] MEMORY_GUARD: GPU $gpu_id reached ${gpu_pct}% (limit ${MEMORY_GUARD_GPU_PCT}%), terminating pid $train_pid" >> "$MEMORY_GUARD_LOG"
              kill -TERM "$train_pid" 2>/dev/null || true
              return 0
            fi
          fi
          ;;
      esac
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)

    available_kib=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)
    if [[ "$available_kib" =~ ^[0-9]+$ ]]; then
      echo "[$now] host_mem_available_kib=$available_kib" >> "$MEMORY_GUARD_LOG"
      if (( available_kib > 0 && available_kib <= host_limit_kib )); then
        echo "[$now] MEMORY_GUARD: host MemAvailable=${available_kib}KiB (limit ${host_limit_kib}KiB), terminating pid $train_pid" >> "$MEMORY_GUARD_LOG"
        kill -TERM "$train_pid" 2>/dev/null || true
        return 0
      fi
    fi

    # Also protect against a container cgroup limit when one is configured.
    local cgroup_current cgroup_max cgroup_pct
    cgroup_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    cgroup_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
    if [[ "$cgroup_current" =~ ^[0-9]+$ && "$cgroup_max" =~ ^[0-9]+$ && "$cgroup_max" -gt 0 ]]; then
      cgroup_pct=$((cgroup_current * 100 / cgroup_max))
      echo "[$now] cgroup_memory_current=$cgroup_current cgroup_memory_max=$cgroup_max used_pct=$cgroup_pct" >> "$MEMORY_GUARD_LOG"
      if (( cgroup_pct >= MEMORY_GUARD_CGROUP_PCT )); then
        echo "[$now] MEMORY_GUARD: cgroup memory reached ${cgroup_pct}% (limit ${MEMORY_GUARD_CGROUP_PCT}%), terminating pid $train_pid" >> "$MEMORY_GUARD_LOG"
        kill -TERM "$train_pid" 2>/dev/null || true
        return 0
      fi
    fi

    sleep "$MEMORY_GUARD_INTERVAL"
  done
  echo "# memory_guard_end=$(date --iso-8601=seconds)" >> "$MEMORY_GUARD_LOG"
}

# Redirect all training output through tee while retaining the Python PID for
# the guard and for reliable cleanup.  This replaces a pipeline whose PID can
# otherwise be only the intermediate `tee` process.
exec > >(tee -a "$LOG_DIR/terminal.log") 2>&1
echo "memory_guard_enable=$MEMORY_GUARD_ENABLE visible_gpus=$CUDA_VISIBLE_DEVICES"

python cookbooks/mobilegym/train.py \
  rllm/backend=verl \
  trainer.project_name=mobilegym \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='[console,swanlab]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.default_local_dir="checkpoints/mobilegym/$EXPERIMENT_NAME/$RUN_NAME" \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.disable_log_stats=false \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=12 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34000 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
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
  +monitor=true \
  +env_pool_size=96 \
  +env_num_browsers=32 \
  &
TRAIN_PID=$!

GUARD_PID=""
if [[ "$MEMORY_GUARD_ENABLE" != "0" ]]; then
  memory_guard "$TRAIN_PID" &
  GUARD_PID=$!
fi

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e

if [[ -n "$GUARD_PID" ]]; then
  kill "$GUARD_PID" 2>/dev/null || true
  wait "$GUARD_PID" 2>/dev/null || true
fi

echo "training_exit_status=$TRAIN_STATUS"
exit "$TRAIN_STATUS"
