# Batched environment initialization

The MobileGym training integration enables environment warm-up by default.
`ENV_INIT_BATCH_SIZE=8` limits each startup wave to eight environments. Each
wave completes page startup, simulator/data readiness checks, and a first
observation (including a screenshot) before the next wave starts. Browser and
context allocation retain the existing sequential creation logic.

The pool is published to rollout workflows only after all waves complete.
Warm-up does not change the total environment count, rollout concurrency,
task sampling, or rewards. It preserves the fresh-page flag, so the first
task can reuse the initialized page. Task-specific setup still runs normally;
this is not a guarantee against all later screenshot timeouts.

Configuration passed through the training entry point:

| Shell variable | Hydra field | Default |
| --- | --- | --- |
| `ENV_INIT_BATCH_SIZE` | `env_init_batch_size` | `8` |
| `ENV_INIT_WARMUP` | `env_init_warmup` | `true` |

With 192 environments, startup has 24 waves of eight. Logs report
`EnvPool initialization: ... ready` and finally `pool ready`. Initialization
errors abort startup and clean up the pool instead of admitting failed
environments into training. The standalone `EnvPool` API keeps warm-up disabled
unless requested with `warmup=True`.

These settings apply on the next process start; they do not update an existing
running pool. Start training manually after activating the `mobilegym` conda
environment and entering the `mobilegym-rl` directory:

```bash
ENV_INIT_BATCH_SIZE=8 ENV_INIT_WARMUP=true \
EXPERIMENT_NAME=mobilegym-qwen3-vl-4b-train-SAMPLE-DAPO-WARMUP \
bash cookbooks/mobilegym/train_qwen3_vl_4b_verl_sample.sh
```
