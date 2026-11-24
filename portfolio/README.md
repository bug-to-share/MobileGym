# MobileGym online reinforcement learning experiments

This branch extends [MobileGym](https://github.com/Purewhiter/mobilegym) with Qwen3-VL-4B training experiments. The actor reads screenshots and text history, then controls the browser simulator through coordinate and text actions. The benchmark checks the final application state and the termination action. GRPO uses the resulting trajectory reward to update the policy.

## Project showcase

Download [index.html](index.html) from GitHub and open it in a browser for the interactive Chinese résumé, experiment charts, interview questions, and literature review. The core page works offline. To open its companion files and source links as well, clone the branch or download and extract the branch ZIP.

- [Résumé text](resume_entry.md)
- [Experiment summary CSV](experiments.csv)
- [GUI Agent research notes](gui_agent_research.md) and [paper versions](gui_agent_sources.json)

## Changes in this branch

| Area | Implementation |
| --- | --- |
| Rewards | Selectable terminal progress and binary success rewards, plus an option to zero out false completion. [Evaluator](../mobilegym-rl/cookbooks/mobilegym/evaluator.py) |
| Sampling | Collect additional rollout groups when too few groups contain a reward difference. [Trainer](../mobilegym-rl/rllm/experimental/unified_trainer.py) |
| Task generation | Reconstruct task wording from templates with seeded parameter sampling. [Workflow](../mobilegym-rl/cookbooks/mobilegym/mobilegym_flow.py) |
| Environment startup | Start and warm browser environments in configurable batches. [Initialization guide](../mobilegym-rl/cookbooks/mobilegym/INITIALIZATION.md) |

The reward modes in this branch use terminal state and termination signals. The research notes describe proposed process rewards and step credit separately.

## Run the simulator and training

Use Node.js 22 or later to build the frontend. The GPU trainer uses Python 3.12, CUDA, and the dependency versions in the [training setup guide](../mobilegym-rl/README.md#prerequisites). The Nginx gateway requires the local server prerequisites described in the [simulator README](../README.md).

From the repository root, activate the Python environment described in the setup guide, then run:

```bash
(cd mobilegym-rl && bash install.sh)
npm ci
npm run build
./scripts/server/start_nginx_gateway.sh
curl -fsSk https://localhost:4180/ -o /dev/null
```

From the repository root, launch the baseline or false-completion control:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl_false_complete.sh
```

The six-device defaults reflect the historical training configuration. Set `CUDA_VISIBLE_DEVICES` for the current machine and choose batch sizes compatible with its GPU count and memory. `MODEL_PATH` defaults to `Qwen/Qwen3-VL-4B-Instruct` and accepts a local model path. The launcher accepts Hydra overrides after the script name. Logs, checkpoints, TensorBoard events, and the gateway trace database are generated locally.

| Experiment | Launcher under `mobilegym-rl/cookbooks/mobilegym/` |
| --- | --- |
| Terminal progress GRPO | [Baseline](../mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl.sh) |
| False completion zeroed | [False-completion control](../mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl_false_complete.sh) |
| Dynamic sampling with terminal progress | [Dynamic sampling](../mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl_sample.sh), set `SAMPLE_N=4` for the archived N4 run |
| Terminal progress with larger PPO mini-batch | [Larger mini-batch](../mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl_grpo_minibsz96.sh) |
| Binary success GRPO | [Binary GRPO](../mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl_result_only_grpo.sh), set `TEST_FREQ=3` to enable periodic validation |
| Binary success with dynamic sampling | [Binary dynamic sampling](../mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_verl_result_only_dapo.sh) |

These are launch presets for follow-up runs. Exact replication of historical checkpoints requires the original sampled instances, dependency versions, resume state, and runtime configuration.

## Results

In the historical false-completion control, strict validation success increased from **23/256 (8.98%)** at step 0 to **56/256 (21.88%)** at steps 12 and 18, an increase of **12.89 percentage points**. It ended at **36/256 (14.06%)** on the last recorded validation. The experiment matched task IDs and prompt text across the compared checkpoints. Full initial-state snapshots were unavailable.

| Validation slice at step 12 | Strict successes |
| --- | ---: |
| Single app | 50/163 (30.67%) |
| Cross app | 6/93 (6.45%) |

These values describe one training run. Six exploratory runs varied in task sampling, batch settings, resume points, or infrastructure errors, so the observed peaks are unsuitable for estimating individual treatment effects. See [the experiment CSV](experiments.csv) and the experiment section of the [HTML showcase](index.html#experiments).

## Attribution and licenses

The simulator, task benchmark, and original training integration come from the MobileGym project. The training stack includes [rLLM](https://github.com/rllm-org/rllm) and [verl](https://github.com/volcengine/verl). The repository retains the upstream [code license](../LICENSE), [data license](../LICENSE-DATA), and [notices](../NOTICE).
