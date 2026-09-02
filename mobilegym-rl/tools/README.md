# MobileGym 统一 rollout 评测

`analyze_results.py` 是消融实验共用的统计入口。输入一个 `results.jsonl`，会生成：

- `summary.json`：总体成功率、`progress/goal_success/success/passed`，`false_complete/overdue/wrong_answer_submitted`，步数和长度均值/P50/P90/P95，以及终止原因计数；
- `grouped_metrics.csv`：按 `suite`、`difficulty`、`scope`、`composition` 聚合的成功率、信号率、步数、长度和 reward 方差；
- `task_rollouts.csv`：每个 task 的 rollout 数、成功数/成功率、reward 均值/方差，以及步数和异常计数；
- `records.csv`：逐条归一化记录，便于二次分析；
- `reward.jsonl`（训练新 run 自动生成）：rLLM evaluator 计算后的真实 reward 及逐项惩罚明细；
- PNG：四个分组成功率、终止原因、步数分布、task reward 方差。

基本用法（建议在宿主机执行）：

```bash
python tools/analyze_results.py \
  logs/mobilegym/mobilegym-qwen3-vl-4b-train/run_20260901_041451/train_step_20_epoch_1/results.jsonl \
  -o logs/mobilegym/mobilegym-qwen3-vl-4b-train/run_20260901_041451/train_step_20_epoch_1/analysis_p0
```

默认按每个 task 期望 8 次 rollout 标记 `complete_rollouts`；smoke test 或其他配置可覆盖：

```bash
python tools/analyze_results.py RESULTS_DIR -o analysis --expected-rollouts 1
```

旧版结果没有持久化 reward 和 `wrong_answer_submitted` 时，脚本按
`cookbooks/mobilegym/evaluator.py` 的规则重建 reward，并从 `judge.issues` 推导错误提交。
新 run 若存在同目录 `reward.jsonl`，脚本优先使用其中的真实 reward。
同 run 下存在 `trajectory/` 时，会读取 response/prompt 文件；默认是正则 token 估计。
若需要模型 tokenizer 的精确长度，可增加 `--tokenizer /path/to/tokenizer`。
