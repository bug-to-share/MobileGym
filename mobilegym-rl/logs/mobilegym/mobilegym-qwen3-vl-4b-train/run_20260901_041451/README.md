# 训练 run 归档

这是 `Qwen3-VL-4B-Instruct` 在 MobileGym 上的首次完整训练记录，run 名称为
`run_20260901_041451`。

已纳入 Git 的内容：

- 每个 train/val step 的 `results.jsonl`、`summary.json`、`meta.json`；
- 每个 step 的 `console.log`、顶层 `terminal.log`；
- 对应 TensorBoard event 文件：`mobilegym-rl/tensorboard_log/mobilegym/mobilegym-qwen3-vl-4b-train/`。

未纳入 Git 的内容：

- `trajectory/` 截图和 prompt/response 文件；
- `browser_logs/`；
- checkpoints、tensorboard 临时缓存和 core dump。

这些文件体积较大，且不影响读取训练曲线与重新分析结果。分析命令：

```bash
python mobilegym-rl/tools/analyze_results.py \
  mobilegym-rl/logs/mobilegym/mobilegym-qwen3-vl-4b-train/run_20260901_041451/train_step_20_epoch_1/results.jsonl \
  -o /tmp/mobilegym-step20-analysis
```
