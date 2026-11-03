# history/ — 真实历史预测记录（快照）

各模型在真实行情上的历史预测记录。由 `tools/export_history.py` 从本机日志导出，用来查看模型过往的预测与实际结果。

为公开发布做了两处清理：
- **去掉原始行情列**：SPX 开高低收、收盘价、VIX 点位。这些属于数据商的行情数据。
- **路径改为相对路径**：所有本机绝对路径都改成了相对项目根的路径。

保留的内容：预测值、各类别概率、决策、真实标签（波动档 / 涨跌方向 / 趋势日 / 反转）、次日收益率。

| 文件 | 内容 | 覆盖时间 |
|---|---|---|
| `aio_v1_prediction_log.csv` | aio_v1（冻结基线）：波动 / 方向 / 趋势 / 反转 | 2024-07 起 |
| `spx_vdtr_v2_prediction_log_live.csv` / `_replay.csv` | SPX VDTR v2.1 | 2024-07 起 |
| `us500_vdtr_v2_prediction_log_live.csv` / `_replay.csv` | US500 VDTR v2.1 | 2024-07 起 |
| `trend_mode_daily_predictions.csv` | 开盘后趋势：剩余时段 / 尾盘 150 分钟，balanced 与 precision 两种阈值 | 2026-03 起 |
| `nextday_similarity_results.json` | 次日相似路径：每日预测的路径、最相似历史日、预期振幅（全部是百分比） | |
| `nextday_similarity_rolling_evaluation.json` | 次日相似路径的滚动评估 | |

## 怎么看（重要）

- `date` 是被预测的交易日（t+1），`src_date` 是用来做预测的特征日（t）。
- **replay**：每天日跑时，用当前模型把整段历史重新预测一遍的结果。模型是用包含这段时间在内的数据训练的，所以 replay 有一部分是**样本内**的，命中率会偏高，**不能当作实盘表现**。
- **live**：预测列保留第一次写入时的值，只有真实结果列会后续回填。但 live 文件最初是从 replay 初始化的，早期的行同样不是当时实时记录的。
- 要评估样本外表现，请只看各模型最后一次训练之后的日期；或者用 `models evaluation` 按时间窗口统计。
- 字段解释：`*_pred` 是预测值；`real_*` 是真实标签；`*_P_*` 是各类别概率；`*_decision` 是按阈值得出的交易决策；`*_correct` 表示预测是否命中。

## 更新方法

```bash
python tools/export_history.py
```

运行后再 `git commit` 即可。
