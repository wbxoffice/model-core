# Daily remaining trend prediction 子项目

本目录用于日常 remaining trend 预测与相关实验脚本管理（不影响原 `SPX VDTR model v2.0` 下脚本继续运行）。

项目重命名说明：
- 旧目录名：`trend test`
- 新目录名：`Daily remaining trend prediction`

## 目录结构
- `scripts/`：实验训练与优化脚本
- `models/`：实验模型文件（pkl）
- `results/`：评估、阈值扫描、最新信号等结果文件
- `docs/`：实验说明文档

## 当前归档来源
来源目录：`SPX VDTR model v2.0`

已归档脚本：
- `train_trend_experiment_intraday.py`
- `train_trend_after_open90_experiment.py`
- `optimize_trend_after_open90.py`
- `run_trend_modes_daily.py`
- `backtest_walkforward_after_open90.py`
- `backtest_walkforward_boosting_compare.py`
- `evaluate_trend_mode_predictions.py`

已归档模型：
- `spx_trend_open90_model_exp_atr08_v1.pkl`
- `spx_trend_close150_model_exp_atr08_v1.pkl`
- `spx_trend_remaining_after_open90_model_exp_atr08_v1.pkl`
- `spx_trend_close150_after_open90_model_exp_atr08_v1.pkl`

已归档结果：
- `trend_intraday_experiment_summary.csv`
- `trend_intraday_feature_importance_atr08.csv`
- `trend_intraday_label_diagnostics_atr08.csv`
- `trend_after_open90_experiment_summary.csv`
- `trend_after_open90_feature_importance_atr08.csv`
- `trend_after_open90_label_diagnostics_atr08.csv`
- `trend_after_open90_latest_prediction.csv`
- `trend_after_open90_threshold_scan.csv`
- `trend_after_open90_optimization_summary.csv`
- `trend_after_open90_latest_signal_recommendation.csv`

更新时间：2026-06-11

## 日常运行（两种模式）
- Python:
  - `python scripts/run_trend_modes_daily.py`
- Windows bat:
  - `run_trend_modes_daily.bat`（先做 5min incremental merge，再运行 daily 预测）
  - `retrain_trend_models.bat`（先 precheck 显示读取源与样本量，再确认是否重训）
- macOS:
  - `run_trend_modes_daily.sh` / `run_trend_modes_daily.command`
  - `retrain_trend_models.sh` / `retrain_trend_models.command`
  - `evaluate_trend_mode_predictions.sh` / `evaluate_trend_mode_predictions.command`

推荐使用 bat / sh / command 入口，而不是直接运行 Python 脚本，因为入口会完成数据准备和校验。

### 日常预测数据准备

`run_trend_modes_daily.bat` 与 `run_trend_modes_daily.sh` 在预测前执行：

1. 按 `America/New_York` 计算当前交易日期，检查 `spx_vdtr_v2` daily feature 是否已覆盖该 `target_date`（不再使用运行机器的上海本地日期）。
2. 若已覆盖，跳过重复日线导入和同步。
3. 若未覆盖，运行 `scripts/prepare_daily_sources_for_trend.py`：
   - 更新父项目 `exported data -> data/latest data -> data`。
   - 同步 `SP_SPX, 1D.csv`、`DIX.xlsx`、`CBOE_DLY_VIX.csv`、`event_calendar.csv` 到 `spx_vdtr_v2/data`。
4. 强制检测 5min intraday 数据源：
   - 仅接受 5min；执行增量 merge，保留不含未完成当日的稳定历史文件，同时生成“完整历史 + 当日实时柱”的预测输入后再预测。
   - 探测到 30min 时日常预测直接失败；30min 只保留给模型训练/历史研究流程使用。
5. 直接运行 `python scripts/run_trend_modes_daily.py` 时也会无条件刷新一次 5min 合并输入；bat/sh/command 入口已在前一步刷新，因此会传入内部跳过标记，避免重复合并。

日线准备脚本手动开关：

```bash
python scripts/prepare_daily_sources_for_trend.py --check-only
python scripts/prepare_daily_sources_for_trend.py --force-update
python scripts/prepare_daily_sources_for_trend.py --target-date YYYY-MM-DD
```

输出：
- 结果：`results/trend_mode_daily_predictions.csv`
- 主模式结果：`results/trend_mode_daily_primary.csv`（`mode=balanced`）
- 日志：`log/run_trend_modes_YYYYMMDD_HHMMSS.txt`
- 5min 稳定历史：`data/SPX_5min_merged.csv`（排除当前纽约交易日，供历史维护）
- 5min 实际预测输入：`data/SPX_5min_prediction_input.csv`（稳定历史 + 当前纽约交易日实时柱）
- 数据读取优先级（intraday）：
  1. 先按原始源优先级探测频率：`exported data` 5min → `exported data` 30min → `data/trend` 5min → `data/trend` 30min → VDTR fallback。
  2. 若探测为 5min，模型实际读取 `data/SPX_5min_prediction_input.csv`，不再直接读取只有少量日期的 exported 文件。
  3. 若最高优先级可用源为 30min，日常预测失败并提示导入 5min 文件，不再用 30min 兜底。
- 文件清理规则：
  - 自动清理同前缀旧文件仅对 `data/trend` 生效；
  - `exported data` 不会被自动删除文件。
- 当天校验：
  - “今日”统一指纽约日期；5min 必须完整包含 `06:30–07:55 PT` 的18根柱。
  - 当天0根、少于18根、缺失中间柱、重复柱或错位柱都视为失败，以非零退出码中止，不生成预测。
  - 文件补齐后重新运行即可再次合并、校验并预测；不会因为此前失败而锁死该交易日。
  - 若在纽约 `09:30 ET` 开盘前运行且源文件没有纽约当天数据，自动选择源文件中最近一个已覆盖正常收盘的交易日，作为 `postclose` 运行。这兼容上海次日白天补跑前一交易日盘后预测。
  - 纽约开盘后若没有当天数据，仍按当天缺失失败，不会静默使用昨天的数据。早收盘日可在确认文件完整后显式使用 `--run-mode postclose`。

### 盘中与盘后运行阶段

- 默认 `--run-mode auto`：
  - 5min 当日源覆盖到 `12:55 PT` 或之后，标记为 `postclose`；否则标记为 `intraday`。
- 可手动指定：

```bash
python scripts/run_trend_modes_daily.py --run-mode intraday
python scripts/run_trend_modes_daily.py --run-mode postclose
```

- 早收盘日自动判断可能仍显示 `intraday`，可在确认文件已完整后使用 `--run-mode postclose`。
- `trend_mode_daily_predictions.csv` 和 `trend_mode_daily_primary.csv` 使用 `date + label + mode + run_mode` 作为复合键：
  - 同一交易日的 `intraday` 与 `postclose` 保存在同一文件、互不覆盖。
  - 同一交易日、同一 `run_mode` 重跑时，只保留该阶段的最新结果。
  - 历史旧行自动标记为 `run_mode=legacy`。
- 每行同时记录 `run_timestamp_ny`、`run_timestamp_utc`、`source_max_timestamp_pt`、`target_date_reason`、open90 实际/预期柱数和完整性状态。
- 评价脚本按 `run_mode` 分组计算，避免把盘中、盘后和旧版结果混在一起。

## Daily 结果字段补充
- `trend_mode_daily_predictions.csv` / `trend_mode_daily_primary.csv` 额外输出：
  - `hist_precision` / `hist_recall`：默认阈值对应的 walk-forward 历史表现
  - `wf_ref_threshold` / `wf_precision` / `wf_recall`：按当前 `p_trend` 取“仍判1”的最近阈值的 walk-forward 参考表现
- daily 日志按每条结果分块输出（基础预测 / yesterday / hist / wf）。
- 运行审计字段：`run_mode`、`run_timestamp_ny`、`run_timestamp_utc`、`source_file`、`source_max_timestamp_pt`、`open90_frequency_minutes`、`open90_expected_bars`、`open90_actual_bars`、`open90_complete`。

## 缺失交易日补录与统一回放

运行：

```bash
python scripts/backfill_trend_mode_predictions.py
```

- 只补录模型 `train_end_date` 之后、正式 NYSE 开市日中缺少 live 记录且已有完整实际标签的日期。
- 不覆盖 `trend_mode_daily_predictions.csv`；补录行单独标记为 `origin=backfill`。
- 同时对 post-train 全区间按修复后的完整5分钟输入统一重放，避免把旧短输入 live 记录与新输入补录混算。
- 输出：
  - `results/trend_mode_backfill_predictions.csv`
  - `results/trend_mode_corrected_replay_predictions.csv`
  - `results/trend_mode_predictions_complete.csv`
  - `evaluation/trend_mode_backfill_summary.csv`
  - `evaluation/trend_mode_backfill_report_2026-08-27.md`
- 可用 `--start YYYY-MM-DD` / `--end YYYY-MM-DD` 限定补录区间。

## 重训前确认（Precheck）
- 运行 `retrain_trend_models.bat` 时会先执行：
  - `python scripts/train_trend_after_open90_experiment.py --precheck`
- precheck 会输出：
  - 实际读取的 intraday 源文件路径
  - 时间覆盖范围与可用交易日
  - remaining / close150 训练样本量、正样本率、类别分布
- 只有人工确认 `y` 后才执行正式重训，避免短样本误覆盖模型。

## 预测结果评价
- Python:
  - 全部记录：`python scripts/evaluate_trend_mode_predictions.py --window all`
  - 最近N个交易日：`python scripts/evaluate_trend_mode_predictions.py --window recent --n_days 30`
- Windows bat:
  - `evaluate_trend_mode_predictions.bat`

输出目录：
- `evaluation/`（detail / summary / report）

## 特征工程规范
- 标准文档：`docs/FEATURE_ENGINEERING_STANDARD.md`
- 规范原则：
  - 训练与预测共用同一特征构建入口
  - 不允许 `08:00` 后数据进入当日预测特征（防止未来泄露）

<!-- AUTO_LATEST_SYNC_START -->
## Latest Sync (2026-04-03)
- 研究子项目，当前不纳入父项目日跑/训练。
- 当前主结果口径：alanced（recall 优先）。
- 数据读取优先级：Model projects/exported data 优先，回退 data/trend。
- 交接入口：THREAD_HANDOFF.md + STATUS_SYNC.md
<!-- AUTO_LATEST_SYNC_END -->

