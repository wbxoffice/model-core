# Model core

从 `Model projects` 提取出的独立核心程序包：SPX / US500 日线模型日跑、开盘后趋势预测、模型更新训练、历史表现回顾，以及 Sigma 分位（ATR 分位阶梯）。
不依赖原项目，整个文件夹可以单独运行。

## 快速开始

1. 建环境（每台机器一次）：Windows 双击 `setup_env.bat`，Mac 双击 `setup_env.command`。
   解释器路径写在 `python_env.txt`，每行一个，取本机第一个存在的。
2. 把 TradingView 导出的文件放进 `exported data/`，包括 SPX 1D/30min/5min、US500 1D、VIX 1D 和 DIX。
3. 双击 `start.bat`（Mac 用 `start.command`），按编号选功能。

## 功能与入口

| 功能 | 入口（Windows / Mac） | 产出 |
|---|---|---|
| 菜单 | `start.bat` / `start.command` | — |
| 每日日跑（收盘后） | `run_prediction_all models.bat` / `run_prediction_all_models.command` | `prediction_log_all_models.xlsx`、`summary_all_projects.txt`、`strategy_all_projects.txt`、`log/nextday_similarity/`、`log/sigma_levels/` |
| 开盘后趋势预测 | `remaining_trend_intraday/run_trend_modes_daily.*` | `remaining_trend_intraday/results/trend_mode_daily_*.csv` |
| Sigma 分位 | `sigma_levels.*` | `log/sigma_levels/`：xlsx 分位表、png 阶梯图、history csv |
| 训练：日线模型 | `train_all_projects.*` | `spx_vdtr_v2/model/`、`us500_vdtr_v2/model/` |
| 训练：开盘后趋势 | `remaining_trend_intraday/retrain_trend_models.*` | `remaining_trend_intraday/models/` |
| 回顾：日线模型 | `models evaluation.bat` / `models_evaluation.command` | 交互式窗口统计 |
| 回顾：开盘后趋势 | `remaining_trend_intraday/evaluate_trend_mode_predictions.*` | `remaining_trend_intraday/evaluation/` |

其他回顾和回测脚本可以直接在命令行运行，包括：
- `nextday_prediction_backtest.py`、`analyze_regime_live_performance.py`
- 各模型目录下的 `analyze_prediction_log.py`、`weekly_monthly_report.py`、`analyze_strategy_PnL.py` 和 `backtest_spread_strangle_dynamic_range_PnL.py`

所有 `.bat` / `.command` / `.sh` 都由 `generate_platform_launchers.py` 根据 `launcher_config.json` 生成，不要手改。要新增或修改入口，改完配置后重新运行生成器。

## 日跑步骤

0. `precheck_exported_manifest.py`：检查导出文件是否齐全、是否刷新。缺少必需项时会询问是否继续。
1. `sync_run_and_aggregate_projects.py`：依次完成以下几件事。
   - 更新 `data/`（`update_parent_data_from_latest_v2.py`）。
   - 依次运行 `aio_v1`、`spx_vdtr_v2`、`us500_vdtr_v2` 的预测和策略。
   - 汇总结果。
2. `predict_nextday_similarity_path.py`：生成次日相似路径。
3. `sigma_levels.py`：计算下一交易日的 Sigma 分位，并把摘要追加到 `strategy_all_projects.txt`。

## Sigma 分位

- 中枢 = 基准日收盘价。
- σ = 基准日 Wilder ATR(14)。
- 分位 = 中枢 ± k·σ。其中 ±0.236 为上行/下行突破点，目标 1–4 分别为 0.618、1、1.236、1.618，表格列到 ±3。

与 TradingView 的 ATR Levels 指标（日线）对比：
- SPX：近 60 日误差中位 0.01 点。
- US500：误差中位约 1 点。原因是 US500 是 CFD，日线在美东 17:00 收盘；部分历史日线是在收盘前导出的，偏差主要来自这些日线。

日期按美东时间判定，与电脑所在时区无关。如果最新一根日线还没收盘（SPX 早于 16:00，US500 早于 17:00），就跳过这根，输出当天盘中适用的分位，并在图和表上注明。
可以用 `python sigma_levels.py --root . --date YYYY-MM-DD` 生成任意交易日的分位。

## 目录

```
shared/                   公共层：策略引擎、PnL 分析/回测、启动引擎(launch.py)、解释器选择
aio_v1/                   冻结基线（不再训练）
spx_vdtr_v2/              SPX 日线：方向 / 趋势 / 反转 / 波动
us500_vdtr_v2/            US500 日线（与 SPX 同一套代码，数据不同）
remaining_trend_intraday/ 开盘后趋势（开盘 90 分钟 → 剩余时段 / 尾盘 150 分钟）
data/                     共享数据（日跑自动更新）；exported data/ 是导出文件投放目录
tools/                    模型格式转换工具（aio_v1 旧 sklearn pickle → 当前版本）
```

## 本机私有设置

`local_settings.json` 不进 git，例如 `{"account_equity": 100000}` 表示用于仓位计算和回测初始资金的账户资金。
没有这个文件时，使用代码里的占位默认值。

## 环境说明

- 所有功能共用一个环境，版本固定在 `requirements.txt`（scikit-learn 1.7.2；模型 pickle 依赖这个版本）。
- aio_v1 的原始模型是用 sklearn 1.1.3 保存的。现在用 `tools/export_rf.py` + `tools/rebuild_rf.py` 无损转换，已核对全部历史预测，与旧环境逐字节一致。
