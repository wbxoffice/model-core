# SPX VDTR model v2.0

SPX 日频四模型系统（Volatility / Direction / Trend / Reversal），支持独立训练、独立预测、策略文本输出。

## 1. 项目说明
- 数据基于 SPX + DIX + VIX + 事件日历特征。
- 主流程脚本：
  - `train_all_models_v2.py`：训练四模型
  - `run_all_models_and_log_v2.py`：生成预测日志
  - `daily_meta_strategy_engine_v3.py`：生成策略建议文本
- 所有路径由 `config.py` 控制，默认相对当前项目目录，可独立运行。

## 2. 目录结构（核心）
```text
SPX VDTR model v2.0/
├─ config.py
├─ feature_engineering_v2_1.py
├─ train_all_models_v2.py
├─ run_all_models_and_log_v2.py
├─ daily_meta_strategy_engine_v3.py
├─ data/
├─ model/
├─ log/
└─ backtest/
```

## 3. 环境要求
- Python 3.11+
- 主要依赖：`pandas`, `numpy`, `scikit-learn`, `openpyxl`

建议使用项目内虚拟环境：
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas numpy scikit-learn openpyxl
```

## 4. 运行前准备
`data/` 至少包含以下文件：
- `SP_SPX, 1D.csv`
- `DIX.xlsx`
- `CBOE_DLY_VIX.csv`
- `event_calendar.csv`

`model/` 需包含训练产物（或先执行训练脚本）：
- `spx_vol_model_v2_latest.pkl`
- `spx_dir_model_v2_latest.pkl`
- `spx_trend_model_v2_latest.pkl`
- `spx_rev_model_v2_latest.pkl`

## 5. 常用命令
训练：
```bash
python train_all_models_v2.py
```

预测与日志输出：
```bash
python run_all_models_and_log_v2.py
```

策略输出：
```bash
python daily_meta_strategy_engine_v3.py
```

## 6. 日志说明（当前版本）
`run_all_models_and_log_v2.py` 会维护三类日志：
- `log/prediction_log_live.csv`：实盘口径（历史预测不回写，仅结算字段更新）
- `log/prediction_log_replay.csv`：回放口径（允许历史覆盖）
- `log/prediction_log.csv`：兼容旧流程（当前镜像 replay）

## 7. 相关文档
- `VDTR_Model_Objectives_v1.0.md`
- （旧项目文档已归档）`History/version_backups/docs_cleanup_2026-04-08/SPX VDTR model v2.0/项目文档/`
- `THREAD_HANDOFF.md`

## 8. 免责声明
本项目仅用于研究与工程实验，不构成任何投资建议。


<!-- AUTO_LATEST_SYNC_START -->
## Latest Sync (2026-04-03)
- 父项目接入：日跑=是，训练=是。
- 日跑脚本：un_all_models_and_log_v2.py
- 训练脚本：	rain_all_models_v2.py
- 交接入口：THREAD_HANDOFF.md
<!-- AUTO_LATEST_SYNC_END -->


