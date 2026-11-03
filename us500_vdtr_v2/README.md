# SPX VDTR model v2.0 (US500)

US500 数据基准版本的 VDTR 四模型系统（Volatility / Direction / Trend / Reversal），支持独立训练、独立预测与策略输出。

## 1. 项目说明
- 与 `SPX VDTR model v2.0` 架构一致，主要区别是价格源使用 `US500_1D.csv`。
- 主流程脚本：
  - `train_all_models_v2.py`
  - `run_all_models_and_log_v2.py`
  - `daily_meta_strategy_engine_v3.py`
- 路径由 `config.py` 相对管理，可独立运行。

## 2. 目录结构（核心）
```text
SPX VDTR model v2.0(US500)/
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

## 3. 依赖安装
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 运行前准备
`data/` 至少包含：
- `US500_1D.csv`
- `DIX.xlsx`
- `CBOE_DLY_VIX.csv`
- `event_calendar.csv`

`model/` 需有对应训练产物（或先执行训练脚本）。

## 5. 常用命令
训练：
```bash
python train_all_models_v2.py
```

预测：
```bash
python run_all_models_and_log_v2.py
```

策略输出：
```bash
python daily_meta_strategy_engine_v3.py
```

## 6. 日志说明
- `log/prediction_log_live.csv`：实盘口径
- `log/prediction_log_replay.csv`：回放口径
- `log/prediction_log.csv`：兼容旧流程

## 7. 免责声明
仅用于研究与工程实验，不构成投资建议。


<!-- AUTO_LATEST_SYNC_START -->
## Latest Sync (2026-04-03)
- 父项目接入：日跑=是，训练=是。
- 日跑脚本：un_all_models_and_log_v2.py
- 训练脚本：	rain_all_models_v2.py
- 交接入口：THREAD_HANDOFF.md
<!-- AUTO_LATEST_SYNC_END -->

