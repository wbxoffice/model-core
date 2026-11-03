# all in one model Fluctuation Direction Trend Reversal

SPX 四模型综合系统（Fluctuation / Direction / Trend / Reversal），用于日频预测与日内策略建议输出。

## 1. 项目定位
- 四个子模型联合输出：
  - 波动分层（regime）
  - 方向（up / neutral / down）
  - 趋势日判断
  - 反转风险判断
- 预测脚本：
  - `run_all_models_and_log.py`
- 策略脚本：
  - `daily_meta_strategy_engine_v3.py`

## 2. 目录结构（核心）
```text
all in one model Fluctuation Direction Trend Reversal/
├─ run_all_models_and_log.py
├─ daily_meta_strategy_engine_v3.py
├─ update_data_files.py
├─ analyze_prediction_log.py
├─ weekly_monthly_report.py
├─ data/
├─ model/
├─ log/
└─ range_quantiles_by_regime.json
```

## 3. 运行依赖
已提供依赖文件：
- `requirements.txt`

版本说明：
- 当前 `model/*.pkl` 为旧版本 sklearn 训练产物，项目按“保留旧模型、不重训”策略维护。
- 因此依赖中将 `scikit-learn` 锁定为 `1.1.3`（用于兼容旧 pkl）。
- 若本机 Python 环境无法安装 `scikit-learn==1.1.3`，建议使用 Python 3.10 虚拟环境。

安装：
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 运行前准备
`data/` 目录需有：
- `SP_SPX, 1D.csv`
- `DIX.xlsx`

`model/` 目录需有对应 `.pkl` 模型文件（与脚本内命名一致）。

## 5. 常用命令
生成预测日志：
```bash
python run_all_models_and_log.py
```

生成策略建议：
```bash
python daily_meta_strategy_engine_v3.py
```

分析历史预测：
```bash
python analyze_prediction_log.py
```

周/月报：
```bash
python weekly_monthly_report.py
```

## 6. 说明
- 该目录适合作为独立仓库发布代码骨架。
- 当前 `.gitignore` 已默认排除数据、模型、日志和历史产物。
- 项目内有说明文件 `0此模型目前不再重新训练.txt`，请按实际维护策略使用训练流程。

## 7. 免责声明
本项目仅用于研究与工程实验，不构成投资建议。

<!-- AUTO_LATEST_SYNC_START -->
## Latest Sync (2026-04-03)
- 父项目接入：日跑=是，训练=否（以 project_registry.json 为准）。
- 日跑脚本：un_all_models_and_log.py
- 交接入口：THREAD_HANDOFF.md
- 若需最新状态说明，请同步维护本项目 README 与父项目 STATUS_SYNC.md。
<!-- AUTO_LATEST_SYNC_END -->

