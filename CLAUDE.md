# CLAUDE.md — Model core

> 供 Claude Code 自动读取。使用说明见 README.md。

## 来历

2026-11-04 从 `Model projects`（同级目录）提取而来，只保留核心功能：
- 日跑：aio_v1 + spx_vdtr_v2 + us500_vdtr_v2 → 汇总 → 次日相似路径 → Sigma 分位
- 开盘后趋势
- 训练
- 回顾和回测

原项目里的 aio_v2、vdtr_path_v1、dip_buy、daytrade_strength 和 experiments 没有带过来。

提取时的验证结果（同一份数据，原项目对比本包）：
- aio_v1 预测日志逐字节一致。
- VDTR 的全部决策列一致，概率只有第 17 位有效数字的浮点噪声。
- summary 和 strategy 文本一致。
- 次日路径只有时间戳不同。
- 开盘后趋势：近 12 个交易日 48 条预测一致，误差 ≤ 2e-16。

## 约定

- 目录布局沿用原项目的相对结构，脚本之间通过 `PROJECT_ROOT / "spx_vdtr_v2"` 这类相对路径互相查找，不要改目录名。
- 统一解释器：`shared/launch.py` 按以下顺序取解释器：环境变量 `MODEL_CORE_PYTHON` > 入口配置 > `python_env.txt` > 当前解释器。各子目录不再有自己的 venv 或 PYTHON_ENV.txt。
- 启动器由 `launcher_config.json` 生成（`generate_platform_launchers.py`，`--check` 用来校验），不要手改。
- 跨模块复用的代码放在 `shared/`，子目录里只保留薄包装。
- 私有设置只放在 `local_settings.json`（已 gitignore），目前只有 `account_equity`。不要把账户资金、个人路径、邮箱写进代码或文档。
- git 管代码、配置、文档和当前使用的模型（见 .gitignore 末尾的例外）。数据和日志只随 OneDrive 同步。
- 日期一律按美东时间判断（不要用本机时区，本机时钟不一定在美国时区）。
- US500 是 CFD，日线在美东 17:00 收盘；SPX 现货在 16:00 收盘。

## 快速定位

| 需求 | 文件 |
|---|---|
| 子模型注册/开关 | `project_registry.json` |
| 入口与步骤 | `launcher_config.json`、`menu.py` |
| Sigma 分位 | `sigma_levels.py` |
| 导出文件前缀映射 | `update_parent_data_from_latest_v2.py::EXPORTED_TO_LATEST`、`predict_nextday_similarity_path.py::_find_latest_exported_30m`、`remaining_trend_intraday/scripts/merge_spx_*` |
