# data/ — 共享数据（程序自动维护）

日跑第 1 步会把 `exported data/` 里的新导出合并进来，各模型再从这里同步到自己的 `data/`。**一般不用手动改这里。**

| 文件 | 内容 | 主要列 |
|---|---|---|
| `SP_SPX, 1D.csv` | SPX 日线 | `time`(YYYY-MM-DD), `open`, `high`, `low`, `close` |
| `US500_1D.csv` | US500（CFD）日线 | 同上 |
| `CBOE_DLY_VIX.csv` | VIX 日线 | 同上 |
| `DIX.xlsx` | DIX / GEX | `date`, `dix`, `gex`, `price` |
| `SP_SPX, 30.csv` | SPX 30 分钟线（次日相似路径用） | `time`(带时区，如 `2026-11-02T06:30:00-08:00`), `open`, `high`, `low`, `close` |
| `event_calendar.csv` | FOMC / OPEX / VIX 到期日历 | `date`, `is_fomc`, … |
| `nyse_open_days.csv` | NYSE 开市日历 | `date`, … |
| `latest data/` | 最近约 300 个交易日的中转副本，供导出合并使用 | |
| `trend/` | 可选：手工放入的盘中源文件 | |

说明：
- 导出文件里多余的 TradingView 指标列（波浪、MA、ATR 分位等）会原样保留，程序只读上表列出的列。
- 第一次使用、手头还没有真实数据时，运行 `python tools/install_sample_data.py` 安装示例数据（合成数据，不是真实行情）。
- 仓库只上传本说明文件，数据文件本身不进 git。
