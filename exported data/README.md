# exported data/ — 每天把 TradingView 导出文件放这里

日跑第 0 步会检查这里的文件是否齐全、是否已刷新到上一个交易日；第 1 步把它们合并进 `data/`，导入成功后删除源文件。

## 需要的文件

文件名开头要对上（后面的随机后缀如 `_aeb67` 不用管），文件格式为 CSV。

| 内容 | 是否必需 | 文件名开头（任一） | 由谁使用 |
|---|---|---|---|
| SPX 日线 | 必需 | `SP_DLY_SPX, 1D`（也认旧前缀 `SPCFD_SPX, 1D` / `SP_SPX, 1D` / `CBOE_DLY_SPX, 1D`） | 日跑 |
| SPX 30 分钟 | 必需 | `SP_DLY_SPX, 30`（旧前缀同上） | 次日相似路径 |
| US500 日线 | 必需 | `CAPITALCOM_SPX500, 1D` / `CAPITALCOM_US500, 1D` / `US500, 1D` | 日跑 |
| VIX 日线 | 必需 | `CBOE_DLY_VIX, 1D` | 日跑 |
| DIX | 必需 | `DIX.csv`（列：`date,price,dix,gex`） | 日跑 |
| SPX 5 分钟 | 可选 | `SP_DLY_SPX, 5` | 开盘后趋势（开盘 90 分钟后导出） |

例子：`SP_DLY_SPX, 1D_aeb67.csv`、`SP_DLY_SPX, 30_35b12.csv`、`CBOE_DLY_VIX, 1D_0e4af.csv`。

## 列格式

- 日线：`time,open,high,low,close[,Volume]`，`time` 为日期。
- 分钟线：`time` 带时区，例如 `2026-10-28T08:10:00-07:00`。
- 多出来的指标列可以保留，程序会忽略。

## 注意

- 同一类文件新旧前缀并存时，按最新数据、再按前缀优先级取用：`SP_DLY_SPX` > `SPCFD_SPX` > `SP_SPX` > `CBOE_DLY_SPX`。
- 运行 `tools/install_sample_data.py` 会在这里放一组 `*_sample.csv` 导出样例，用来演示命名和格式。
- 仓库只上传本说明文件。
