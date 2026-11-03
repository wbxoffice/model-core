# VDTR 模型目标文档（Model Objective Specification v1.0）

## 1. 模型体系概述（VDTR Framework）

VDTR 是一个多标签、多任务的日频预测体系，用于预测 SPX 下一交易日的核心价格行为特征，包括：

1. **Volatility Regime**（高/中/低波动）
2. **Direction**（上涨/下跌/横盘）
3. **Trend**（趋势持续）
4. **Reversal**（反转可能）

模型目标不是单纯预测未来，而是提供 **可交易（tradable）** 的结构化决策优势（edge）。

---

## 2. 总体目标（Overall Model Objective）

VDTR 的终极目标：

**为 SPX 0DTE / 1DTE 策略提供稳定、可复用、可解释的交易信号，使策略胜率提升 5–10%，回撤下降 30%+。**

核心关注点：

- 可靠性（Consistency）
- 概率校准（Calibration）
- 可交易性（Tradability）

---

## 3. 四大模型的具体目标（Per-Model Objectives）

### 3.1 Volatility Regime Model（波动预测）

#### 模型职责
- 预测 Low / Mid / High 波动
- 决定是否卖 gamma / 买 gamma
- 决定 0DTE 的最大头寸

#### KPI（关键指标）
| 指标 | 目标 | 原因 |
|------|------|------|
| High-vol recall | ≥ 0.65（理想 0.70） | 抓住大行情日 |
| Low-vol precision | ≥ 0.75 | 避免低波日卖错 gamma |
| Regime accuracy | ≥ 0.55 | 三分类不追求过高 |
| Calibration（P_high > 0.6） | ≥ 0.65 | 强信号必须可靠 |

---

### 3.2 Direction Model（方向 Up/Down/Flat）

#### 模型职责
- 提供方向倾向（右偏/左偏）
- 调整 put/call spread 方向

#### KPI
| 指标 | 目标 |
|------|------|
| 强信号区（P>0.6）准确率 | ≥ 70% |
| Neutral 占比 | 30–40% |
| 上涨/下跌概率校准 | ±10% 以内 |

---

### 3.3 Trend Model（趋势持续）

#### 模型职责
- 判断是否趋势突破
- 决定宽翼 butterfly / debit spread 是否适合

#### KPI
| 指标 | 目标 |
|------|------|
| Trend recall | ≥ 0.55 |
| Trend precision | ≥ 0.55 |

---

### 3.4 Reversal Model（反转）

#### 模型职责
- 判断 gap + close 反向的 reversal 情况
- 控制追涨杀跌风险

#### KPI
| 指标 | 目标 |
|------|------|
| Precision | ≥ 70% |
| Recall | ≥ 40% |
| P_rev > 0.65 时真实发生率 | ≥ 60% |

---

## 4. 策略层目标（Strategy-Level Objectives）

### S1. 胜率提升
| 策略 | 无模型 | 有模型 | 目标 |
|------|--------|--------|------|
| 0DTE spreads | 60–65% | 70–75% | +10% 提升 |

### S2. 回撤下降
- 最大回撤降低 **30–40%**
- 左尾风险（5% worst）改善 **20–30%**

### S3. 提高 R/R（Risk–Reward）
- trend day → debit/butterfly 收益显著改善
- high-vol day → 避免卖 gamma 大亏

### S4. 禁止入场日识别（核心风控）
例如：
- high-vol + reversal → 不做方向腿
- FOMC ±1d → 降低尺寸
- VIX expiry → 避免 early gamma trades

---

## 5. 工程目标（Engineering-Level Objectives）

### E1. 稳定性
- 数据缺失不崩溃
- 异常 spikes 不影响模型

### E2. 版本控制
- 每次 retrain 必须可复现
- 写入版本/训练窗口 meta

### E3. 可解释性
- 支持 SHAP
- 支持 daily explanation

### E4. 自动化
- 自动生成 prediction_log
- 自动生成 summary
- 支持定期 retrain

---

## 6. 风控目标（Risk-Control Objectives）

### R1. 避免 Gamma Risk
- 80% 的高波日提前提示 BUY_GAMMA

### R2. 避免逆势建仓
- P_trend > 0.55 → 避免 countertrend
- P_rev > 0.55 → 避免 breakout chasing

### R3. 动态仓位控制
- 高风险日自动减仓
- 低波日增加 theta 敞口

---

## 7. 长期目标（Lifecycle Objectives）

### L1. Drift Detection
- 每季度检测数据漂移
- 监控 hit-rate & calibration

### L2. Annual Upgrade Cycle
| 版本 | 改进方向 |
|------|---------|
| v2.1 | 精简特征、稳定结构 |
| v2.2 | 增强 GEX/DIX 微结构 & 事件 |
| v3.0 | 模型融合、校准、策略适应 |

---

**文档结束。**
