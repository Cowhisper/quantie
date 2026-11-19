# 关键词
- RSI(2) (2-period RSI)
- 均值回归 (Mean Reversion)
- 超卖 (Oversold)
- Larry Connors
- 200 日均线过滤 (200-day SMA trend filter)
- 固定本金仓位 (Fixed Capital Position)

# 参考文献
- https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2
- https://swingfolio.com/education/level-6-advanced-strategies/rsi-2-strategy
- https://www.quantitativo.com/p/squeezing-more-profits-with-cumulative
- Connors & Alvarez《Short Term Trading Strategies That Work》(2008)
  （RSI(2) 规则原始出处；另见 Warrior Trading 公开资料的同源体系 https://www.warriortrading.com/）

# 策略描述

短期极端超卖后买入、反弹至短期均线即离场的高胜率均值回归策略。
出自 Connors & Alvarez《Short Term Trading Strategies That Work》(2008)：
长期上升趋势中的短期深度回调是均值回归机会而非趋势反转，RSI(2) 对
最近 2 日的涨跌极灵敏，<10 即代表近 2 日几乎单边下跌。原文对指数 ETF
（SPY 1995–2008）回测胜率约 77%，后续公开复测胜率多在 65%~75%，
是文献中胜率最高的日线策略之一；持仓周期短（通常 2~7 天）。

## 买入

### 买入倾向排序
策略逐股扫描信号，不感知账户状态；所有股票的买入信号汇总到账户层
（`lib/account.py` 的 `simulate_account`），在资金与持仓上限（`engine.max_positions`，
默认 10 只）约束下按评分降序成交——评分高（最可能赚钱）的信号优先获得资金与仓位，
评分相同按股票代码排序保证确定性。
- 排序公式：`score` = 超卖深度归一化 = (`rsi_buy` − RSI(2)) / `rsi_buy`。
  Connors 原文的统计结论是 RSI(2) 读数越低、随后预期收益越高（其 <5 加仓规则
  的依据），故以超卖深度排序：RSI(2) 越接近 0 评分越接近 1，越浅越接近 0。
  写入 `Trade.score`。
- 超参注释：`rsi_window`=2（RSI 周期，"RSI(2)" 名称来源；周期越短对单日
  极端波动越灵敏），`rsi_buy`=10（超卖阈值，原文规则；原文 RSI(2)<5 时
  加仓第二笔，本仓位的固定本金单笔模型无法表达加仓，只取首笔），
  `trend_window`=200（长期趋势均线窗口，原文规则；回调必须发生在上升趋势中，
  下跌趋势中的超卖多是基本面恶化、会继续阴跌），
  `exit_ma_window`=5（离场参考均线，原文规则），
  `max_chase_pct`=0.05（次日高开追价上限，见下）
  `rsi_floor`=2.0（RSI(2) 下限，A 股实证约束：信号日 RSI(2) 不得低于
  该值——RSI(2)→0 多为连续跌停/基本面恶化的下跌飞刀，会继续阴跌；
  baseline 回测中 RSI(2)<2 的成交胜率仅 ~54% 且收益为负，2~10 区间
  ~58% 为正，"越深超卖越好"的原文结论在 A 股个股上不成立），
  `max_fall5_pct`=0.25（近 5 日累计跌幅上限；崩盘式下跌的超卖不是
  均值回归机会，baseline 中近 5 日跌超 25% 的信号胜率仅 ~36%），
  `skip_st`=true（排除 ST/*ST 个股；退市风险不符合均值回归前提，
  baseline 中 ST 股信号胜率 ~47% 显著偏低）
- 最低买入要求：信号日同时满足——收盘价高于 `trend_window` 日均线、
  RSI(`rsi_window`) < `rsi_buy`、RSI(`rsi_window`) ≥ `rsi_floor`、
  近 5 日累计跌幅不超过 `max_fall5_pct`、非 ST/*ST（`skip_st`）、
  信号日处于回测区间内、命中全部附加市场过滤条件
  （`conditions`，如价格区间）、固定本金按次日开盘价能买入至少 1 手；
  低于任一要求不触发买入。
  次日开盘价相对信号日收盘价高开超过 `max_chase_pct`（默认 5%）放弃——
  超卖反弹的隔夜空间已被兑现，追高买入则盈亏比恶化。
  满足要求的信号也不保证成交：当日资金或持仓不足时，低分信号被跳过（整笔作废）。
- 数据要求：`trend_window`=200 需要信号日前约一年的历史数据，回测/预测的
  `warmup`（加载信号日前 N 天数据算指标）应 ≥ 250，否则区间前期长期均线为
  NaN、无信号。
- 成交假设：信号一律由收盘数据确认、次日开盘成交——回测模拟每天开盘前决策，
  以次日开盘价买入，避免"按当日收盘价成交"的前视偏差（A 股 T+1 下买入当日
  也不可卖出）；信号日之后无交易数据则不入场。原版按信号日收盘买入，
  此处为仓库统一的无未来函数近似。

### 买入量
固定本金仓位（`lib/position.py` 的 `FixedCapitalPosition`）：
买入股数 = `capital` // 次日开盘价，向下取整到整手（`lot_size`，默认 100 股/手）；
不足 1 手放弃本次买入。本金与手数可在配置段 `FixedCapitalPosition` 调整。

## 卖出

### 卖出条件
持仓次日起每日收盘依次检查（A 股 T+1：买入当日不可卖出），满足任一条件即
次日开盘卖出：
- 反弹到位：收盘价站上 `exit_ma_window` 日均线（默认 5 日）——原文离场规则，
  短期均线被收复即意味着超卖修复完成（原文按当日收盘离场，此处为次日开盘的
  无未来函数近似）
- 止损：收盘价跌破买入价的 (1 − `stop_loss_pct`)（默认 10%）——原文不设止损
  （200 日均线过滤器即其风控，且对象为指数 ETF 无退市风险）；A 股个股可能
  连续阴跌不触发"站上 5 日线"，必须显式兜底
- 时间止损：持仓满 `max_hold_days` 天（默认 10 天，原文持仓通常 2~7 天）
- 回测区间结束：日期到达 `end_time` 或最后一条数据，按当日收盘强制平仓

### 卖出量
全部卖出（卖出数量 = 买入数量，见 `FixedCapitalPosition.sell_amount`）。
