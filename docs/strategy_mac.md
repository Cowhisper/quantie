# 关键词
- 均线金叉 (Moving Average Golden Cross)
- 均线死叉 (Moving Average Death Cross)
- 移动平均线 (MA)
- 止损 (Stop Loss)
- 固定本金仓位 (Fixed Capital Position)

# 参考文献
- https://www.investopedia.com/terms/g/goldencross.asp
- https://www.investopedia.com/terms/d/deathcross.asp

# 策略描述

## 买入

### 买入倾向排序
策略逐股扫描信号，不感知账户状态；所有股票的买入信号汇总到账户层
（`lib/account.py` 的 `simulate_account`），在资金与持仓上限（`engine.max_positions`，
默认 10 只）约束下按评分降序成交——评分高（最可能赚钱）的信号优先获得资金与仓位，
评分相同按股票代码排序保证确定性。
- 排序公式：`score` = 金叉日快线相对慢线的乖离率 = (fast_ma − slow_ma) / slow_ma。
  上穿瞬间快线高出慢线越多，短期动能越强、预期收益越高；写入 `Trade.score`。
- 超参注释：`fast_window`=5（快线窗口），`slow_window`=20（慢线窗口）
- 最低买入要求：当日 5 日均线上穿 20 日均线（金叉，当日收盘确认），
  且信号日处于回测区间内、命中全部附加市场过滤条件（`conditions`，如价格区间），
  且固定本金按次日开盘价能买入至少 1 手；低于该要求不触发买入。
  满足要求的信号也不保证成交：当日资金或持仓不足时，低分信号被跳过（整笔作废）。
- 成交假设：信号一律由收盘数据确认、次日开盘成交——回测模拟每天开盘前决策，
  以次日开盘价买入，避免"按当日收盘价成交"的前视偏差；
  信号日之后无交易数据（如区间末金叉）则不入场。

### 买入量
固定本金仓位（`lib/position.py` 的 `FixedCapitalPosition`）：
买入股数 = `capital` // 次日开盘价，向下取整到整手（`lot_size`，默认 100 股/手）；
不足 1 手放弃本次买入。本金与手数可在配置段 `FixedCapitalPosition` 调整。

## 卖出

### 卖出条件
持仓期间每日收盘依次检查，满足任一条件即触发离场（次日开盘卖出）：
- 死叉：5 日均线下穿 20 日均线
- 止损：收盘价跌破买入价的 (1 - `stop_loss_pct`)（默认 8%）
- 回测区间结束：日期到达 `end_time` 或最后一条数据，按当日收盘强制平仓
  （A 股 T+1：买入当日不可卖出，自买入次日起检查卖出条件）

### 卖出量
全部卖出（卖出数量 = 买入数量，见 `FixedCapitalPosition.sell_amount`）。
