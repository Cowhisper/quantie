# 关键词
- Ross Cameron
- Gap and Go
- 小盘动量 (Small-cap Momentum)
- 跳空缺口 (Gap Up)
- 相对放量 (Relative Volume, RVOL)
- 固定本金仓位 (Fixed Capital Position)

# 参考文献
- https://www.warriortrading.com/gap-go/
- https://kriminiltrading.com/blogs/must-read-economic-market-books/how-to-day-trade-the-plain-truth-by-ross-cameron-book-summary-review
- https://scalpeltrade.com/guides/momentum-method

# 策略描述

## 买入

### 买入倾向排序
单票独立回测，无跨股票排序；每只股票按日期顺序独立扫描盘前信号（日线近似）。
- 排序公式：不适用（逐票扫描；其原方法论为盘前扫描器排序，见参考文献）
- 超参注释：`gap_min_pct`=0.04（跳空高开下限，公开标准 >4%），
  `rel_vol_min`=2.0（相对放量倍数），`vol_window`=20（均量窗口，不含当日），
  `max_chase_pct`=0.05（次日高开追价上限）
- 最低买入要求：信号日同时满足——跳空高开 ≥ `gap_min_pct`、
  成交量 ≥ `rel_vol_min` 倍 `vol_window` 日均量、
  收盘价命中全部附加市场过滤条件（`conditions`，如价格区间，对应其 $1-20 低价股偏好）、
  信号日在回测区间内；
  次日开盘价相对信号日收盘价高开超过 `max_chase_pct` 放弃（不追高），
  固定本金按次日开盘价能买入至少 1 手；低于任一要求不触发买入。

### 买入量
固定本金仓位（`lib/position.py` 的 `FixedCapitalPosition`）：
买入股数 = `capital` // 次日开盘价，向下取整到整手（`lot_size`，默认 100 股/手）；
不足 1 手放弃本次买入。本金与手数可在配置段 `FixedCapitalPosition` 调整。

## 卖出

### 卖出条件
持仓期间每日收盘依次检查，满足任一条件即卖出：
- 止损：收盘价跌破买入价的 (1 - `stop_loss_pct`)（默认 5%）
- 突破失败：收盘价跌破信号日最低价（其原方法论止损位于启动 K 线低点下方）
- 移动止盈：曾盈利（持仓期最高价 > 买入价）后，收盘价自最高价回撤超过 `trail_pct`（默认 5%），动能衰减离场
- 时间止损：持仓满 `max_hold_days` 天（默认 5 天，代理其日内平仓纪律）
- 回测区间结束：日期到达 `end_time` 或最后一条数据，强制平仓

### 卖出量
全部卖出（卖出数量 = 买入数量，见 `FixedCapitalPosition.sell_amount`）。
