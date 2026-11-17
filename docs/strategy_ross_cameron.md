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
策略逐股扫描盘前信号（日线近似），不感知账户状态；所有股票的买入信号汇总到账户层
（`lib/account.py` 的 `simulate_account`），在资金与持仓上限（`engine.max_positions`，
默认 10 只）约束下按评分降序成交——评分高（最可能赚钱）的信号优先获得资金与仓位，
评分相同按股票代码排序保证确定性。
- 排序公式：`score` = 三个特征分桶胜率映射的等权平均（标定自 turn/涨幅过滤后的
  全量信号统计，优化目标以胜率优先）：
  - 缺口 = 信号日开盘价 / 前收盘价 − 1：≤6%→0.419，≤9%→0.386，
    ≤15%→0.415，>15%→0.389（6~9% 区间最弱）
  - 相对放量 = 信号日成交量 / 过去 `vol_window` 日均量：
    ≤3→0.416，≤5→0.409，≤10→0.392，>10→0.400（2~5 倍温和放量最好）
  - 股价 = 信号日收盘价：<5 元→0.472，<10 元→0.404，<20 元→0.400，
    <50 元→0.390，≥50 元→0.403（低价小盘最好，对应其 $1-20 低价股偏好）
  旧公式（缺口 × 放量）与收益零/负相关（rank corr ≈ −0.03），已废弃。
  统计依据：过滤后信号 4073 笔，股价<5 元胜率 47.2% vs 20~50 元 39.0%。
  写入 `Trade.score`。
- 超参注释：`gap_min_pct`=0.04（跳空高开下限，公开标准 >4%），
  `rel_vol_min`=2.0（相对放量倍数），`vol_window`=20（均量窗口，不含当日），
  `max_turn_pct`=15（信号日换手率上限 %，高换手的拥挤交易次日退潮概率大：
  全量信号统计 turn>20% 的次日止损率 26%，turn≤3% 仅 9%），
  `max_signal_pct`=9（信号日涨幅上限 %，不追已涨停/接近涨停的延续：
  涨幅 9%+ 的信号次日止损率约 26%，6~9% 区间胜率最高 43.3%），
  `min_close_pos`=0.2（信号日收盘位于当日振幅的位置下限：高开但收在
  低位的信号日=全天抛压，次日延续下跌——收在振幅最低 20% 的信号次日
  51% 跌破信号日低点，收在最高 20% 仅 16%），
  `max_chase_pct`=0.05（次日高开追价上限）
- 最低买入要求：信号日同时满足——跳空高开 ≥ `gap_min_pct`、
  成交量 ≥ `rel_vol_min` 倍 `vol_window` 日均量、
  换手率 ≤ `max_turn_pct`、涨幅 ≤ `max_signal_pct`、
  收盘价位于当日振幅上半部（(close−low)/(high−low) ≥ `min_close_pos`，默认 0.2：
  高开但收在低位的信号日=全天抛压，次日延续下跌概率大——收在振幅最低 20% 的
  信号次日 51% 跌破信号日低点，收在最高 20% 仅 16%）、
  收盘价命中全部附加市场过滤条件（`conditions`，如价格区间，对应其 $1-20 低价股偏好）、
  信号日在回测区间内；
  次日开盘价相对信号日收盘价高开超过 `max_chase_pct` 放弃（不追高），
  固定本金按次日开盘价能买入至少 1 手；低于任一要求不触发买入。
  满足要求的信号也不保证成交：当日资金或持仓不足时，低分信号被跳过（整笔作废）。

### 买入量
固定本金仓位（`lib/position.py` 的 `FixedCapitalPosition`）：
买入股数 = `capital` // 次日开盘价，向下取整到整手（`lot_size`，默认 100 股/手）；
不足 1 手放弃本次买入。本金与手数可在配置段 `FixedCapitalPosition` 调整。

## 卖出

### 卖出条件
持仓次日起每日收盘依次检查（A 股 T+1：买入当日不可卖出），满足任一条件即卖出：
- 止损：收盘价跌破买入价的 (1 - `stop_loss_pct`)（默认 10%）
- 突破失败：收盘价跌破信号日最低价（其原方法论止损位于启动 K 线低点下方；
  仅持仓前 `breakout_exit_days` 个交易日检查，默认 1）
- 移动止盈：曾盈利（持仓期最高价 > 买入价）后，收盘价自最高价回撤超过 `trail_pct`（默认 5%），动能衰减离场
- 时间止损：持仓满 `max_hold_days` 天（默认 5 天，代理其日内平仓纪律）
- 回测区间结束：日期到达 `end_time` 或最后一条数据，强制平仓

### 卖出量
全部卖出（卖出数量 = 买入数量，见 `FixedCapitalPosition.sell_amount`）。
