"""Ross Cameron 的 Gap & Go 小盘动量策略（日线近似实现）。

其原策略是美股小盘股日内动量交易（Warrior Trading 公开资料，详见 docs/strategy_ross_cameron.md）：
盘前扫描高开缺口股（>4%），要求低价（$1-20）、低流通盘（<10-20M）、
相对放量（2 倍以上）与明确催化剂（财报/新闻/公告）；开盘后做突破买入
（盘前高点 / 1 分钟开盘区间突破 / 红翻绿），止损设在启动 K 线低点下方，
动能衰减即离场，日内平仓；单笔风险固定且很小，每日有亏损上限。

本仓库只有 A 股日线数据，故按日线近似：
- 以「跳空高开 + 相对放量 + 价格区间」代理盘前扫描（流通盘/催化剂无数据），
  其中跳空与放量实现为内置扫描条件，价格等附加条件通过 conditions 注入
- 信号日收盘确认，次日开盘买入，高开追价超过上限则放弃
- 以「时间止损 + 区间末强制平仓」代理其日内平仓纪律（A 股 T+1 下
  买入当日不可卖出，持仓自次日起每日收盘检查离场条件）
"""

from typing import List, Optional

import pandas as pd
from yacrs import register

from lib.conditions.base import BaseCondition
from lib.conditions.date_range import DateRangeCondition
from lib.conditions.gap import GapUpCondition
from lib.conditions.relative_volume import RelativeVolumeCondition
from lib.position import BasePosition, FixedCapitalPosition
from lib.trade import Trade

from .base import BaseStrategy


@register()
class RossCameronGapAndGoStrategy(BaseStrategy):
    """
    Ross Cameron "Gap and Go" 策略的日线近似版（说明文档见 docs/strategy_ross_cameron.md）。

    扫描条件（信号日，须在回测区间内）：
    - 跳空高开 ≥ gap_min_pct（默认 4%，其公开扫描标准，GapUpCondition）
    - 相对放量 ≥ rel_vol_min 倍 vol_window 日均量（默认 2 倍，RelativeVolumeCondition）
    - conditions 中附加的市场过滤条件
      （如 PriceRangeCondition 价格区间，对应其 $1-20 低价股偏好）

    入场：信号日次日以开盘价买入（买入量按固定本金仓位计算），
    次日相对信号日收盘价高开超过 max_chase_pct（默认 5%）则放弃（不追高）。

    出场（持仓次日起每日收盘依次检查，全仓卖出；A 股 T+1：买入当日不可卖出）：
    - 收盘跌破买入价 stop_loss_pct：止损
    - 收盘跌破信号日最低价：突破失败，离场
    - 曾盈利后自持仓期最高价回撤 trail_pct：移动止盈（动能衰减）
    - 持仓满 max_hold_days 天：时间止损（其日内平仓的日线代理）
    - 回测区间结束：强制平仓

    买入信号附带倾向评分（Trade.score，见 docs/strategy_ross_cameron.md##买入倾向排序）：
    账户层（lib/account.py）在资金/持仓约束下按评分降序决定哪些信号成交。
    """

    name = "Ross Cameron Gap&Go"
    description = ("小盘动量：高开缺口+相对放量扫描，次日开盘突破买入，"
                   "止损/移动止盈/时间止损离场（Ross Cameron 日线近似）")

    BUY_REASONS = {
        "gap_go": "跳空高开+相对放量，次日开盘突破买入（Gap and Go）",
    }
    SELL_REASONS = {
        "stop_loss": "收盘跌破止损位，强制离场",
        "failed_breakout": "跌回信号日低点下方，突破失败",
        "trailing_stop": "盈利后自高点回撤，动能衰减止盈",
        "time_stop": "持仓超过时间上限，强制离场",
        "end_of_backtest": "回测区间结束，强制平仓",
    }

    def __init__(self, gap_min_pct: float = 0.04, rel_vol_min: float = 2.0,
                 vol_window: int = 20,
                 max_chase_pct: float = 0.05,
                 stop_loss_pct: float = 0.05, trail_pct: float = 0.05,
                 max_hold_days: int = 5,
                 conditions: Optional[List[BaseCondition]] = None,
                 position: Optional[BasePosition] = None) -> None:
        super().__init__()
        self.gap_min_pct = gap_min_pct
        self.rel_vol_min = rel_vol_min
        self.vol_window = vol_window
        self.max_chase_pct = max_chase_pct
        self.stop_loss_pct = stop_loss_pct
        self.trail_pct = trail_pct
        self.max_hold_days = max_hold_days
        self.conditions = list(conditions) if conditions else []
        self.position = position if position is not None else FixedCapitalPosition()

    def run(self, df: pd.DataFrame, start_time, end_time) -> List[Trade]:
        end_ts = pd.Timestamp(end_time)

        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            return []

        close = df["close"]
        open_ = df["open"]
        low = df["low"]
        volume = df["volume"]
        # 相对放量（与 RelativeVolumeCondition 同口径：当日量 / 过去 vol_window 日均量，不含当日）
        prev_avg_volume = volume.shift(1).rolling(self.vol_window).mean()
        rel_volume = volume / prev_avg_volume

        # 最低买入要求（docs/strategy_ross_cameron.md##买入）：日期区间 + 跳空高开 +
        # 相对放量 + 全部附加条件取交集；每个条件都在完整 df 上计算，
        # 保证依赖历史数据的条件（如均量）不受影响
        conditions = [DateRangeCondition(start_time, end_time),
                      GapUpCondition(self.gap_min_pct),
                      RelativeVolumeCondition(self.rel_vol_min, self.vol_window),
                      *self.conditions]
        signal_dates = set(df["date"])
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])

        ticker = df["ticker"].iloc[0]
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        trades: List[Trade] = []
        position_open = False
        entry_i = -1  # 买入日下标（信号日次日，开盘价成交），当日不可卖出（T+1）

        for i in range(len(df)):
            date = df["date"].iloc[i]
            price = close.iloc[i]

            if not position_open:
                # 信号日须满足全部扫描条件，且次日有 K 线可入场
                if i + 1 >= len(df) or date not in signal_dates:
                    continue

                next_open = open_.iloc[i + 1]
                entry_date = df["date"].iloc[i + 1]
                if pd.isna(next_open):
                    continue
                entry_price = float(next_open)
                # 买入点条件（docs/strategy_ross_cameron.md##买入）：次日开盘突破买入；
                # 负高开（低开）视为回调允许入场，只限制向上追高
                chase_pct = entry_price / float(price) - 1
                if entry_date > end_ts or chase_pct > self.max_chase_pct:
                    continue

                # 买入量（docs/strategy_ross_cameron.md##买入）：固定本金，不足一手放弃
                amount = self.position.buy_amount(entry_price)
                if amount <= 0:
                    continue

                # 买入倾向评分（docs/strategy_ross_cameron.md##买入倾向排序）：信号日
                # 跳空幅度 × 相对放量倍数，缺口越大、放量越足评分越高
                # （账户层按评分配资金/仓位）
                prev_close = float(close.iloc[i - 1]) if i > 0 else 0.0
                gap_pct = (float(open_.iloc[i]) / prev_close - 1) if prev_close else 0.0
                rv = rel_volume.iloc[i]
                score = gap_pct * (float(rv) if not pd.isna(rv) else 0.0)

                trades.append(Trade(
                    ticker=ticker,
                    name=name,
                    buy_time=entry_date.strftime("%Y-%m-%d"),
                    buy_amount=amount,
                    buy_price=entry_price,
                    buy_reason=self.BUY_REASONS["gap_go"],
                    score=score,
                ))
                position_open = True
                entry_i = i + 1
                signal_low = float(low.iloc[i])
                peak_price = entry_price
                hold_days = 1
            else:
                trade = trades[-1]
                if i <= entry_i:
                    continue  # 买入当日不可卖出（T+1），自次日起检查卖出
                hold_days += 1
                peak_price = max(peak_price, float(price))

                # 卖出条件（docs/strategy_ross_cameron.md##卖出）：按优先级依次检查
                sell_reason = None
                if price < trade.buy_price * (1 - self.stop_loss_pct):
                    sell_reason = self.SELL_REASONS["stop_loss"]
                elif price < signal_low:
                    sell_reason = self.SELL_REASONS["failed_breakout"]
                elif (peak_price > trade.buy_price
                        and price < peak_price * (1 - self.trail_pct)):
                    sell_reason = self.SELL_REASONS["trailing_stop"]
                elif hold_days >= self.max_hold_days:
                    sell_reason = self.SELL_REASONS["time_stop"]
                elif date >= end_ts or i == len(df) - 1:
                    sell_reason = self.SELL_REASONS["end_of_backtest"]

                if sell_reason is not None:
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    # 卖出量（docs/strategy_ross_cameron.md##卖出）：全仓卖出
                    trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason
                    position_open = False

        return trades
