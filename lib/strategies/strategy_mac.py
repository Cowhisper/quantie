"""示例策略：均线金叉买入，死叉 / 止损 / 区间结束卖出。

信号统一由收盘数据确认、次日开盘成交：金叉 / 死叉 / 止损都在信号日收盘
判定，回测假设次日开盘前完成决策，以次日开盘价买入 / 卖出——避免
"用当日收盘算出的信号按当日收盘价成交"的前视偏差（A 股 T+1 下买入当日
也不可卖出）。
"""

from typing import List, Optional

import pandas as pd
from yacrs import register

from lib.conditions.base import BaseCondition
from lib.conditions.date_range import DateRangeCondition
from lib.position import BasePosition, FixedCapitalPosition
from lib.trade import Trade

from .base import BaseStrategy


@register()
class MovingAverageCrossStrategy(BaseStrategy):
    """
    示例策略（说明文档见 docs/strategy_mac.md）：
    - 5 日均线上穿 20 日均线（金叉）后，次日开盘买入，买入量按固定本金仓位计算
    - 5 日均线下穿 20 日均线（死叉）确认后，次日开盘卖出
    - 收盘跌破买入价 8% 止损，次日开盘卖出
    - 回测区间结束时按当日收盘强制平仓
    - 卖出默认全仓卖出

    买入信号需满足 conditions 中附加的市场过滤条件
    （如 PriceRangeCondition 价格区间）；日期区间由 run 的回测窗口自动附加。

    买入信号附带倾向评分（Trade.score，见 docs/strategy_mac.md##买入倾向排序）：
    账户层（lib/account.py）在资金/持仓约束下按评分降序决定哪些信号成交。
    """

    name = "均线金叉策略"
    description = "5日均线上穿20日均线后次日开盘买入，死叉/止损确认后次日开盘卖出，区间结束强制平仓"

    BUY_REASONS = {
        "golden_cross": "5日均线上穿20日均线（金叉），次日开盘买入",
    }
    SELL_REASONS = {
        "death_cross": "5日均线下穿20日均线（死叉），次日开盘卖出",
        "stop_loss": "收盘跌破买入价8%，次日开盘止损",
        "end_of_backtest": "回测区间结束，强制平仓",
    }

    def __init__(self, fast_window: int = 5, slow_window: int = 20,
                 stop_loss_pct: float = 0.08,
                 conditions: Optional[List[BaseCondition]] = None,
                 position: Optional[BasePosition] = None) -> None:
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.stop_loss_pct = stop_loss_pct
        self.conditions = list(conditions) if conditions else []
        self.position = position if position is not None else FixedCapitalPosition()

    def run(self, df: pd.DataFrame, start_time, end_time) -> List[Trade]:
        end_ts = pd.Timestamp(end_time)

        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            return []

        close = df["close"]
        open_ = df["open"]
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        dates = df["date"]

        # 最低买入要求（docs/strategy_mac.md##买入）：日期区间 + 全部附加条件取交集，
        # 每个条件都在完整 df 上计算，保证依赖历史数据的条件（如均量）不受影响
        conditions = [DateRangeCondition(start_time, end_time), *self.conditions]
        signal_dates = set(dates)
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])

        ticker = df["ticker"].iloc[0]
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        trades: List[Trade] = []
        position_open = False
        entry_i = -1  # 买入日下标（信号日次日，开盘价成交），当日不可卖出（T+1）

        for i in range(len(df)):
            date = dates.iloc[i]
            price = close.iloc[i]

            if not position_open:
                # 买入点条件（docs/strategy_mac.md##买入）：5 日均线上穿 20 日均线（金叉）。
                # 金叉由当日收盘确认，次日开盘前决策、以次日开盘价成交
                if bool(golden_cross.iloc[i]) and date in signal_dates:
                    if i + 1 >= len(df):
                        continue
                    entry_i = i + 1
                    entry_date = dates.iloc[entry_i]
                    next_open = open_.iloc[entry_i]
                    if entry_date > end_ts or pd.isna(next_open):
                        continue
                    entry_price = float(next_open)
                    # 买入量（docs/strategy_mac.md##买入）：固定本金，不足一手放弃
                    amount = self.position.buy_amount(entry_price)
                    if amount <= 0:
                        continue
                    # 买入倾向评分（docs/strategy_mac.md##买入倾向排序）：金叉日快线
                    # 相对慢线的乖离率，上穿动能越强评分越高（账户层按评分配资金/仓位）
                    slow = float(slow_ma.iloc[i])
                    score = ((float(fast_ma.iloc[i]) - slow) / slow) if slow else 0.0
                    trades.append(Trade(
                        ticker=ticker,
                        name=name,
                        buy_time=entry_date.strftime("%Y-%m-%d"),
                        buy_amount=amount,
                        buy_price=entry_price,
                        buy_reason=self.BUY_REASONS["golden_cross"],
                        score=score,
                    ))
                    position_open = True
            else:
                if i <= entry_i:
                    continue  # 买入当日不可卖出（T+1），自次日起检查卖出
                trade = trades[-1]
                # 卖出条件（docs/strategy_mac.md##卖出）：死叉 / 止损，当日收盘确认后
                # 次日开盘卖出；信号日之后无交易数据则按当日收盘强制平仓
                sell_reason = None
                if bool(death_cross.iloc[i]):
                    sell_reason = self.SELL_REASONS["death_cross"]
                elif price < trade.buy_price * (1 - self.stop_loss_pct):
                    sell_reason = self.SELL_REASONS["stop_loss"]

                if sell_reason is not None:
                    if i + 1 < len(df) and dates.iloc[i + 1] <= end_ts:
                        trade.sell_time = dates.iloc[i + 1].strftime("%Y-%m-%d")
                        # 卖出量（docs/strategy_mac.md##卖出）：全仓卖出
                        trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                        trade.sell_price = float(open_.iloc[i + 1])
                        trade.sell_reason = sell_reason
                        position_open = False
                        continue
                    sell_reason = self.SELL_REASONS["end_of_backtest"]
                if date >= end_ts or i == len(df) - 1:
                    # 回测区间结束 / 最后一条数据：按当日收盘强制平仓
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    # 卖出量（docs/strategy_mac.md##卖出）：全仓卖出
                    trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason or self.SELL_REASONS["end_of_backtest"]
                    position_open = False

        return trades
