"""示例策略：均线金叉买入，死叉 / 止损 / 区间结束卖出。"""

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
    - 5 日均线上穿 20 日均线（金叉）时买入，买入量按固定本金仓位计算
    - 5 日均线下穿 20 日均线（死叉）时卖出
    - 持仓期间跌破买入价 8% 止损
    - 回测区间结束时强制平仓
    - 卖出默认全仓卖出

    买入信号需满足 conditions 中附加的市场过滤条件
    （如 PriceRangeCondition 价格区间）；日期区间由 run 的回测窗口自动附加。
    """

    name = "均线金叉策略"
    description = "5日均线上穿20日均线买入，死叉/止损/区间结束卖出"

    BUY_REASONS = {
        "golden_cross": "5日均线上穿20日均线（金叉）",
    }
    SELL_REASONS = {
        "death_cross": "5日均线下穿20日均线（死叉）",
        "stop_loss": "跌破买入价8%，止损",
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
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # 最低买入要求（docs/strategy_mac.md##买入）：日期区间 + 全部附加条件取交集，
        # 每个条件都在完整 df 上计算，保证依赖历史数据的条件（如均量）不受影响
        conditions = [DateRangeCondition(start_time, end_time), *self.conditions]
        signal_dates = set(df["date"])
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])

        ticker = df["ticker"].iloc[0]
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        trades: List[Trade] = []
        position_open = False

        for i in range(len(df)):
            date = df["date"].iloc[i]
            price = close.iloc[i]

            if not position_open:
                # 买入点条件（docs/strategy_mac.md##买入）：5 日均线上穿 20 日均线（金叉）
                if bool(golden_cross.iloc[i]) and date in signal_dates:
                    # 买入量（docs/strategy_mac.md##买入）：固定本金，不足一手放弃
                    amount = self.position.buy_amount(float(price))
                    if amount <= 0:
                        continue
                    trades.append(Trade(
                        ticker=ticker,
                        name=name,
                        buy_time=date.strftime("%Y-%m-%d"),
                        buy_amount=amount,
                        buy_price=float(price),
                        buy_reason=self.BUY_REASONS["golden_cross"],
                    ))
                    position_open = True
            else:
                trade = trades[-1]
                # 卖出条件（docs/strategy_mac.md##卖出）：死叉 / 止损 / 区间结束，任一命中
                sell_reason = None
                if bool(death_cross.iloc[i]):
                    sell_reason = self.SELL_REASONS["death_cross"]
                elif price < trade.buy_price * (1 - self.stop_loss_pct):
                    sell_reason = self.SELL_REASONS["stop_loss"]
                elif date >= end_ts or i == len(df) - 1:
                    sell_reason = self.SELL_REASONS["end_of_backtest"]

                if sell_reason is not None:
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    # 卖出量（docs/strategy_mac.md##卖出）：全仓卖出
                    trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason
                    position_open = False

        return trades
