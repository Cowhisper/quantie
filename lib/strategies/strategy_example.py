"""示例策略：均线金叉买入，死叉 / 止损 / 区间结束卖出。"""

from typing import List

import pandas as pd

from ..trade import Trade
from .base import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):
    """
    示例策略：
    - 5 日均线上穿 20 日均线（金叉）时以当日收盘价买入
    - 5 日均线下穿 20 日均线（死叉）时卖出
    - 持仓期间跌破买入价 8% 止损
    - 回测区间结束时强制平仓
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
                 stop_loss_pct: float = 0.08) -> None:
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.stop_loss_pct = stop_loss_pct

    def run(self, df: pd.DataFrame, start_time, end_time) -> List[Trade]:
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)

        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            return []

        close = df["close"]
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        ticker = df["ticker"].iloc[0]
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        trades: List[Trade] = []
        position_open = False

        for i in range(len(df)):
            date = df["date"].iloc[i]
            price = close.iloc[i]
            in_range = start_ts <= date <= end_ts

            if not position_open:
                if in_range and bool(golden_cross.iloc[i]):
                    trades.append(Trade(
                        ticker=ticker,
                        name=name,
                        buy_time=date.strftime("%Y-%m-%d"),
                        buy_amount=100,  # 示例：固定买入100股
                        buy_price=float(price),
                        buy_reason=self.BUY_REASONS["golden_cross"],
                    ))
                    position_open = True
            else:
                trade = trades[-1]
                sell_reason = None
                if bool(death_cross.iloc[i]):
                    sell_reason = self.SELL_REASONS["death_cross"]
                elif price < trade.buy_price * (1 - self.stop_loss_pct):
                    sell_reason = self.SELL_REASONS["stop_loss"]
                elif date >= end_ts or i == len(df) - 1:
                    sell_reason = self.SELL_REASONS["end_of_backtest"]

                if sell_reason is not None:
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    trade.sell_amount = trade.buy_amount
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason
                    position_open = False

        return trades
