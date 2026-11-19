"""Larry Connors 的 RSI(2) 超卖均值回归策略（日线版）。

出自 Connors & Alvarez《Short Term Trading Strategies That Work》(2008)
（Warrior Trading 创始人同宗文献体系，公开资料详见 docs/strategy_connors_rsi2.md）：
长期上涨形态中的短期深度回调会均值回归，RSI(2) 是捕捉这类极端超卖的
灵敏指标——原文对指数 ETF 回测胜率 65%~77%（SPY 1995-2008 约 77%），
属于文献中胜率最高的日线策略之一。

原规则：
- 买入：收盘价高于 200 日均线（长期上升趋势），且 RSI(2) < 10；
  RSI(2) < 5 时原文加仓第二笔（本仓位的 FixedCapitalPosition 单笔固定
  本金无法表达加仓，故只取首笔，见 docs/strategy_connors_rsi2.md##买入量）
- 卖出：收盘价站上 5 日均线，反弹目标即达到；原文不设止损
  （200 日均线过滤器即风控），持仓均值回归通常在数日内完成

本仓库的 A 股日线近似（与 strategy_mac.py 同一套无未来函数约定）：
- 信号日收盘确认（RSI(2) 超卖 + 200 日均线上方），次日开盘买入；
  原版按信号日收盘价成交，此处改为次日开盘，避免"当日收盘算信号、
  当日收盘成交"的前视偏差（A 股 T+1 下买入当日也不可卖出）
- 次日高开超过 max_chase_pct（默认 5%）放弃——超卖的隔夜反弹空间
  已被兑现，追高则盈亏比恶化
- 增加收盘止损 stop_loss_pct（默认 10%）与时间止损 max_hold_days
  （默认 10 天）：原文对象是指数 ETF，无退市风险；A 股个股可能连续
  阴跌不触发"站上 5 日线"，需要显式兜底（见 docs/strategy_connors_rsi2.md##卖出）
"""

from typing import List, Optional, Tuple

import pandas as pd
from yacrs import register

from lib.conditions.base import BaseCondition
from lib.conditions.date_range import DateRangeCondition
from lib.position import BasePosition, FixedCapitalPosition
from lib.trade import Trade

from .base import BaseStrategy


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI：涨跌额经 alpha=1/period 的指数平滑（ewm adjust=False，
    与 ta-lib/通达信 RSI 在预热期后基本一致），RS = 均涨/均跌。
    无下跌（均跌=0）时 RSI=100；无涨无跌时为 NaN（不产生信号）。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)


@register()
class ConnorsRSI2ReversionStrategy(BaseStrategy):
    """
    Larry Connors RSI(2) 超卖均值回归策略（说明文档见 docs/strategy_connors_rsi2.md）。

    扫描条件（信号日，须在回测区间内）：
    - 收盘价高于长期均线 trend_window（默认 200 日，原文规则：上升趋势中
      的回调才均值回归；下跌趋势中的超卖多是基本面恶化，继续阴跌概率大）
    - RSI(rsi_window=2) < rsi_buy（默认 10，原文规则；RSI(2) 对单日
      极端波动极灵敏，<10 表示近 2 日几乎单边下跌）

    入场：信号日次日以开盘价买入（买入量按固定本金仓位计算），
    次日高开超过 max_chase_pct（默认 5%）放弃——超卖反弹隔夜已兑现。

    出场（持仓次日起每日收盘依次检查，全仓卖出；A 股 T+1：买入当日不可卖出）：
    - 收盘站上 exit_ma_window 日均线（默认 5 日）：反弹目标达到，离场
    - 收盘跌破买入价 stop_loss_pct（默认 10%）：止损（A 股个股风控兜底）
    - 持仓满 max_hold_days 天（默认 10 天）：时间止损
    - 回测区间结束：强制平仓

    买入信号附带倾向评分（Trade.score，见 docs/strategy_connors_rsi2.md##买入倾向排序）：
    账户层（lib/account.py）在资金/持仓约束下按评分降序决定哪些信号成交。
    """

    name = "Connors RSI(2) 超卖回归"
    description = ("上升趋势中 RSI(2) 深度超卖买入，收盘站上 5 日均线离场，"
                   "止损/时间止损兜底（Larry Connors 日线版）")

    BUY_REASONS = {
        "rsi2_oversold": "RSI(2) 超卖且处长期均线上方，次日开盘买入（Connors 均值回归）",
    }
    SELL_REASONS = {
        "ma_exit": "收盘站上 5 日均线，反弹目标达到",
        "stop_loss": "收盘跌破止损位，强制离场",
        "time_stop": "持仓超过时间上限，强制离场",
        "end_of_backtest": "回测区间结束，强制平仓",
    }

    def __init__(self, rsi_window: int = 2, rsi_buy: float = 10.0,
                 trend_window: int = 200, exit_ma_window: int = 5,
                 max_chase_pct: float = 0.05, stop_loss_pct: float = 0.10,
                 max_hold_days: int = 10,
                 conditions: Optional[List[BaseCondition]] = None,
                 position: Optional[BasePosition] = None) -> None:
        super().__init__()
        self.rsi_window = rsi_window
        self.rsi_buy = rsi_buy
        self.trend_window = trend_window
        self.exit_ma_window = exit_ma_window
        self.max_chase_pct = max_chase_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self.conditions = list(conditions) if conditions else []
        self.position = position if position is not None else FixedCapitalPosition()

    def _signal_dates(self, df: pd.DataFrame, start_time, end_time
                      ) -> Tuple[set, List[Tuple[str, int]]]:
        """信号日集合（docs/strategy_connors_rsi2.md##买入）：日期区间 + 全部
        附加条件取交集，再过长期均线/RSI(2) 超卖过滤；每个条件都在完整 df 上
        计算，保证依赖历史数据的条件不受影响。同时返回逐阶段剩余行数
        [(阶段名, 剩余行数), ...]，供 forecast 在 0 信号时拆解原因。"""
        conditions = [DateRangeCondition(start_time, end_time),
                      *self.conditions]
        signal_dates = set(df["date"])
        stages: List[Tuple[str, int]] = [("候选K线", len(df))]
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])
            stages.append((cond.description, len(signal_dates)))

        close = df["close"]
        trend_ma = close.rolling(self.trend_window).mean()
        rsi = _rsi(close, self.rsi_window)
        filters: List[Tuple[str, pd.Series]] = [
            (f"收盘价高于 MA{self.trend_window}（长期上升趋势）",
             close > trend_ma),
            (f"RSI({self.rsi_window}) < {self.rsi_buy:g}（深度超卖）",
             rsi < self.rsi_buy),
        ]
        for label, mask in filters:
            signal_dates &= set(df.loc[mask, "date"])
            stages.append((label, len(signal_dates)))
        return signal_dates, stages

    def signal_day_breakdown(self, df: pd.DataFrame, start_time, end_time
                             ) -> List[Tuple[str, int]]:
        """诊断接口（forecast 0 信号时调用）：返回逐阶段剩余信号日行数，
        阶段与 run() 的信号判定完全一致（每阶段均在完整 df 上计算后取交集）。"""
        _, stages = self._signal_dates(df, start_time, end_time)
        return stages

    def run(self, df: pd.DataFrame, start_time, end_time) -> List[Trade]:
        end_ts = pd.Timestamp(end_time)

        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            return []

        close = df["close"]
        open_ = df["open"]
        exit_ma = close.rolling(self.exit_ma_window).mean()
        rsi = _rsi(close, self.rsi_window)

        signal_dates, _ = self._signal_dates(df, start_time, end_time)

        ticker = df["ticker"].iloc[0]
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        trades: List[Trade] = []
        position_open = False
        entry_i = -1  # 买入日下标（信号日次日，开盘价成交），当日不可卖出（T+1）

        for i in range(len(df)):
            date = df["date"].iloc[i]
            price = close.iloc[i]

            if not position_open:
                # 买入点条件（docs/strategy_connors_rsi2.md##买入）：长期均线上方
                # + RSI(2) 深度超卖，当日收盘确认，次日开盘前决策、以次日开盘价成交
                if i + 1 >= len(df) or date not in signal_dates:
                    continue

                entry_i = i + 1
                entry_date = df["date"].iloc[entry_i]
                next_open = open_.iloc[entry_i]
                if pd.isna(next_open):
                    continue
                entry_price = float(next_open)
                # 追价保护（docs/strategy_connors_rsi2.md##买入）：次日高开超过
                # max_chase_pct 说明超卖反弹已在隔夜兑现，盈亏比恶化，不追
                chase_pct = entry_price / float(price) - 1
                if entry_date > end_ts or chase_pct > self.max_chase_pct:
                    continue

                # 买入量（docs/strategy_connors_rsi2.md##买入）：固定本金，不足一手放弃
                amount = self.position.buy_amount(entry_price)
                if amount <= 0:
                    continue

                # 买入倾向评分（docs/strategy_connors_rsi2.md##买入倾向排序）：超卖深度
                # 归一化 = (rsi_buy − RSI2) / rsi_buy。Connors 原文 RSI(2) 越低
                # 预期收益越高（原文 <5 加仓规则的依据），故越深度超卖评分越高；
                # 账户层按评分配资金/仓位
                r = float(rsi.iloc[i])
                score = (self.rsi_buy - r) / self.rsi_buy

                trades.append(Trade(
                    ticker=ticker,
                    name=name,
                    buy_time=entry_date.strftime("%Y-%m-%d"),
                    buy_amount=amount,
                    buy_price=entry_price,
                    buy_reason=self.BUY_REASONS["rsi2_oversold"],
                    score=score,
                ))
                position_open = True
                hold_days = 1
            else:
                if i <= entry_i:
                    continue  # 买入当日不可卖出（T+1），自次日起检查卖出
                hold_days += 1
                trade = trades[-1]

                # 卖出条件（docs/strategy_connors_rsi2.md##卖出）：按优先级依次检查，
                # 当日收盘确认后次日开盘卖出（无未来函数；回测区间末无次日数据则
                # 按当日收盘强制平仓）
                sell_reason = None
                if price < trade.buy_price * (1 - self.stop_loss_pct):
                    sell_reason = self.SELL_REASONS["stop_loss"]
                elif price > exit_ma.iloc[i]:
                    sell_reason = self.SELL_REASONS["ma_exit"]
                elif hold_days >= self.max_hold_days:
                    sell_reason = self.SELL_REASONS["time_stop"]

                if sell_reason is not None:
                    if i + 1 < len(df) and df["date"].iloc[i + 1] <= end_ts:
                        trade.sell_time = df["date"].iloc[i + 1].strftime("%Y-%m-%d")
                        # 卖出量（docs/strategy_connors_rsi2.md##卖出）：全仓卖出
                        trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                        trade.sell_price = float(open_.iloc[i + 1])
                        trade.sell_reason = sell_reason
                        position_open = False
                        continue
                    sell_reason = self.SELL_REASONS["end_of_backtest"]
                if date >= end_ts or i == len(df) - 1:
                    # 回测区间结束 / 最后一条数据：按当日收盘强制平仓
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    # 卖出量（docs/strategy_connors_rsi2.md##卖出）：全仓卖出
                    trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason or self.SELL_REASONS["end_of_backtest"]
                    position_open = False

        return trades
