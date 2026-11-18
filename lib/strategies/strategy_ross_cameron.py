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

from typing import List, Optional, Tuple

import pandas as pd
from yacrs import register

from lib.conditions.base import BaseCondition
from lib.conditions.date_range import DateRangeCondition
from lib.conditions.gap import GapUpCondition
from lib.conditions.relative_volume import RelativeVolumeCondition
from lib.position import BasePosition, FixedCapitalPosition
from lib.trade import Trade

from .base import BaseStrategy


def _gap_win(gap_pct: float) -> float:
    """缺口幅度 -> 信号统计标定的分桶胜率（全过滤条件后的全量信号回测）。"""
    if gap_pct <= 0.06:
        return 0.465
    if gap_pct <= 0.09:
        return 0.404
    if gap_pct <= 0.15:
        return 0.417
    return 0.417


def _rv_win(rel_vol: float) -> float:
    """相对放量倍数 -> 分桶胜率：2~3 倍温和放量最好，爆量最差。"""
    if rel_vol <= 3.0:
        return 0.458
    if rel_vol <= 5.0:
        return 0.440
    if rel_vol <= 10.0:
        return 0.454
    return 0.400


def _price_win(price: float) -> float:
    """信号日收盘价 -> 分桶胜率：低价小盘最好（其原方法论 $1-20 低价股偏好）。"""
    if price < 5.0:
        return 0.490
    if price < 10.0:
        return 0.441
    if price < 20.0:
        return 0.461
    if price < 50.0:
        return 0.439
    return 0.413


@register()
class RossCameronGapAndGoStrategy(BaseStrategy):
    """
    Ross Cameron "Gap and Go" 策略的日线近似版（说明文档见 docs/strategy_ross_cameron.md）。

    扫描条件（信号日，须在回测区间内）：
    - 跳空高开 ≥ gap_min_pct（默认 4%，其公开扫描标准，GapUpCondition）
    - 相对放量 ≥ rel_vol_min 倍 vol_window 日均量（默认 2 倍，RelativeVolumeCondition）
    - 换手率 ≤ max_turn_pct（默认 15%）：高换手=拥挤交易，次日退潮概率大
      （信号统计：turn>20% 次日止损率 26% vs turn≤3% 仅 9%）
    - 信号日涨幅 ≤ max_signal_pct（默认 9%）：不追已涨停/接近涨停的延续，
      动能透支后次日回归（涨幅 9%+ 的信号次日止损率 ~26%）
    - 信号日收盘价位于当日振幅上半部（(close−low)/(high−low) ≥ min_close_pos，
      默认 0.2）：收在低位=高开全天抛压，次日延续下跌概率大（统计：收在振幅
      最低 20% 的信号次日 51% 跌破信号日低点，收在最高 20% 仅 16%）
    - conditions 中附加的市场过滤条件
      （如 PriceRangeCondition 价格区间，对应其 $1-20 低价股偏好）

    入场：信号日次日以开盘价买入（买入量按固定本金仓位计算），
    次日相对信号日收盘价高开超过 max_chase_pct（默认 5%）则放弃（不追高），
    低开超过 min_chase_pct（默认 -3%）同样放弃（缺口开盘即退潮，动能证伪）。

    出场（持仓次日起每日收盘依次检查，全仓卖出；A 股 T+1：买入当日不可卖出）：
    - 收盘跌破买入价 stop_loss_pct：止损
    - 收盘跌破信号日最低价：突破失败，离场（仅持仓前 breakout_exit_days 个交易日
      检查，默认 1：跌破后该参考线失去意义，之后交给止损/移动止盈/时间止损）
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
                 max_turn_pct: float = 15.0, max_signal_pct: float = 9.0,
                 min_close_pos: float = 0.2,
                 min_chase_pct: float = -0.03, max_chase_pct: float = 0.05,
                 stop_loss_pct: float = 0.10, trail_pct: float = 0.05,
                 breakout_exit_days: int = 1,
                 max_hold_days: int = 5,
                 conditions: Optional[List[BaseCondition]] = None,
                 position: Optional[BasePosition] = None) -> None:
        super().__init__()
        self.gap_min_pct = gap_min_pct
        self.rel_vol_min = rel_vol_min
        self.vol_window = vol_window
        self.max_turn_pct = float(max_turn_pct)
        self.max_signal_pct = float(max_signal_pct)
        self.min_close_pos = float(min_close_pos)
        self.min_chase_pct = float(min_chase_pct)
        self.max_chase_pct = max_chase_pct
        self.stop_loss_pct = stop_loss_pct
        self.trail_pct = trail_pct
        self.breakout_exit_days = int(breakout_exit_days)
        self.max_hold_days = max_hold_days
        self.conditions = list(conditions) if conditions else []
        self.position = position if position is not None else FixedCapitalPosition()

    def _signal_dates(self, df: pd.DataFrame, start_time, end_time
                      ) -> Tuple[set, List[Tuple[str, int]]]:
        """信号日集合（docs/strategy_ross_cameron.md##买入）：日期区间 + 跳空高开 +
        相对放量 + 全部附加条件取交集，再过换手率/涨幅/收盘位置过滤；每个条件
        都在完整 df 上计算，保证依赖历史数据的条件（如均量）不受影响。
        同时返回逐阶段剩余行数 [(阶段名, 剩余行数), ...]，供 forecast 在
        0 信号时拆解原因（阶段顺序即过滤顺序，逐阶段取日期交集）。"""
        conditions = [DateRangeCondition(start_time, end_time),
                      GapUpCondition(self.gap_min_pct),
                      RelativeVolumeCondition(self.rel_vol_min, self.vol_window),
                      *self.conditions]
        signal_dates = set(df["date"])
        # 换手率/涨幅过滤（统计标定）：高换手的拥挤交易、涨停延续的动能透支
        # 都会显著提高次日失败概率；turn/pctChg 缺失（NaN）的行不满足过滤
        filters: List[Tuple[str, pd.Series]] = [
            (f"换手≤{self.max_turn_pct:g}% 且 涨幅≤{self.max_signal_pct:g}%",
             (df["turn"] <= self.max_turn_pct)
             & (df["pctChg"] <= self.max_signal_pct)),
        ]
        bar_range = df["high"] - df["low"]
        close_pos = (df["close"] - df["low"]) / bar_range.where(bar_range > 0)
        # 收盘位置过滤：收在当日振幅低位的全天抛压，次日延续下跌概率大；
        # 一字零振幅（high=low）时 close_pos 为 NaN，不满足过滤
        filters.append((f"收盘位于振幅上部≥{self.min_close_pos:g}",
                        close_pos >= self.min_close_pos))

        stages: List[Tuple[str, int]] = [("候选K线", len(df))]
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])
            stages.append((cond.description, len(signal_dates)))
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
        low = df["low"]
        # 相对放量（与 RelativeVolumeCondition 同口径：当日量 / 过去 vol_window 日均量，不含当日）
        prev_avg_volume = df["volume"].shift(1).rolling(self.vol_window).mean()
        rel_volume = df["volume"] / prev_avg_volume

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
                # 信号日须满足全部扫描条件，且次日有 K 线可入场
                if i + 1 >= len(df) or date not in signal_dates:
                    continue

                next_open = open_.iloc[i + 1]
                entry_date = df["date"].iloc[i + 1]
                if pd.isna(next_open):
                    continue
                entry_price = float(next_open)
                # 买入点条件（docs/strategy_ross_cameron.md##买入）：次日开盘突破买入；
                # 高开超过 max_chase_pct 不追；低开超过 min_chase_pct（默认 -3%）说明
                # 缺口已在开盘退潮、动能证伪，放弃（统计标定：低开>3% 的入场次日
                # 失败率显著更高）
                chase_pct = entry_price / float(price) - 1
                if (entry_date > end_ts or chase_pct > self.max_chase_pct
                        or chase_pct < self.min_chase_pct):
                    continue

                # 买入量（docs/strategy_ross_cameron.md##买入）：固定本金，不足一手放弃
                amount = self.position.buy_amount(entry_price)
                if amount <= 0:
                    continue

                # 买入倾向评分（docs/strategy_ross_cameron.md##买入倾向排序）：缺口 /
                # 相对放量 / 股价三个特征的分桶胜率映射等权平均（标定自 turn/涨幅
                # 过滤后的全量信号统计，以胜率为目标）——温和缺口 + 温和放量 + 低价
                # 优先；旧公式（缺口 × 放量）与收益零/负相关，已废弃。
                # 账户层按评分配资金/仓位
                prev_close = float(close.iloc[i - 1]) if i > 0 else 0.0
                gap_pct = (float(open_.iloc[i]) / prev_close - 1) if prev_close else 0.0
                rv = rel_volume.iloc[i]
                rv_v = float(rv) if not pd.isna(rv) else 0.0
                score = (_gap_win(gap_pct) + _rv_win(rv_v)
                         + _price_win(float(price))) / 3.0

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
                elif (price < signal_low
                      and hold_days - 1 <= self.breakout_exit_days):
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
