"""Gianluca Bruni 的亚盘箱体突破策略（Asian Session Box，日线近似实现）。

Gianluca Bruni：意大利西西里出身、现居迪拜的 price action 交易员，
主做外汇与指数（prop firm 账户，Morpheus Education 创始人）。其核心
工具为亚盘箱体（Asian Session Box）：亚洲时段低波动收敛出窄幅箱体，
伦敦开盘流动性涌入、放量突破箱体后跟进；关键区分 accepted expansion
（突破被接受）与 rejected break（假突破跌回箱体）；止损放箱体中点/
对侧，目标按箱体高度投影或固定盈亏比；并按星期几的历史统计调整
风险（99% 机械化："If X, I go in. If Y, I don't go in."）。
公开资料详见 docs/strategy_gianluca.md。

本仓库只有 A 股日线数据，故按日线近似：
- 「亚盘箱体」近似为过去 box_window 个交易日的窄幅横盘区间（低波动
  收敛，Darvas 箱体同源）；「伦敦开盘放量突破」近似为信号日收盘站上
  箱体上沿且相对放量——收盘确认接受、次日开盘成交，避免前视偏差
  （A 股 T+1 下买入当日也不可卖出）
- 「accepted vs rejected」机械化：收盘须收在箱体上沿之外且位于当日
  振幅上半部；买入后跌回箱体内部即判定突破失败离场（仅持仓初期检查）
- 「按星期调整风险」机械化为 skip_weekdays 黑名单（默认空）
- 止损/仓位改为仓库统一约定：固定本金仓位 + 固定百分比止损 + 突破
  失败退出 + 移动止盈 + 固定止盈（度量目标/盈亏比的日线近似）+ 时间止损
"""

from typing import List, Optional, Tuple

import pandas as pd
from yacrs import register

from lib.conditions.base import BaseCondition
from lib.conditions.date_range import DateRangeCondition
from lib.position import BasePosition, FixedCapitalPosition
from lib.trade import Trade

from .base import BaseStrategy


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@register()
class GianlucaAsianBoxStrategy(BaseStrategy):
    """
    Gianluca Bruni "亚盘箱体突破" 策略的日线近似版
    （说明文档见 docs/strategy_gianluca.md）。

    扫描条件（信号日，须在回测区间内）：
    - 箱体：过去 box_window 日（不含当日）最高价/最低价 − 1 ≤
      max_box_width_pct（默认 20 日窗口、宽度 ≤ 15%）——亚盘低波动
      收敛的日线代理，宽箱体没有突破价值
    - 突破：收盘价 > 箱体上沿（盘中插破不算，须收盘确认 =
      accepted expansion）
    - 放量：当日量 ≥ vol_window（默认 20）日均量 × breakout_vol_mult
      （默认 1.5）——伦敦开盘流动性涌入的代理，无量突破多为假突破
    - 接受：收盘位于当日振幅上半部（(close−low)/(high−low) ≥
      min_close_pos，默认 0.5）——收在下半部 = rejected break 的
      抛压特征；一字零振幅不产生信号
    - 质量过滤（仓库实证沿用）：换手率 ≤ max_turn_pct（默认 15%，
      高换手=拥挤交易）、涨幅 ≤ max_signal_pct（默认 9%，不追涨停
      延续的动能透支）
    - 非 ST/*ST（skip_st）；信号日星期几不在 skip_weekdays 内
      （默认空，原文按星期统计调整风险的机械化）
    - conditions 中附加的市场过滤条件（如 PriceRangeCondition）

    入场：信号日次日以开盘价买入（买入量按固定本金仓位计算），
    次日相对信号日收盘高开超过 max_chase_pct（默认 5%）放弃（不追高），
    低开超过 min_chase_pct（默认 −3%）同样放弃（突破隔夜即退潮、
    动能证伪）。

    出场（持仓次日起每日收盘依次检查，全仓卖出；A 股 T+1：买入当日
    不可卖出）：
    - 收盘跌破买入价 stop_loss_pct（默认 8%）：止损（原文箱体中点/
      对侧止损的固定百分比近似）
    - 收盘跌回箱体内部（< 信号日箱体上沿）：突破失败（rejected break），
      仅持仓前 breakout_exit_days 个交易日检查（默认 2，走远后参考线
      失效，交给止损/移动止盈/止盈）
    - 曾盈利后自持仓期最高价回撤 trail_pct（默认 6%）：移动止盈
      （动能衰减）
    - 收盘 ≥ 买入价 × (1 + take_profit_pct)（默认 15%）：度量目标
      止盈（原文箱体高度投影 / 固定盈亏比 1:2 的日线近似）
    - 持仓满 max_hold_days 天（默认 10 个交易日）：时间止损（原文
      日内平仓纪律的日线代理）
    - 回测区间结束：强制平仓

    买入信号附带倾向评分（Trade.score，见 docs/strategy_gianluca.md
    ##买入倾向排序）：箱体窄度 / 突破放量 / 收盘强度 / 突破新鲜度
    四因子加权；账户层（lib/account.py）在资金/持仓约束下按评分
    降序决定哪些信号成交。
    """

    name = "Gianluca 亚盘箱体突破"
    description = ("窄幅横盘箱体 + 收盘放量突破上沿买入，突破失败/止损/"
                   "移动止盈/度量目标止盈离场（Gianluca Bruni 亚盘箱体日线近似）")

    BUY_REASONS = {
        "box_breakout": "窄幅箱体放量突破上沿，次日开盘买入（亚盘箱体突破）",
    }
    SELL_REASONS = {
        "stop_loss": "收盘跌破止损位，强制离场",
        "failed_breakout": "收盘跌回箱体内部，突破未被接受（rejected break）",
        "trailing_stop": "盈利后自高点回撤，动能衰减止盈",
        "take_profit": "到达度量目标位，止盈离场",
        "time_stop": "持仓超过时间上限，强制离场",
        "end_of_backtest": "回测区间结束，强制平仓",
    }

    def __init__(self, box_window: int = 20, max_box_width_pct: float = 0.15,
                 vol_window: int = 20, breakout_vol_mult: float = 1.5,
                 min_close_pos: float = 0.5,
                 max_turn_pct: float = 15.0, max_signal_pct: float = 9.0,
                 skip_st: bool = True, skip_weekdays: Optional[List[int]] = None,
                 min_chase_pct: float = -0.03, max_chase_pct: float = 0.05,
                 stop_loss_pct: float = 0.08, trail_pct: float = 0.06,
                 take_profit_pct: float = 0.15, breakout_exit_days: int = 2,
                 max_hold_days: int = 10,
                 conditions: Optional[List[BaseCondition]] = None,
                 position: Optional[BasePosition] = None) -> None:
        super().__init__()
        # yacrs 命令行覆盖进来的值可能是字符串，统一转换
        self.box_window = int(box_window)
        self.max_box_width_pct = float(max_box_width_pct)
        self.vol_window = int(vol_window)
        self.breakout_vol_mult = float(breakout_vol_mult)
        self.min_close_pos = float(min_close_pos)
        self.max_turn_pct = float(max_turn_pct)
        self.max_signal_pct = float(max_signal_pct)
        if isinstance(skip_st, str):
            skip_st = skip_st.strip().lower() not in ("false", "0", "no")
        self.skip_st = bool(skip_st)
        # skip_weekdays 兼容 "[2,3]" 形式的字符串覆盖
        if isinstance(skip_weekdays, str):
            skip_weekdays = [int(x) for x in skip_weekdays.strip("[]").split(",")
                             if x.strip()]
        self.skip_weekdays = {int(d) for d in (skip_weekdays or [])}
        self.min_chase_pct = float(min_chase_pct)
        self.max_chase_pct = float(max_chase_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trail_pct = float(trail_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.breakout_exit_days = int(breakout_exit_days)
        self.max_hold_days = int(max_hold_days)
        self.conditions = list(conditions) if conditions else []
        self.position = position if position is not None else FixedCapitalPosition()

    def _signal_dates(self, df: pd.DataFrame, start_time, end_time
                      ) -> Tuple[set, List[Tuple[str, int]]]:
        """信号日集合（docs/strategy_gianluca.md##买入）：日期区间 + 箱体 +
        突破 + 放量 + 全部附加条件取交集，再过收盘位置/换手率/涨幅/排雷
        过滤；每个条件都在完整 df 上计算，保证依赖历史数据的窗口指标
        不受影响。同时返回逐阶段剩余行数 [(阶段名, 剩余行数), ...]，
        供 forecast 在 0 信号时拆解原因（阶段顺序即过滤顺序）。"""
        close, high, low = df["close"], df["high"], df["low"]
        # 箱体（docs/strategy_gianluca.md##买入①）：过去 box_window 日
        # （不含当日）高低点区间，宽度 ≤ 上限——亚盘低波动收敛的日线代理
        box_high = high.shift(1).rolling(self.box_window).max()
        box_low = low.shift(1).rolling(self.box_window).min()
        box_width = box_high / box_low - 1
        # 相对放量（与 RelativeVolumeCondition 同口径：当日量 / 过去
        # vol_window 日均量，不含当日）
        prev_avg_volume = df["volume"].shift(1).rolling(self.vol_window).mean()
        rel_volume = df["volume"] / prev_avg_volume
        bar_range = high - low
        close_pos = (close - low) / bar_range.where(bar_range > 0)

        conditions = [DateRangeCondition(start_time, end_time), *self.conditions]
        filters: List[Tuple[str, pd.Series]] = [
            (f"箱体宽度≤{self.max_box_width_pct:.0%}（{self.box_window}日窄幅收敛）",
             box_width <= self.max_box_width_pct),
            ("收盘突破箱体上沿（accepted expansion）", close > box_high),
            (f"相对放量≥{self.breakout_vol_mult:g}倍（{self.vol_window}日均量）",
             rel_volume >= self.breakout_vol_mult),
            (f"收盘位于振幅上半部≥{self.min_close_pos:g}（拒绝 rejected break）",
             close_pos >= self.min_close_pos),
            (f"换手≤{self.max_turn_pct:g}% 且 涨幅≤{self.max_signal_pct:g}%",
             (df["turn"] <= self.max_turn_pct)
             & (df["pctChg"] <= self.max_signal_pct)),
        ]
        if self.skip_st:
            filters.append(("排除 ST/*ST 个股（退市风险，不符合突破前提）",
                            ~df["name"].astype(str).str.contains("ST", na=False)))
        if self.skip_weekdays:
            # 星期风险调整（docs/strategy_gianluca.md##买入⑥）：原文按星期
            # 统计调整风险敞口的机械化，黑名单内星期几不入场
            weekday = pd.to_datetime(df["date"]).dt.weekday
            filters.append((f"星期黑名单 {sorted(self.skip_weekdays)} 不入场",
                            ~weekday.isin(self.skip_weekdays)))

        signal_dates = set(df["date"])
        stages: List[Tuple[str, int]] = [("候选K线", len(df))]
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])
            stages.append((cond.description, len(signal_dates)))
        for label, mask in filters:
            signal_dates &= set(df.loc[mask.fillna(False), "date"])
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

        close, high, low = df["close"], df["high"], df["low"]
        open_ = df["open"]
        # 与 _signal_dates 同口径的窗口指标（评分与突破失败退出复用）
        box_high_s = high.shift(1).rolling(self.box_window).max()
        box_low_s = low.shift(1).rolling(self.box_window).min()
        box_width_s = box_high_s / box_low_s - 1
        prev_avg_volume = df["volume"].shift(1).rolling(self.vol_window).mean()
        rel_volume_s = df["volume"] / prev_avg_volume
        bar_range = high - low
        close_pos_s = (close - low) / bar_range.where(bar_range > 0)

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
                # 买入点条件（docs/strategy_gianluca.md##买入）：次日开盘买入；
                # 高开超过 max_chase_pct 不追；低开超过 min_chase_pct（默认 −3%）
                # 说明突破隔夜即退潮、动能证伪，放弃
                chase_pct = entry_price / float(price) - 1
                if (entry_date > end_ts or chase_pct > self.max_chase_pct
                        or chase_pct < self.min_chase_pct):
                    continue

                # 买入量（docs/strategy_gianluca.md##买入量）：固定本金，不足一手放弃
                amount = self.position.buy_amount(entry_price)
                if amount <= 0:
                    continue

                # 买入倾向评分（docs/strategy_gianluca.md##买入倾向排序）：
                # 箱体窄度 / 突破放量 / 收盘强度 / 突破新鲜度四因子加权，
                # 账户层按评分配资金/仓位
                bw = float(box_width_s.iloc[i])
                rv = float(rel_volume_s.iloc[i])
                cp = close_pos_s.iloc[i]
                bh = float(box_high_s.iloc[i])
                t_box = 1.0 - _clip(bw / self.max_box_width_pct)
                t_vol = _clip((rv / self.breakout_vol_mult - 1.0) / 2.0)
                t_close = _clip(float(cp)) if not pd.isna(cp) else 0.0
                t_fresh = 1.0 - _clip((float(price) / bh - 1.0) / 0.05)
                score = (0.30 * t_box + 0.30 * t_vol
                         + 0.20 * t_close + 0.20 * t_fresh)

                trades.append(Trade(
                    ticker=ticker,
                    name=name,
                    buy_time=entry_date.strftime("%Y-%m-%d"),
                    buy_amount=amount,
                    buy_price=entry_price,
                    buy_reason=self.BUY_REASONS["box_breakout"],
                    score=score,
                ))
                position_open = True
                entry_i = i + 1
                signal_box_high = bh  # 突破失败参考线：信号日箱体上沿
                peak_price = entry_price
                hold_days = 1
            else:
                trade = trades[-1]
                if i <= entry_i:
                    continue  # 买入当日不可卖出（T+1），自次日起检查卖出
                hold_days += 1
                peak_price = max(peak_price, float(price))

                # 卖出条件（docs/strategy_gianluca.md##卖出）：按优先级依次检查
                sell_reason = None
                if price < trade.buy_price * (1 - self.stop_loss_pct):
                    sell_reason = self.SELL_REASONS["stop_loss"]
                elif (price < signal_box_high
                      and hold_days - 1 <= self.breakout_exit_days):
                    sell_reason = self.SELL_REASONS["failed_breakout"]
                elif (peak_price > trade.buy_price
                        and price < peak_price * (1 - self.trail_pct)):
                    sell_reason = self.SELL_REASONS["trailing_stop"]
                elif price >= trade.buy_price * (1 + self.take_profit_pct):
                    sell_reason = self.SELL_REASONS["take_profit"]
                elif hold_days >= self.max_hold_days:
                    sell_reason = self.SELL_REASONS["time_stop"]
                elif date >= end_ts or i == len(df) - 1:
                    sell_reason = self.SELL_REASONS["end_of_backtest"]

                if sell_reason is not None:
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    # 卖出量（docs/strategy_gianluca.md##卖出量）：全仓卖出
                    trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason
                    position_open = False

        return trades
