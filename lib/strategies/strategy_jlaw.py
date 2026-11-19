"""J Law（罗伟森）的多重优势交易系统 / VCP 突破回踩策略（日线版）。

J Law：香港前中学教师，2024 年美国投资锦标赛百万美元以上股票组冠军
（+353.9%，打破 Minervini 334.8% 纪录），2025 年以 +252.3% 卫冕，两年复利
+1,499%。体系师承 Mark Minervini 的 VCP（波动收缩形态），核心思想是
"Multiple Edge Trading Area"：只在多重独立买入理由（前高阻力转支撑 +
10/20 日均线带 + 趋势线）重合的价位进场；止损放在形态失效处（3%~5%），
单笔风险约账户 0.35%；卖点分趁强（阻力位/风险收益比 2~2.5 倍分批止盈）、
趁弱（跌破 10/20 日均线）、趋势持有三种（公开资料详见 docs/strategy_jlaw.md）。

本仓库的 A 股日线近似（与 strategy_connors_rsi2.py 同一套无未来函数约定）：
- 信号一律收盘确认、次日开盘成交：VCP 末段收缩与缩量按"截至昨日"状态判定
  （近 10 日振幅 ≤ 前 20 日振幅 × contraction_shrink 且 ≤ max_last_contraction_pct、
  近 5 日均量 ≤ 20 日均量 × volume_dry_ratio），当日仅作触发——突破=收盘站上
  近 base_window 日最高价（Pivot）且放量 ≥ breakout_vol_mult 倍均量；
  回踩=突破后 pullback_window 日内首次收盘回到 MA20 带 ±ma_zone_pct 内、
  不破 Pivot、量 ≤ 突破日量 × pullback_vol_ratio
- 选股带 Stage 2 趋势模板（收盘 > MA50 > MA200 且两线向上）、催化代理
  （rally_window 日内最低价至 Pivot 涨幅 ≥ min_rally_pct，对应其"先有放量急涨"
  的 momentum move）、流动性过滤（20 日平均成交额 ≥ min_amount，近似其
  >1 亿美元/日美元成交额要求）与 ST 排除
- 卖出机械化：跌破信号日止损价（max(买入价 × (1−stop_loss_pct)，多重优势区
  下沿 × (1−stop_buffer_pct)) 二者取更近）/ 跌破 MA10 带（趁弱卖）/
  固定止盈 take_profit_pct（趁强卖，R:R 2~2.5 的日线近似）/ 时间止损 /
  区间末强制平仓；「趋势持有/零成本减仓」需人工判断股票类型，不在机械实现内
- 原文按"单笔风险 0.35% 倒推股数"，本仓库统一用固定本金仓位（见文档##买入量）

超参经 exp/baseline.yaml 回测校准（2026-11-19，分支 autostrategy/Exp-jlaw-20261119-01，
详见 docs/strategy_jlaw.md 末尾的 A 股实证校准记录）。
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from yacrs import register

from lib.conditions.base import BaseCondition
from lib.conditions.date_range import DateRangeCondition
from lib.position import BasePosition, FixedCapitalPosition
from lib.trade import Trade

from .base import BaseStrategy


@register()
class JLawMultipleEdgeStrategy(BaseStrategy):
    """
    J Law 多重优势交易系统 / VCP 突破回踩（说明文档见 docs/strategy_jlaw.md）。

    扫描条件（信号日，须在回测区间内；VCP 形态与缩量状态均截至昨日判定）：
    - 趋势模板：收盘 > MA50 > MA200，且 MA50 / MA200 分别近 10 / 20 日向上
      （trend_fast / trend_slow / fast_rise_days / slow_rise_days；
      Stage 2 上升趋势，原文选股第一步）
    - 催化代理：rally_window 日内最低价至 Pivot 涨幅 ≥ min_rally_pct
      （默认 120 日 / 30%：先有放量急涨才有 VCP 整理，对应 momentum move）
    - VCP 收缩：近 45 日自 Pivot 最大回撤 ≤ max_base_depth_pct，且近
      contraction_recent 日振幅 ≤ 前 contraction_prior 日振幅 ×
      contraction_shrink 且 ≤ max_last_contraction_pct（回调逐次变浅的量化）
    - 量能：近 dryup_days 日均量 ≤ dryup_ref 日均量 × volume_dry_ratio
      （末段回调缩量，抛压耗尽）；20 日平均成交额 ≥ min_amount（流动性）
    - 非 ST/*ST（skip_st）

    买点（entry_mode 控制，默认 both）：
    - 突破（vcp_breakout）：收盘 > Pivot（近 base_window 日最高，不含当日）
      且当日量 ≥ breakout_vol_mult × 20 日均量（放量突破）
    - 回踩（vcp_pullback）：突破后 pullback_window 日内首次收盘回到
      MA ma_zone 带 ±ma_zone_pct 内、收盘 ≥ Pivot（前高转支撑不破）、
      当日量 ≤ 突破日量 × pullback_vol_ratio、收盘 > MA50（多重优势区域）

    入场：信号日次日以开盘价买入（买入量按固定本金仓位计算），次日高开超过
    max_chase_pct（默认 5%）放弃——突破隔夜兑现后盈亏比恶化。

    出场（持仓次日起每日收盘依次检查，全仓卖出；A 股 T+1：买入当日不可卖出）：
    - 收盘跌破止损价（信号日算出：max(买入价 × (1−stop_loss_pct)，多重优势区
      下沿 × (1−stop_buffer_pct)) 取更近者，风险上限默认 8%，对应原文 3%~5% 的
      A 股放宽——baseline 实证 5%/8%/10% 三点 16.32%/18.90%/14.76%，8% 为峰）
    - 收盘 < MA exit_ma_window × (1 − exit_ma_buffer_pct)（趁弱卖，默认 10 日线）
    - 收盘 ≥ 买入价 × (1 + take_profit_pct)（趁强卖，默认 +20%）
    - 持仓满 max_hold_days 天（默认 60 个交易日，波段~中线）
    - 回测区间结束：强制平仓

    买入信号附带倾向评分（Trade.score，见 docs/strategy_jlaw.md##买入倾向排序）：
    5 因子加权（末段收缩紧度/量能枯度/买点质量/距 52 周高位/趋势强度），
    与 J Law 的 edge 叠加逻辑对齐；账户层（lib/account.py）在资金/持仓约束下
    按评分降序决定哪些信号成交。
    """

    name = "J Law VCP 多重优势"
    description = ("Stage 2 趋势中 VCP 收缩形态放量突破/缩量回踩买入，跌破均线带趁弱卖、"
                   "固定止盈趁强卖、形态失效止损（J Law 多重优势交易系统日线版）")

    BUY_REASONS = {
        "vcp_breakout": "VCP 放量突破 Pivot（近45日新高），次日开盘买入（J Law 突破买法）",
        "vcp_pullback": "突破后首次缩量回踩 20 日均线带（前高转支撑），次日开盘买入（J Law 回踩买法）",
    }
    SELL_REASONS = {
        "stop_loss": "收盘跌破止损位（形态失效），强制离场",
        "ma_break": "收盘跌破持仓均线带（趁弱卖），离场",
        "take_profit": "收盘达到固定止盈位（趁强卖），离场",
        "time_stop": "持仓超过时间上限，强制离场",
        "end_of_backtest": "回测区间结束，强制平仓",
    }

    def __init__(self, trend_fast: int = 50, trend_slow: int = 200,
                 fast_rise_days: int = 10, slow_rise_days: int = 20,
                 rally_window: int = 120, min_rally_pct: float = 0.30,
                 base_window: int = 45, max_base_depth_pct: float = 0.30,
                 contraction_recent: int = 10, contraction_prior: int = 20,
                 contraction_shrink: float = 0.75,
                 max_last_contraction_pct: float = 0.22,
                 dryup_days: int = 5, dryup_ref: int = 20,
                 volume_dry_ratio: float = 0.8,
                 entry_mode: str = "both",
                 breakout_vol_mult: float = 2.0, pullback_window: int = 10,
                 ma_zone: int = 20, ma_zone_pct: float = 0.02,
                 pullback_vol_ratio: float = 0.8,
                 min_amount: float = 1e8, skip_st: bool = True,
                 max_chase_pct: float = 0.05, stop_loss_pct: float = 0.08,
                 stop_buffer_pct: float = 0.02, take_profit_pct: float = 0.20,
                 exit_ma_window: int = 10, exit_ma_buffer_pct: float = 0.01,
                 max_hold_days: int = 60,
                 conditions: Optional[List[BaseCondition]] = None,
                 position: Optional[BasePosition] = None) -> None:
        super().__init__()
        if entry_mode not in ("breakout", "pullback", "both"):
            raise ValueError(f"entry_mode 必须是 breakout/pullback/both，got {entry_mode}")
        self.trend_fast = int(trend_fast)
        self.trend_slow = int(trend_slow)
        self.fast_rise_days = int(fast_rise_days)
        self.slow_rise_days = int(slow_rise_days)
        self.rally_window = int(rally_window)
        self.min_rally_pct = float(min_rally_pct)
        self.base_window = int(base_window)
        self.max_base_depth_pct = float(max_base_depth_pct)
        self.contraction_recent = int(contraction_recent)
        self.contraction_prior = int(contraction_prior)
        self.contraction_shrink = float(contraction_shrink)
        self.max_last_contraction_pct = float(max_last_contraction_pct)
        self.dryup_days = int(dryup_days)
        self.dryup_ref = int(dryup_ref)
        self.volume_dry_ratio = float(volume_dry_ratio)
        self.entry_mode = entry_mode
        self.breakout_vol_mult = float(breakout_vol_mult)
        self.pullback_window = int(pullback_window)
        self.ma_zone = int(ma_zone)
        self.ma_zone_pct = float(ma_zone_pct)
        self.pullback_vol_ratio = float(pullback_vol_ratio)
        self.min_amount = float(min_amount)
        self.skip_st = bool(skip_st)
        self.max_chase_pct = float(max_chase_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.stop_buffer_pct = float(stop_buffer_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.exit_ma_window = int(exit_ma_window)
        self.exit_ma_buffer_pct = float(exit_ma_buffer_pct)
        self.max_hold_days = int(max_hold_days)
        self.conditions = list(conditions) if conditions else []
        self.position = position if position is not None else FixedCapitalPosition()

    # ------------------------------------------------------------------
    # 指标与信号
    # ------------------------------------------------------------------

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """VCP 形态所需的全部滚动指标。形态/缩量状态一律截至昨日（shift 1），
        当日收盘仅作触发判定，保证信号无未来函数。"""
        close, high, low = df["close"], df["high"], df["low"]
        vol, amt = df["volume"], df["amount"]
        ind = pd.DataFrame(index=df.index)
        ind["ma_fast"] = close.rolling(self.trend_fast).mean()
        ind["ma_slow"] = close.rolling(self.trend_slow).mean()
        ind["ma_zone"] = close.rolling(self.ma_zone).mean()
        # Pivot = 前 base_window 日最高（不含当日），突破即收盘站上它
        ind["pivot"] = high.rolling(self.base_window).max().shift(1)
        ind["base_low"] = low.rolling(self.base_window).min().shift(1)
        # VCP 两段振幅：近段 vs 其前段（均截至昨日）
        hi_r = high.rolling(self.contraction_recent).max().shift(1)
        lo_r = low.rolling(self.contraction_recent).min().shift(1)
        hi_p = high.rolling(self.contraction_prior).max().shift(
            self.contraction_recent + 1)
        lo_p = low.rolling(self.contraction_prior).min().shift(
            self.contraction_recent + 1)
        ind["depth_recent"] = hi_r / lo_r - 1.0
        ind["depth_prior"] = hi_p / lo_p - 1.0
        # 量能：近 dryup_days 日均量 / dryup_ref 日均量（截至昨日）
        ind["vol_ma"] = vol.rolling(self.dryup_ref).mean().shift(1)
        ind["dry_ratio"] = (vol.rolling(self.dryup_days).mean().shift(1)
                            / ind["vol_ma"])
        ind["amt_ma"] = amt.rolling(self.dryup_ref).mean().shift(1)
        ind["high_250"] = close.rolling(250).max().shift(1)
        # 催化代理：rally_window 日最低（截至昨日）至 Pivot 的涨幅
        ind["rally_low"] = low.rolling(self.rally_window).min().shift(1)
        return ind

    def _setup_ok(self, df: pd.DataFrame, ind: pd.DataFrame,
                  signal_dates: set) -> pd.Series:
        """最低买入要求中"形态侧"的全部过滤（截至昨日状态），
        不含当日触发与日期区间/附加条件（见 docs/strategy_jlaw.md##买入）。"""
        close = df["close"]
        trend_ok = ((close > ind["ma_fast"])
                    & (ind["ma_fast"] > ind["ma_slow"])
                    & (ind["ma_fast"] > ind["ma_fast"].shift(self.fast_rise_days))
                    & (ind["ma_slow"] > ind["ma_slow"].shift(self.slow_rise_days)))
        rally_ok = (ind["pivot"] / ind["rally_low"] - 1.0) >= self.min_rally_pct
        base_ok = ((ind["pivot"] / ind["base_low"] - 1.0)
                   <= self.max_base_depth_pct)
        contract_ok = ((ind["depth_recent"] <= self.max_last_contraction_pct)
                       & (ind["depth_recent"]
                          <= ind["depth_prior"] * self.contraction_shrink))
        dry_ok = ind["dry_ratio"] <= self.volume_dry_ratio
        amt_ok = ind["amt_ma"] >= self.min_amount
        ok = (trend_ok & rally_ok & base_ok & contract_ok & dry_ok & amt_ok)
        if self.skip_st:
            ok &= ~df["name"].astype(str).str.contains("ST", na=False)
        in_range = pd.Series(df["date"].isin(signal_dates), index=df.index)
        return ok & in_range

    def _breakout_trigger(self, df: pd.DataFrame, ind: pd.DataFrame,
                          setup_ok: pd.Series) -> pd.Series:
        """突破买法当日触发：收盘站上 Pivot 且放量 ≥ breakout_vol_mult 倍均量。"""
        close, vol = df["close"], df["volume"]
        return (setup_ok
                & (close > ind["pivot"])
                & (vol >= self.breakout_vol_mult * ind["vol_ma"]))

    def _triggers(self, df: pd.DataFrame, ind: pd.DataFrame,
                  setup_ok: pd.Series) -> pd.Series:
        """全部买点触发日（"breakout" / "pullback" / None），逐日扫描：
        突破日更新跟踪状态；突破后 pullback_window 日内首次收盘回到 MA 带内、
        不破 Pivot、缩量即回踩买点；收盘跌破 Pivot（前高转支撑失效）则作废。"""
        breakout = self._breakout_trigger(df, ind, setup_ok)
        close, low, vol = df["close"], df["low"], df["volume"]
        ma_zone = ind["ma_zone"]
        modes: List[Optional[str]] = [None] * len(df)
        last_day, last_pivot, last_vol = -1, 0.0, 0.0
        want_breakout = self.entry_mode in ("breakout", "both")
        want_pullback = self.entry_mode in ("pullback", "both")
        for i in range(len(df)):
            c = float(close.iloc[i])
            if last_day >= 0 and c < last_pivot:
                last_day = -1  # 收盘跌破 Pivot：前高转支撑失效，回踩 setup 作废
            if bool(breakout.iloc[i]):
                if want_breakout:
                    modes[i] = "breakout"
                last_day = i
                last_pivot = float(ind["pivot"].iloc[i])
                last_vol = float(vol.iloc[i])
                continue
            if (want_pullback and last_day >= 0
                    and i - last_day <= self.pullback_window
                    and c >= last_pivot
                    and c > float(ind["ma_fast"].iloc[i])
                    and float(low.iloc[i]) <= ma_zone.iloc[i] * (1 + self.ma_zone_pct)
                    and c >= ma_zone.iloc[i] * (1 - self.ma_zone_pct)
                    and float(vol.iloc[i]) <= last_vol * self.pullback_vol_ratio):
                modes[i] = "pullback"
        return pd.Series(modes, index=df.index, dtype=object)

    def _signal_dates(self, df: pd.DataFrame, start_time, end_time
                      ) -> Tuple[set, List[Tuple[str, int]], pd.DataFrame,
                                 pd.Series, pd.Series]:
        """信号日集合 + 逐阶段剩余行数（forecast 0 信号诊断用）。
        阶段与 run() 的判定完全一致：日期区间 + 附加条件 → 趋势模板 → 催化 →
        VCP 收缩（基底深度 + 逐次变浅）→ 缩量与流动性 → ST → 买点触发。"""
        conditions = [DateRangeCondition(start_time, end_time),
                      *self.conditions]
        signal_dates = set(df["date"])
        stages: List[Tuple[str, int]] = [("候选K线", len(df))]
        for cond in conditions:
            signal_dates &= set(cond(df)["date"])
            stages.append((cond.description, len(signal_dates)))

        ind = self._indicators(df)
        in_range = pd.Series(df["date"].isin(signal_dates), index=df.index)

        close = df["close"]
        trend_ok = ((close > ind["ma_fast"])
                    & (ind["ma_fast"] > ind["ma_slow"])
                    & (ind["ma_fast"] > ind["ma_fast"].shift(self.fast_rise_days))
                    & (ind["ma_slow"] > ind["ma_slow"].shift(self.slow_rise_days)))
        stages.append((f"趋势模板：收盘>MA{self.trend_fast}>MA{self.trend_slow} 且两线向上",
                       int((in_range & trend_ok).sum())))
        rally_ok = (ind["pivot"] / ind["rally_low"] - 1.0) >= self.min_rally_pct
        stages.append((f"催化：近{self.rally_window}日最低价至Pivot涨幅≥{self.min_rally_pct:.0%}",
                       int((in_range & trend_ok & rally_ok).sum())))
        base_ok = ((ind["pivot"] / ind["base_low"] - 1.0)
                   <= self.max_base_depth_pct)
        contract_ok = ((ind["depth_recent"] <= self.max_last_contraction_pct)
                       & (ind["depth_recent"]
                          <= ind["depth_prior"] * self.contraction_shrink))
        stages.append(("VCP 收缩：基底深度≤{:.0%} 且近{}日振幅≤前{}日×{} 且 ≤{:.0%}".format(
            self.max_base_depth_pct, self.contraction_recent,
            self.contraction_prior, self.contraction_shrink,
            self.max_last_contraction_pct),
            int((in_range & trend_ok & rally_ok & base_ok & contract_ok).sum())))
        dry_ok = ind["dry_ratio"] <= self.volume_dry_ratio
        amt_ok = ind["amt_ma"] >= self.min_amount
        setup_mid = (in_range & trend_ok & rally_ok & base_ok & contract_ok
                     & dry_ok & amt_ok)
        stages.append((f"缩量：近{self.dryup_days}日均量≤{self.dryup_ref}日均量×{self.volume_dry_ratio} "
                       f"且流动性≥{self.min_amount / 1e8:g}亿",
                       int(setup_mid.sum())))
        if self.skip_st:
            setup_mid &= ~df["name"].astype(str).str.contains("ST", na=False)
            stages.append(("排除 ST/*ST", int(setup_mid.sum())))

        setup_ok = (trend_ok & rally_ok & base_ok & contract_ok & dry_ok
                    & amt_ok & in_range)
        if self.skip_st:
            setup_ok &= ~df["name"].astype(str).str.contains("ST", na=False)
        triggers = self._triggers(df, ind, setup_ok)
        hit = triggers.notna()
        stages.append(("买点触发（突破/回踩）", int(hit.sum())))
        return set(df.loc[hit, "date"]), stages, ind, setup_ok, triggers

    def signal_day_breakdown(self, df: pd.DataFrame, start_time, end_time
                             ) -> List[Tuple[str, int]]:
        """诊断接口（forecast 0 信号时调用）：返回逐阶段剩余信号日行数，
        阶段与 run() 的信号判定完全一致。"""
        _, stages, _, _, _ = self._signal_dates(df, start_time, end_time)
        return stages

    # ------------------------------------------------------------------
    # 回测
    # ------------------------------------------------------------------

    def _score(self, df: pd.DataFrame, ind: pd.DataFrame, i: int,
               mode: str) -> float:
        """买入倾向评分（docs/strategy_jlaw.md##买入倾向排序）：5 因子加权，
        全部取信号日可得数据（形态因子截至昨日）。"""
        close = df["close"]
        depth = float(ind["depth_recent"].iloc[i])
        dry = float(ind["dry_ratio"].iloc[i])
        t_contract = 1.0 - min(max(depth / self.max_last_contraction_pct, 0.0), 1.0)
        t_dry = 1.0 - min(max(dry, 0.0), 1.0)
        if mode == "pullback":
            zone = float(ind["ma_zone"].iloc[i])
            dist = abs(float(close.iloc[i]) - zone) / (zone * self.ma_zone_pct)
            t_entry = 1.0 - min(max(dist, 0.0), 1.0)
        else:
            vol_mult = (float(df["volume"].iloc[i]) / float(ind["vol_ma"].iloc[i])
                        if ind["vol_ma"].iloc[i] else 0.0)
            t_entry = min(max((vol_mult - 1.0) / 2.0, 0.0), 1.0)
        high_250 = float(ind["high_250"].iloc[i])
        t_high = (min(max(float(close.iloc[i]) / high_250, 0.0), 1.0)
                  if high_250 else 0.0)
        ma_slow = float(ind["ma_slow"].iloc[i])
        t_trend = (min(max((float(close.iloc[i]) / ma_slow - 1.0) / 0.5, 0.0), 1.0)
                   if ma_slow else 0.0)
        return (0.30 * t_contract + 0.20 * t_dry + 0.20 * t_entry
                + 0.15 * t_high + 0.15 * t_trend)

    def run(self, df: pd.DataFrame, start_time, end_time) -> List[Trade]:
        end_ts = pd.Timestamp(end_time)

        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            return []

        _, _, ind, setup_ok, triggers = self._signal_dates(df, start_time, end_time)

        close = df["close"]
        open_ = df["open"]
        exit_ma = close.rolling(self.exit_ma_window).mean()
        dates = df["date"]

        ticker = df["ticker"].iloc[0]
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        trades: List[Trade] = []
        position_open = False
        entry_i = -1  # 买入日下标（信号日次日，开盘价成交），当日不可卖出（T+1）
        hold_days = 0
        stop_price = 0.0

        for i in range(len(df)):
            date = dates.iloc[i]
            price = close.iloc[i]
            mode = triggers.iloc[i]

            if not position_open:
                if mode is None:
                    continue
                if i + 1 >= len(df):
                    continue
                # 买入点条件（docs/strategy_jlaw.md##买入）：VCP 形态（截至昨日）
                # + 当日触发（突破/回踩），次日开盘前决策、以次日开盘价成交
                entry_i = i + 1
                entry_date = dates.iloc[entry_i]
                next_open = open_.iloc[entry_i]
                if pd.isna(next_open):
                    continue
                entry_price = float(next_open)
                # 追价保护（docs/strategy_jlaw.md##买入）：次日高开超过
                # max_chase_pct 说明突破/回踩的隔夜空间已被兑现，盈亏比恶化，不追
                chase_pct = entry_price / float(price) - 1
                if entry_date > end_ts or chase_pct > self.max_chase_pct:
                    continue

                # 买入量（docs/strategy_jlaw.md##买入）：固定本金，不足一手放弃
                amount = self.position.buy_amount(entry_price)
                if amount <= 0:
                    continue

                # 止损（docs/strategy_jlaw.md##卖出）：形态失效价 = max(买入价 ×
                # (1−stop_loss_pct)，多重优势区下沿 × (1−stop_buffer_pct))，
                # 二者取更近（更高）者——止损放支撑区下方，风险上限 5%
                zone_low = min(float(ind["pivot"].iloc[i]),
                               float(ind["ma_zone"].iloc[i]))
                stop_price = max(entry_price * (1 - self.stop_loss_pct),
                                 zone_low * (1 - self.stop_buffer_pct))

                score = self._score(df, ind, i, str(mode))
                trades.append(Trade(
                    ticker=ticker,
                    name=name,
                    buy_time=entry_date.strftime("%Y-%m-%d"),
                    buy_amount=amount,
                    buy_price=entry_price,
                    buy_reason=self.BUY_REASONS[f"vcp_{mode}"],
                    score=score,
                ))
                position_open = True
                hold_days = 1
            else:
                if i <= entry_i:
                    continue  # 买入当日不可卖出（T+1），自次日起检查卖出
                hold_days += 1
                trade = trades[-1]

                # 卖出条件（docs/strategy_jlaw.md##卖出）：按优先级依次检查，
                # 当日收盘确认后次日开盘卖出（无未来函数；回测区间末无次日数据
                # 则按当日收盘强制平仓）
                sell_reason = None
                if price < stop_price:
                    sell_reason = self.SELL_REASONS["stop_loss"]
                elif price < float(exit_ma.iloc[i]) * (1 - self.exit_ma_buffer_pct):
                    sell_reason = self.SELL_REASONS["ma_break"]
                elif price >= trade.buy_price * (1 + self.take_profit_pct):
                    sell_reason = self.SELL_REASONS["take_profit"]
                elif hold_days >= self.max_hold_days:
                    sell_reason = self.SELL_REASONS["time_stop"]

                if sell_reason is not None:
                    if i + 1 < len(df) and dates.iloc[i + 1] <= end_ts:
                        trade.sell_time = dates.iloc[i + 1].strftime("%Y-%m-%d")
                        # 卖出量（docs/strategy_jlaw.md##卖出）：全仓卖出
                        trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                        trade.sell_price = float(open_.iloc[i + 1])
                        trade.sell_reason = sell_reason
                        position_open = False
                        continue
                    sell_reason = self.SELL_REASONS["end_of_backtest"]
                if date >= end_ts or i == len(df) - 1:
                    # 回测区间结束 / 最后一条数据：按当日收盘强制平仓
                    trade.sell_time = date.strftime("%Y-%m-%d")
                    # 卖出量（docs/strategy_jlaw.md##卖出）：全仓卖出
                    trade.sell_amount = self.position.sell_amount(trade.buy_amount)
                    trade.sell_price = float(price)
                    trade.sell_reason = sell_reason or self.SELL_REASONS["end_of_backtest"]
                    position_open = False

        return trades
