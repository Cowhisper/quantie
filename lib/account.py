"""账户级模拟：在策略产生的买入信号上执行资金与持仓约束。

策略逐股独立扫描信号（lib/strategies/），不了解账户状态；本模块按交易日
推进，把信号变成实际成交，保证：

- T+1（A 股）：当日买入次一交易日起才可卖出。策略信号若把卖出日定在
  买入日或之前（当日回转），顺延至买入日后该股票下一个有行情的交易日、
  以当日开盘价成交；区间内已无后续行情则保留持仓至期末盯市（卖出腿作废）；
- 现金不为负：买入日按买入价扣减现金，现金不足的信号整笔跳过；
- 持仓数上限（max_positions）：卖出前空出的仓位按买入倾向评分
  （Trade.score，策略给出的"最可能赚钱"程度）降序配给当日买入信号，
  评分相同按股票代码排序保证确定性；
- 卖出信号全部执行（全仓卖出，回笼现金、空出仓位）。

被跳过的信号及其卖出腿整笔作废（不返回、不产生任何现金变动）。
"""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import pandas as pd

from lib.trade import Trade


def simulate_account(trades: List[Trade], stock_data: Dict[str, pd.DataFrame],
                     start_time, end_time,
                     initial_capital: float,
                     max_positions: int = 10,
                     ) -> Tuple[List[Trade], int]:
    """
    按交易日推进账户模拟，返回 (实际成交的交易列表, 被跳过的买入信号数)。

    每个交易日：先执行当日卖出（cash += 卖出价 × 卖出量，空出仓位），
    再把当日买入信号按 score 降序依次尝试成交——持仓未满且
    买入价 × 买入量 <= 剩余现金 则成交，否则跳过。
    成交的买入按 A 股 T+1 规则校正卖出腿：卖出日早于/等于买入日的信号
    顺延至买入日后该股票下一个有行情的交易日、按其开盘价重定价成交
    （区间内无后续行情则作废卖出腿，持仓保留至期末盯市；校正对 Trade
    对象整体复制，传入的信号对象不被修改）。
    传入的 trades 列表不被修改；成交列表保持传入顺序。
    """
    start_ts, end_ts = pd.Timestamp(start_time), pd.Timestamp(end_time)
    initial_capital = float(initial_capital)
    max_positions = int(max_positions)

    all_dates = sorted({d for df in stock_data.values() for d in df["date"]
                        if start_ts <= d <= end_ts})

    buys_by_day: Dict[str, List[Trade]] = {}
    for t in trades:
        if not t.buy_time:
            continue
        buys_by_day.setdefault(t.buy_time, []).append(t)

    def enforce_t1(t: Trade) -> Trade:
        """T+1 校正（见模块 docstring）：返回校正后的 Trade，不改入参。"""
        if not t.sell_time or t.sell_time > t.buy_time:
            return t
        buy_ts = pd.Timestamp(t.buy_time)
        g = stock_data.get(t.ticker)
        if g is not None:
            later = g[(g["date"] > buy_ts) & (g["date"] <= end_ts)]
            for row in later.itertuples():
                open_p = float(row.open)
                if not pd.isna(open_p):
                    return replace(t, sell_time=row.date.strftime("%Y-%m-%d"),
                                   sell_price=open_p)
        return replace(t, sell_time=None, sell_amount=None,
                       sell_price=None, sell_reason=None)

    cash = initial_capital
    open_count = 0
    accepted: List[Trade] = []
    sells_by_day: Dict[str, List[Trade]] = {}

    for d in all_dates:
        ds = d.strftime("%Y-%m-%d")
        # 先卖出：回笼现金、空出仓位
        for t in sells_by_day.pop(ds, []):
            cash += (t.sell_price or 0.0) * (t.sell_amount or 0.0)
            open_count -= 1
        # 再买入：评分高者优先配给资金与仓位
        cands = buys_by_day.get(ds, [])
        cands.sort(key=lambda t: (-(t.score or 0.0), t.ticker))
        for t in cands:
            cost = t.buy_price * t.buy_amount
            if open_count >= max_positions or cost > cash + 1e-9:
                continue  # 资金/持仓不足：跳过（跳过总数在结尾按候选-成交统计）
            cash -= cost
            open_count += 1
            t = enforce_t1(t)  # T+1：当日买入的卖出腿不可当日执行
            accepted.append(t)
            if t.sell_time:
                sells_by_day.setdefault(t.sell_time, []).append(t)

    # 被跳过的信号数 = 候选信号总数 - 成交数（区间外日期的信号不会被遍历到）
    total_candidates = sum(len(v) for v in buys_by_day.values())
    skipped = total_candidates - len(accepted)
    return accepted, skipped
