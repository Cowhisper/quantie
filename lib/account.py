"""账户级模拟：在策略产生的买入信号上执行资金与持仓约束。

策略逐股独立扫描信号（lib/strategies/），不了解账户状态；本模块按交易日
推进，把信号变成实际成交，保证：

- 现金不为负：买入日按买入价扣减现金，现金不足的信号整笔跳过；
- 持仓数上限（max_positions）：卖出前空出的仓位按买入倾向评分
  （Trade.score，策略给出的"最可能赚钱"程度）降序配给当日买入信号，
  评分相同按股票代码排序保证确定性；
- 卖出信号全部执行（全仓卖出，回笼现金、空出仓位）。

被跳过的信号及其卖出腿整笔作废（不返回、不产生任何现金变动）。
"""

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
            accepted.append(t)
            if t.sell_time:
                sells_by_day.setdefault(t.sell_time, []).append(t)
                if t.sell_time == ds:  # 当日回转：立即了结，不占仓位
                    cash += (t.sell_price or 0.0) * (t.sell_amount or 0.0)
                    open_count -= 1

    # 被跳过的信号数 = 候选信号总数 - 成交数（区间外日期的信号不会被遍历到）
    total_candidates = sum(len(v) for v in buys_by_day.values())
    skipped = total_candidates - len(accepted)
    return accepted, skipped
