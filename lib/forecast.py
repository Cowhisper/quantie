"""明日买入预测的扫描内核（quantie.py forecast / daily 两个任务共用）。

口径：信号日 = 库中最新交易日（lib/download.last_trade_date），每只股票
的 K 线末尾追加一根「买入日」合成 K 线（开盘 = 信号日收盘价，仅作参考价），
复用策略的 run() 产出买入日 = 预测日的信号 —— 与回测完全同一套信号逻辑。
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from lib.engine import build_strategies, load_data
from lib.download import last_trade_date, next_trade_date
from lib.strategies.base import BaseStrategy
from lib.trade import Trade

FORECAST_COLUMNS = ["股票代码", "股票名称", "建议股数", "参考价格",
                    "评分", "信号日期", "买入原因"]


def _append_entry_bar(g: pd.DataFrame, entry_date) -> pd.DataFrame:
    """
    在单只股票K线末尾追加 entry_date 的合成K线（开盘 = 信号日收盘价，
    其余行情字段 NaN）。策略 run() 要求信号日次日有K线才发信号，这根合成K线
    让「信号日 = 库中最后一个交易日」的信号把买入挂到 entry_date；
    合成开盘价即参考价（实际成交以次日开盘价为准）。
    """
    last = g.iloc[-1]
    bar = {col: float("nan") for col in g.columns}
    bar["ticker"] = last["ticker"]
    bar["name"] = last["name"] if "name" in g.columns else last["ticker"]
    if "sector" in g.columns:
        bar["sector"] = last["sector"]
    bar["date"] = pd.Timestamp(entry_date)
    bar["open"] = float(last["close"])
    return pd.concat([g, pd.DataFrame([bar])], ignore_index=True)


def forecast_strategy(strategy: BaseStrategy,
                      stock_data: Dict[str, pd.DataFrame],
                      entry_date: str, signal_date: pd.Timestamp) -> List[Trade]:
    """
    扫描全部股票，返回买入日 = entry_date 的信号（信号日均为 signal_date），
    按评分降序、股票代码升序排列。只有行情停在 signal_date（库中最新交易日）
    的股票才可能入选（更早停牌的个股信号已过期，且停牌无法买入）。
    """
    picks: List[Trade] = []
    for ticker, g in stock_data.items():
        g = g.sort_values("date").reset_index(drop=True)
        if g.empty or g["date"].iloc[-1] != signal_date:
            continue
        g_ext = _append_entry_bar(g, entry_date)
        for t in strategy.run(g_ext, g_ext["date"].iloc[0], entry_date):
            if t.buy_time == entry_date:
                picks.append(t)
    picks.sort(key=lambda t: (-(t.score or 0.0), t.ticker))
    return picks


def _explain_zero_picks(strategy: BaseStrategy,
                        stock_data: Dict[str, pd.DataFrame],
                        signal_ts: pd.Timestamp) -> None:
    """策略选出 0 只时，按策略自身的过滤阶段拆解剩余股票数，区分
    「当日确实无形态」与「数据/逻辑异常」。策略需实现
    signal_day_breakdown(df, start, end)（如 RossCameronGapAndGoStrategy）；
    这里把信号日限制在 signal_ts 单日，逐股累加各阶段剩余行数。"""
    breakdown = getattr(strategy, "signal_day_breakdown", None)
    if breakdown is None:
        return
    totals: Dict[str, int] = {}
    for g in stock_data.values():
        if g.empty or g["date"].iloc[-1] != signal_ts:
            continue
        for label, n in breakdown(g, signal_ts, signal_ts):
            totals[label] = totals.get(label, 0) + n
    if not totals:
        return
    flow = " → ".join(f"{label} {n}" for label, n in totals.items())
    print(f"    [诊断] 信号日逐阶段剩余股票数: {flow}")


def write_forecast_csv(picks: List[Trade], path: str, signal_date: str) -> str:
    """写出单策略的明日买入清单 CSV（utf-8-sig，Excel 可直接打开）。"""
    rows = [{
        "股票代码": t.ticker,
        "股票名称": t.name,
        "建议股数": int(t.buy_amount),
        "参考价格": round(t.buy_price, 3),
        "评分": round(t.score, 4),
        "信号日期": signal_date,
        "买入原因": t.buy_reason,
    } for t in picks]
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows, columns=FORECAST_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig")
    return path


def _as_list(v, default: List[str]) -> List[str]:
    """命令行 key=value 传进来的列表是字符串："[ma,ross]" 按列表解析，
    单值按一个元素处理；None 用默认值。"""
    if v is None:
        return list(default)
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(v)
                return [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                # 元素不带引号（如 [ma,ross]）时按逗号切分
                return [x.strip().strip("'\"") for x in v[1:-1].split(",")
                        if x.strip()]
        return [v]
    return [str(x) for x in v]


def run_scan(db, warmup=120, strategies=None, conditions=None,
             date=None, max_positions=10):
    """第 2 步策略扫描：信号日 = 库中最新交易日，扫描各策略在预测买入日的
    信号。返回 (signal_date, entry_date, results, stock_data)，
    results = [(策略短名, 策略实例, picks)]（picks 已按 max_positions 截断）。
    forecast / daily 两个任务共用本函数，保证预测口径一致。"""
    strategy_names = _as_list(strategies, ["ross"])
    if isinstance(conditions, str):
        conditions = None  # 命令行不便传映射，条件映射请写在配置文件里

    signal_date = last_trade_date(db)
    if not signal_date:
        raise SystemExit("数据库中没有数据，请先运行数据下载（quantie.py download）"
                         "或先执行数据更新")
    if date:
        entry_date = date
    else:
        # 买入日 = 信号日的下一个交易日（与策略 run() 的入场逻辑一致：
        # 凌晨盘前跑 -> 今天开盘买；盘后跑且数据最新 -> 明天）。
        # 交易日历不可用时退回「运行日 +1 天」的近似
        entry_date = next_trade_date(signal_date)
        if entry_date is None:
            entry_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"    [警告] 交易日历查询失败，买入日退回运行日+1天 = {entry_date}")
    if entry_date <= signal_date:
        raise SystemExit(f"买入日（{entry_date}）必须晚于库中最新交易日 {signal_date}")
    # 数据陈旧检查：买入日已过去（含今天已收盘）说明库中缺最近交易日的行情，
    # 此时产出的挂单在模拟账户入场校验时对不上策略买点，应先更新数据再预测
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if entry_date < today or (entry_date == today and now.hour >= 15):
        print(f"    [警告] 买入日 {entry_date} 早于当前时间（库中最新行情仅到 "
              f"{signal_date}），数据可能不是最新，建议先更新数据再重新预测")

    df = load_data(db, signal_date, signal_date, warmup_days=int(warmup))
    if df.empty:
        raise SystemExit(f"数据库中 {signal_date} 前无数据，无法计算指标")
    stock_data: Dict[str, pd.DataFrame] = {
        ticker: g.reset_index(drop=True)
        for ticker, g in df.groupby("ticker", sort=False)
    }
    signal_ts = pd.Timestamp(signal_date)

    results = []
    for short_name, strategy in zip(strategy_names,
                                    build_strategies(strategy_names, conditions)):
        picks = forecast_strategy(strategy, stock_data, entry_date, signal_ts)
        if max_positions > 0:
            picks = picks[:max_positions]
        results.append((short_name, strategy, picks))
    return signal_date, entry_date, results, stock_data
