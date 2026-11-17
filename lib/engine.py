"""回测评估框架：运行策略并生成 HTML 可视化报告。"""

import html
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from yacrs import _C, cli, configurable, register

import lib  # noqa: F401  import lib 即触发所有策略/条件的 yacrs 注册
from lib.account import simulate_account
from lib.strategies.base import BaseStrategy
from lib.trade import Trade

DEFAULT_DB = "a_stock_daily.db"

OVERVIEW_METRICS = [
    "回测股票数", "总交易次数", "盈利次数", "亏损次数",
    "胜率", "总收益额", "总收益率", "最大单笔盈利", "最大单笔亏损", "平均收益率",
]

DEFAULT_INITIAL_CAPITAL = 1_000_000.0

DAY_COLUMNS = ["日期", "总资产", "现金", "持仓金额", "当日收益率",
               "收益额", "累计收益率", "累计收益额", "持仓股票数"]

HOLDING_COLUMNS = [
    "股票代码", "名称", "买入日期", "卖出日期", "买入价格/股数",
    "卖出价格/股数", "开盘价格", "收盘价格", "当日收益率", "持有收益率",
    "卖出原因", "买入原因",
]

# 配置文件中可调的策略注册表：短名 -> 类名
# （类用 yacrs @register 注册在类上，经 lib/__init__.py 导入完成注册；
#  实例通过 configurable()(类名) 创建，构造参数自动绑定同名配置段）
STRATEGY_REGISTRY: Dict[str, str] = {
    "ma": "MovingAverageCrossStrategy",
    "ross": "RossCameronGapAndGoStrategy",
}

CONDITION_NAMES = (
    "DateRangeCondition",
    "GapUpCondition",
    "PriceRangeCondition",
    "RelativeVolumeCondition",
)


def _instantiate(cls_name: str, **kwargs):
    """按注册名实例化（yacrs：configurable()(类名)(**kwargs)）。"""
    return configurable()(cls_name)(**kwargs)


def build_strategies(strategy_names: List[str],
                     condition_map: Optional[Dict[str, List[str]]] = None
                     ) -> List[BaseStrategy]:
    """按配置的策略名/条件名实例化策略；条件与策略的构造参数自动绑定
    各自的同名配置段（如 PriceRangeCondition.min_price）。"""
    condition_map = condition_map or {}
    strategies: List[BaseStrategy] = []
    for name in strategy_names:
        if name not in STRATEGY_REGISTRY:
            raise SystemExit(f"未知策略: {name}（可选: {', '.join(STRATEGY_REGISTRY)}）")
        cls_name = STRATEGY_REGISTRY[name]
        kwargs: Dict[str, object] = {}
        # yacrs 登记的构造参数名；支持 conditions 的策略才装配条件
        if "conditions" in _C._configurables[cls_name]["kwargs"]:
            conds = []
            # 注意 yacrs Node 重写了 get()（点路径取值且缺失时报错），这里用 dict 语义
            cond_names = condition_map[name] if name in condition_map else []
            for cond_name in cond_names:
                if cond_name not in CONDITION_NAMES:
                    raise SystemExit(
                        f"未知条件: {cond_name}（可选: {', '.join(CONDITION_NAMES)}）")
                conds.append(_instantiate(cond_name))
            kwargs["conditions"] = conds
        strategies.append(_instantiate(cls_name, **kwargs))
    return strategies


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------

def load_data(db_path: str, start_time: str, end_time: str,
              warmup_days: int = 120,
              limit: int = 0) -> pd.DataFrame:
    """从 SQLite 加载日线数据，附带 start_time 之前的历史数据用于指标预热。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    load_start = (pd.Timestamp(start_time) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM stock_daily WHERE date >= ? AND date <= ? ORDER BY ticker, date",
        conn, params=(load_start, end_time),
    )
    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    # 历史数据遗留：旧版下载脚本曾把 tradeStatus='1' 误存为 name
    # （见 fix_stock_names.py），无有效名称时退回股票代码展示
    bad_name = (df["name"].isna()
                | df["name"].astype(str).str.strip().isin(("", "1", "0")))
    df.loc[bad_name, "name"] = df.loc[bad_name, "ticker"]
    if limit and limit > 0:
        tickers = sorted(df["ticker"].unique())[:limit]
        df = df[df["ticker"].isin(tickers)]
    return df


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------

def compute_overview(trades: List[Trade], stock_count: int,
                     initial_capital: float = 0.0,
                     total_pnl: Optional[float] = None) -> Dict[str, float]:
    """
    概览统计。总收益额 / 总收益率为账户口径：total_pnl = 已实现盈亏
    （各笔 (卖出价 − 买入价) × 卖出量）+ 期末未平仓浮盈（按期末前最近收盘），
    总收益率 = total_pnl / initial_capital，与交易明细日表的累计收益额 /
    累计收益率（末日值）完全一致。未传 total_pnl 时退回信号口径
    （各笔收益率之和，仅用于无账户信息的场景）。
    """
    closed = [t for t in trades if t.return_pct is not None]
    returns = [t.return_pct for t in closed]
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)

    if total_pnl is None:
        total_pnl_v: float = float(np.sum(returns)) if returns else 0.0
        total_ret: float = total_pnl_v
    else:
        total_pnl_v = round(float(total_pnl), 2)
        total_ret = (float(total_pnl) / initial_capital) if initial_capital else 0.0

    return {
        "回测股票数": stock_count,
        "总交易次数": len(closed),
        "盈利次数": wins,
        "亏损次数": losses,
        "胜率": wins / len(closed) if closed else 0.0,
        "总收益额": total_pnl_v,
        "总收益率": total_ret,
        "最大单笔盈利": max(returns) if returns else 0.0,
        "最大单笔亏损": min(returns) if returns else 0.0,
        "平均收益率": float(np.mean(returns)) if returns else 0.0,
    }


PCT_METRICS = {"胜率", "总收益率", "最大单笔盈利", "最大单笔亏损", "平均收益率"}
MONEY_METRICS = {"总收益额"}


def format_metric(key: str, value) -> str:
    if key in PCT_METRICS:
        return f"{value * 100:.2f}%"
    if key in MONEY_METRICS:
        return f"{value:,.2f}"
    return str(value)


def account_total_pnl(trades: List[Trade], stock_data: Dict[str, pd.DataFrame],
                      end_time) -> float:
    """账户总盈亏 = 已实现盈亏（各笔 (卖出价 − 买入价) × 卖出量）
    + 期末未平仓浮盈（按 end_time 前最近收盘价盯市），
    与交易明细日表末日的累计收益额同口径。"""
    end_ts = pd.Timestamp(end_time)
    pnl = 0.0
    for t in trades:
        if not t.buy_time:
            continue
        if t.sell_price is not None:
            pnl += (t.sell_price - t.buy_price) * (t.sell_amount or t.buy_amount)
        else:
            g = stock_data.get(t.ticker)
            if g is None:
                continue
            gg = g[g["date"] <= end_ts]
            if gg.empty:
                continue
            last_close = float(gg["close"].iloc[-1])
            pnl += (last_close - t.buy_price) * t.buy_amount
    return pnl


def compute_equity(trades: List[Trade], stock_data: Dict[str, pd.DataFrame],
                   start_time, end_time) -> Dict[str, list]:
    """
    账户盈亏曲线（等额定仓、逐日盯市）：浮盈按持仓期每日收盘价结算，
    卖出日按实际卖出价计入已实现盈亏。返回日线、月末权益与月度盈亏。
    """
    start_ts, end_ts = pd.Timestamp(start_time), pd.Timestamp(end_time)
    closed = [t for t in trades if t.return_pct is not None]
    all_dates = sorted({d for df in stock_data.values() for d in df["date"]
                        if start_ts <= d <= end_ts})
    if not closed or not all_dates:
        return {"daily": [], "monthly": [], "monthly_pnl": []}

    dates_ns = np.array(all_dates, dtype="datetime64[ns]")
    n = len(all_dates)
    equity = np.zeros(n)
    close_cache: Dict[str, np.ndarray] = {}

    def close_arr(ticker: str) -> np.ndarray:
        if ticker not in close_cache:
            df = stock_data[ticker]
            price_map = dict(zip(df["date"], df["close"]))
            close_cache[ticker] = np.array(
                [price_map.get(d, np.nan) for d in all_dates], dtype=float)
        return close_cache[ticker]

    for t in closed:
        arr = close_arr(t.ticker)
        i0 = int(np.searchsorted(dates_ns, np.datetime64(pd.Timestamp(t.buy_time))))
        i1 = int(np.searchsorted(dates_ns, np.datetime64(pd.Timestamp(t.sell_time))))
        i0 = min(max(i0, 0), n - 1)
        i1 = min(max(i1, 0), n - 1)
        if i1 < i0:
            continue
        # 买入日（含）至卖出日前一日：持仓盯市浮盈
        seg = arr[i0:i1]
        equity[i0:i1] += np.where(np.isnan(seg), 0.0, (seg - t.buy_price) * t.buy_amount)
        # 卖出日：按实际卖出价计入已实现盈亏
        equity[i1] += (t.sell_price - t.buy_price) * t.sell_amount

    daily = [[d.strftime("%Y-%m-%d"), round(float(v), 2)]
             for d, v in zip(all_dates, equity)]
    series = pd.Series(equity, index=pd.to_datetime(all_dates))
    monthly = series.resample("ME").last()
    monthly_list = [[d.strftime("%Y-%m"), round(float(v), 2)]
                    for d, v in monthly.items()]
    monthly_pnl = [[d.strftime("%Y-%m"), round(float(v), 2)]
                   for d, v in monthly.diff().items() if not np.isnan(v)]
    return {"daily": daily, "monthly": monthly_list, "monthly_pnl": monthly_pnl}


def compute_total_assets(days: List[dict]) -> Dict[str, list]:
    """总资产变化曲线：日线（每日收盘总资产）与月线（月末总资产）。"""
    if not days:
        return {"assets_daily": [], "assets_monthly": []}
    series = pd.Series([d["total_assets"] for d in days],
                       index=pd.to_datetime([d["date"] for d in days]))
    monthly = series.resample("ME").last()
    return {
        "assets_daily": [[d["date"], d["total_assets"]] for d in days],
        "assets_monthly": [[i.strftime("%Y-%m"), round(float(v), 2)]
                           for i, v in monthly.items()],
    }


# ----------------------------------------------------------------------
# 逐日账户状态
# ----------------------------------------------------------------------

def compute_daily_account(trades: List[Trade], stock_data: Dict[str, pd.DataFrame],
                          start_time, end_time,
                          initial_capital: float = DEFAULT_INITIAL_CAPITAL
                          ) -> List[dict]:
    """
    逐日账户状态（逐日盯市）：以 initial_capital 为初始资金，买入日按买入价
    扣减现金、卖出日按卖出价回笼现金，持仓按当日收盘价（停牌沿用最近收盘价）
    计入持仓金额。返回回测区间内每个交易日的账户快照（含当日持仓列表）。

    配合 lib/account.py 的账户模拟（Engine.evaluate 会先对策略信号执行
    资金/持仓约束再传入 trades），现金不会为负、持仓数不超过上限。

    每个交易日一行：
        {date, total_assets, cash, position_value,
         day_return_pct, day_pnl, cum_return_pct, cum_pnl, holdings}
    day_return_pct / day_pnl 为当日相对前一交易日总资产的收益率 / 收益额
    （首日相对初始资金）；cum_* 为相对初始资金的累计口径。
    holding 含 trade_idx（trades 列表下标，用于取 K 线图）、当日开盘/收盘价
    （open_price / close_price，停牌无 K 线为 None）；
    day_return_pct 为该股票当日收益率（买入日相对买入价、持仓中日间涨跌、
    卖出日相对前收），holding_return_pct 为持有期累计收益率
    （卖出日为已实现）；卖出字段（sell_time/sell_price/sell_amount/
    sell_reason）仅在卖出当天给出，其余持仓日为 None。
    """
    start_ts, end_ts = pd.Timestamp(start_time), pd.Timestamp(end_time)
    initial_capital = float(initial_capital)

    all_dates = sorted({d for df in stock_data.values() for d in df["date"]
                        if start_ts <= d <= end_ts})

    dated = []  # (idx, trade, buy_ts, sell_ts)
    for idx, t in enumerate(trades):
        if not t.buy_time:
            continue
        buy_ts = pd.Timestamp(t.buy_time)
        sell_ts = pd.Timestamp(t.sell_time) if t.sell_time else None
        dated.append((idx, t, buy_ts, sell_ts))
    by_idx = {item[0]: item for item in dated}

    buys_by_day: Dict[str, list] = {}
    sells_by_day: Dict[str, list] = {}
    for item in dated:
        buys_by_day.setdefault(item[2].strftime("%Y-%m-%d"), []).append(item)
        if item[3] is not None:
            sells_by_day.setdefault(item[3].strftime("%Y-%m-%d"), []).append(item)

    bar_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def bars(ticker: str):
        if ticker not in bar_cache:
            df = stock_data[ticker]
            bar_cache[ticker] = (df["date"].values,
                                 df["open"].values.astype(float),
                                 df["close"].values.astype(float))
        return bar_cache[ticker]

    def close_on(ticker: str, d) -> float:
        """ticker 在日期 d（含）之前最近一个交易日的收盘价，无则 nan。"""
        dates_arr, _, closes = bars(ticker)
        j = int(np.searchsorted(dates_arr, np.datetime64(d), side="right")) - 1
        return float(closes[j]) if j >= 0 else float("nan")

    def prev_close_on(ticker: str, d) -> float:
        """ticker 在日期 d 之前最近一个交易日（严格在前）的收盘价，无则 nan。"""
        dates_arr, _, closes = bars(ticker)
        j = int(np.searchsorted(dates_arr, np.datetime64(d), side="left")) - 1
        return float(closes[j]) if j >= 0 else float("nan")

    def bar_on(ticker: str, d) -> Tuple[Optional[float], Optional[float]]:
        """ticker 在日期 d 当日的 (开盘, 收盘)，当日无 K 线（停牌）为 (None, None)。"""
        dates_arr, opens, closes = bars(ticker)
        j = int(np.searchsorted(dates_arr, np.datetime64(d), side="left"))
        if j < len(dates_arr) and dates_arr[j] == np.datetime64(d):
            return float(opens[j]), float(closes[j])
        return None, None

    cash = initial_capital
    open_idx = set()
    days = []
    prev_total = initial_capital
    for d in all_dates:
        ds = d.strftime("%Y-%m-%d")
        # 先卖出回笼现金，再买入占用现金（T+1：卖出日必晚于买入日，
        # 由 lib/account.py 的账户模拟保证，买入日即占用仓位）
        for idx, t, _, _ in sells_by_day.get(ds, []):
            cash += (t.sell_price or 0.0) * (t.sell_amount or 0.0)
            open_idx.discard(idx)
        for idx, t, _, _ in buys_by_day.get(ds, []):
            cash -= t.buy_price * t.buy_amount
            open_idx.add(idx)

        position_value = 0.0
        for idx in open_idx:
            close = close_on(by_idx[idx][1].ticker, d)
            if not np.isnan(close):
                position_value += close * by_idx[idx][1].buy_amount

        holdings = []
        for idx, t, buy_ts, sell_ts in dated:
            if buy_ts > d or (sell_ts is not None and sell_ts < d):
                continue  # 尚未买入或已经卖出的交易日不在当日持仓中
            sold_today = sell_ts == d
            open_p, close_p = bar_on(t.ticker, d)
            prev_close = prev_close_on(t.ticker, d)
            if sold_today:
                hold_ret = ((t.sell_price - t.buy_price) / t.buy_price
                            if t.sell_price is not None and t.buy_price else None)
                if np.isnan(prev_close) or not prev_close:
                    # 卖出日无前收（上市首日等）：以买入价为当日基准
                    day_ret = hold_ret
                else:
                    day_ret = (t.sell_price - prev_close) / prev_close
            else:
                if close_p is None:
                    # 停牌当日：沿用最近收盘价算持有收益率，当日收益率不显示
                    close = close_on(t.ticker, d)
                    day_ret = None
                elif buy_ts == d:
                    close = close_p
                    day_ret = ((close_p - t.buy_price) / t.buy_price
                               if t.buy_price else None)
                else:
                    close = close_p
                    day_ret = ((close_p - prev_close) / prev_close
                               if not np.isnan(prev_close) and prev_close else None)
                hold_ret = ((close - t.buy_price) / t.buy_price
                            if not np.isnan(close) and t.buy_price else None)
            holdings.append({
                "trade_idx": idx,
                "ticker": t.ticker,
                "name": t.name,
                "buy_time": t.buy_time,
                "buy_price": t.buy_price,
                "buy_amount": t.buy_amount,
                "buy_reason": t.buy_reason,
                "sell_time": t.sell_time if sold_today else None,
                "sell_price": t.sell_price if sold_today else None,
                "sell_amount": t.sell_amount if sold_today else None,
                "sell_reason": t.sell_reason if sold_today else None,
                "open_price": open_p,
                "close_price": close_p,
                "day_return_pct": day_ret,
                "holding_return_pct": hold_ret,
            })

        total = cash + position_value
        day_pnl = total - prev_total
        days.append({
            "date": ds,
            "total_assets": round(total, 2),
            "cash": round(cash, 2),
            "position_value": round(position_value, 2),
            "day_return_pct": day_pnl / prev_total if prev_total else 0.0,
            "day_pnl": round(day_pnl, 2),
            "cum_return_pct": ((total - initial_capital) / initial_capital
                               if initial_capital else 0.0),
            "cum_pnl": round(total - initial_capital, 2),
            "holdings": holdings,
        })
        prev_total = total
    return days


# ----------------------------------------------------------------------
# K线图 (内联 SVG，无需外部依赖)
# ----------------------------------------------------------------------

def _candlestick_svg(window: pd.DataFrame, buy_time: str, sell_time: str,
                     width: int = 880, height: int = 300) -> str:
    """根据窗口内的日线数据绘制 K 线 SVG，标注买卖点。"""
    if window.empty:
        return "<p>无图表数据</p>"

    pad_left, pad_right, pad_top, pad_bottom = 50, 20, 15, 25
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    high_max = float(window["high"].max())
    low_min = float(window["low"].min())
    span = (high_max - low_min) or 1.0

    n = len(window)
    slot = plot_w / max(n, 1)
    body_w = max(min(slot * 0.6, 14.0), 2.0)

    def y(price: float) -> float:
        return pad_top + (high_max - price) / span * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']

    # 网格线 + 价格刻度
    for j in range(5):
        price = low_min + span * j / 4
        yy = y(price)
        parts.append(f'<line x1="{pad_left}" y1="{yy:.1f}" x2="{width - pad_right}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_left - 5}" y="{yy + 3:.1f}" font-size="9" text-anchor="end" fill="#999">{price:.2f}</text>')

    for i, row in enumerate(window.itertuples()):
        cx = pad_left + slot * i + slot / 2
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        color = "#e0342f" if c >= o else "#21a366"  # 红涨绿跌
        parts.append(f'<line x1="{cx:.1f}" y1="{y(h):.1f}" x2="{cx:.1f}" y2="{y(l):.1f}" stroke="{color}" stroke-width="1"/>')
        top, bh = (y(o), y(c) - y(o)) if c >= o else (y(c), y(o) - y(c))
        parts.append(f'<rect x="{cx - body_w / 2:.1f}" y="{top:.1f}" width="{body_w:.1f}" height="{max(bh, 1.0):.1f}" fill="{color}"/>')
        if i % max(n // 8, 1) == 0:
            parts.append(f'<text x="{cx:.1f}" y="{height - 8}" font-size="8" text-anchor="middle" fill="#999">{row.date.strftime("%m-%d")}</text>')

    # 标注买卖点
    for date_str, label, color in ((buy_time, "B", "#1f77b4"), (sell_time, "S", "#ff7f0e")):
        if not date_str:
            continue
        ts = pd.Timestamp(date_str)
        idx = int((window["date"] == ts).idxmax()) if (window["date"] == ts).any() else None
        if idx is None:
            continue
        pos = window.index.get_loc(idx)
        row = window.iloc[pos]
        cx = pad_left + slot * pos + slot / 2
        cy = y(float(row.low))
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy - 8:.1f}" font-size="10" font-weight="bold" text-anchor="middle" fill="{color}">{label}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _holding_window(df_stock: pd.DataFrame, buy_time: str, sell_time: str,
                    pre_days: int = 10) -> pd.DataFrame:
    """K 线窗口：买入前最多 pre_days 个交易日 ~ 卖出日（未卖出则到数据末尾）。"""
    dates = df_stock["date"]
    i0 = int(dates.searchsorted(pd.Timestamp(buy_time)))
    start_pos = max(0, i0 - pre_days)
    mask = dates >= dates.iloc[start_pos]
    if sell_time:
        mask &= dates <= pd.Timestamp(sell_time)
    return df_stock[mask].reset_index(drop=True)


def _holding_chart(t: Trade, stock_data: Dict[str, pd.DataFrame]) -> Optional[str]:
    """单笔持仓的 K 线 SVG（持有期 + 买入前最多 10 个交易日）。"""
    df_stock = stock_data.get(t.ticker)
    if df_stock is None or not t.buy_time:
        return None
    window = _holding_window(df_stock, t.buy_time, t.sell_time)
    return _candlestick_svg(window, t.buy_time, t.sell_time)


# ----------------------------------------------------------------------
# HTML 报告
# ----------------------------------------------------------------------


def _line_chart_svg(series: List[list], title: str,
                    width: int = 430, height: int = 200) -> str:
    """折线图 SVG：series = [[label, value], ...]（账户盈亏日线）。"""
    if not series:
        return f"<div class='chart-empty'>{html.escape(title)}：无数据</div>"
    pad_l, pad_r, pad_t, pad_b = 56, 12, 24, 22
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    values = [v for _, v in series]
    vmax, vmin = max(values), min(values)
    span = (vmax - vmin) or 1.0
    n = len(series)

    def x(i: int) -> float:
        return pad_l + plot_w * i / max(n - 1, 1)

    def y(v: float) -> float:
        return pad_t + (vmax - v) / span * plot_h

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(series))
    parts = [f'<svg viewBox="0 0 {width} {height}" class="eq-chart" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<text x="{width / 2}" y="14" font-size="11" text-anchor="middle" fill="#333">{html.escape(title)}</text>')
    for j in range(4):
        vv = vmin + span * j / 3
        yy = y(vv)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l - 4}" y="{yy + 3:.1f}" font-size="8" text-anchor="end" fill="#999">{vv:,.0f}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>')
    for i in (0, n // 2, n - 1):
        parts.append(f'<text x="{x(i):.1f}" y="{height - 6}" font-size="8" text-anchor="middle" fill="#999">{series[i][0]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _bar_chart_svg(series: List[list], title: str,
                   width: int = 430, height: int = 200) -> str:
    """柱状图 SVG：series = [[label, value], ...]，红盈绿亏（账户盈亏月度）。"""
    if not series:
        return f"<div class='chart-empty'>{html.escape(title)}：无数据</div>"
    pad_l, pad_r, pad_t, pad_b = 56, 12, 24, 22
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    vmax = max(max(v for _, v in series), 0.0)
    vmin = min(min(v for _, v in series), 0.0)
    span = (vmax - vmin) or 1.0
    n = len(series)
    slot = plot_w / n
    bar_w = max(min(slot * 0.6, 30.0), 2.0)

    def y(v: float) -> float:
        return pad_t + (vmax - v) / span * plot_h

    y0 = y(0.0)  # 零线
    parts = [f'<svg viewBox="0 0 {width} {height}" class="eq-chart" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<text x="{width / 2}" y="14" font-size="11" text-anchor="middle" fill="#333">{html.escape(title)}</text>')
    for j in range(4):
        vv = vmin + span * j / 3
        yy = y(vv)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l - 4}" y="{yy + 3:.1f}" font-size="8" text-anchor="end" fill="#999">{vv:,.0f}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width - pad_r}" y2="{y0:.1f}" stroke="#bbb"/>')
    for i, (_, v) in enumerate(series):
        cx = pad_l + slot * i + slot / 2
        top, bh = (y(v), y0 - y(v)) if v >= 0 else (y0, y(v) - y0)
        color = "#e0342f" if v >= 0 else "#21a366"
        parts.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{max(bh, 0.5):.1f}" fill="{color}"/>')
    for i in (0, n // 2, n - 1):
        cx = pad_l + slot * i + slot / 2
        parts.append(f'<text x="{cx:.1f}" y="{height - 6}" font-size="8" text-anchor="middle" fill="#999">{series[i][0]}</text>')
    parts.append("</svg>")
    return "".join(parts)


_PAGE_CSS = """
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #f5f6fa; color: #333; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 32px; }
.card { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.desc { color: #888; font-size: 13px; margin: 4px 0 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
th, td { border: 1px solid #eee; padding: 6px 10px; text-align: center; white-space: nowrap; }
th { background: #fafbfc; }
.table-wrap { overflow-x: auto; }
tr.day-row, tr.holding-row { cursor: pointer; }
tr.day-row:hover, tr.holding-row:hover { background: #f0f7ff; }
tr.day-detail > td { padding: 8px 14px 14px; background: #fcfcfd; text-align: left; }
tr.day-detail.hidden, tr.holding-chart.hidden { display: none; }
table.holding-table { font-size: 12px; }
table.holding-table th { background: #f5f7fa; font-weight: normal; color: #666; }
tr.holding-chart td { padding: 8px; background: #fff; text-align: left; }
.pos { color: #e0342f; } .neg { color: #21a366; }
.toggle-hint { color: #bbb; font-size: 11px; font-weight: normal; margin-left: 8px; }
.tabs { margin: 16px 0; border-bottom: 2px solid #e4e7ed; }
.tabs button { background: none; border: none; padding: 10px 20px; font-size: 15px;
               cursor: pointer; color: #666; border-bottom: 2px solid transparent;
               margin-bottom: -2px; }
.tabs button.active { color: #1f77b4; border-bottom-color: #1f77b4; font-weight: bold; }
.charts { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
.eq-chart { flex: 1; min-width: 380px; background: #fcfcfd; border: 1px solid #f0f0f0;
            border-radius: 6px; }
.chart-empty { flex: 1; min-width: 380px; color: #999; font-size: 12px; text-align: center;
               padding: 40px 0; border: 1px dashed #eee; border-radius: 6px; }
.pager { margin: 12px 0; text-align: center; font-size: 13px; color: #666; }
.pager button { padding: 4px 14px; margin: 0 8px; border: 1px solid #dcdfe6;
                background: #fff; border-radius: 4px; cursor: pointer; }
.pager button:disabled { color: #c0c4cc; cursor: not-allowed; }
.pager select { padding: 4px 8px; margin: 0 8px; border: 1px solid #dcdfe6;
                background: #fff; border-radius: 4px; font-size: 13px; color: #666; }
"""

_PAGE_JS = """
// 策略标签页切换（每个策略一个独立报告）
document.querySelectorAll('.tabs button').forEach(function (btn) {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.tabs button').forEach(function (b) {
      b.classList.toggle('active', b === btn);
    });
    document.querySelectorAll('.strategy-section').forEach(function (sec) {
      sec.style.display = (sec.id === btn.dataset.target) ? 'block' : 'none';
    });
  });
});

// 每个策略交易明细按月分页：上一月 / 下一月按钮 + 月份下拉框跳转，
// 每个月显示其全部交易日的账户明细
document.querySelectorAll('.strategy-section').forEach(function (sec) {
  var units = Array.prototype.slice.call(sec.querySelectorAll('tbody.unit'));
  var months = [];
  units.forEach(function (u) {
    var m = u.dataset.month;
    if (months.indexOf(m) === -1) months.push(m);
  });
  var mi = 0;
  var info = sec.querySelector('.pager .info');
  var prev = sec.querySelector('.pager .prev');
  var next = sec.querySelector('.pager .next');
  var jump = sec.querySelector('.pager .month-jump');
  months.forEach(function (m) {
    var opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    jump.appendChild(opt);
  });
  function render() {
    if (!months.length) {
      info.textContent = '无数据';
      prev.disabled = next.disabled = true;
      jump.disabled = true;
      return;
    }
    var cur = months[mi];
    var count = 0;
    units.forEach(function (u) {
      var show = u.dataset.month === cur;
      if (show) count++;
      u.style.display = show ? 'table-row-group' : 'none';
      if (!show) {  // 切换月时折叠展开的持仓明细与 K 线图
        var d = u.querySelector('.day-detail');
        if (d) d.classList.add('hidden');
        u.querySelectorAll('.holding-chart').forEach(function (c) {
          c.classList.add('hidden');
        });
      }
    });
    jump.value = cur;
    info.textContent = cur + '（共 ' + count + ' 个交易日）';
    prev.disabled = mi <= 0;
    next.disabled = mi >= months.length - 1;
  }
  prev.addEventListener('click', function () { if (mi > 0) { mi--; render(); } });
  next.addEventListener('click', function () { if (mi < months.length - 1) { mi++; render(); } });
  jump.addEventListener('change', function () {
    var idx = months.indexOf(jump.value);
    if (idx !== -1) { mi = idx; render(); }
  });
  render();
});

// 点击交易日行展开 / 折叠当日持仓明细；点击持仓行展开 / 折叠 K 线图
// （K 线图按交易去重存放在隐藏的 chart-pool 中，首次点开时克隆到行内）
document.addEventListener('click', function (e) {
  var hrow = e.target.closest('tr.holding-row');
  if (hrow) {
    var chartRow = hrow.nextElementSibling;
    if (chartRow && chartRow.classList.contains('holding-chart')) {
      var cell = chartRow.querySelector('.chart-cell');
      if (cell && !cell.hasChildNodes()) {
        var sec = hrow.closest('.strategy-section');
        var src = sec.querySelector(".chart-src[data-trade='" + cell.dataset.trade + "']");
        if (src) {
          cell.appendChild(src.firstElementChild.cloneNode(true));
        } else {
          cell.innerHTML = "<span class='desc'>（图表已省略，持仓记录过多）</span>";
        }
      }
      chartRow.classList.toggle('hidden');
    }
    return;
  }
  var drow = e.target.closest('tr.day-row');
  if (!drow) return;
  var detail = drow.nextElementSibling;
  if (detail && detail.classList.contains('day-detail')) {
    detail.classList.toggle('hidden');
  }
});
"""


def _render_overview(overview: Dict[str, float]) -> str:
    header = "".join(f"<th>{k}</th>" for k in OVERVIEW_METRICS)
    values = "".join(
        f"<td>{format_metric(k, overview[k])}</td>"
        for k in OVERVIEW_METRICS
    )
    table = f"<table><tr>{header}</tr><tr>{values}</tr></table>"
    return f"<div class='table-wrap'>{table}</div>"


def _render_daily_table(days: List[dict], trades: List[Trade],
                        stock_data: Dict[str, pd.DataFrame],
                        max_charts: int = 0) -> str:
    """渲染按交易日的账户明细表（每个交易日一个 tbody 单元，带 data-month
    供前端按月分页）：首列行为 总资产/现金/持仓金额/收益率/持仓股票数；
    点击行展开当日持仓明细表（每只股票一行，卖出字段仅在卖出当天有值），
    再点击持仓行展开 K 线图。
    K 线图按交易去重后集中放在隐藏的 chart-pool 中（同一笔交易在持仓期内
    每天点开都是同一张图），点击时克隆到对应位置；max_charts > 0 时只为前
    max_charts 笔交易嵌入 K 线图。"""
    esc = html.escape
    charts_left: Optional[int] = max_charts if max_charts > 0 else None
    chart_pool: Dict[int, str] = {}
    units = []
    for day in days:
        def money_cls(v: float) -> str:
            return "pos" if v > 0 else ("neg" if v < 0 else "")

        day_row = (
            f"<tr class='day-row'>"
            f"<td>{day['date']}</td>"
            f"<td>{day['total_assets']:,.2f}</td>"
            f"<td>{day['cash']:,.2f}</td>"
            f"<td>{day['position_value']:,.2f}</td>"
            f"<td class='{money_cls(day['day_return_pct'])}'>{day['day_return_pct'] * 100:+.2f}%</td>"
            f"<td class='{money_cls(day['day_pnl'])}'>{day['day_pnl']:,.2f}</td>"
            f"<td class='{money_cls(day['cum_return_pct'])}'>{day['cum_return_pct'] * 100:+.2f}%</td>"
            f"<td class='{money_cls(day['cum_pnl'])}'>{day['cum_pnl']:,.2f}</td>"
            f"<td>{len(day['holdings'])}</td>"
            f"</tr>"
        )

        holding_units = []
        for h in day["holdings"]:
            idx = h["trade_idx"]
            dret = h["day_return_pct"]
            hret = h["holding_return_pct"]
            dret_str = f"{dret * 100:+.2f}%" if dret is not None else "-"
            hret_str = f"{hret * 100:+.2f}%" if hret is not None else "-"
            buy_pa = f"{h['buy_price']:.2f} / {h['buy_amount']:,.0f}"
            sell_pa = (f"{h['sell_price']:.2f} / {h['sell_amount']:,.0f}"
                       if h["sell_time"] else "-")
            open_str = f"{h['open_price']:.2f}" if h["open_price"] is not None else "-"
            close_str = f"{h['close_price']:.2f}" if h["close_price"] is not None else "-"
            hrow = (
                f"<tr class='holding-row'>"
                f"<td>{esc(h['ticker'])}</td><td>{esc(h['name'])}</td>"
                f"<td>{esc(h['buy_time'] or '-')}</td>"
                f"<td>{esc(h['sell_time'] or '-')}</td>"
                f"<td>{buy_pa}</td><td>{sell_pa}</td>"
                f"<td>{open_str}</td><td>{close_str}</td>"
                f"<td class='{money_cls(dret or 0.0)}'>{dret_str}</td>"
                f"<td class='{money_cls(hret or 0.0)}'>{hret_str}</td>"
                f"<td>{esc(h['sell_reason'] or '-')}</td>"
                f"<td>{esc(h['buy_reason'] or '-')}</td>"
                f"</tr>"
            )
            if (charts_left is None or charts_left > 0) and idx not in chart_pool:
                chart = _holding_chart(trades[idx], stock_data)
                if chart is not None:
                    chart_pool[idx] = chart
                    if charts_left is not None:
                        charts_left -= 1
            holding_units.append(
                f"<tbody class='holding-unit'>{hrow}"
                f"<tr class='holding-chart hidden'>"
                f"<td colspan='{len(HOLDING_COLUMNS)}'>"
                f"<div class='chart-cell' data-trade='{idx}'></div></td></tr>"
                f"</tbody>"
            )
        if holding_units:
            holdings_table = (
                f"<table class='holding-table'>"
                f"<thead><tr>{''.join(f'<th>{c}</th>' for c in HOLDING_COLUMNS)}</tr></thead>"
                f"{''.join(holding_units)}</table>"
            )
        else:
            holdings_table = "<span class='desc'>当日无持仓</span>"
        detail_row = (
            f"<tr class='day-detail hidden'>"
            f"<td colspan='{len(DAY_COLUMNS)}'>{holdings_table}</td></tr>"
        )
        units.append(
            f"<tbody class='unit' style='display:none' data-month='{day['date'][:7]}'>"
            f"{day_row}{detail_row}</tbody>")

    pool = (f"<div class='chart-pool' style='display:none'>"
            + "".join(f"<div class='chart-src' data-trade='{i}'>{svg}</div>"
                      for i, svg in chart_pool.items())
            + "</div>")
    header = "".join(f"<th>{c}</th>" for c in DAY_COLUMNS)
    table = f"<table><thead><tr>{header}</tr></thead>{''.join(units)}</table>"
    pager = ("<div class='pager'><button class='prev'>&lt; 上一月</button>"
             "<select class='month-jump'></select>"
             "<span class='info'></span>"
             "<button class='next'>下一月 &gt;</button></div>")
    return f"<div class='table-wrap'>{table}</div>{pager}{pool}"


def generate_report(results: Dict[str, dict], output_path: str,
                    max_charts: int = 0, start_time=None, end_time=None,
                    initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> str:
    """
    生成 HTML 报告。results 格式:
        {策略名: {"overview": {...}, "trades": [Trade...], "stock_data": {ticker: df}}}
    每个策略一个标签页（独立报告），页首为概览 + 账户盈亏日/月图，
    交易明细按月分页显示（每月一页，支持上一月/下一月及月份下拉跳转；
    点击行展开当日持仓明细，再点击持仓行展开 K 线图）；
    max_charts > 0 时只为前 max_charts 笔交易嵌入 K 线图（同一笔交易在持仓期内
    共用同一张图）。
    传入 start_time / end_time 时渲染账户盈亏曲线，否则跳过。
    """
    tabs = []
    sections = []
    for si, (name, data) in enumerate(results.items()):
        overview_html = _render_overview(data["overview"])
        days = compute_daily_account(data["trades"], data["stock_data"],
                                     start_time, end_time,
                                     initial_capital=initial_capital)
        charts_html = ""
        if start_time and end_time:
            eq = compute_equity(data["trades"], data["stock_data"],
                                start_time, end_time)
            ta = compute_total_assets(days)
            charts_html = (
                f"<div class='charts'>"
                f"{_line_chart_svg(eq['daily'], '账户盈亏（日线）')}"
                f"{_bar_chart_svg(eq['monthly_pnl'], '账户盈亏（月度）')}"
                f"{_line_chart_svg(ta['assets_daily'], '总资产（日线）')}"
                f"{_line_chart_svg(ta['assets_monthly'], '总资产（月线）')}"
                f"</div>"
            )
        trades_html = _render_daily_table(days, data["trades"],
                                          data["stock_data"],
                                          max_charts=max_charts)
        hint = ("<span class='toggle-hint'>"
                "(点击交易日行展开当日持仓，点击持仓行展开 K 线图)</span>")
        active = "active" if si == 0 else ""
        display = "block" if si == 0 else "none"
        tabs.append(
            f"<button data-target='sec{si}' class='{active}'>{html.escape(name)}</button>"
        )
        sections.append(
            f"<div class='card strategy-section' id='sec{si}' style='display:{display}'>"
            f"<h2>{html.escape(name)}{hint}</h2>"
            f"<p class='desc'>{html.escape(data.get('description', ''))}</p>"
            f"{overview_html}{charts_html}<h2>交易明细（按月分页）</h2>{trades_html}</div>"
        )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>策略回测报告</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<div class="container">
<h1>策略回测报告</h1>
<div class="tabs">{''.join(tabs)}</div>
{''.join(sections)}
</div>
<script>{_PAGE_JS}</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    return output_path


# ----------------------------------------------------------------------
# 交易明细 Excel 导出
# ----------------------------------------------------------------------

TRADE_COLUMNS = ["策略", "股票代码", "股票名称", "买入日期", "买入价格", "买入数量",
                 "买入原因", "买入开盘价", "买入收盘价", "买入当日收益率",
                 "卖出日期", "卖出价格", "卖出数量", "卖出原因",
                 "持有收益率", "收益额", "评分"]


def _sheet_names(names: List[str]) -> Dict[str, str]:
    """策略名 -> 合法且互不重复的 sheet 名
    （Excel 限制：不超过 31 字符，不含 []:*?/\\）。"""
    used = set()
    out: Dict[str, str] = {}
    for name in names:
        base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "sheet"
        s, i = base, 2
        while s in used:
            suffix = f"_{i}"
            s = base[:31 - len(suffix)] + suffix
            i += 1
        used.add(s)
        out[name] = s
    return out


def _buy_day_bar(t: Trade, stock_data: Dict[str, pd.DataFrame]
                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """买入日当日的 (开盘, 收盘, 当日收益率)；当日无 K 线（停牌）或无买入日时
    返回 (None, None, None)。当日收益率 = (买入日收盘 − 买入价) / 买入价。"""
    df = stock_data.get(t.ticker)
    if df is None or not t.buy_time:
        return None, None, None
    row = df[df["date"] == pd.Timestamp(t.buy_time)]
    if row.empty:
        return None, None, None
    open_p = float(row["open"].iloc[0])
    close_p = float(row["close"].iloc[0])
    day_ret = ((close_p - t.buy_price) / t.buy_price) if t.buy_price else None
    return open_p, close_p, day_ret


def export_trades_xlsx(results: Dict[str, dict], path: str) -> str:
    """把每个策略的成交交易导出为一个 Excel 工作簿：每个策略一个 sheet，
    每笔交易一行（列见 TRADE_COLUMNS），供后续分析。serve / 静态报告两种
    模式都调用本函数，产出内容只取决于回测结果，因此完全一致。
    持有收益率 / 买入当日收益率为小数（如 0.0835 = 8.35%），未平仓交易为空；
    收益额 = (卖出价 − 买入价) × 卖出数量，未平仓为空。"""
    wb = Workbook()
    sheet_names = _sheet_names(list(results))
    first = True
    for name, data in results.items():
        ws = wb.active if first else wb.create_sheet()
        first = False
        ws.title = sheet_names[name]
        ws.append(TRADE_COLUMNS)
        stock_data = data["stock_data"]
        for t in data["trades"]:
            pnl = ((t.sell_price - t.buy_price) * (t.sell_amount or t.buy_amount)
                   if t.sell_price is not None else None)
            open_p, close_p, day_ret = _buy_day_bar(t, stock_data)
            ws.append([
                name, t.ticker, t.name, t.buy_time or "",
                t.buy_price, t.buy_amount, t.buy_reason,
                open_p if open_p is not None else "",
                close_p if close_p is not None else "",
                round(day_ret, 6) if day_ret is not None else "",
                t.sell_time or "",
                t.sell_price if t.sell_price is not None else "",
                t.sell_amount if t.sell_amount is not None else "",
                t.sell_reason or "",
                round(t.return_pct, 6) if t.return_pct is not None else "",
                round(pnl, 2) if pnl is not None else "",
                t.score,
            ])
    wb.save(path)
    return path


# ----------------------------------------------------------------------
# 回测引擎
# ----------------------------------------------------------------------

class Engine:
    """对一组策略在相同数据区间上执行回测（含账户级资金/持仓约束模拟）。"""

    def __init__(self, strategies: Optional[List[BaseStrategy]] = None,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 max_positions: int = 10) -> None:
        default_name = STRATEGY_REGISTRY["ma"]
        self.strategies = strategies if strategies is not None else [_instantiate(default_name)]
        self.initial_capital = float(initial_capital)
        self.max_positions = int(max_positions)

    def evaluate(self, df: pd.DataFrame, start_time: str, end_time: str) -> Dict[str, dict]:
        start_ts, end_ts = pd.Timestamp(start_time), pd.Timestamp(end_time)
        stock_data: Dict[str, pd.DataFrame] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            stock_data[ticker] = group.reset_index(drop=True)

        # 回测股票数：区间内（含预热数据之外）有数据的股票数
        stock_count = sum(
            1 for g in stock_data.values() if (g["date"] >= start_ts).any()
        )

        results: Dict[str, dict] = {}
        for strategy in self.strategies:
            print(f"[{strategy.name}] 开始回测 ({stock_count} 只股票)...")
            signals: List[Trade] = []
            for ticker, g in stock_data.items():
                if not (g["date"] >= start_ts).any() or not (g["date"] <= end_ts).any():
                    continue
                signals.extend(strategy.run(g, start_ts, end_ts))
            # 账户模拟：卖出全执行；买入按 score 降序在资金/持仓约束下成交
            trades, skipped = simulate_account(signals, stock_data, start_ts, end_ts,
                                               self.initial_capital,
                                               self.max_positions)
            trades.sort(key=lambda t: (t.buy_time or "", t.ticker))

            overview = compute_overview(
                trades, stock_count, initial_capital=self.initial_capital,
                total_pnl=account_total_pnl(trades, stock_data, end_ts))
            print(f"[{strategy.name}] 信号 {len(signals)} 笔 -> 成交 "
                  f"{overview['总交易次数']} 笔（资金/持仓限制跳过 {skipped} 笔）| "
                  f"胜率 {format_metric('胜率', overview['胜率'])} | "
                  f"总收益率 {format_metric('总收益率', overview['总收益率'])}")

            results[strategy.name] = {
                "overview": overview,
                "trades": trades,
                "stock_data": stock_data,
                "description": strategy.description,
            }
        return results


# ----------------------------------------------------------------------
# CLI（yacrs：-c 加载 yaml/json/toml 配置，key=value 命令行覆盖）
# ----------------------------------------------------------------------

@cli("engine")
@register("engine")
def main(db=DEFAULT_DB, start=None, end=None, output="strategy_report.html",
         trades_xlsx="trades.xlsx",
         warmup=120, limit=0, max_charts=1000,
         initial_capital=DEFAULT_INITIAL_CAPITAL, max_positions=10,
         strategies=None, conditions=None,
         serve=False, host="127.0.0.1", port=8000) -> None:
    """
    回测入口。所有参数在配置文件中设置（见 strategy_test.yaml），
    也可以用 key=value 在命令行覆盖：
        python -m lib.engine -c strategy_test.yaml engine.start=2025-01-01
    serve=true 时启动 Web 服务（Vue 前端）展示报告，否则生成静态 HTML。
    两种模式都会把成交交易导出为 trades_xlsx：一个 Excel 工作簿，
    每个策略一个 sheet，每笔交易一行（含买卖信息与持有收益；
    空字符串表示不导出）。
    initial_capital 为账户初始资金；max_positions 为最大持仓股票数，
    资金与仓位按策略买入倾向评分（Trade.score）优先配给高分信号，
    保证现金不为负（lib/account.py）。
    """
    if not start or not end:
        raise SystemExit("请在配置文件中设置 engine.start / engine.end（回测起止日期 YYYY-MM-DD）")

    # 命令行 key=value 覆盖进来的值是字符串，数值参数统一转换
    warmup, limit, max_charts = int(warmup), int(limit), int(max_charts)
    initial_capital = float(initial_capital)
    max_positions = int(max_positions)
    port = int(port)
    if isinstance(serve, str):  # 命令行覆盖 "false"/"0" 等字符串时按布尔解析
        serve = serve.lower() not in ("0", "false", "no", "")
    if isinstance(strategies, str):  # 命令行覆盖单值时按一个策略处理
        strategies = [strategies]
    strategy_names = list(strategies) if strategies else ["ma"]

    df = load_data(db, start, end, warmup_days=warmup, limit=limit)
    if df.empty:
        print("数据库中没有数据，请先运行 download_data.py")
        return

    engine = Engine(strategies=build_strategies(strategy_names, conditions),
                    initial_capital=initial_capital, max_positions=max_positions)
    results = engine.evaluate(df, start, end)

    if trades_xlsx:
        path = export_trades_xlsx(results, trades_xlsx)
        print(f"交易明细已导出: {path}")

    if serve:
        from lib.server import serve as serve_report  # 延迟导入避免循环
        serve_report(results, start, end, host, port,
                     initial_capital=initial_capital)
    elif output:
        path = generate_report(results, output, max_charts=max_charts,
                               start_time=start, end_time=end,
                               initial_capital=initial_capital)
        print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
