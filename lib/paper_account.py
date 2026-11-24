"""本地模拟账户（paper trading）：把每日预测清单落到真实行情上，虚拟跟踪成交。

quantie.py 的 forecast / daily 任务在策略扫描后调用 sync_account() 同步账户：

1. 补账：自上次同步的次一交易日起，逐交易日 D 推进到信号日：
   - 开盘：执行买入日 = D 的预测挂单（读 output_dir/{D}-{策略短名}.csv）。
     以策略在真实数据上的重跑校验入场（数据截至信号日，末尾追加一根
     合成K线）：开盘超出追价区间等被策略拒绝时挂单作废——与回测完全同
     一套信号逻辑；成交价 = D 日开盘价、股数 = 策略仓位管理的计算值。
   - 收盘：对每只持仓（买入日 < D，A 股 T+1）重跑其策略，策略给出卖出日
     = D 时按卖出价全仓平仓，卖出原因以策略为准。
   - 末尾的合成K线（日期 = 最后交易日+1，开盘 = 最后收盘，其余 NaN）让
     策略的「回测结束强制平仓」落在未来K线上而不污染真实 bar——真实 bar
     上的卖出信号才是要执行的。
2. 盯市：每个补账交易日按收盘价记一条账户快照（daily 表）；
   总收益额 = 已实现盈亏 + 期末持仓按最近收盘的浮盈（与引擎 overview 同口径）。
3. 账户约束：现金不为负、持仓不超过 max_positions，买入按 Trade.score
   降序配给资金与仓位（与 lib/account.py 同一口径）。

状态持久化在独立 SQLite 库（默认 paper_account.db，*.db 已被 .gitignore）：
- trades：全部交易记录（status = open/closed/cancelled），含买卖字段、
  所属策略短名；cancelled 的 note 记录作废原因（入场校验未过/现金不足/
  持仓已满/策略未运行）
- daily：每个补账日的账户快照（总资产/现金/持仓市值/持仓数/盈亏统计）
- meta：现金、本金、上次同步交易日
"""

import glob
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.engine import HOLDING_COLUMNS, account_total_pnl, compute_overview
from lib.strategies.base import BaseStrategy
from lib.trade import Trade

DEFAULT_ACCOUNT_DB = "paper_account.db"
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_MAX_POSITIONS = 10
EPS = 1e-9

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    buy_time TEXT,
    buy_price REAL,
    buy_amount REAL,
    buy_reason TEXT,
    score REAL,
    sell_time TEXT,
    sell_price REAL,
    sell_amount REAL,
    sell_reason TEXT,
    status TEXT NOT NULL,          -- open / closed / cancelled
    note TEXT                      -- 作废原因（cancelled 时）
);
CREATE TABLE IF NOT EXISTS daily (
    date TEXT PRIMARY KEY,
    total_assets REAL NOT NULL,
    cash REAL NOT NULL,
    position_value REAL NOT NULL,
    holdings_count INTEGER NOT NULL,
    total_trades INTEGER NOT NULL,   -- 已平仓笔数
    win_count INTEGER NOT NULL,
    loss_count INTEGER NOT NULL,
    total_pnl REAL NOT NULL,         -- 已实现 + 持仓浮盈（盯市）
    total_return_pct REAL NOT NULL   -- total_pnl / 本金
);
"""


# ----------------------------------------------------------------------
# 账户数据库
# ----------------------------------------------------------------------

def init_account_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _meta_get(conn: sqlite3.Connection, key: str,
              default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 (key, str(value)))


# ----------------------------------------------------------------------
# 行情读取（按股票全历史，含指标预热；持仓股数很少，代价可忽略）
# ----------------------------------------------------------------------

def _load_ticker(market_db: str, ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(market_db)
    df = pd.read_sql("SELECT * FROM stock_daily WHERE ticker = ? ORDER BY date",
                     conn, params=(ticker,))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _trading_days(market_db: str, through: str) -> List[str]:
    conn = sqlite3.connect(market_db)
    rows = conn.execute("SELECT DISTINCT date FROM stock_daily WHERE date <= ? "
                        "ORDER BY date", (through,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _append_sentinel_bar(g: pd.DataFrame) -> pd.DataFrame:
    """末尾追加一根合成K线：日期 = 最后交易日+1、开盘 = 最后收盘、其余 NaN。
    策略的「回测结束强制平仓」会落在它上面（卖出日 > 信号日而被忽略），
    真实 bar 上的卖出信号才是账户要执行的。"""
    last = g.iloc[-1]
    bar = {col: float("nan") for col in g.columns}
    bar["ticker"] = last["ticker"]
    if "name" in g.columns:
        bar["name"] = last["name"]
    if "sector" in g.columns:
        bar["sector"] = last["sector"]
    bar["date"] = last["date"] + pd.Timedelta(days=1)
    bar["open"] = float(last["close"])
    return pd.concat([g, pd.DataFrame([bar])], ignore_index=True)


def _rerun(strategy: BaseStrategy, g: pd.DataFrame) -> List[Trade]:
    """在单只股票真实数据（含末尾合成K线）上重跑策略，返回全部交易轨迹。"""
    g_ext = _append_sentinel_bar(g.sort_values("date").reset_index(drop=True))
    return strategy.run(g_ext, g_ext["date"].iloc[0], g_ext["date"].iloc[-1])


def _find_trade(trades: List[Trade], buy_time: str) -> Optional[Trade]:
    for t in trades:
        if t.buy_time == buy_time:
            return t
    return None


# ----------------------------------------------------------------------
# 单日价格工具（与 lib/engine.compute_daily_account 同口径）
# ----------------------------------------------------------------------

def _bar_on(g: pd.DataFrame, d) -> Tuple[Optional[float], Optional[float]]:
    """(开盘, 收盘)；当日无 K 线（停牌）为 (None, None)。"""
    dates = g["date"].values
    j = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(d)), side="left"))
    if j < len(dates) and dates[j] == np.datetime64(pd.Timestamp(d)):
        return float(g["open"].iloc[j]), float(g["close"].iloc[j])
    return None, None


def _close_on(g: pd.DataFrame, d) -> float:
    """d（含）之前最近收盘，无则 nan。"""
    dates = g["date"].values
    j = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(d)), side="right")) - 1
    return float(g["close"].iloc[j]) if j >= 0 else float("nan")


def _prev_close_on(g: pd.DataFrame, d) -> float:
    """d 之前最近收盘（严格在前），无则 nan。"""
    dates = g["date"].values
    j = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(d)), side="left")) - 1
    return float(g["close"].iloc[j]) if j >= 0 else float("nan")


# ----------------------------------------------------------------------
# 预测挂单（quantie.py forecast / daily 任务写出的 {买入日}-{策略短名}.csv）
# ----------------------------------------------------------------------

def _load_pending_picks(output_dir: str, day: str
                        ) -> List[Tuple[str, str, str, float, float, int]]:
    """读 output_dir 中买入日 = day 的全部预测 CSV，返回
    [(策略短名, 代码, 名称, 评分, 参考价格, 建议股数), ...]。"""
    picks: List[Tuple[str, str, str, float, float, int]] = []
    for path in sorted(glob.glob(os.path.join(output_dir, f"{day}-*.csv"))):
        short = os.path.basename(path)[len(f"{day}-"):-len(".csv")]
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype={"股票代码": str})
        except Exception:
            continue
        for r in df.itertuples(index=False):
            picks.append((short, str(r.股票代码), str(r.股票名称),
                          float(r.评分), float(r.参考价格), int(r.建议股数)))
    return picks


# ----------------------------------------------------------------------
# 交易记录读写
# ----------------------------------------------------------------------

def _insert_trade(conn: sqlite3.Connection, strategy: str, t: Trade,
                  status: str, note: Optional[str] = None) -> None:
    # 成交（open）时不落卖出腿：重跑策略时卖出腿已被提前算出（数据截至
    # 信号日），但卖出尚未发生——卖出字段在卖出真正执行时才回填，否则
    # 中间快照会把它误计为已平仓
    sell_time = sell_price = sell_amount = sell_reason = None
    if status == "closed":
        sell_time, sell_price = t.sell_time, t.sell_price
        sell_amount, sell_reason = t.sell_amount, t.sell_reason
    conn.execute("""
        INSERT INTO trades (strategy, ticker, name, buy_time, buy_price, buy_amount,
                            buy_reason, score, sell_time, sell_price, sell_amount,
                            sell_reason, status, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (strategy, t.ticker, t.name, t.buy_time, t.buy_price, t.buy_amount,
          t.buy_reason, t.score, sell_time, sell_price, sell_amount,
          sell_reason, status, note))


def _open_rows(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM trades WHERE status = 'open' ORDER BY buy_time, ticker"
    ).fetchall()


def _to_trade(row: sqlite3.Row) -> Trade:
    return Trade(
        ticker=row["ticker"], name=row["name"] or row["ticker"],
        buy_time=row["buy_time"], buy_amount=row["buy_amount"] or 0.0,
        buy_price=row["buy_price"] or 0.0, buy_reason=row["buy_reason"] or "",
        sell_time=row["sell_time"], sell_amount=row["sell_amount"],
        sell_price=row["sell_price"], sell_reason=row["sell_reason"],
        score=row["score"] or 0.0,
    )


# ----------------------------------------------------------------------
# 补账：逐交易日推进
# ----------------------------------------------------------------------

def _process_day(conn: sqlite3.Connection, market_db: str, day: str,
                 cash: float, strategies: Dict[str, BaseStrategy],
                 output_dir: str, max_positions: int,
                 rerun_cache: Dict[str, List[Trade]],
                 filled: List[dict], sold: List[dict]) -> float:
    """推进一个交易日：开盘执行当日挂单 → 收盘检查持仓卖出。返回新现金。"""

    def trades_of(strategy: BaseStrategy, ticker: str) -> List[Trade]:
        if ticker not in rerun_cache:
            g = _load_ticker(market_db, ticker)
            rerun_cache[ticker] = _rerun(strategy, g) if not g.empty else []
        return rerun_cache[ticker]

    # ---- 开盘：执行买入日 = day 的挂单（按评分降序配给资金/仓位）----
    picks = _load_pending_picks(output_dir, day)
    if picks:
        print(f"    [{day}] 开盘：{len(picks)} 条挂单待执行")
    cands: List[Tuple[float, Trade, str]] = []
    for short, ticker, name, score, ref_price, shares in picks:
        strategy = strategies.get(short)
        if strategy is None:
            print(f"        {ticker} 取消：策略 {short} 本次未运行，无法校验入场")
            _insert_trade(conn, short or "?", Trade(ticker=ticker, name=name,
                          buy_time=day, buy_price=ref_price, buy_amount=shares,
                          score=score), "cancelled", "策略未运行")
            continue
        t = _find_trade(trades_of(strategy, ticker), day)
        if t is None:
            print(f"        {ticker} 取消：策略重扫未给出入场"
                  f"（如开盘超出追价区间）")
            _insert_trade(conn, short, Trade(ticker=ticker, name=name,
                          buy_time=day, buy_price=ref_price, buy_amount=shares,
                          score=score), "cancelled", "入场校验未过")
            continue
        cands.append((t.score or score or 0.0, t, short))
    cands.sort(key=lambda x: (-x[0], x[1].ticker))

    open_count = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status = 'open'").fetchone()[0]
    for score, t, short in cands:
        cost = t.buy_price * t.buy_amount
        if open_count >= max_positions:
            print(f"        {t.ticker} 取消：持仓数已达上限 {max_positions}")
            _insert_trade(conn, short, t, "cancelled", "持仓数已达上限")
            continue
        if cost > cash + EPS:
            print(f"        {t.ticker} 取消：现金不足"
                  f"（需 {cost:,.2f}，余 {cash:,.2f}）")
            _insert_trade(conn, short, t, "cancelled", "现金不足")
            continue
        cash -= cost
        open_count += 1
        _insert_trade(conn, short, t, "open")
        print(f"        {t.ticker} {t.name} 成交 {int(t.buy_amount)} 股"
              f" @ {t.buy_price:.2f}（{cost:,.2f} 元）")
        filled.append({"ticker": t.ticker, "name": t.name, "action": "买入"})

    # ---- 收盘：持仓卖出检查（T+1：买入日 = day 的当日不可卖）----
    for row in _open_rows(conn):
        if row["buy_time"] >= day:
            continue  # 当日买入（T+1）；历史上买入的才检查
        strategy = strategies.get(row["strategy"])
        if strategy is None:
            print(f"        {row['ticker']} 持仓卖出检查跳过："
                  f"策略 {row['strategy']} 本次未运行")
            continue
        t = _find_trade(trades_of(strategy, row["ticker"]), row["buy_time"])
        if t is None:
            print(f"        {row['ticker']} 警告：持仓重跑未见买入记录，保持持仓")
            continue
        if t.sell_time == day:
            amount = t.sell_amount if t.sell_amount is not None else t.buy_amount
            cash += (t.sell_price or 0.0) * amount
            conn.execute("""
                UPDATE trades SET status = 'closed', sell_time = ?, sell_price = ?,
                                  sell_amount = ?, sell_reason = ?
                WHERE id = ?
            """, (t.sell_time, t.sell_price, t.sell_amount, t.sell_reason,
                  row["id"]))
            pnl = (t.sell_price - t.buy_price) * amount
            print(f"        {t.ticker} {t.name} 卖出 {int(amount)} 股"
                  f" @ {t.sell_price:.2f}（{t.sell_reason}，盈亏 {pnl:+,.2f}）")
            sold.append({"ticker": t.ticker, "name": t.name, "action": "卖出"})
    return cash


def _write_snapshot(conn: sqlite3.Connection, market_db: str, day: str,
                    cash: float, initial_capital: float,
                    gcache: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """按 day 收盘价盯市，写当日账户快照，返回快照主要字段。"""
    open_rows = _open_rows(conn)
    # cancelled 挂单不参与盈亏统计（其代码可能与持仓相同，盯市时会误算）
    all_rows = conn.execute(
        "SELECT * FROM trades WHERE status IN ('open', 'closed')").fetchall()
    trades = [_to_trade(r) for r in all_rows]

    position_value = 0.0
    for row in open_rows:
        g = gcache.get(row["ticker"])
        if g is None:
            g = _load_ticker(market_db, row["ticker"])
            gcache[row["ticker"]] = g
        if g.empty:
            continue
        close = _close_on(g, day)
        if close == close:  # not NaN
            position_value += close * (row["buy_amount"] or 0.0)

    held_data = {row["ticker"]: gcache[row["ticker"]]
                 for row in open_rows if row["ticker"] in gcache}
    total_pnl = account_total_pnl(trades, held_data, pd.Timestamp(day))
    overview = compute_overview(trades, 0, initial_capital=initial_capital,
                                total_pnl=total_pnl)
    total_assets = cash + position_value
    snap = {
        "date": day,
        "total_assets": round(total_assets, 2),
        "cash": round(cash, 2),
        "position_value": round(position_value, 2),
        "holdings_count": len(open_rows),
        "total_trades": int(overview["总交易次数"]),
        "win_count": int(overview["盈利次数"]),
        "loss_count": int(overview["亏损次数"]),
        "win_rate": overview["胜率"],
        "total_pnl": overview["总收益额"],
        "total_return_pct": overview["总收益率"],
    }
    conn.execute("""
        INSERT OR REPLACE INTO daily (date, total_assets, cash, position_value,
            holdings_count, total_trades, win_count, loss_count, total_pnl,
            total_return_pct)
        VALUES (:date, :total_assets, :cash, :position_value, :holdings_count,
            :total_trades, :win_count, :loss_count, :total_pnl, :total_return_pct)
    """, snap)
    return snap


# ----------------------------------------------------------------------
# 对外接口：同步账户 / 读取报告 / 消息格式化
# ----------------------------------------------------------------------

def sync_account(market_db: str, strategies: Dict[str, BaseStrategy],
                 signal_date: str, output_dir: str = "forecast",
                 account_db: str = DEFAULT_ACCOUNT_DB,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 max_positions: int = DEFAULT_MAX_POSITIONS) -> dict:
    """把账户补账到信号日并返回报告（结构见 build_report）。

    strategies = {策略短名: 策略实例}（与本次扫描同一批实例，保证参数/条件一致）。
    首次同步不回溯历史：只处理信号日当天（仍会执行信号日的挂单，即昨日
    预测）；其后每天增量推进。跳过报告里可看到被取消的挂单及原因。
    """
    initial_capital = float(initial_capital)
    max_positions = int(max_positions)
    conn = init_account_db(account_db)
    if _meta_get(conn, "cash") is None:
        _meta_set(conn, "initial_capital", initial_capital)
    cash = float(_meta_get(conn, "cash", initial_capital))
    last_sync = _meta_get(conn, "last_sync")

    days = _trading_days(market_db, signal_date)
    if last_sync:
        days = [d for d in days if d > last_sync]
    else:
        days = days[-1:]  # 首次同步从信号日开始，不回溯
    if not days:
        conn.close()
        print(f">>> 模拟账户同步：{account_db} 已是最新"
              f"（上次同步 {last_sync}，信号日 {signal_date}），无需补账")
        return build_report(account_db, market_db, signal_date,
                            initial_capital=initial_capital)

    print(f">>> 模拟账户同步（{account_db}，本金 "
          f"{initial_capital:,.2f}）：补账 {days[0]} ~ {days[-1]}"
          f"（{len(days)} 个交易日）")
    gcache: Dict[str, pd.DataFrame] = {}
    filled: List[dict] = []
    sold: List[dict] = []
    last_snap: Optional[Dict[str, float]] = None
    for day in days:
        rerun_cache: Dict[str, List[Trade]] = {}
        cash = _process_day(conn, market_db, day, cash, strategies,
                            output_dir, max_positions, rerun_cache,
                            filled, sold)
        last_snap = _write_snapshot(conn, market_db, day, cash,
                                    initial_capital, gcache)
        _meta_set(conn, "last_sync", day)
    _meta_set(conn, "cash", cash)
    conn.commit()
    conn.close()
    print(f"    账户同步完成：总资产 {last_snap['total_assets']:,.2f} | "
          f"现金 {last_snap['cash']:,.2f} | 持仓 "
          f"{last_snap['holdings_count']} 只 | 累计盈亏 "
          f"{last_snap['total_pnl']:+,.2f}（{last_snap['total_return_pct'] * 100:+.2f}%）")

    report = build_report(account_db, market_db, signal_date,
                          sold_since=days[0], initial_capital=initial_capital)
    report["filled"] = filled
    report["sold_ops"] = sold
    return report


def _holding_row(t: Trade, g: pd.DataFrame, as_of: str,
                 sold: bool) -> Dict[str, object]:
    """单笔持仓/卖出记录 → 交易明细行（engine server 模式同口径字段）。"""
    open_p, close_p = (None, None)
    day_ret = hold_ret = None
    if not g.empty:
        if sold:
            open_p, close_p = _bar_on(g, t.sell_time)
            prev_close = _prev_close_on(g, t.sell_time)
            if t.sell_price is not None and prev_close == prev_close and prev_close:
                day_ret = (t.sell_price - prev_close) / prev_close
            hold_ret = t.return_pct
        else:
            open_p, close_p = _bar_on(g, as_of)
            close = _close_on(g, as_of)
            prev_close = _prev_close_on(g, as_of)
            if close_p is not None:
                if t.buy_time == as_of:
                    day_ret = ((close_p - t.buy_price) / t.buy_price
                               if t.buy_price else None)
                elif prev_close == prev_close and prev_close:
                    day_ret = (close_p - prev_close) / prev_close
            if close == close and t.buy_price:
                hold_ret = (close - t.buy_price) / t.buy_price
    return {
        "ticker": t.ticker, "name": t.name,
        "buy_time": t.buy_time, "sell_time": t.sell_time if sold else None,
        "buy_price": t.buy_price, "buy_amount": t.buy_amount,
        "sell_price": t.sell_price if sold else None,
        "sell_amount": t.sell_amount if sold else None,
        "open_price": open_p, "close_price": close_p,
        "day_return_pct": day_ret, "holding_return_pct": hold_ret,
        "sell_reason": t.sell_reason if sold else None,
        "buy_reason": t.buy_reason,
    }


def build_report(account_db: str, market_db: str, as_of: Optional[str] = None,
                 sold_since: Optional[str] = None,
                 initial_capital: Optional[float] = None) -> dict:
    """从账户库重建某日的报告（不落库）：
    当前持仓（交易明细口径）、一段时间内卖出记录、账户统计与快照。
    as_of 缺省取账户库最近一个快照日；sold_since 缺省取 as_of 前 7 天
    （供 daily.mode=push 只推送时复用）。"""
    if not os.path.exists(account_db):
        return {}
    conn = init_account_db(account_db)
    if as_of is None:
        row = conn.execute("SELECT MAX(date) FROM daily").fetchone()
        as_of = row[0] if row and row[0] else None
        if as_of is None:
            conn.close()
            return {}
    if initial_capital is None:
        initial_capital = float(_meta_get(conn, "initial_capital",
                                          DEFAULT_INITIAL_CAPITAL))
    if sold_since is None:
        sold_since = (pd.Timestamp(as_of) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    cash = float(_meta_get(conn, "cash", initial_capital))
    # cancelled 挂单不算交易（盈亏/胜率统计均排除）
    rows = conn.execute(
        "SELECT * FROM trades WHERE status IN ('open', 'closed')").fetchall()
    conn.close()

    gcache: Dict[str, pd.DataFrame] = {}
    holdings, sold_rows = [], []
    trades: List[Trade] = []
    for row in rows:
        t = _to_trade(row)
        trades.append(t)
        if row["status"] == "open":
            if row["ticker"] not in gcache:
                gcache[row["ticker"]] = _load_ticker(market_db, row["ticker"])
            holdings.append(_holding_row(t, gcache[row["ticker"]], as_of,
                                         sold=False))
        elif (row["status"] == "closed"
              and (sold_since is None or (t.sell_time or "") >= sold_since)):
            if row["ticker"] not in gcache:
                gcache[row["ticker"]] = _load_ticker(market_db, row["ticker"])
            sold_rows.append(_holding_row(t, gcache[row["ticker"]], as_of,
                                          sold=True))

    held_data = {r["ticker"]: gcache[r["ticker"]] for r in rows
                 if r["status"] == "open" and r["ticker"] in gcache}
    total_pnl = account_total_pnl(trades, held_data, pd.Timestamp(as_of))
    overview = compute_overview(trades, 0, initial_capital=initial_capital,
                                total_pnl=total_pnl)
    position_value = 0.0
    for h in holdings:
        g = gcache.get(h["ticker"])
        if g is None or g.empty:
            continue
        close = _close_on(g, as_of)
        if close == close:
            position_value += close * (h["buy_amount"] or 0.0)
    stats = {k: overview[k] for k in
             ("总交易次数", "盈利次数", "亏损次数", "胜率", "总收益额", "总收益率")}
    return {
        "as_of": as_of,
        "initial_capital": initial_capital,
        "cash": cash,
        "position_value": round(position_value, 2),
        "total_assets": round(cash + position_value, 2),
        "holdings_count": len(holdings),
        "stats": stats,
        "holdings": holdings,
        "sold": sold_rows,
    }


# ----------------------------------------------------------------------
# 企业微信消息格式化（markdown，与 TASKS.md 消息拼接）
# ----------------------------------------------------------------------

def _fmt_pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:+.2f}%"


def _fmt_row(h: Dict[str, object], reason_len: int = 18) -> str:
    def reason(s: Optional[str]) -> str:
        s = s or "-"
        return s if len(s) <= reason_len else s[:reason_len] + "…"

    buy_pa = f"{h['buy_price']:.2f} / {h['buy_amount']:,.0f}"
    sell_pa = (f"{h['sell_price']:.2f} / {h['sell_amount']:,.0f}"
               if h["sell_time"] else "-")
    open_s = f"{h['open_price']:.2f}" if h["open_price"] is not None else "-"
    close_s = f"{h['close_price']:.2f}" if h["close_price"] is not None else "-"
    return "\t".join([
        str(h["ticker"]), str(h["name"]),
        str(h["buy_time"] or "-"),
        str(h["sell_time"] or "-"),
        buy_pa, sell_pa, open_s, close_s,
        _fmt_pct(h["day_return_pct"]), _fmt_pct(h["holding_return_pct"]),
        reason(h["sell_reason"] if h["sell_time"] else None),
        reason(h["buy_reason"]),
    ])


def report_markdown(report: dict, max_sold: int = 5) -> str:
    """持仓明细（engine server 模式交易明细同列）+ 本周期卖出 + 账户概览。"""
    if not report:
        return ""
    header = "\t".join(HOLDING_COLUMNS)
    lines: List[str] = []
    holdings = report["holdings"]
    lines.append(f"## 持仓明细（{len(holdings)} 只）")
    if holdings:
        lines.append(header)
        lines += [_fmt_row(h) for h in holdings]
    else:
        lines.append("（当前无持仓）")

    sold = report["sold"]
    shown = sold[-max_sold:]
    lines.append("")
    lines.append(f"## 本周期卖出（{len(sold)} 只）")
    if shown:
        lines.append(header)
        lines += [_fmt_row(h) for h in shown]
        if len(sold) > len(shown):
            lines.append(f"（仅显示最近 {len(shown)} 只，其余见账户库）")
    else:
        lines.append("（本周期无卖出）")

    s = report["stats"]
    lines.append("")
    lines.append(f"## 模拟账户（本金 {report['initial_capital']:,.2f}）")
    lines.append(f"总资产 {report['total_assets']:,.2f}｜现金 {report['cash']:,.2f}"
                 f"｜持仓市值 {report['position_value']:,.2f}")
    lines.append(f"总交易次数 {s['总交易次数']}｜盈利次数 {s['盈利次数']}｜"
                 f"亏损次数 {s['亏损次数']}｜胜率 {s['胜率'] * 100:.2f}%")
    lines.append(f"总收益额 {s['总收益额']:+,.2f}｜"
                 f"总收益率 {s['总收益率'] * 100:+.2f}%")
    return "\n".join(lines)
