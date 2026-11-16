#!/usr/bin/env python3
"""
修正 stock_daily 中的错误股票名称（历史数据遗留问题）。

背景：2026-08-13 之前的 download_data.py 在获取全 A 股代码表时取错列
（把 tradeStatus='1' 当成了 code_name，见 git 提交 722a745），导致此前
下载的 stock_daily.name 大部分为 '1'；stock_meta.name 在修复后的增量
下载中已被改正。本脚本把 stock_daily.name 从 stock_meta 回填，
并可选地从 baostock 在线补齐 stock_meta 中仍为 '1'/空的少数名称。

用法:
    python fix_stock_names.py [db路径]                    # 离线回填（默认 a_stock_daily.db）
    python fix_stock_names.py db --sibling a_stock_dev.db # 再用兄弟库的名称回填一轮
    python fix_stock_names.py db --online                 # 最后尝试 baostock 在线补齐
"""

import sqlite3
import sys
from datetime import datetime, timedelta

BAD_NAMES = ("1", "", "0")


def backfill_from_meta(conn) -> int:
    """stock_daily.name ∈ BAD_NAMES 的行，用 stock_meta.name 回填。返回修正行数。"""
    cur = conn.execute("""
        UPDATE stock_daily
        SET name = (SELECT m.name FROM stock_meta m
                    WHERE m.ticker = stock_daily.ticker)
        WHERE name IN ('1', '', '0')
          AND EXISTS (SELECT 1 FROM stock_meta m
                      WHERE m.ticker = stock_daily.ticker
                        AND m.name NOT IN ('1', '', '0'))
    """)
    return cur.rowcount


def backfill_from_sibling(conn, sibling_path: str) -> int:
    """兄弟库（同源下载的另一个 db）中已有正确名称的，回填 stock_daily/stock_meta。"""
    sib = sqlite3.connect(sibling_path)
    fixed = 0
    try:
        for (ticker,) in conn.execute(
                "SELECT DISTINCT ticker FROM stock_daily WHERE name IN ('1', '', '0')"):
            row = sib.execute(
                "SELECT name FROM stock_daily WHERE ticker = ? "
                "AND name NOT IN ('1', '', '0') LIMIT 1", (ticker,)).fetchone()
            if not row or not row[0]:
                continue
            conn.execute("UPDATE stock_daily SET name = ? WHERE ticker = ? "
                         "AND name IN ('1', '', '0')", (row[0], ticker))
            conn.execute("UPDATE stock_meta SET name = ? WHERE ticker = ?",
                         (row[0], ticker))
            fixed += 1
    finally:
        sib.close()
    return fixed


def remaining_bad_tickers(conn):
    """stock_daily 中名称仍异常的去重 ticker 列表。"""
    return [r[0] for r in conn.execute("""
        SELECT DISTINCT ticker FROM stock_daily WHERE name IN ('1', '', '0')
    """)]


def fix_online(conn, db_path: str) -> None:
    """从 baostock 拉代码表，补齐 stock_daily/stock_meta 中仍为异常的名称。"""
    import baostock as bs

    last_date = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
    day = datetime.strptime(last_date, "%Y-%m-%d")
    names = {}
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock 登录失败: {lg.error_msg}（跳过在线补齐）")
        return
    try:
        for _ in range(10):  # 向前回退直到命中交易日
            rs = bs.query_all_stock(day=day.strftime("%Y-%m-%d"))
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                names[row[rs.fields.index("code")]] = row[rs.fields.index("code_name")]
            if names:
                break
            day -= timedelta(days=1)
    finally:
        bs.logout()
    if not names:
        print("baostock 代码表为空（跳过在线补齐）")
        return

    fixed = 0
    for ticker in remaining_bad_tickers(conn):
        name = names.get(ticker)
        if not name or name in BAD_NAMES:
            continue  # 已退市/查不到：保留原值
        conn.execute("UPDATE stock_daily SET name = ? WHERE ticker = ? AND name IN ('1','','0')",
                     (name, ticker))
        conn.execute("INSERT OR REPLACE INTO stock_meta "
                     "(ticker, name, sector, first_date, last_date, record_count, update_time) "
                     "SELECT ticker, ?, sector, first_date, last_date, record_count, datetime('now') "
                     "FROM stock_meta WHERE ticker = ?", (name, ticker))
        fixed += 1
    conn.commit()
    print(f"在线补齐 {fixed} 只")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    db_path = args[0] if args else "a_stock_daily.db"
    sibling = args[1] if len(args) > 1 else None
    online = "--online" in sys.argv

    conn = sqlite3.connect(db_path)
    n = backfill_from_meta(conn)
    conn.commit()
    print(f"离线回填: {n} 行")
    if sibling:
        n = backfill_from_sibling(conn, sibling)
        conn.commit()
        print(f"兄弟库回填: {n} 只")
    if online:
        fix_online(conn, db_path)
    left = remaining_bad_tickers(conn)
    conn.close()
    print(f"仍有异常名称的 ticker: {len(left)} 只{('，如 ' + ', '.join(left[:5])) if left else ''}")
    print(f"数据库: {db_path}")


if __name__ == "__main__":
    main()
