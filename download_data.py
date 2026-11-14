#!/usr/bin/env python3
"""
A股日频数据下载脚本 —— Baostock版
- 国内源，不限流，完全免费
- 支持前复权/后复权
- 自动去重、断点续传、增量更新
- 支持 --full-refresh 强制全量刷新
- 支持 --start/--end 指定下载日期范围（--end 默认为昨天）
- 支持 --chunk N 分块下载：每 N 天一轮，逐轮遍历全部股票，日志显示 ETA
- 写入使用 INSERT OR REPLACE，重复下载自动去重
"""

import os
import sys
import time
import sqlite3
import argparse
import pandas as pd
from datetime import datetime, timedelta

# ==================== 配置区 ====================
DB_PATH = "a_stock_daily.db"
START_DATE = "2024-01-01"

STOCKS = [
    ("sh.600519", "贵州茅台", "白酒"),
    ("sz.002594", "比亚迪", "新能源"),
    ("sz.300750", "宁德时代", "电池"),
    ("sz.000858", "五粮液", "白酒"),
    ("sh.601318", "中国平安", "保险"),
    ("sh.600036", "招商银行", "银行"),
    ("sz.000333", "美的集团", "家电"),
    ("sh.600900", "长江电力", "电力"),
    ("sz.002415", "海康威视", "安防"),
    ("sh.601012", "隆基绿能", "光伏"),
    ("sh.600276", "恒瑞医药", "医药"),
    ("sz.000001", "平安银行", "银行"),
    ("sz.300059", "东方财富", "券商"),
    ("sh.601888", "中国中免", "免税"),
    ("sz.002230", "科大讯飞", "AI"),
    ("sh.600030", "中信证券", "券商"),
    ("sz.000568", "泸州老窖", "白酒"),
    ("sz.002714", "牧原股份", "养殖"),
    ("sh.601899", "紫金矿业", "有色"),
    ("sz.300760", "迈瑞医疗", "医疗器械"),
]


def get_stock_list():
    import baostock as bs

    bs.login()
    rs = bs.query_all_stock(day="2024-07-01")
    stocks = []
    while (rs.error_code == '0') & rs.next():
        row = rs.get_row_data()
        code, name = row[rs.fields.index('code')], row[rs.fields.index('code_name')]
        if code.startswith("sh.") or code.startswith("sz."):
            stocks.append((code, name, "-"))
    bs.logout()
    return stocks


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            ticker TEXT, name TEXT, sector TEXT,
            date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, amount REAL, turn REAL, pctChg REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS stock_meta (
            ticker TEXT PRIMARY KEY, name TEXT, sector TEXT,
            first_date TEXT, last_date TEXT, record_count INTEGER,
            update_time TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS download_log (
            ticker TEXT PRIMARY KEY, status TEXT, message TEXT,
            attempt_count INTEGER DEFAULT 0, last_attempt TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_stock_status():
    """获取每只股票的最后更新日期"""
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT ticker, last_date FROM stock_meta")
    result = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return result


def calc_date_range(ticker, last_date, args):
    """
    计算该股票本次需要下载的日期范围
    返回: (start_date, end_date, is_incremental)
    """
    if args.full_refresh or last_date is None:
        return args.start, args.end, False

    last_dt = datetime.strptime(last_date, "%Y-%m-%d")
    next_dt = last_dt + timedelta(days=1)
    start = next_dt.strftime("%Y-%m-%d")
    end = args.end

    if start > end:
        return None, None, True

    return start, end, True


def daterange_chunks(start_date, end_date, chunk_days):
    """将日期范围按 chunk_days 天切分，逐段产出 (chunk_start, chunk_end)"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)


def download_stock(bs, code, name, sector, stock_status, args, date_range=None):
    """下载单只股票
    date_range: 显式指定 (start, end) 时按该范围下载（分块模式），否则按增量逻辑计算
    """

    last_date = stock_status.get(code)
    if date_range is not None:
        start_date, end_date, is_incr = date_range[0], date_range[1], False
    else:
        start_date, end_date, is_incr = calc_date_range(code, last_date, args)

    if start_date is None:
        return code, 0, "UP-TO-DATE"

    fields_str = "date,open,high,low,close,volume,amount,turn,pctChg"

    if args.dry_run:
        action = "增量" if is_incr else "全量"
        return code, 0, f"DRY-RUN({action} {start_date}~{end_date})"

    try:
        rs = bs.query_history_k_data_plus(
            code, fields_str,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        if rs.error_code != '0':
            return code, 0, f"DOWNLOAD: {rs.error_msg}"

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return code, 0, "无数据"

        df = pd.DataFrame(data_list, columns=rs.fields)

        numeric_cols = ['open', 'high', 'low', 'close', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

        df['ticker'] = code
        df['name'] = name
        df['sector'] = sector

    except Exception as e:
        return code, 0, f"DOWNLOAD: {str(e)[:100]}"

    try:
        cols = ['ticker', 'name', 'sector', 'date', 'open', 'high', 'low', 'close',
                'volume', 'amount', 'turn', 'pctChg']
        conn = sqlite3.connect(DB_PATH)
        # NaN/NA 转为 None，重复记录由主键 (ticker, date) 覆盖，保证可重复下载
        records = df[cols].astype(object).where(df[cols].notna(), None).values.tolist()
        conn.executemany(
            f"INSERT OR REPLACE INTO stock_daily ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            records
        )

        c = conn.cursor()
        c.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM stock_daily WHERE ticker = ?", (code,))
        first_d, last_d, cnt = c.fetchone()

        c.execute("""
            INSERT OR REPLACE INTO stock_meta
            (ticker, name, sector, first_date, last_date, record_count, update_time)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (code, name, sector, first_d, last_d, cnt))

        if date_range is not None:
            action = "分块"
        else:
            action = "增量" if is_incr else "全量"
        c.execute("""
            INSERT OR REPLACE INTO download_log (ticker, status, message, attempt_count, last_attempt)
            VALUES (?, 'success', ?, 1, datetime('now'))
        """, (code, action))

        conn.commit()
        conn.close()

        return code, len(df), f"SUCCESS({action})"

    except Exception as e:
        return code, 0, f"EXCEPTION: {str(e)[:100]}"


def format_eta(seconds):
    """将秒数格式化为 h:mm:ss 或 m:ss"""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def run_per_stock_download(bs, stocks, stock_status, args):
    """普通模式：逐只股票下载（每只股票一次性下载其日期范围）"""
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, (code, name, sector) in enumerate(stocks, 1):
        result = download_stock(bs, code, name, sector, stock_status, args)

        if "SUCCESS" in result[2]:
            status_icon = "✅"
            success_count += 1
        elif "UP-TO-DATE" in result[2] or "DRY-RUN" in result[2]:
            status_icon = "⏭️"
            skip_count += 1
        else:
            status_icon = "❌"
            fail_count += 1

        print(f"[{i}/{len(stocks)}] {status_icon} {code} {name}: {result[1]}条 | {result[2]}")
        time.sleep(0.3)

    return success_count, skip_count, fail_count


def run_chunked_download(bs, stocks, args):
    """分块模式：外层按日期分块迭代，内层遍历全部股票，日志显示 ETA"""
    chunks = list(daterange_chunks(args.start, args.end, args.chunk))
    total_tasks = len(chunks) * len(stocks)

    if args.dry_run:
        print(f"计划: {len(chunks)} 轮 x {len(stocks)} 只 = {total_tasks} 个下载任务")
        stock_status = get_stock_status()
        for ci, (cs, ce) in enumerate(chunks, 1):
            if args.full_refresh:
                need = len(stocks)
            else:
                need = sum(1 for code, _, _ in stocks
                           if not (stock_status.get(code) and stock_status[code] >= ce))
            print(f"[{ci}/{len(chunks)}] {cs} ~ {ce}: 需下载 {need}/{len(stocks)} 只")
        return 0, 0, 0

    print(f"共分 {len(chunks)} 轮 x {len(stocks)} 只 = {total_tasks} 个下载任务")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_rows = 0
    done_tasks = 0
    t_start = time.time()

    for ci, (cs, ce) in enumerate(chunks, 1):
        stock_status = get_stock_status()  # 每轮刷新，利用前面轮次已下载的数据
        for si, (code, name, sector) in enumerate(stocks, 1):
            prefix = f"[{ci}/{len(chunks)} {cs}~{ce}] [{si}/{len(stocks)}]"

            last_date = stock_status.get(code)
            if not args.full_refresh and last_date and last_date >= ce:
                skip_count += 1
                rows = 0
                status = "⏭️ 已覆盖"
            else:
                result = download_stock(bs, code, name, sector, stock_status, args,
                                        date_range=(cs, ce))
                rows = result[1]
                total_rows += rows
                if "SUCCESS" in result[2]:
                    success_count += 1
                    status = "✅"
                elif "无数据" in result[2]:
                    skip_count += 1
                    status = "⏭️ 无数据"
                else:
                    fail_count += 1
                    status = f"❌ {result[2]}"
                time.sleep(0.3)

            done_tasks += 1
            elapsed = time.time() - t_start
            remaining = elapsed / done_tasks * (total_tasks - done_tasks)
            eta_clock = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
            print(f"{prefix} {status} {code} {name}: {rows}条 | "
                  f"已用 {format_eta(elapsed)} | 预计剩余 {format_eta(remaining)} (ETA {eta_clock})")

    print()
    print(f"分块下载完成: 共 {total_rows} 条 | ✅{success_count} | ⏭️跳过{skip_count} | ❌失败{fail_count}")

    return success_count, skip_count, fail_count


def batch_download(args):
    import baostock as bs

    init_db()

    if args.all:
        stocks = get_stock_list()
    else:
        stocks = STOCKS

    sep = "=" * 60
    print(sep)
    print("Baostock A股日频数据下载")
    print(f"日期范围: {args.start} ~ {args.end}")
    mode_str = "全量刷新" if args.full_refresh else "增量更新"
    print(f"模式: {mode_str}")
    if args.chunk:
        print(f"分块下载: 每 {args.chunk} 天一轮，逐轮下载全部股票，直至全部完成")
    if args.dry_run:
        print("👁️  预览模式: 只显示，不写入")
    print(f"股票池: {len(stocks)} 只")
    print(sep)
    print()

    lg = bs.login()
    print(f"登录结果: {lg.error_msg}")
    if lg.error_code != '0':
        print("登录失败，请检查网络")
        return

    if args.chunk:
        success_count, skip_count, fail_count = run_chunked_download(bs, stocks, args)
    else:
        stock_status = get_stock_status()
        success_count, skip_count, fail_count = run_per_stock_download(bs, stocks, stock_status, args)

    bs.logout()

    print()
    print(sep)
    print(f"完成: ✅{success_count} | ⏭️跳过/预览{skip_count} | ❌失败{fail_count}")
    print(f"数据库: {DB_PATH}")
    print(sep)

    if not args.dry_run:
        final_status = get_stock_status()
        print()
        print("📋 各股票最新数据日期:")
        for code, name, _ in stocks[:20]:
            last = final_status.get(code, "从未下载")
            print(f"   {code} {name}: {last}")


def main():

    global DB_PATH
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="A股日频数据下载 —— 增量更新版")
    parser.add_argument("--full-refresh", "--full", dest="full_refresh", action="store_true",
                        help="强制全量刷新（忽略缓存，重新下载全部历史）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式：只显示会下载哪些，不实际写入")
    parser.add_argument("--all", action="store_true",
                        help="下载全A股（默认只下载配置的核心股票池）")
    parser.add_argument("--db", default=DB_PATH, help=f"数据库路径 (默认: {DB_PATH})")
    parser.add_argument("--start", default=START_DATE,
                        help=f"下载起始日期 YYYY-MM-DD (默认: {START_DATE})")
    parser.add_argument("--end", default=yesterday,
                        help=f"下载结束日期 YYYY-MM-DD (默认: 昨天，即 {yesterday})")
    parser.add_argument("--chunk", type=int, default=None, metavar="N",
                        help="分块下载：每 N 天一轮，逐轮下载全部股票，直至全部完成")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error(f"--start ({args.start}) 不能晚于 --end ({args.end})")
    if args.chunk is not None and args.chunk <= 0:
        parser.error("--chunk 必须是大于 0 的整数")

    DB_PATH = args.db

    batch_download(args)


if __name__ == "__main__":
    main()
