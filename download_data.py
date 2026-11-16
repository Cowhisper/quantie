#!/usr/bin/env python3
"""
A股日频数据下载脚本 —— Baostock 版（yacrs 配置驱动）

- 国内源，不限流，完全免费，前复权
- 增量更新：每只股票只下载数据库中最后日期之后的数据，已覆盖的自动跳过
- 每只股票一次请求覆盖全部缺失区间（不做日期分块），请求数 = 股票数
- 每只股票写入后立即提交，中途断掉不丢已完成的部分（天然断点续传）
- 用法（所有参数见 download.yaml 的 download 段，命令行 key=value 可覆盖任意项）:
      python download_data.py -c download.yaml
      python download_data.py -c download.yaml download.end=2026-11-15 download.all=true
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
from yacrs import cli, register

DEFAULT_DB = "a_stock_daily.db"
DEFAULT_START = "2024-01-01"

# 默认核心股票池（配置 download.stocks 为空时生效）
DEFAULT_STOCKS = [
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

FIELDS = "date,open,high,low,close,volume,amount,turn,pctChg"
NUMERIC_COLS = ["open", "high", "low", "close", "amount", "turn", "pctChg"]
DAILY_COLS = ["ticker", "name", "sector", "date", "open", "high", "low", "close",
              "volume", "amount", "turn", "pctChg"]
DATE_FMT = "%Y-%m-%d"
# 头部缺口超过该天数才认为 start 被有意前移、需要补齐历史；避免把起始日落在
# 节假日（如 2024-01-01，首个交易日 01-02）误判为缺口而每天全量重下
GAP_FILL_DAYS = 7


# ----------------------------------------------------------------------
# 数据库
# ----------------------------------------------------------------------

def init_db(db_path):
    conn = sqlite3.connect(db_path)
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
    return conn


def get_stock_status(conn):
    """ticker -> (first_date, last_date)，即库中已下载数据覆盖的区间"""
    c = conn.execute("SELECT ticker, first_date, last_date FROM stock_meta")
    return {row[0]: (row[1], row[2]) for row in c.fetchall()}


def save_stock(conn, code, name, sector, df):
    """写入一只股票的数据并更新元信息（调用方已 commit 过表结构）。"""
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    df["ticker"] = code
    df["name"] = name
    df["sector"] = sector

    # NaN/NA 转为 None，重复记录由主键 (ticker, date) 覆盖，保证可重复下载
    records = (df[DAILY_COLS].astype(object)
               .where(df[DAILY_COLS].notna(), None).values.tolist())
    conn.executemany(
        f"INSERT OR REPLACE INTO stock_daily ({', '.join(DAILY_COLS)}) "
        f"VALUES ({', '.join(['?'] * len(DAILY_COLS))})",
        records,
    )

    first_d, last_d, cnt = conn.execute(
        "SELECT MIN(date), MAX(date), COUNT(*) FROM stock_daily WHERE ticker = ?",
        (code,)).fetchone()
    conn.execute("""
        INSERT OR REPLACE INTO stock_meta
        (ticker, name, sector, first_date, last_date, record_count, update_time)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (code, name, sector, first_d, last_d, cnt))


def log_success(conn, code, action):
    conn.execute("""
        INSERT OR REPLACE INTO download_log (ticker, status, message, attempt_count, last_attempt)
        VALUES (?, 'success', ?, 1, datetime('now'))
    """, (code, action))


# ----------------------------------------------------------------------
# 下载
# ----------------------------------------------------------------------

def fetch_stock(bs, code, start_date, end_date):
    """下载单只股票 [start_date, end_date] 的日线，返回 DataFrame（可能为空）。"""
    rs = bs.query_history_k_data_plus(
        code, FIELDS,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2",
    )
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def get_stock_list(bs, end_date):
    """全 A 股代码表（取 end_date 当天及之前最近一个有数据的交易日）。"""
    day = datetime.strptime(end_date, "%Y-%m-%d")
    for _ in range(10):  # 向前回退直到命中交易日
        rs = bs.query_all_stock(day=day.strftime("%Y-%m-%d"))
        stocks = []
        while (rs.error_code == "0") & rs.next():
            row = rs.get_row_data()
            code = row[rs.fields.index("code")]
            if code.startswith("sh.") or code.startswith("sz."):
                stocks.append((code, row[rs.fields.index("code_name")], "-"))
        if stocks:
            return stocks
        day -= timedelta(days=1)
    raise RuntimeError("无法获取全 A 股代码表（连续 10 天均非交易日？）")


def plan_range(covered, start, end, full_refresh):
    """计算该股票本次需要下载的区间。covered=(first_date, last_date) 或 None，
    返回 (start, end, 动作) 或 None（已覆盖）。"""
    if full_refresh or covered is None:
        return start, end, "全量"
    first_date, last_date = covered
    if last_date >= end:
        return None
    if last_date < start:
        # 已有数据全在窗口之前：重下整个窗口（重复部分主键去重）
        return start, end, "全量"
    # 尾部缺口 (last_date, end] 必补
    next_day = (datetime.strptime(last_date, DATE_FMT) + timedelta(days=1)
                ).strftime(DATE_FMT)
    # 头部缺口 [start, first_date)：仅当 start 被有意前移超过 GAP_FILL_DAYS 才
    # 从头补齐，否则视为起始日落在节假日，照常增量即可
    if first_date:
        head_gap = (datetime.strptime(first_date, DATE_FMT)
                    - datetime.strptime(start, DATE_FMT))
        if head_gap > timedelta(days=GAP_FILL_DAYS):
            return start, end, "增量"
    return next_day, end, "增量"


def format_eta(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _as_bool(v):
    return v if isinstance(v, bool) else str(v).lower() not in ("0", "false", "no", "")


# ----------------------------------------------------------------------
# 入口（yacrs：-c 加载 yaml/json/toml 配置，key=value 命令行覆盖）
# ----------------------------------------------------------------------

@cli("download")
@register("download")
def main(db=DEFAULT_DB, start=DEFAULT_START, end=None,
         all=False, full_refresh=False, dry_run=False, sleep=0.0,
         stocks=None) -> None:
    """
    数据下载入口。所有参数在配置文件 download.yaml 的 download 段中设置，
    也可以用 key=value 在命令行覆盖：
        python download_data.py -c download.yaml download.end=2026-11-15
    每日增量更新只需直接运行（end 默认为昨天），已下载到库的数据自动跳过。
    """
    all, full_refresh, dry_run = _as_bool(all), _as_bool(full_refresh), _as_bool(dry_run)
    sleep = float(sleep)
    end = end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if start > end:
        raise SystemExit(f"download.start ({start}) 不能晚于 download.end ({end})")

    sep = "=" * 60
    print(sep)
    print("Baostock A股日频数据下载")
    print(f"日期范围: {start} ~ {end} | 模式: {'全量刷新' if full_refresh else '增量更新'}"
          f"{' | 👁️ 预览（不写入）' if dry_run else ''}")
    print(f"数据库: {db}")
    print(sep)
    print()

    # 解析股票池：全 A 股需联网取代码表，其余直接来自配置/默认值
    import baostock as bs
    if all or not dry_run:
        lg = bs.login()
        if lg.error_code != "0":
            raise SystemExit(f"Baostock 登录失败: {lg.error_msg}")

    if all:
        pool = get_stock_list(bs, end)
    elif stocks:
        pool = [tuple(s) for s in stocks]
    else:
        pool = DEFAULT_STOCKS
    print(f"股票池: {len(pool)} 只（{'全 A 股' if all else '配置文件 stocks'}）")
    print()

    if dry_run:
        # 预览不写入：只根据库中已有进度列出每只股票将下载的区间
        conn = sqlite3.connect(db) if os.path.exists(db) else None
        status = get_stock_status(conn) if conn else {}
        for code, name, _ in pool:
            planned = plan_range(status.get(code), start, end, full_refresh)
            if planned is None:
                print(f"⏭️  {code} {name}: 已覆盖至 {status[code][1]}")
            else:
                print(f"📥 {code} {name}: {planned[2]} {planned[0]} ~ {planned[1]}")
        if conn:
            conn.close()
        if all:
            bs.logout()
        return

    conn = init_db(db)
    status = get_stock_status(conn)

    success_count = skip_count = fail_count = total_rows = 0
    t_start = time.time()

    for i, (code, name, sector) in enumerate(pool, 1):
        planned = plan_range(status.get(code), start, end, full_refresh)
        if planned is None:
            skip_count += 1
            print(f"[{i}/{len(pool)}] ⏭️  {code} {name}: 已最新 ({status[code][1]})")
            continue
        dl_start, dl_end, action = planned

        try:
            df = fetch_stock(bs, code, dl_start, dl_end)
            if df.empty:
                skip_count += 1
                print(f"[{i}/{len(pool)}] ⏭️  {code} {name}: 区间内无数据")
            else:
                save_stock(conn, code, name, sector, df)
                log_success(conn, code, action)
                conn.commit()
                success_count += 1
                total_rows += len(df)
                print(f"[{i}/{len(pool)}] ✅ {code} {name}: {len(df)}条 ({action} {dl_start}~{dl_end})")
        except Exception as e:
            fail_count += 1
            print(f"[{i}/{len(pool)}] ❌ {code} {name}: {str(e)[:100]}")

        if sleep > 0:
            time.sleep(sleep)

        elapsed = time.time() - t_start
        done = success_count + skip_count + fail_count
        remaining = elapsed / done * (len(pool) - done)
        print(f"      已用 {format_eta(elapsed)} | 预计剩余 {format_eta(remaining)}")

    bs.logout()
    conn.close()

    print()
    print(sep)
    print(f"完成: 共 {total_rows} 条 | ✅{success_count} | ⏭️跳过{skip_count} | ❌失败{fail_count}")
    print(f"数据库: {db}")
    print(sep)


if __name__ == "__main__":
    main()
