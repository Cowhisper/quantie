"""Baostock A股日频数据下载内核（quantie.py 的 download / forecast / daily 任务共用）。

- 国内源，不限流，完全免费，前复权
- 增量更新：每只股票只下载数据库中最后日期之后的数据，已覆盖的自动跳过
- 每只股票一次请求覆盖全部缺失区间（不做日期分块），请求数 = 股票数
- 每只股票写入后立即提交，中途断掉不丢已完成的部分（天然断点续传）
- 板块/行业：全 A 股股票池的行业取自 baostock 证监会行业分类
  （query_stock_industry，每周更新）；配置股票池用配置里的行业。
  每次下载会把行业信息同步进 stock_meta / stock_daily（含历史行）
"""

import os
import re
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd

DEFAULT_DB = "a_stock_daily.db"
DEFAULT_START = "2024-01-01"

# 默认核心股票池（stocks 为空时生效）
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


def _as_bool(v):
    return v if isinstance(v, bool) else str(v).lower() not in ("0", "false", "no", "")


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


def sync_sectors(conn, pool):
    """把股票池的行业信息同步进库：stock_meta / stock_daily 的 sector 更新为
    当前分类（历史行一并刷新，保持同一股票全库口径一致；save_stock 只覆盖
    本次下载区间的行，没有这一步旧的 "-" 会一直留在历史数据里）。
    行业为 "-"（未获取到）的股票不动。返回更新了多少只股票。"""
    known = {code: sector for code, _, sector in pool
             if sector and sector != "-"}
    if not known:
        return 0
    current = dict(conn.execute("SELECT ticker, sector FROM stock_meta"))
    changed = [(code, sector) for code, sector in known.items()
               if code in current and current[code] != sector]
    for code, sector in changed:
        conn.execute("UPDATE stock_meta SET sector = ? WHERE ticker = ?",
                     (sector, code))
        conn.execute("UPDATE stock_daily SET sector = ? "
                     "WHERE ticker = ? AND sector != ?", (sector, code, sector))
    return len(changed)


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


def get_industry_map(bs, date):
    """全 A 股行业分类（baostock query_stock_industry，证监会行业分类，
    每周更新），返回 {code: 行业名称}（已剥离 "J66"/"C36" 这类分类代码
    前缀）。查询失败返回 {}——行业信息是附加数据，不阻断行情下载。"""
    try:
        rs = bs.query_stock_industry(date=date)
        if rs.error_code != "0":
            return {}
        i_code, i_ind = rs.fields.index("code"), rs.fields.index("industry")
        result = {}
        while rs.next():
            row = rs.get_row_data()
            industry = re.sub(r"^[A-Z]\d+", "", row[i_ind]).strip()
            if industry:
                result[row[i_code]] = industry
        return result
    except Exception:
        return {}


def get_stock_list(bs, end_date):
    """全 A 股代码表（取 end_date 当天及之前最近一个有数据的交易日），
    行业取自 baostock 证监会行业分类，未收录的股票为 "-"。"""
    industry = get_industry_map(bs, end_date)
    day = datetime.strptime(end_date, "%Y-%m-%d")
    for _ in range(10):  # 向前回退直到命中交易日
        rs = bs.query_all_stock(day=day.strftime("%Y-%m-%d"))
        stocks = []
        while (rs.error_code == "0") & rs.next():
            row = rs.get_row_data()
            code = row[rs.fields.index("code")]
            if code.startswith("sh.") or code.startswith("sz."):
                stocks.append((code, row[rs.fields.index("code_name")],
                               industry.get(code, "-")))
        if stocks:
            return stocks
        day -= timedelta(days=1)
    raise RuntimeError("无法获取全 A 股代码表（连续 10 天均非交易日？）")


def resolve_pool(bs, end_date, all=False, stocks=None):
    """解析股票池：全 A 股需联网取代码表，其余直接来自配置/默认值。"""
    if all:
        return get_stock_list(bs, end_date)
    if stocks:
        return [tuple(s) for s in stocks]
    return DEFAULT_STOCKS


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


# ----------------------------------------------------------------------
# 增量下载主流程（quantie.py 的 download / forecast / daily 任务共用）
# ----------------------------------------------------------------------

def run_download(db=DEFAULT_DB, start=DEFAULT_START, end=None,
                 all=False, full_refresh=False, sleep=0.0,
                 stocks=None, bs=None, pool=None) -> dict:
    """
    增量下载主流程：登录 baostock，逐只下载股票池并写入 db（每只股票
    写完即提交）。返回 {"success", "skip", "fail", "rows"} 统计。
    all/full_refresh 等布尔参数请调用方自行归一（见 _as_bool）。
    bs / pool 可由调用方传入复用（如 download 任务已登录并解析过
    股票池），否则本函数自行登录 baostock、自行解析股票池。
    """
    sleep = float(sleep)
    end = end or (datetime.now() - timedelta(days=1)).strftime(DATE_FMT)
    if start > end:
        raise SystemExit(f"下载起始日期 ({start}) 不能晚于结束日期 ({end})")

    own_session = bs is None
    if bs is None:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise SystemExit(f"Baostock 登录失败: {lg.error_msg}")

    if pool is None:
        pool = resolve_pool(bs, end, all=all, stocks=stocks)
    print(f"股票池: {len(pool)} 只（{'全 A 股' if all else '配置文件 stocks'}）")
    print()

    conn = init_db(db)
    status = get_stock_status(conn)

    n_sector = sync_sectors(conn, pool)
    if n_sector:
        conn.commit()
        print(f"行业分类同步: {n_sector} 只股票的板块信息已更新")
        print()

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

    if own_session:
        bs.logout()
    conn.close()
    return {"success": success_count, "skip": skip_count,
            "fail": fail_count, "rows": total_rows}


def last_trade_date(db_path) -> str | None:
    """库中全部行情的最大日期（最后一个交易日），空库返回 None。"""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
    conn.close()
    return row[0] if row else None


def next_trade_date(date_str, bs=None) -> str | None:
    """date_str 之后的第一个交易日（baostock 交易日历，含节假日），
    查询失败（未安装/未联网/登录失败）返回 None，由调用方决定兜底。
    bs 可传入已登录的 baostock 模块复用连接，否则自行登录并登出。"""
    try:
        own_login = bs is None
        if own_login:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                return None
        try:
            end = (datetime.strptime(date_str, DATE_FMT)
                   + timedelta(days=15)).strftime(DATE_FMT)
            rs = bs.query_trade_dates(start_date=date_str, end_date=end)
            if rs.error_code != "0":
                return None
            while rs.next():
                day, is_trading = rs.get_row_data()[:2]
                if day > date_str and is_trading == "1":
                    return day
            return None
        finally:
            if own_login:
                bs.logout()
    except Exception:
        return None
