"""quantie 统一命令行工具的任务入口层（quantie.py 的五个子命令分发到这里）。

quantie.py 把配置文件中与子命令同名的配置段（download/engine/forecast/
daily/export）作为关键字参数传给这里的任务函数；策略/条件类参数仍由
yacrs 按同名配置段自动绑定（配置文件加载时已并入 yacrs 全局配置 _C）。
"""

import glob
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta

import pandas as pd

from lib.download import (DATE_FMT, DEFAULT_DB, DEFAULT_START, _as_bool,
                          get_stock_status, plan_range, resolve_pool,
                          run_download)
from lib.engine import (Engine, build_strategies, export_trades_xlsx,
                        generate_report, load_data)
from lib.forecast import (_explain_zero_picks, run_scan, write_forecast_csv)
from lib.notify import DEFAULT_WEBHOOK, send_markdown, send_report_image
from lib.paper_account import (DEFAULT_ACCOUNT_DB, DEFAULT_INITIAL_CAPITAL,
                               build_report, report_markdown, sync_account)
from lib.report_img import render_report_image


# ----------------------------------------------------------------------
# download：增量下载行情数据（内核 lib/download.py）
# ----------------------------------------------------------------------

def download_task(db=DEFAULT_DB, start=DEFAULT_START, end=None,
                  all=False, full_refresh=False, dry_run=False, sleep=0.0,
                  stocks=None) -> None:
    """数据下载入口（配置段 download）。每日增量更新只需直接运行
    （end 默认为昨天），已下载到库的数据自动跳过。"""
    all, full_refresh, dry_run = (_as_bool(all), _as_bool(full_refresh),
                                  _as_bool(dry_run))
    end = end or (datetime.now() - timedelta(days=1)).strftime(DATE_FMT)
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

    pool = resolve_pool(bs, end, all=all, stocks=stocks)
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

    stats = run_download(db=db, start=start, end=end, all=all,
                         full_refresh=full_refresh, sleep=float(sleep),
                         stocks=stocks, bs=bs, pool=pool)
    bs.logout()

    print()
    print(sep)
    print(f"完成: 共 {stats['rows']} 条 | ✅{stats['success']} | "
          f"⏭️跳过{stats['skip']} | ❌失败{stats['fail']}")
    print(f"数据库: {db}")
    print(sep)


# ----------------------------------------------------------------------
# backtest：策略回测 + 报告（引擎内核 lib/engine.py）
# ----------------------------------------------------------------------

def backtest_task(db=DEFAULT_DB, start=None, end=None,
                  output="exp/strategy_report.html",
                  trades_xlsx="exp/trades.xlsx",
                  warmup=120, limit=0, max_charts=1000,
                  initial_capital=100000, max_positions=10,
                  strategies=None, conditions=None,
                  serve=False, host="127.0.0.1", port=8000) -> None:
    """回测入口（配置段 engine）。serve=true 启动 Web 服务（Vue 前端）
    展示报告，否则生成静态 HTML；两种模式都会把成交交易导出为
    trades_xlsx（空字符串表示不导出）。initial_capital 为账户初始资金；
    max_positions 为最大持仓股票数，资金与仓位按策略买入倾向评分
    （Trade.score）优先配给高分信号，保证现金不为负（lib/account.py）。"""
    if not start or not end:
        raise SystemExit("请在配置文件 engine 段设置 start / end"
                         "（回测起止日期 YYYY-MM-DD）")

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
        print("数据库中没有数据，请先运行数据下载（quantie.py download）")
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


# ----------------------------------------------------------------------
# forecast：预测下一交易日买入清单（扫描内核 lib/forecast.py）
# ----------------------------------------------------------------------

def forecast_task(db=DEFAULT_DB, download=True, start=DEFAULT_START, end=None,
                  all=False, full_refresh=False, sleep=0.0, stocks=None,
                  warmup=120, strategies=None, conditions=None,
                  output_dir="forecast", date=None, max_positions=10,
                  paper=True, paper_db=DEFAULT_ACCOUNT_DB,
                  paper_capital=DEFAULT_INITIAL_CAPITAL,
                  paper_max_positions=10) -> None:
    """明日买入预测入口（配置段 forecast）。
    db/start/end/all/full_refresh/sleep/stocks 为增量下载参数（语义与
    download 一致；end 默认为今天，盘后运行可取到当日行情）；warmup 为
    指标预热天数；strategies 为要预测的策略短名（ma/ross/rsi2/jlaw）；
    conditions 为每个策略附加的市场过滤条件（类名 -> 自动绑定同名配置段）；
    output_dir 为 CSV 输出目录；date 为预测买入日（默认信号日的下一个
    交易日，经 baostock 交易日历计算），即 CSV
    文件名 {date}-{策略}.csv 中的日期；max_positions 为每只策略最多输出
    的股票数（按评分降序取前 N，0 = 不限）；paper* 为本地模拟账户参数
    （扫描后同步：执行昨日挂单、检查持仓卖出、按日盯市并打印持仓明细与
    账户概览，paper=false 可关闭）。"""
    download, all, full_refresh = (_as_bool(download), _as_bool(all),
                                   _as_bool(full_refresh))
    paper = _as_bool(paper)
    sleep = float(sleep)
    warmup = int(warmup)
    max_positions = int(max_positions)
    paper_capital = float(paper_capital)
    paper_max_positions = int(paper_max_positions)

    today = datetime.now().strftime("%Y-%m-%d")
    end = end or today  # 预测需要最新行情，默认下载到今天

    sep = "=" * 60
    print(sep)
    print("明日买入预测")
    print(f"数据库: {db} | 下载: {'开' if download else '关'}"
          f"{'（全量刷新）' if full_refresh else ''}")
    print(sep)
    print()

    # 第 1 步：增量更新行情数据（与 download 同一内核）
    if download:
        print(">>> 第 1 步：增量更新行情数据")
        stats = run_download(db=db, start=start, end=end, all=all,
                             full_refresh=full_refresh, sleep=sleep,
                             stocks=stocks)
        print(f"    下载完成: 新增 {stats['rows']} 条 | "
              f"✅{stats['success']} ⏭️{stats['skip']} ❌{stats['fail']}")
    else:
        print(">>> 第 1 步：跳过数据下载（forecast.download=false）")
    print()

    # 第 2 步：策略扫描，产出预测买入日的信号
    print(">>> 第 2 步：策略扫描")
    signal_date, entry_date, results, stock_data = run_scan(
        db=db, warmup=warmup, strategies=strategies, conditions=conditions,
        date=date, max_positions=max_positions)
    signal_ts = pd.Timestamp(signal_date)
    print(f"    信号日 {signal_date} -> 预测买入日 {entry_date}，"
          f"{len(stock_data)} 只股票")
    print("    参考价格 = 信号日收盘价，实际成交以买入日开盘价为准")
    for short_name, strategy, picks in results:
        if not picks:
            _explain_zero_picks(strategy, stock_data, signal_ts)
        path = write_forecast_csv(picks, os.path.join(
            output_dir, f"{entry_date}-{short_name}.csv"), signal_date)
        print(f"[{strategy.name}] {len(picks)} 只 -> {path}")
        for t in picks:
            print(f"    {t.ticker} {t.name}  {int(t.buy_amount)} 股"
                  f"  @参考 {t.buy_price:.2f}  评分 {t.score:.3f}")
    print()

    # 第 3 步：本地模拟账户同步（执行昨日挂单、检查持仓卖出、按日盯市），
    # 并打印当前持仓（交易明细口径）与账户概览
    if paper:
        strategies_map = {short: strategy for short, strategy, _ in results}
        report = sync_account(
            market_db=db, strategies=strategies_map, signal_date=signal_date,
            output_dir=output_dir, account_db=paper_db,
            initial_capital=paper_capital, max_positions=paper_max_positions)
        print()
        print(report_markdown(report))
    else:
        print(">>> 模拟账户同步已关闭（forecast.paper=false）")
    print()
    print(sep)


# ----------------------------------------------------------------------
# daily：每日任务（TASKS.md 流程：数据更新 → 预测 → 推送）
# ----------------------------------------------------------------------

def build_ops_message(entry_date: str, rows) -> str:
    """按 TASKS.md 格式组装企业微信 markdown 消息。
    rows = [(股票名称, 股票代码, 操作), ...]（操作如 买入/卖出）。"""
    lines = [f"# {entry_date} 股票买入卖出操作", "",
             "股票\t股票代码\t操作"]
    if rows:
        lines += [f"{name}\t{ticker}\t{action}" for name, ticker, action in rows]
    else:
        lines.append("（今日无买入信号）")
    return "\n".join(lines)


def is_weekend(date_str: str) -> bool:
    """date_str（YYYY-MM-DD）是否为周末（周六/周日，A 股休市）。"""
    return datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5


def load_latest_picks(output_dir: str):
    """读取 output_dir 中最新买入日的预测 CSV，返回 (买入日, rows)，
    rows = [(股票名称, 股票代码, "买入"), ...]。手动推送（daily.mode=push）
    用：不重新预测，直接推送已生成的结果。无 CSV 时 SystemExit。"""
    parsed = []
    for path in glob.glob(os.path.join(output_dir, "*.csv")):
        base = os.path.basename(path)[:-4]  # 去掉 .csv
        # 文件名 {买入日}-{策略短名}.csv：日期在前（含连字符），用正则切分
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(.+)", base)
        if m:
            parsed.append((m.group(1), m.group(2), path))
    if not parsed:
        raise SystemExit(f"{output_dir} 中没有预测 CSV，请先用 full 模式运行一次")
    latest = max(entry for entry, _, _ in parsed)
    rows = []
    for entry, _, path in parsed:
        if entry != latest:
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"股票代码": str})
        for r in df.itertuples(index=False):
            rows.append((r.股票名称, r.股票代码, "买入"))
    return latest, rows


def daily_task(mode="full", push=True, webhook=DEFAULT_WEBHOOK, max_retries=5,
               retry_sleep=30, db=DEFAULT_DB, start=DEFAULT_START, end=None,
               all=False, full_refresh=False, sleep=0.0, stocks=None,
               warmup=120, strategies=None, conditions=None,
               output_dir="forecast", date=None, max_positions=10,
               paper=True, paper_db=DEFAULT_ACCOUNT_DB,
               paper_capital=DEFAULT_INITIAL_CAPITAL,
               paper_max_positions=10) -> None:
    """每日任务入口（配置段 daily，TASKS.md 流程）。mode=full：数据更新
    （失败重试）→ 预测 → 模拟账户同步 → 推送；mode=push：只推送最新已生成
    的预测结果（手动触发，附模拟账户当前状态）。db/start/end/all/
    full_refresh/sleep/stocks 为增量下载参数（语义与 download 一致；
    end 默认今天——闭市后（如 20:00）跑可取到当日行情，信号日 = 今天；
    凌晨盘前跑当日尚无数据，自动落在昨天）；warmup/strategies/conditions/
    output_dir/date/max_positions 为预测参数（语义与 forecast 一致）；
    paper* 为本地模拟账户参数（本金默认 10 万，同步持仓/成交记录，推送
    消息附上持仓明细与账户概览；paper=false 可关闭）；push=跑完后是否
    推送；webhook=企业微信机器人地址；max_retries/retry_sleep=下载失败
    重试次数与间隔秒数（0=不限次数重试）。"""
    mode = str(mode).lower()
    if mode not in ("full", "push"):
        raise SystemExit(f"daily.mode 只能是 full 或 push，got {mode}")
    push = _as_bool(push)
    paper = _as_bool(paper)
    max_retries = int(max_retries)
    retry_sleep = float(retry_sleep)
    paper_capital = float(paper_capital)
    paper_max_positions = int(paper_max_positions)

    sep = "=" * 60
    print(sep)
    print(f"每日任务（mode={mode}）")
    print(sep)
    print()

    if mode == "push":
        # 手动触发：直接推送 output_dir 中最新买入日的预测结果
        entry_date, rows = load_latest_picks(output_dir)
        print(f">>> 手动推送：{output_dir} 中买入日 {entry_date} 的预测结果"
              f"（{len(rows)} 条买入）")
        report = build_report(paper_db, db) if paper else None
        if paper and not report:
            print(f"    （账户库 {paper_db} 暂无记录，本次消息不含账户信息）")
    else:
        # 第 1 步：增量更新行情数据，失败重试至全部成功（TASKS.md 要求）
        end = end or datetime.now().strftime(DATE_FMT)
        attempt = 0
        while True:
            attempt += 1
            print(f">>> 第 1 步：增量更新行情数据（第 {attempt} 轮）")
            stats = run_download(db=db, start=start, end=end,
                                 all=_as_bool(all),
                                 full_refresh=_as_bool(full_refresh),
                                 sleep=float(sleep), stocks=stocks)
            print(f"    下载完成: 新增 {stats['rows']} 条 | "
                  f"✅{stats['success']} ⏭️{stats['skip']} ❌{stats['fail']}")
            if stats["fail"] == 0:
                break
            if max_retries > 0 and attempt >= max_retries:
                raise SystemExit(
                    f"重试 {attempt} 轮后仍有 {stats['fail']} 只股票下载失败，"
                    f"中止每日任务（避免基于不完整的数据产出信号）")
            print(f"    {stats['fail']} 只下载失败，{retry_sleep:g}s 后重试...")
            time.sleep(retry_sleep)
        print()

        # 第 2 步：策略扫描（与 forecast 同一 run_scan，口径一致）
        signal_date, entry_date, results, stock_data = run_scan(
            db=db, warmup=warmup, strategies=strategies, conditions=conditions,
            date=date, max_positions=max_positions)
        signal_ts = pd.Timestamp(signal_date)
        print(f">>> 第 2 步：策略扫描（信号日 {signal_date} -> 买入日 "
              f"{entry_date}）")
        rows = []
        for short_name, strategy, picks in results:
            if not picks:
                _explain_zero_picks(strategy, stock_data, signal_ts)
            path = write_forecast_csv(picks, os.path.join(
                output_dir, f"{entry_date}-{short_name}.csv"), signal_date)
            print(f"[{strategy.name}] {len(picks)} 只 -> {path}")
            for t in picks:
                print(f"    {t.ticker} {t.name}  {int(t.buy_amount)} 股"
                      f"  @参考 {t.buy_price:.2f}  评分 {t.score:.3f}")
                rows.append((t.name, t.ticker, "买入"))
        print()

        # 第 3 步：本地模拟账户同步（执行昨日挂单、检查持仓卖出、盯市快照）
        report = None
        if paper:
            strategies_map = {short: strategy for short, strategy, _ in results}
            report = sync_account(
                market_db=db, strategies=strategies_map, signal_date=signal_date,
                output_dir=output_dir, account_db=paper_db,
                initial_capital=paper_capital, max_positions=paper_max_positions)
        else:
            print(">>> 模拟账户同步已关闭（daily.paper=false）")
        print()

    if not push:
        print("daily.push=false，跳过推送")
        return
    if is_weekend(entry_date):
        print(f"买入日 {entry_date} 为周末休市，跳过推送")
        return
    send_markdown(build_ops_message(entry_date, rows), webhook=webhook)
    print(f">>> 已推送文字消息（买入日 {entry_date}，{len(rows)} 条买入）")
    if report:
        # 企业微信 markdown 不支持表格：持仓明细/卖出记录/账户概览渲染成
        # 图片以 image 消息发送；PIL/字体不可用时退回纯文本 markdown
        png = render_report_image(
            report, os.path.join(tempfile.gettempdir(),
                                 f"daily_report_{report['as_of']}.png"),
            ops_rows=rows, ops_date=entry_date)
        if png and send_report_image(png, webhook=webhook):
            print(">>> 已推送账户报告图片（持仓明细/本周期卖出/模拟账户）")
        else:
            send_markdown(report_markdown(report), webhook=webhook)
            print(">>> 图片渲染不可用，已改发纯文本账户明细")
    print(sep)


# ----------------------------------------------------------------------
# export：数据库导出 CSV（导出函数在 export_db.py，保持不变）
# ----------------------------------------------------------------------

def export_task(db=DEFAULT_DB, output_dir="csv_export", mode="all") -> None:
    """数据导出入口（配置段 export）。mode: all=全量合并 /
    by_stock=按股票拆分 / by_sector=按板块拆分 / meta=元信息 /
    summary=统计摘要 / everything=全部。"""
    import export_db

    if not os.path.exists(db):
        raise SystemExit(f"数据库不存在: {db}（请先运行 quantie.py download）")

    print(f"{'=' * 60}")
    print(f"📦 数据库: {db}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔄 导出模式: {mode}")
    print(f"{'=' * 60}")

    if mode == "all":
        export_db.export_all(db, output_dir)
    elif mode == "by_stock":
        export_db.export_by_stock(db, output_dir)
    elif mode == "by_sector":
        export_db.export_by_sector(db, output_dir)
    elif mode == "meta":
        export_db.export_meta(db, output_dir)
    elif mode == "summary":
        export_db.export_summary(db, output_dir)
    elif mode == "everything":
        print(">>> 模式1: 全量合并")
        export_db.export_all(db, output_dir)
        print()
        print(">>> 模式2: 按股票拆分")
        export_db.export_by_stock(db, os.path.join(output_dir, "by_stock"))
        print()
        print(">>> 模式3: 按板块拆分")
        export_db.export_by_sector(db, os.path.join(output_dir, "by_sector"))
        print()
        print(">>> 模式4: 元信息")
        export_db.export_meta(db, output_dir)
        print()
        print(">>> 模式5: 统计摘要")
        export_db.export_summary(db, output_dir)
    else:
        raise SystemExit(f"未知导出模式: {mode}（可选: all/by_stock/by_sector/"
                         f"meta/summary/everything）")

    print(f"{'=' * 60}")
    print(f"✅ 导出完成！文件保存在: {output_dir}/")
    print(f"{'=' * 60}")
