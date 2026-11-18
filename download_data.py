#!/usr/bin/env python3
"""
A股日频数据下载脚本 —— Baostock 版（yacrs 配置驱动）

- 国内源，不限流，完全免费，前复权
- 增量更新：每只股票只下载数据库中最后日期之后的数据，已覆盖的自动跳过
- 每只股票一次请求覆盖全部缺失区间（不做日期分块），请求数 = 股票数
- 每只股票写入后立即提交，中途断掉不丢已完成的部分（天然断点续传）
- 下载内核在 lib/download.py（forecast.py 的增量更新复用同一内核）
- 用法（所有参数见 download.yaml 的 download 段，命令行 key=value 可覆盖任意项）:
      python download_data.py -c download.yaml
      python download_data.py -c download.yaml download.end=2026-11-15 download.all=true
"""

import os
import sqlite3
from datetime import datetime, timedelta

from yacrs import cli, register

from lib.download import (DATE_FMT, DEFAULT_DB, DEFAULT_START, _as_bool,
                          get_stock_status, plan_range, resolve_pool,
                          run_download)


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
                         full_refresh=full_refresh, sleep=sleep, stocks=stocks,
                         bs=bs, pool=pool)
    bs.logout()

    print()
    print(sep)
    print(f"完成: 共 {stats['rows']} 条 | ✅{stats['success']} | "
          f"⏭️跳过{stats['skip']} | ❌失败{stats['fail']}")
    print(f"数据库: {db}")
    print(sep)


if __name__ == "__main__":
    main()
