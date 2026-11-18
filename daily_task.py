#!/usr/bin/env python3
"""每日任务：增量更新行情（失败自动重试）→ 策略预测 → 企业微信推送（TASKS.md）。

两种模式（daily.mode）：
- full（默认）：数据更新 → 预测 → 推送。供定时任务使用（crontab 每天 02:00，
  见 configs/daily.yaml 头部注释）；下载失败按 TASKS.md 要求重试直至全部成功
  （受 max_retries 限制，0 = 不限次数）。
- push：只推送。手动触发场景：不更新数据、不重新预测，直接推送 output_dir
  中最新买入日已生成的预测 CSV（比如白天已经跑过预测，只想重发消息）。

消息格式见 TASKS.md（企业微信 markdown）：
    # <date> 股票买入卖出操作
    股票\t股票代码\t操作
    xxx\tsh.xxxxx\t买入

注意：仓库目前没有真实持仓记录，卖出无从谈起，消息只包含策略预测的
「买入」部分；无买入信号时推送「（今日无买入信号）」。

用法（所有参数见 configs/daily.yaml 的 daily 段，key=value 可覆盖）：
      python daily_task.py -c configs/daily.yaml                      # 完整流程
      python daily_task.py -c configs/daily.yaml daily.push=false     # 完整流程，不推送
      python daily_task.py -c configs/daily.yaml daily.mode=push      # 手动触发推送
"""

import glob
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
from yacrs import cli, register

import forecast
from forecast import run_scan, write_forecast_csv
import lib  # noqa: F401  import lib 即触发所有策略/条件的 yacrs 注册
from lib.download import DEFAULT_DB, DEFAULT_START, _as_bool, run_download
from lib.notify import DEFAULT_WEBHOOK, send_markdown


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


# ----------------------------------------------------------------------
# 入口（yacrs：-c 加载 yaml/json/toml 配置，key=value 命令行覆盖）
# ----------------------------------------------------------------------

@cli("daily")
@register("daily")
def main(mode="full", push=True, webhook=DEFAULT_WEBHOOK, max_retries=5,
         retry_sleep=30, db=DEFAULT_DB, start=DEFAULT_START, end=None,
         all=False, full_refresh=False, sleep=0.0, stocks=None,
         warmup=120, strategies=None, conditions=None,
         output_dir="forecast", date=None, max_positions=10) -> None:
    """每日任务入口（TASKS.md 流程）。mode=full：数据更新（失败重试）→
    预测 → 推送；mode=push：只推送最新已生成的预测结果（手动触发）。
    db/start/end/all/full_refresh/sleep/stocks 为增量下载参数（语义与
    download_data.py 一致；end 默认昨天——凌晨运行时昨日数据已完整，盘后
    手动跑可用 end=<当天> 或 forecast.py 取到当日行情）；warmup/strategies/
    conditions/output_dir/date/max_positions 为预测参数（语义与 forecast.py
    一致）；push=跑完后是否推送；webhook=企业微信机器人地址；max_retries/
    retry_sleep=下载失败重试次数与间隔秒数（0=不限次数重试）。"""
    mode = str(mode).lower()
    if mode not in ("full", "push"):
        raise SystemExit(f"daily.mode 只能是 full 或 push，got {mode}")
    push = _as_bool(push)
    max_retries = int(max_retries)
    retry_sleep = float(retry_sleep)

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
    else:
        # 第 1 步：增量更新行情数据，失败重试至全部成功（TASKS.md 要求）
        end = end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
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

        # 第 2 步：策略扫描（与 forecast.py 同一 run_scan，口径一致）
        signal_date, entry_date, results, stock_data = run_scan(
            db=db, warmup=warmup, strategies=strategies, conditions=conditions,
            date=date, max_positions=max_positions)
        signal_ts = pd.Timestamp(signal_date)
        print(f">>> 第 2 步：策略扫描（信号日 {signal_date} -> 买入日 "
              f"{entry_date}）")
        rows = []
        for short_name, strategy, picks in results:
            if not picks:
                forecast._explain_zero_picks(strategy, stock_data, signal_ts)
            path = write_forecast_csv(picks, os.path.join(
                output_dir, f"{entry_date}-{short_name}.csv"), signal_date)
            print(f"[{strategy.name}] {len(picks)} 只 -> {path}")
            for t in picks:
                print(f"    {t.ticker} {t.name}  {int(t.buy_amount)} 股"
                      f"  @参考 {t.buy_price:.2f}  评分 {t.score:.3f}")
                rows.append((t.name, t.ticker, "买入"))
        print()

    if not push:
        print("daily.push=false，跳过推送")
        return
    message = build_ops_message(entry_date, rows)
    send_markdown(message, webhook=webhook)
    print(f">>> 已推送到企业微信（买入日 {entry_date}，{len(rows)} 条买入）")
    print(sep)


if __name__ == "__main__":
    main()
