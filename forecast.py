#!/usr/bin/env python3
"""
明日买入预测脚本（yacrs 配置驱动）

步骤：
1. 增量更新行情数据（与 download_data.py 同一下载内核 lib/download.py，
   已覆盖的数据自动跳过；forecast.download=false 可跳过本步）
2. 策略扫描：信号日 = 库中最后一个交易日，在每只股票K线末尾追加一根
   「买入日」合成K线（开盘 = 信号日收盘价，仅作参考价），复用策略的 run()
   产出买入日 = 预测日的信号 —— 与回测完全同一套信号逻辑，保证预测口径一致
3. 建议股数 = 策略自带仓位管理按参考价计算
   （FixedCapitalPosition：固定本金向下取整到整手）
4. 每个策略输出一个 CSV：{output_dir}/{买入日}-{策略短名}.csv
   （列：股票代码/股票名称/建议股数/参考价格/评分/信号日期/买入原因）

注意：参考价格 = 信号日收盘价，实际成交以明日开盘价为准；若明日开盘
大幅高开/低开超出策略的追价区间（如 Ross 的 min/max_chase_pct），应放弃买入。

用法（所有参数见 configs/forecast.yaml 的 forecast 段，key=value 可覆盖）:
      python forecast.py -c configs/forecast.yaml
      python forecast.py -c configs/forecast.yaml forecast.strategies=[ma,ross] forecast.date=2026-11-18
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from yacrs import cli, register

import lib  # noqa: F401  import lib 即触发所有策略/条件的 yacrs 注册
from lib.download import (DEFAULT_DB, DEFAULT_START, _as_bool,
                          last_trade_date, run_download)
from lib.engine import build_strategies, load_data
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


def run_scan(db=DEFAULT_DB, warmup=120, strategies=None, conditions=None,
             date=None, max_positions=10):
    """第 2 步策略扫描：信号日 = 库中最新交易日，扫描各策略在预测买入日的
    信号。返回 (signal_date, entry_date, results, stock_data)，
    results = [(策略短名, 策略实例, picks)]（picks 已按 max_positions 截断）。
    forecast.py 与 daily_task.py（每日任务）共用本函数，保证预测口径一致。"""
    strategy_names = _as_list(strategies, ["ross"])
    if isinstance(conditions, str):
        conditions = None  # 命令行不便传映射，条件映射请写在配置文件里

    signal_date = last_trade_date(db)
    if not signal_date:
        raise SystemExit("数据库中没有数据，请先运行 download_data.py 或先执行数据更新")
    entry_date = date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    if entry_date <= signal_date:
        raise SystemExit(f"买入日（{entry_date}）必须晚于库中最新交易日 {signal_date}")

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


# ----------------------------------------------------------------------
# 入口（yacrs：-c 加载 yaml/json/toml 配置，key=value 命令行覆盖）
# ----------------------------------------------------------------------

@cli("forecast")
@register("forecast")
def main(db=DEFAULT_DB, download=True, start=DEFAULT_START, end=None,
         all=False, full_refresh=False, sleep=0.0, stocks=None,
         warmup=120, strategies=None, conditions=None,
         output_dir="forecast", date=None, max_positions=10) -> None:
    """
    明日买入预测入口。所有参数在配置文件 configs/forecast.yaml 的 forecast
    段中设置，也可以用 key=value 在命令行覆盖：
        python forecast.py -c configs/forecast.yaml forecast.strategies=[ross]
    db/start/end/all/full_refresh/sleep/stocks 为增量下载参数（语义与
    download_data.py 一致；end 默认为今天，盘后运行可取到当日行情）；
    warmup 为指标预热天数；strategies 为要预测的策略短名（ma/ross）；
    conditions 为每个策略附加的市场过滤条件（类名 -> 自动绑定同名配置段）；
    output_dir 为 CSV 输出目录；date 为预测买入日（默认明天），即 CSV
    文件名 {date}-{策略}.csv 中的日期；max_positions 为每只策略最多输出
    的股票数（按评分降序取前 N，0 = 不限）。
    """
    download, all, full_refresh = (_as_bool(download), _as_bool(all),
                                   _as_bool(full_refresh))
    sleep = float(sleep)
    warmup = int(warmup)
    max_positions = int(max_positions)

    today = datetime.now().strftime("%Y-%m-%d")
    end = end or today  # 预测需要最新行情，默认下载到今天

    sep = "=" * 60
    print(sep)
    print("明日买入预测")
    print(f"数据库: {db} | 下载: {'开' if download else '关'}"
          f"{'（全量刷新）' if full_refresh else ''}")
    print(sep)
    print()

    # 第 1 步：增量更新行情数据（与 download_data.py 同一内核）
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
    print(sep)


if __name__ == "__main__":
    main()
