#!/usr/bin/env python3
"""
策略测试框架

功能：
1. 自动发现 strategies/ 目录下所有继承自 BaseStrategy 的策略类。
2. 在指定时间区间内运行每个策略，扫描全市场买入信号。
3. 对触发信号的股票进行回测，验证策略输出。
4. 生成可视化 HTML 报告，列出每只股票代码与买入原因。

用法：
    python test_framework.py --start 2024-01-01 --end 2024-12-31
"""

import argparse
import importlib
import inspect
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class StrategyTestFramework:
    """
    策略测试框架。

    参数：
        db_path: SQLite 数据库路径
        start_date: 测试开始日期，格式 YYYY-MM-DD；为空则使用数据库最早日期
        end_date: 测试结束日期，格式 YYYY-MM-DD；为空则使用数据库最晚日期
        min_days: 单只股票最少需要多少天数据才参与计算
        initial_capital: 回测初始资金
    """

    def __init__(
        self,
        db_path: str = "a_stock_daily.db",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_days: int = 60,
        initial_capital: float = 100000,
    ):
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.min_days = min_days
        self.initial_capital = initial_capital
        self.strategies: List[BaseStrategy] = self._discover_strategies()

    # ------------------------------------------------------------------
    # 策略发现
    # ------------------------------------------------------------------
    def _discover_strategies(self) -> List[BaseStrategy]:
        """
        自动发现 strategies/ 目录下所有继承自 BaseStrategy 的非抽象策略类。
        """
        strategies = []
        strategies_dir = os.path.join(os.path.dirname(__file__), "strategies")

        if not os.path.isdir(strategies_dir):
            raise FileNotFoundError(f"策略目录不存在: {strategies_dir}")

        for filename in sorted(os.listdir(strategies_dir)):
            if not filename.endswith(".py"):
                continue
            if filename in ("__init__.py", "base.py"):
                continue

            module_name = f"strategies.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"  跳过 {filename}: {e}")
                continue

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseStrategy)
                    and obj is not BaseStrategy
                    and not inspect.isabstract(obj)
                ):
                    instance = obj()
                    strategies.append(instance)

        return strategies

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def _verify_scan_results(self, df: pd.DataFrame) -> None:
        """
        验证扫描结果是否包含必要字段、无空值、日期在范围内。
        """
        required_cols = ["ticker", "name", "sector", "date", "open", "close", "pctchg", "reason"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"扫描结果缺少必要列: {missing}")

        if df.empty:
            return

        if df["ticker"].isna().any():
            raise ValueError("ticker 列存在空值")
        if df["date"].isna().any():
            raise ValueError("date 列存在空值")
        if df["reason"].isna().any():
            raise ValueError("reason 列存在空值")

        if self.start_date:
            min_date = pd.to_datetime(self.start_date)
            if (df["date"] < min_date).any():
                raise ValueError(f"存在早于开始日期 {self.start_date} 的信号")
        if self.end_date:
            max_date = pd.to_datetime(self.end_date)
            if (df["date"] > max_date).any():
                raise ValueError(f"存在晚于结束日期 {self.end_date} 的信号")

    def _verify_backtest_stats(self, stats: Dict) -> None:
        """
        验证回测绩效统计结构正确。
        """
        required_keys = ["总交易次数", "盈利次数", "亏损次数", "胜率", "总收益率"]
        missing = [k for k in required_keys if k not in stats]
        if missing:
            raise ValueError(f"回测绩效缺少必要字段: {missing}")

    # ------------------------------------------------------------------
    # 运行测试
    # ------------------------------------------------------------------
    def run_all(self) -> Dict:
        """
        对所有发现的策略运行扫描与回测，返回结构化结果。
        """
        results = {}

        print("=" * 60)
        print("策略测试框架")
        print("=" * 60)
        print(f"数据库: {self.db_path}")
        print(f"时间区间: {self.start_date or '最早'} ~ {self.end_date or '最晚'}")
        print(f"发现策略数: {len(self.strategies)}")

        for strategy in self.strategies:
            print("\n" + "-" * 60)
            print(f"[{strategy.name}] {strategy.description}")
            print("-" * 60)

            # 1. 全市场扫描
            print("[1] 扫描买入信号...")
            selected = strategy.scan_market(
                db_path=self.db_path,
                start_date=self.start_date,
                end_date=self.end_date,
                min_days=self.min_days,
            )
            self._verify_scan_results(selected)
            print(f"    共 {len(selected)} 条买入信号")

            # 2. 对触发信号的股票逐一回测
            print("[2] 回测触发信号的股票...")
            backtest_results = []
            if not selected.empty:
                # 加载全部数据一次，避免重复读取数据库；为指标计算预留历史数据
                load_start = None
                if self.start_date is not None:
                    load_start = (
                        pd.to_datetime(self.start_date)
                        - pd.Timedelta(days=self.min_days * 2)
                    ).strftime("%Y-%m-%d")
                all_data = strategy._load_all_data(
                    self.db_path, load_start, self.end_date
                )
                tickers = selected["ticker"].unique()
                for ticker in tickers:
                    df_stock = all_data[all_data["ticker"] == ticker].copy()
                    analyzed = strategy.analyze_stock(df_stock, self.min_days)
                    if analyzed is None or analyzed.empty:
                        continue
                    trades, stats = strategy.backtest(
                        analyzed, initial_capital=self.initial_capital
                    )
                    self._verify_backtest_stats(stats)
                    backtest_results.append(
                        {
                            "ticker": ticker,
                            "name": analyzed["name"].iloc[-1]
                            if "name" in analyzed.columns
                            else ticker,
                            "trades": trades,
                            "stats": stats,
                            "df": analyzed[["date", "open", "high", "low", "close", "volume", "pctchg"]].copy(),
                        }
                    )

            # 3. 聚合绩效
            aggregated = self._aggregate_performance(backtest_results)
            print(f"    回测股票数: {len(backtest_results)}")
            print(f"    总交易次数: {aggregated['总交易次数']}")
            print(f"    总收益率: {aggregated['总收益率']*100:.2f}%")

            results[strategy.name] = {
                "selected": selected,
                "backtests": backtest_results,
                "aggregated": aggregated,
            }

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        return results

    def _aggregate_performance(self, backtest_results: List[Dict]) -> Dict:
        """
        聚合多只股票回测结果。
        """
        total_sells = 0
        total_wins = 0
        total_losses = 0
        total_profit = 0
        max_win = 0
        max_loss = 0
        all_profits = []

        for item in backtest_results:
            stats = item["stats"]
            total_sells += stats["总交易次数"]
            total_wins += stats["盈利次数"]
            total_losses += stats["亏损次数"]
            if stats["总交易次数"] > 0:
                # 用每笔交易的平均收益率近似聚合收益
                total_profit += stats["平均收益率"] * stats["总交易次数"]
                all_profits.extend([stats["平均收益率"]] * stats["总交易次数"])
            max_win = max(max_win, stats["最大单笔盈利"])
            max_loss = min(max_loss, stats["最大单笔亏损"])

        avg_return = (total_profit / total_sells) if total_sells > 0 else 0
        win_rate = (total_wins / total_sells) if total_sells > 0 else 0

        return {
            "回测股票数": len(backtest_results),
            "总交易次数": total_sells,
            "盈利次数": total_wins,
            "亏损次数": total_losses,
            "胜率": win_rate,
            "总收益率": avg_return,
            "最大单笔盈利": max_win,
            "最大单笔亏损": max_loss,
            "平均收益率": avg_return,
        }

    # ------------------------------------------------------------------
    # 可视化报告
    # ------------------------------------------------------------------
    def _build_holding_table(self, holding_df: pd.DataFrame) -> str:
        """
        生成持有期间日线数据的 HTML 子表。
        """
        if holding_df.empty:
            return "<p class='no-data'>无持有期间数据</p>"

        rows = ""
        for _, r in holding_df.iterrows():
            pct = r.get("pctchg", 0.0)
            pct_class = "up" if pct >= 0 else "down"
            pct_sign = "+" if pct >= 0 else ""
            rows += (
                f"<tr>"
                f"<td>{r['date'].strftime('%Y-%m-%d')}</td>"
                f"<td>{r['open']:.2f}</td>"
                f"<td>{r['high']:.2f}</td>"
                f"<td>{r['low']:.2f}</td>"
                f"<td>{r['close']:.2f}</td>"
                f"<td>{r.get('volume', 0):,.0f}</td>"
                f"<td class='{pct_class}'>{pct_sign}{pct:.2f}%</td>"
                f"</tr>"
            )

        return f"""
        <div class="holding-table-wrap">
            <table class="holding-table">
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>开盘</th>
                        <th>最高</th>
                        <th>最低</th>
                        <th>收盘</th>
                        <th>成交量</th>
                        <th>涨跌幅</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _build_trades_table(self, backtests: List[Dict]) -> List[Dict]:
        """
        将每个股票的回测交易记录配对为完整的买入-卖出交易列表，
        并附带持有期间的日线数据。
        """
        completed = []
        for item in backtests:
            ticker = item["ticker"]
            name = item["name"]
            df = item.get("df")
            pending_buy = None
            for trade in item["trades"]:
                if trade["action"] == "BUY":
                    pending_buy = trade
                elif trade["action"] == "SELL" and pending_buy is not None:
                    buy_date = pending_buy["date"]
                    sell_date = trade["date"]
                    holding_df = pd.DataFrame()
                    if df is not None and not df.empty:
                        mask = (df["date"] >= buy_date) & (df["date"] <= sell_date)
                        holding_df = df.loc[mask].copy()
                    completed.append(
                        {
                            "ticker": ticker,
                            "name": name,
                            "buy_date": buy_date,
                            "sell_date": sell_date,
                            "buy_price": pending_buy["price"],
                            "sell_price": trade["price"],
                            "profit_pct": trade.get("profit_pct", 0.0),
                            "buy_reason": pending_buy.get("reason", "-"),
                            "sell_reason": trade.get("reason", "-"),
                            "holding_df": holding_df,
                        }
                    )
                    pending_buy = None
        return completed

    def generate_report(
        self, results: Dict, output_path: str = "strategy_test_report.html"
    ) -> str:
        """
        生成 HTML 可视化报告，按每笔交易维度展示买入日期、卖出日期、
        买入价格、卖出价格、收益率及买卖原因。
        """
        rows = []
        for strategy_name, data in results.items():
            selected = data["selected"]
            aggregated = data["aggregated"]

            if selected.empty:
                rows.append(
                    f"""
                    <div class="strategy-card">
                        <h2>{strategy_name}</h2>
                        <p class="meta">{self._strategy_desc(strategy_name)}</p>
                        <p>该区间未触发任何买入信号。</p>
                    </div>
                    """
                )
                continue

            # 聚合每笔完整交易（买入 -> 卖出）
            trades_table = self._build_trades_table(data["backtests"])

            table_rows = ""
            for idx, t in enumerate(trades_table):
                pct = t["profit_pct"]
                pct_class = "up" if pct >= 0 else "down"
                pct_sign = "+" if pct >= 0 else ""
                detail_id = f"trade-detail-{strategy_name.replace(' ', '-')}-{idx}"

                # 持有期间日线数据子表
                holding_html = self._build_holding_table(t["holding_df"])

                table_rows += (
                    f"<tr class='trade-row' onclick=\"document.getElementById('{detail_id}').classList.toggle('hidden')\">"
                    f"<td>{t['ticker']}</td>"
                    f"<td>{t['name']}</td>"
                    f"<td>{t['buy_date'].strftime('%Y-%m-%d')}</td>"
                    f"<td>{t['sell_date'].strftime('%Y-%m-%d')}</td>"
                    f"<td>{t['buy_price']:.2f}</td>"
                    f"<td>{t['sell_price']:.2f}</td>"
                    f"<td class='{pct_class}'>{pct_sign}{pct*100:.2f}%</td>"
                    f"<td>{t['sell_reason']}</td>"
                    f"<td>{t['buy_reason']}</td>"
                    f"</tr>"
                    f"<tr id='{detail_id}' class='trade-detail hidden'>"
                    f"<td colspan='9'>{holding_html}</td>"
                    f"</tr>"
                )

            agg_html = ""
            for k, v in aggregated.items():
                if isinstance(v, float):
                    if k in ("胜率", "总收益率", "平均收益率"):
                        agg_html += f"<span class='metric'><b>{k}:</b> {v*100:.2f}%</span>"
                    else:
                        agg_html += f"<span class='metric'><b>{k}:</b> {v:.4f}</span>"
                else:
                    agg_html += f"<span class='metric'><b>{k}:</b> {v}</span>"

            rows.append(
                f"""
                <div class="strategy-card">
                    <h2>{strategy_name}</h2>
                    <p class="meta">{self._strategy_desc(strategy_name)}</p>
                    <div class="metrics">{agg_html}</div>
                    <table>
                        <thead>
                            <tr>
                                <th>股票代码</th>
                                <th>名称</th>
                                <th>买入日期</th>
                                <th>卖出日期</th>
                                <th>买入价格</th>
                                <th>卖出价格</th>
                                <th>收益率</th>
                                <th>卖出原因</th>
                                <th>买入原因</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
                """
            )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>策略测试结果</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f6f8fa;
            color: #24292f;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 24px;
            margin-bottom: 8px;
            color: #0969da;
        }}
        .summary {{
            color: #57606a;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .strategy-card {{
            background: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .strategy-card h2 {{
            font-size: 18px;
            margin-bottom: 4px;
            color: #1f2328;
        }}
        .meta {{
            color: #57606a;
            font-size: 13px;
            margin-bottom: 16px;
        }}
        .metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            font-size: 13px;
        }}
        .metric {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 16px;
            padding: 4px 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #d0d7de;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
            color: #57606a;
        }}
        tr:hover td {{
            background: #f6f8fa;
        }}
        .up {{ color: #d93026; font-weight: 600; }}
        .down {{ color: #1a7f37; font-weight: 600; }}
        .reason {{
            max-width: 300px;
            word-break: break-word;
        }}
        .trade-row {{
            cursor: pointer;
        }}
        .trade-row:hover td {{
            background: #eef4fb;
        }}
        .trade-detail.hidden {{
            display: none;
        }}
        .trade-detail td {{
            padding: 0;
            border-bottom: none;
            background: #f6f8fa;
        }}
        .holding-table-wrap {{
            padding: 12px 24px 16px 24px;
        }}
        .holding-table {{
            width: 100%;
            font-size: 12px;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            overflow: hidden;
        }}
        .holding-table th {{
            background: #e7effc;
            color: #1f2328;
            padding: 8px 10px;
        }}
        .holding-table td {{
            padding: 7px 10px;
            border-bottom: 1px solid #e1e4e8;
            background: #ffffff;
        }}
        .holding-table tr:last-child td {{
            border-bottom: none;
        }}
        .no-data {{
            padding: 12px 24px;
            color: #57606a;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>策略测试报告</h1>
        <div class="summary">
            数据库: {self.db_path} | 时间区间: {self.start_date or '最早'} ~ {self.end_date or '最晚'} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        {''.join(rows)}
    </div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return os.path.abspath(output_path)

    def _strategy_desc(self, strategy_name: str) -> str:
        for s in self.strategies:
            if s.name == strategy_name:
                return s.description
        return ""


# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="策略测试框架")
    parser.add_argument(
        "--db", default="a_stock_daily.db", help="SQLite 数据库路径"
    )
    parser.add_argument(
        "--start",
        default=None,
        help="测试开始日期 (YYYY-MM-DD)，默认使用数据库最早日期",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="测试结束日期 (YYYY-MM-DD)，默认使用数据库最晚日期",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=60,
        help="单只股票最少数据天数",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="回测初始资金",
    )
    parser.add_argument(
        "--output",
        default="strategy_test_report.html",
        help="输出 HTML 报告路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    framework = StrategyTestFramework(
        db_path=args.db,
        start_date=args.start,
        end_date=args.end,
        min_days=args.min_days,
        initial_capital=args.capital,
    )

    results = framework.run_all()
    report_path = framework.generate_report(results, output_path=args.output)
    print(f"\n报告已生成: {report_path}")


if __name__ == "__main__":
    main()
