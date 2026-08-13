from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sqlite3
import os


class BaseStrategy(ABC):
    """
    策略基类。

    所有量化策略必须继承此类，并实现：
    - prepare_data: 计算该策略所需的技术指标
    - generate_signals: 根据指标生成买卖信号（至少添加 Buy_Signal 列）
    - get_signal_reason: 返回触发买入信号的具体原因（用于可视化）
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.description = ""

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标。输入 df 必须包含基础 OHLCV 列。
        返回添加完指标后的 df。
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。必须返回包含 'Buy_Signal' 列的 df。
        建议同时实现 'Sell_Signal' 或止损逻辑。
        """
        pass

    def get_signal_reason(self, row) -> str:
        """
        返回某一行触发买入信号的原因描述。
        子类应重写以提供更具体的理由。
        """
        return "信号触发"

    # ------------------------------------------------------------------
    # 通用数据加载
    # ------------------------------------------------------------------
    def _load_all_data(
        self,
        db_path: str = "a_stock_daily.db",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        从 SQLite 数据库加载日线数据，支持按日期范围过滤。
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库不存在: {db_path}")

        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM stock_daily ORDER BY ticker, date"
        df = pd.read_sql(query, conn)
        conn.close()

        # 统一列名
        df.columns = [col.lower() for col in df.columns]
        df["date"] = pd.to_datetime(df["date"])

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        return df

    def analyze_stock(
        self, df_stock: pd.DataFrame, min_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        对单只股票计算指标并生成信号。
        """
        df_stock = df_stock.sort_values("date").reset_index(drop=True)

        if len(df_stock) < min_days:
            return None

        df_stock = self.prepare_data(df_stock)
        df_stock = self.generate_signals(df_stock)
        return df_stock

    # ------------------------------------------------------------------
    # 全市场扫描
    # ------------------------------------------------------------------
    def scan_market(
        self,
        db_path: str = "a_stock_daily.db",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_days: int = 60,
    ) -> pd.DataFrame:
        """
        扫描全市场，返回在指定时间范围内触发买入信号的股票列表。
        结果包含股票代码、名称、行业、日期、收盘价和买入原因。
        """
        # 为计算技术指标预留足够历史数据（取 start_date 前约 2*min_days 个自然日）
        load_start = None
        if start_date is not None:
            load_start = (pd.to_datetime(start_date) - timedelta(days=min_days * 2)).strftime("%Y-%m-%d")

        df = self._load_all_data(db_path, load_start, end_date)

        if df.empty:
            print("数据库中没有数据")
            return pd.DataFrame(
                columns=[
                    "ticker", "name", "sector", "date",
                    "open", "close", "pctchg", "reason"
                ]
            )

        selected = []
        for ticker, group in df.groupby("ticker", sort=False):
            analyzed = self.analyze_stock(group.copy(), min_days)
            if analyzed is None or analyzed.empty:
                continue

            signal_rows = analyzed[analyzed["Buy_Signal"] == True]
            for _, row in signal_rows.iterrows():
                selected.append(
                    {
                        "ticker": ticker,
                        "name": row.get("name", ticker),
                        "sector": row.get("sector", "-"),
                        "date": row["date"],
                        "open": row.get("open", 0.0),
                        "close": row["close"],
                        "pctchg": row.get("pctchg", 0.0),
                        "reason": self.get_signal_reason(row),
                    }
                )

        if selected:
            result = pd.DataFrame(selected)
        else:
            result = pd.DataFrame(
                columns=[
                    "ticker", "name", "sector", "date",
                    "open", "close", "pctchg", "reason"
                ]
            )

        # 只保留指定区间内的信号
        if start_date is not None and not result.empty:
            result = result[result["date"] >= pd.to_datetime(start_date)]
        if end_date is not None and not result.empty:
            result = result[result["date"] <= pd.to_datetime(end_date)]

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 默认回测引擎
    # ------------------------------------------------------------------
    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000,
        stop_loss_pct: float = 0.02,
        trailing_stop_pct: float = 0.08,
    ) -> Tuple[List[Dict], Dict]:
        """
        对单只股票数据执行简化回测。
        按 Buy_Signal 以收盘价买入，使用固定止损 + 移动止盈。
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        df = df[df["Buy_Signal"].notna()]

        trades = []
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        highest_price = 0

        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close_price = row["close"]

            if position == 0 and row["Buy_Signal"]:
                position = capital / close_price
                entry_price = close_price
                entry_date = date
                highest_price = close_price
                capital = 0
                trades.append(
                    {
                        "date": date,
                        "action": "BUY",
                        "price": close_price,
                        "shares": position,
                        "capital_used": position * close_price,
                        "reason": self.get_signal_reason(row),
                    }
                )
                continue

            if position > 0:
                if close_price > highest_price:
                    highest_price = close_price

                current_value = position * close_price
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                trailing_stop_price = highest_price * (1 - trailing_stop_pct)

                sell_signal = False
                sell_reason = ""

                if close_price < stop_loss_price:
                    sell_signal = True
                    sell_reason = "固定止损"
                elif close_price < trailing_stop_price:
                    sell_signal = True
                    sell_reason = "移动止盈"
                elif i == len(df) - 1:
                    sell_signal = True
                    sell_reason = "回测结束"

                if sell_signal:
                    capital = current_value
                    trades.append(
                        {
                            "date": date,
                            "action": "SELL",
                            "price": close_price,
                            "shares": position,
                            "capital_received": current_value,
                            "reason": sell_reason,
                            "profit": current_value - position * entry_price,
                            "profit_pct": (current_value - position * entry_price)
                            / (position * entry_price),
                        }
                    )
                    position = 0
                    entry_price = 0
                    highest_price = 0

        if position > 0:
            last_close = df.iloc[-1]["close"]
            capital = position * last_close
            trades.append(
                {
                    "date": df.iloc[-1]["date"],
                    "action": "SELL",
                    "price": last_close,
                    "shares": position,
                    "capital_received": capital,
                    "reason": "回测结束平仓",
                    "profit": capital - position * entry_price,
                    "profit_pct": (capital - position * entry_price)
                    / (position * entry_price),
                }
            )

        stats = self._calculate_performance(trades, initial_capital)
        return trades, stats

    def _calculate_performance(self, trades: List[Dict], initial_capital: float) -> Dict:
        """
        计算回测绩效指标。
        """
        sells = [t for t in trades if t["action"] == "SELL"]

        if not sells:
            return {
                "总交易次数": 0,
                "盈利次数": 0,
                "亏损次数": 0,
                "胜率": 0,
                "总收益率": 0,
                "最大单笔盈利": 0,
                "最大单笔亏损": 0,
                "平均收益率": 0,
                "总资金": initial_capital,
                "最大回撤": 0,
            }

        profits = [t["profit_pct"] for t in sells]
        win_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]

        total_return = (
            sum(t["capital_received"] for t in sells) - initial_capital
        ) / initial_capital

        return {
            "总交易次数": len(sells),
            "盈利次数": len(win_trades),
            "亏损次数": len(loss_trades),
            "胜率": len(win_trades) / len(sells) if sells else 0,
            "总收益率": total_return,
            "最大单笔盈利": max(profits) if profits else 0,
            "最大单笔亏损": min(profits) if profits else 0,
            "平均收益率": np.mean(profits) if profits else 0,
            "总资金": initial_capital * (1 + total_return),
            "最大回撤": 0,
        }
