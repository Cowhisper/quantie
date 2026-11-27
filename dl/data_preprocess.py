#!/usr/bin/env python3
"""
数据预处理脚本: 将 a_stock_daily.db 的日线行情转换为训练格式

每只股票输出一个 npz 文件 (默认 dl/data/{ticker}.npz):
    x      float32 (N, 7)  原始逐日特征 [time_norm, open, high, low, close, volume, turn]
                           time_norm 为当日在一年中的相对位置, 归一化到 [-1, 1]
    sector int32   (N,)    行业 id (词表见 meta.json 的 sector2id)
    ret    float32 (N,)    当日收益率 close/prev_close - 1 (首日为 NaN)
    dates  int32   (N,)    交易日期 yyyymmdd
meta.json 记录行业词表 sector2id / 特征名 x_features / 每只股票的有效行数

这里只做格式转换和基础清洗 (去重/去空/去停牌占位行),
窗口归一化与异常样本过滤在 data.py 中完成
"""
import argparse
import json
import os
import sqlite3

import numpy as np
import pandas as pd

X_FEATURES = ["time_norm", "open", "high", "low", "close", "volume", "turn"]

COLUMNS = "sector, date, open, high, low, close, volume, turn"


def clean_stock(df):
    """单只股票基础清洗: 去重/去空/去非交易日占位行, 返回 None 表示不可用"""
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] > 0)]  # volume=0 为停牌占位行
    df["turn"] = df["turn"].fillna(0.0)
    if len(df) == 0:
        return None
    return df.reset_index(drop=True)


def build_stock_arrays(df, sector2id, unknown_sector_id):
    dates = pd.to_datetime(df["date"])
    day_of_year = dates.dt.dayofyear.to_numpy()
    time_norm = np.clip((day_of_year - 1) / 365.0 * 2.0 - 1.0, -1.0, 1.0)

    x = np.column_stack([
        time_norm,
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        df["volume"].to_numpy(),
        df["turn"].to_numpy(),
    ]).astype(np.float32)

    close = df["close"].to_numpy(dtype=np.float64)
    ret = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0]).astype(np.float32)

    sector = df["sector"].map(sector2id).fillna(unknown_sector_id).to_numpy(dtype=np.int32)
    date_int = dates.dt.strftime("%Y%m%d").to_numpy(dtype=np.int32)
    return x, sector, ret, date_int


def main():
    parser = argparse.ArgumentParser(description="a_stock_daily.db -> 训练数据")
    parser.add_argument("--db", default="a_stock_daily.db", help="行情数据库路径")
    parser.add_argument("--out", default="dl/data", help="输出目录")
    parser.add_argument("--min_rows", type=int, default=30,
                        help="有效交易日少于此值的股票跳过")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 只股票 (调试用)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    sectors = sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT sector FROM stock_daily") if isinstance(r[0], str))
    sector2id = {s: i for i, s in enumerate(sectors)}
    unknown_sector_id = sector2id.get("-", 0)

    tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM stock_daily")]
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"股票数 {len(tickers)}, 行业数 {len(sector2id)}")

    os.makedirs(args.out, exist_ok=True)
    meta = {"sector2id": sector2id, "x_features": X_FEATURES, "tickers": {}}

    n_skip = 0
    for k, ticker in enumerate(tickers):
        df = pd.read_sql(
            f"SELECT {COLUMNS} FROM stock_daily WHERE ticker = ? ORDER BY date",
            conn, params=(ticker,))
        sdf = clean_stock(df)
        if sdf is None or len(sdf) < args.min_rows:
            n_skip += 1
            continue
        x, sector, ret, date_int = build_stock_arrays(sdf, sector2id, unknown_sector_id)
        np.savez(os.path.join(args.out, ticker.replace(".", "_") + ".npz"),
                 x=x, sector=sector, ret=ret, dates=date_int)
        meta["tickers"][ticker] = int(len(sdf))
        if (k + 1) % 1000 == 0:
            print(f"  进度 {k + 1}/{len(tickers)}")
    conn.close()

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"完成: {len(meta['tickers'])} 只股票 -> {args.out}/ (跳过 {n_skip} 只)")


if __name__ == "__main__":
    main()
