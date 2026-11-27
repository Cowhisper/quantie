# * 100!/usr/bin/env python3
"""
预测脚本: 以截至信号日的最近 seq_len 个交易日行情, 预测预测日的涨跌标签

默认以全市场最新交易日为信号日 (--date 可指定历史信号日, 窗口截至该日),
信号日无 bar 的股票视为停牌自动剔除, 控制台打印看涨概率最高的前 N 条, 全市场预测写入 CSV
(列: 涨跌幅/预测标签/各标签预测概率 等, 见 dl/README.md)

用法:
    python dl/predict.py --ckpt dl/checkpoints/best.pt
    python dl/predict.py --tickers sh.600519 sz.000001 --date 2026-11-26
"""
import argparse
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import torch

from data import CLS_NAMES, N_CONT, RET_CLIP, load_stocks, window_features
from log_util import fmt_eta, setup_logger
from net import load_model


def load_names(db_path):
    """股票名称用于展示; db 不存在或名称为历史脏数据 ('1') 时回退为代码"""
    if not db_path or not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    names = dict(conn.execute("SELECT ticker, name FROM stock_meta"))
    conn.close()
    return {t: n for t, n in names.items() if n and n != "1"}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="以最近行情预测明日涨跌标签")
    p.add_argument("--ckpt", default="dl/checkpoints/best.pt")
    p.add_argument("--data_dir", default="dl/data")
    p.add_argument("--db", default="a_stock_daily.db", help="查股票名称用, 可为空")
    p.add_argument("--date", default=None,
                   help="信号日 yyyymmdd 或 YYYY-MM-DD, 默认全市场最新交易日")
    p.add_argument("--tickers", nargs="*", default=None, help="只预测指定股票代码")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--top", type=int, default=50, help="控制台打印前 N 条")
    p.add_argument("--out_csv", default="dl/predict.csv")
    p.add_argument("--log_interval", type=int, default=100,
                   help="每 N 个 iteration 打印一次进度和 ETA, 0 关闭")
    p.add_argument("--log_file", default="", help="默认 <out_csv目录>/predict.log")
    args = p.parse_args()

    log_file = args.log_file or os.path.join(
        os.path.dirname(os.path.abspath(args.out_csv)), "predict.log")
    logger = setup_logger("predict", log_file)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device, n_cont=N_CONT,
                             num_classes=len(CLS_NAMES))
    seq_len = ckpt["config"]["seq_len"]

    _, stocks = load_stocks(args.data_dir)
    if args.tickers:
        stocks = {t: stocks[t] for t in args.tickers if t in stocks}
    if not stocks:
        logger.error("没有可预测的股票")
        return
    signal_day = (int(args.date.replace("-", "")) if args.date
                  else max(int(s[3][-1]) for s in stocks.values()))

    # 构造输入窗口: 截至信号日的 seq_len 日, 与训练/评估的特征口径完全一致
    max_span = seq_len * 2 + 15
    tickers, feats, sectors = [], [], []
    for ticker, (x, sector, ret, dates, ord_days) in stocks.items():
        i = int(np.searchsorted(dates, signal_day))
        if i >= len(dates) or int(dates[i]) != signal_day:
            continue  # 信号日停牌/无数据
        if i + 1 < seq_len:
            continue  # 信号日之前的历史不足
        if ord_days[i] - ord_days[i - seq_len + 1] > max_span:
            continue  # 窗口跨长期停牌
        tickers.append(ticker)
        feats.append(window_features(x, i + 1, seq_len))
        sectors.append(sector[i - seq_len + 1: i + 1].astype(np.int64))
    logger.info(f"模型 {args.ckpt} (epoch {ckpt.get('epoch', '?')}) | 设备 {device} | "
                f"信号日 {signal_day} | 可预测股票 {len(tickers)} 只")
    if not tickers:
        logger.error("信号日无可预测股票")
        return

    x_all = torch.from_numpy(np.stack(feats))
    s_all = torch.from_numpy(np.stack(sectors))
    probs, exp_rets = [], []
    n_batch = (len(tickers) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for bi, i in enumerate(range(0, len(tickers), args.batch_size)):
        logits, pred = model(x_all[i:i + args.batch_size].to(device),
                             s_all[i:i + args.batch_size].to(device))
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        exp_rets.append(pred.cpu().numpy())
        done = bi + 1
        if args.log_interval and done % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (n_batch - done)
            logger.info(f"  predict [{done}/{n_batch}] | {done / elapsed:.0f} it/s "
                        f"| ETA {fmt_eta(eta)}")
    probs = np.concatenate(probs)
    exp_ret = np.concatenate(exp_rets) * RET_CLIP  # 归一化值换算回收益率
    pred_cls = probs.argmax(axis=-1)
    bull = probs[:, 4:].sum(axis=1)   # 涨 + 大涨 + 近乎涨停
    bear = probs[:, :3].sum(axis=1)   # 近乎跌停 + 大跌 + 跌

    names = load_names(args.db)
    rows = []
    for i in np.argsort(-bull):
        t = tickers[i]
        rows.append({
            "股票代码": t, "股票名称": names.get(t, t),
            "信号日期": str(signal_day), "预测标签": CLS_NAMES[pred_cls[i]],
            "预测涨跌幅": f"{exp_ret[i]:.2%}",
            "看涨概率": f"{bull[i]:.2%}", "看跌概率": f"{bear[i]:.2%}",
            **{f"P({cls})": f"{pr:.2%}" for cls, pr in zip(CLS_NAMES, probs[i])},
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"全市场预测已写入 {args.out_csv} ({len(rows)} 只)")
    logger.info(f"看涨概率 Top {min(args.top, len(rows))}:")
    for r in rows[:args.top]:
        logger.info(f"  {r['股票代码']} {r['股票名称']} | {r['预测标签']} | "
                    f"涨跌幅 {r['预测涨跌幅']} | 看涨 {r['看涨概率']}")
    logger.info(f"日志已写入 {log_file}")


if __name__ == "__main__":
    main()
