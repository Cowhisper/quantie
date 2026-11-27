"""
PyTorch Dataset: 从 data_preprocess.py 的输出构建 (输入窗口, 预测日标签) 样本

口径定义 (见 dl/README.md):
    信号日 = 取信号的当天, 预测日 = 信号日的下一交易日
    涨跌幅 = (预测日收盘价 - 信号日收盘价) / 信号日收盘价

样本定义:
    输入: 截至信号日的连续 seq_len 个交易日 (含信号日),
          每个交易日特征 C = [time_norm, open, high, low, close, volume, turn]
          + 行业 Embedding; 归一化:
          - time_norm: 日期在一年中的相对位置, [-1, 1]
          - 价格 open/high/low/close: 按窗口内价格最大/最小值归一化到 [-1, 1]
          - 交易量: 按窗口内最大/最小值归一化到 [-1, 1]
          - 换手率: 按窗口内最大/最小值归一化到 [-1, 1]
    标签: 预测日涨跌幅 ret (= close[预测日]/close[信号日] - 1, 预处理时已算好)
          分类标签 7 类: 近乎跌停/大跌/跌/波动/涨/大涨/近乎涨停
          回归标签: 涨跌幅按 [-20%, +20%] 截断归一化到 [-1, 1]

异常数据处理:
    - 涨跌幅缺失/非有限 (NaN/inf) 的预测日样本剔除
    - 窗口跨越长期停牌 (日历跨度超过 max_span_days) 的样本剔除
    - 窗口内价格/交易量/换手率无波动 (max=min) 时对应特征置 0
    - 特征经 min-max 归一化天然有界, IPO 等极端行情不会溢出
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLS_NAMES = ["近乎跌停", "大跌", "跌", "波动", "涨", "大涨", "近乎涨停"]
CLS_BINS = [-0.15, -0.10, -0.03, 0.03, 0.10, 0.15]  # 涨跌幅分箱边界, 6 界 7 类

N_CONT = 7          # 连续特征维度 [time_norm, open, high, low, close, volume, turn]
RET_CLIP = 0.20     # 标签涨跌幅截断, 对应归一化到 [-1, 1]


def _min_max_norm(v):
    """按窗口内最大/最小值归一化到 [-1, 1], 窗口无波动 (max=min) 时返回 0"""
    v_min, v_max = v.min(), v.max()
    rng = v_max - v_min
    if rng < 1e-12:
        return np.zeros_like(v)
    return 2.0 * (v - v_min) / rng - 1.0


def window_features(x, t, seq_len):
    """预测日 t 的输入窗口特征 (seq_len, N_CONT), 需要 t >= seq_len

    x 为 load_stocks 返回的原始特征数组, 窗口为截至信号日 (t-1) 的连续
    seq_len 个交易日; 归一化口径见模块 docstring
    """
    win = x[t - seq_len: t]                     # (T, 7), 最后一行为信号日
    feats = np.column_stack([
        win[:, 0],                              # time_norm 已在 [-1, 1]
        _min_max_norm(win[:, 1:5]),             # open/high/low/close 共用窗口价格区间
        _min_max_norm(win[:, 5]),               # volume
        _min_max_norm(win[:, 6]),               # turn
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


def load_stocks(data_dir):
    """读取预处理输出, 返回 (meta, {ticker: (x, sector, ret, dates, ord_days)})"""
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    stocks = {}
    for ticker in sorted(meta["tickers"]):
        path = os.path.join(data_dir, ticker.replace(".", "_") + ".npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            x, sector, ret, dates = z["x"], z["sector"], z["ret"], z["dates"]
        ord_days = (pd.to_datetime(dates.astype(str), format="%Y%m%d")
                    .to_numpy().astype("datetime64[D]").astype(np.int64))
        stocks[ticker] = (x, sector, ret, dates, ord_days)
    return meta, stocks


class StockDataset(Dataset):
    """(输入窗口, 预测日涨跌幅标签) 数据集, 同一批 stocks 可在 train/val 间共享"""

    def __init__(self, stocks, meta, seq_len=30, date_min=None, date_max=None,
                 max_span_days=None):
        """
        stocks:       load_stocks 返回的字典
        seq_len:      输入窗口长度 T (截至信号日, 含信号日, 默认 30 见 dl/README.md)
        date_min/date_max: 预测日日期范围 (yyyymmdd int), 用于按时间切分训练/验证集
        max_span_days: 窗口允许的最大日历跨度, 默认 seq_len*2+15 (剔除跨长期停牌的样本)
        """
        self.stocks = stocks
        self.tickers = list(stocks.keys())
        self.seq_len = seq_len
        self.n_sectors = len(meta["sector2id"])
        self.max_span_days = max_span_days if max_span_days is not None else seq_len * 2 + 15

        index, labels = [], []
        for si, ticker in enumerate(self.tickers):
            _, _, ret, dates, ord_days = stocks[ticker]
            n = len(dates)
            if n < seq_len + 1:
                continue
            # t >= seq_len: 窗口 [t-seq_len, t-1] 为截至信号日 (t-1) 的连续 seq_len 日
            t = np.arange(seq_len, n, dtype=np.int64)
            ok = np.isfinite(ret[t])
            ok &= (ord_days[t] - ord_days[t - seq_len]) <= self.max_span_days
            if date_min is not None:
                ok &= dates[t] >= date_min
            if date_max is not None:
                ok &= dates[t] <= date_max
            t = t[ok]
            if len(t) == 0:
                continue
            index.append(np.column_stack([np.full(len(t), si, dtype=np.int64), t]))
            labels.append(np.digitize(ret[t], CLS_BINS).astype(np.int64))

        self.index = (np.concatenate(index) if index
                      else np.zeros((0, 2), dtype=np.int64))
        self.cls_labels = (np.concatenate(labels) if labels
                           else np.zeros(0, dtype=np.int64))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, t = self.index[idx]
        x, sector, ret, _, _ = self.stocks[self.tickers[si]]
        feats = window_features(x, t, self.seq_len)

        r = float(np.clip(ret[t], -RET_CLIP, RET_CLIP))
        return {
            "x": torch.from_numpy(feats),                                   # (T, C)
            "sector": torch.from_numpy(                                     # (T,)
                sector[t - self.seq_len: t].astype(np.int64)),
            "cls": torch.tensor(int(np.digitize(r, CLS_BINS)),
                                dtype=torch.long),
            "reg": torch.tensor(r / RET_CLIP, dtype=torch.float32),         # [-1, 1]
            "ret": torch.tensor(float(ret[t]), dtype=torch.float32),        # 原始涨跌幅, 评估用
        }


def build_datasets(data_dir, seq_len=30, val_start="2026-01-01", max_span_days=None):
    """按预测日日期切分训练/验证集, 返回 (train_ds, val_ds), 股票数据只加载一次"""
    meta, stocks = load_stocks(data_dir)
    val_ymd = int(str(val_start).replace("-", ""))
    train = StockDataset(stocks, meta, seq_len, date_max=val_ymd - 1,
                         max_span_days=max_span_days)
    val = StockDataset(stocks, meta, seq_len, date_min=val_ymd,
                       max_span_days=max_span_days)
    return train, val
