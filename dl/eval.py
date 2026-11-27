#!/usr/bin/env python3
"""
评估脚本: 在给定数据切分上评估模型, 输出所有标签类别的 precision / recall / f1

用法:
    python dl/eval.py --ckpt dl/checkpoints/best.pt
    python dl/eval.py --ckpt dl/checkpoints/best.pt --split train --limit 100000
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import CLS_NAMES, N_CONT, build_datasets
from log_util import fmt_eta, setup_logger
from net import load_model


@torch.no_grad()
def collect_predictions(model, loader, device, logger=None, log_interval=0):
    y_true, y_pred, r_true, r_pred = [], [], [], []
    t0 = time.time()
    total = len(loader)
    for step, batch in enumerate(loader):
        logits, pred = model(batch["x"].to(device), batch["sector"].to(device))
        y_true.append(batch["cls"].numpy())
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
        r_true.append(batch["reg"].numpy())
        r_pred.append(pred.cpu().numpy())
        done = step + 1
        if logger and log_interval and done % log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            logger.info(f"  eval [{done}/{total}] | {done / elapsed:.0f} it/s "
                        f"| ETA {fmt_eta(eta)}")
    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(r_true), np.concatenate(r_pred))


def main():
    p = argparse.ArgumentParser(description="评估模型: 各类别 precision / recall")
    p.add_argument("--ckpt", default="dl/checkpoints/best.pt")
    p.add_argument("--data_dir", default="dl/data")
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--val_start", default="2026-01-01", help="与训练一致的时间切分点")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="只随机评估 N 条 (调试)")
    p.add_argument("--log_interval", type=int, default=100,
                   help="每 N 个 iteration 打印一次进度和 ETA, 0 关闭")
    p.add_argument("--log_file", default="", help="默认 <ckpt目录>/eval.log")
    args = p.parse_args()

    log_file = args.log_file or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "eval.log")
    logger = setup_logger("eval", log_file)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device, n_cont=N_CONT,
                             num_classes=len(CLS_NAMES))
    seq_len = ckpt["config"]["seq_len"]

    train_ds, val_ds = build_datasets(args.data_dir, seq_len, args.val_start)
    ds = val_ds if args.split == "val" else train_ds
    if args.limit:
        idx = np.random.default_rng(0).permutation(len(ds))[:args.limit]
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers)

    logger.info(f"模型 {args.ckpt} (epoch {ckpt.get('epoch', '?')}) | 设备 {device} | "
                f"{args.split} 集 {len(ds)} 样本 (val_start={args.val_start})")

    y_true, y_pred, r_true, r_pred = collect_predictions(
        model, loader, device, logger, args.log_interval)

    n_cls = len(CLS_NAMES)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    logger.info(f"总体 acc {np.mean(y_true == y_pred):.2%} | "
                f"回归 mae {np.abs(r_pred - r_true).mean() * 0.2:.2%}")
    logger.info(f"{'标签':<8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    precs, recs, f1s = [], [], []
    for c, name in enumerate(CLS_NAMES):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        logger.info(f"{name:<8}{prec:>10.2%}{rec:>10.2%}{f1:>10.2%}"
                    f"{cm[c, :].sum():>10}")
    logger.info(f"{'macro':<8}{np.mean(precs):>10.2%}{np.mean(recs):>10.2%}"
                f"{np.mean(f1s):>10.2%}")
    logger.info(f"日志已写入 {log_file}")


if __name__ == "__main__":
    main()
