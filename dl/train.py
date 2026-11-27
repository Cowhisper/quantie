#!/usr/bin/env python3
"""
训练脚本: 分类 (softmax + cross entropy) + 回归 (smooth-l1) 联合监督

回归损失支持错配加权 (--reg_mismatch_weight): 预测幅值的分箱与真实标签
不一致的样本, 回归 loss 乘指定倍数, 让回归头聚焦大波动/难样本, 缓解预测幅值收缩

用法:
    python dl/data_preprocess.py     # 先生成 dl/data/
    python dl/train.py               # 训练, 最优模型保存到 dl/checkpoints/best.pt
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import CLS_BINS, CLS_NAMES, N_CONT, RET_CLIP, build_datasets
from log_util import fmt_eta, setup_logger
from net import StockTransformer


def run_epoch(model, loader, device, cls_weight, lambda_reg, optimizer=None,
              logger=None, log_interval=0, desc="", mismatch_weight=1.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    n = correct = 0
    loss_sum = ce_sum = reg_sum = abs_err = 0.0
    t0 = time.time()
    total = len(loader)
    # 回归预测分箱用的边界 (归一化单位), 与分类标签口径一致
    reg_bins = torch.tensor([b / RET_CLIP for b in CLS_BINS],
                            dtype=torch.float32, device=device)
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            x = batch["x"].to(device)
            sector = batch["sector"].to(device)
            y_cls = batch["cls"].to(device)
            y_reg = batch["reg"].to(device)

            logits, pred = model(x, sector)
            ce = F.cross_entropy(logits, y_cls, weight=cls_weight)
            reg_elem = F.smooth_l1_loss(pred, y_reg, reduction="none")
            if mismatch_weight != 1.0:
                # 预测幅值的分箱与真实标签不一致的样本, 回归 loss 加权
                with torch.no_grad():
                    pred_bin = torch.bucketize(pred, reg_bins, right=True)
                    mismatch = pred_bin != y_cls
                    w = 1.0 + mismatch.float() * (mismatch_weight - 1.0)
                reg = (reg_elem * w).mean()
            else:
                reg = reg_elem.mean()
            loss = ce + lambda_reg * reg

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = y_cls.numel()
            n += bs
            loss_sum += loss.item() * bs
            ce_sum += ce.item() * bs
            reg_sum += reg.item() * bs
            correct += (logits.argmax(dim=-1) == y_cls).sum().item()
            abs_err += (pred - y_reg).abs().sum().item()

            done = step + 1
            if logger and log_interval and done % log_interval == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                logger.info(f"  {desc} [{done}/{total}] loss {loss_sum / n:.4f} "
                            f"acc {correct / n:.2%} | {done / elapsed:.0f} it/s "
                            f"| ETA {fmt_eta(eta)}")

    n = max(n, 1)
    return {"loss": loss_sum / n, "ce": ce_sum / n, "reg": reg_sum / n,
            "acc": correct / n, "mae": abs_err / n * 0.2,  # 换算回收益率
            "secs": time.time() - t0}


def fmt(metrics):
    return (f"loss {metrics['loss']:.4f} (ce {metrics['ce']:.4f} reg {metrics['reg']:.4f}) "
            f"acc {metrics['acc']:.2%} mae {metrics['mae']:.2%}")


def main():
    p = argparse.ArgumentParser(description="Transformer 次日收益预测训练")
    p.add_argument("--data_dir", default="dl/data")
    p.add_argument("--out", default="dl/checkpoints")
    p.add_argument("--seq_len", type=int, default=30,
                   help="输入窗口长度 T (截至信号日, 含信号日)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--lambda_reg", type=float, default=1.0, help="回归损失权重")
    p.add_argument("--reg_mismatch_weight", type=float, default=5.0,
                   help="预测分箱与真实标签不一致的样本, 回归 loss 乘此倍数, 1 关闭")
    p.add_argument("--val_start", default="2026-01-01", help="验证集起始日 (按目标日切分)")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_ff", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_class_weight", action="store_true", help="不使用类别均衡权重")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=100,
                   help="每 N 个 iteration 打印一次进度和 ETA, 0 关闭")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_train", type=int, default=0, help="训练集只随机取 N 条 (调试)")
    p.add_argument("--limit_val", type=int, default=0, help="验证集只随机取 N 条 (调试)")
    p.add_argument("--log_file", default="", help="训练日志文件, 默认 <out>/train.log")
    args = p.parse_args()

    log_file = args.log_file or os.path.join(args.out, "train.log")
    logger = setup_logger("train", log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    train_ds, val_ds = build_datasets(args.data_dir, args.seq_len, args.val_start)
    logger.info(f"设备 {device} | 训练集 {len(train_ds)} 样本 | 验证集 {len(val_ds)} 样本 "
                f"(val_start={args.val_start})")
    logger.info(f"参数 {vars(args)}")

    counts = np.bincount(train_ds.cls_labels, minlength=len(CLS_NAMES))
    dist = " ".join(f"{name} {c / max(counts.sum(), 1):.1%}"
                    for name, c in zip(CLS_NAMES, counts))
    logger.info(f"训练集类别分布: {dist}")

    cls_weight = None
    if not args.no_class_weight:
        w = counts.sum() / (len(CLS_NAMES) * np.maximum(counts, 1))
        cls_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.seed)
    if args.limit_train:
        train_ds = Subset(train_ds, rng.permutation(len(train_ds))[:args.limit_train])
    if args.limit_val:
        val_ds = Subset(val_ds, rng.permutation(len(val_ds))[:args.limit_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    model = StockTransformer(
        n_sectors=train_ds.dataset.n_sectors if isinstance(train_ds, Subset)
        else train_ds.n_sectors,
        n_cont=N_CONT, num_classes=len(CLS_NAMES), d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers, dim_ff=args.dim_ff,
        emb_dim=args.emb_dim, dropout=args.dropout, max_len=args.seq_len + 8)
    model.to(device)
    logger.info(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, cls_weight, args.lambda_reg,
                       optimizer, logger, args.log_interval, desc=f"train e{epoch}",
                       mismatch_weight=args.reg_mismatch_weight)
        va = run_epoch(model, val_loader, device, cls_weight, args.lambda_reg,
                       logger=logger, log_interval=args.log_interval,
                       desc=f"val e{epoch}",
                       mismatch_weight=args.reg_mismatch_weight)
        scheduler.step()
        logger.info(f"epoch {epoch}/{args.epochs} | train {fmt(tr)} | val {fmt(va)} "
                    f"| {tr['secs'] + va['secs']:.0f}s")

        state = {"model": model.state_dict(), "config": vars(args),
                 "n_sectors": model.sector_emb.num_embeddings, "epoch": epoch,
                 "val": va}
        torch.save(state, os.path.join(args.out, "last.pt"))
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(state, os.path.join(args.out, "best.pt"))
            logger.info(f"  保存最优模型 -> {args.out}/best.pt")
    logger.info(f"训练结束, 最优 val loss {best_val:.4f}, 日志已写入 {log_file}")


if __name__ == "__main__":
    main()
