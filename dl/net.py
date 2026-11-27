"""
Transformer 预测网络

输入:  x      (B, T, C_cont)  归一化连续特征 (time_norm/open/high/low/close/volume/turn)
       sector (B, T)          行业 id, 经 Embedding 参与学习后与连续特征拼接
输出:  cls_logits (B, 7)      分类 logits (近乎跌停 ~ 近乎涨停)
       reg        (B,)        回归 logits (收益率按 [-20%, +20%] 归一化到 [-1, 1])
"""
import torch
import torch.nn as nn


class StockTransformer(nn.Module):
    def __init__(self, n_sectors, n_cont=7, num_classes=7, d_model=64,
                 nhead=4, num_layers=2, dim_ff=128, emb_dim=16,
                 dropout=0.1, max_len=256):
        super().__init__()
        self.sector_emb = nn.Embedding(n_sectors, emb_dim)
        self.in_proj = nn.Linear(n_cont + emb_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.reg_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sector):
        T = x.shape[1]
        h = torch.cat([x, self.sector_emb(sector)], dim=-1)
        h = self.dropout(self.in_proj(h) + self.pos_emb.weight[:T])
        h = self.encoder(h)
        h = self.out_norm(h[:, -1])  # 最后一个时间步即预测日
        return self.cls_head(h), self.reg_head(h).squeeze(-1)


def load_model(ckpt_path, device="cpu", n_cont=7, num_classes=7):
    """从 train.py 保存的 checkpoint 重建模型, 返回 (model, ckpt)"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = StockTransformer(
        n_sectors=ckpt["n_sectors"], n_cont=n_cont, num_classes=num_classes,
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"], emb_dim=cfg["emb_dim"], dropout=cfg["dropout"],
        max_len=cfg["seq_len"] + 8)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
