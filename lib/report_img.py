"""把模拟账户报告渲染成 PNG 表格图片。

企业微信群机器人 markdown 是有限语法子集、不支持表格，多列明细
（持仓明细/本周期卖出）以 text 发送会在手机上挤成一行——这里用 PIL
绘制成对齐的表格图片，由 lib/notify.send_image 以 image 消息发送。

中文字体按 macOS 常见路径依次尝试（PingFang → 冬青黑 → 黑体），
取第一个可用的；无 PIL 或无可用字体时 render_report_image 返回 None，
调用方退回纯文本 markdown。
"""

import os
from typing import Dict, List, Optional

from lib.engine import HOLDING_COLUMNS

# macOS 中文字体候选（path, ttc index）
_FONT_CANDIDATES = [
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 0),  # 兜底（中文会变方块，仅防崩）
]

_COL_W = {  # 列最小像素宽（按表头字号 15 估）
    "股票代码": 92, "名称": 96, "买入日期": 96, "卖出日期": 96,
    "买入价格/股数": 118, "卖出价格/股数": 118, "开盘价格": 76,
    "收盘价格": 76, "当日收益率": 86, "持有收益率": 86,
    "卖出原因": 150, "买入原因": 170,
}

_PAD_X, _PAD_Y = 9, 6          # 单元格内边距
_MARGIN = 24                   # 页面边距
_LINE_H_EXTRA = 8              # 行高额外留白
_HEADER_BG = (245, 247, 250)
_BORDER = (210, 214, 222)
_TEXT = (51, 51, 51)
_MUTED = (120, 120, 120)
_POS = (224, 52, 47)           # A 股红涨
_NEG = (33, 163, 102)          # 绿跌
_TITLE = (31, 119, 180)
_BG = (255, 255, 255)


def _load_fonts():
    """按候选路径加载 (正文, 加粗/标题) 两字号的 PIL 字体；不可用返回 None。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path, index in _FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return (ImageFont.truetype(path, 15, index=index),
                    ImageFont.truetype(path, 20, index=index))
        except Exception:
            continue
    return None


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:+.2f}%"


def _holding_cells(h: Dict[str, object], reason_len: int = 14) -> List[str]:
    """持仓/卖出记录 → 交易明细 12 列文本（与 engine server 模式同列）。"""

    def reason(s: Optional[str]) -> str:
        s = s or "-"
        return s if len(s) <= reason_len else s[:reason_len] + "…"

    return [
        str(h["ticker"]), str(h["name"]),
        str(h["buy_time"] or "-"), str(h["sell_time"] or "-"),
        f"{h['buy_price']:.2f} / {h['buy_amount']:,.0f}",
        (f"{h['sell_price']:.2f} / {h['sell_amount']:,.0f}"
         if h["sell_time"] else "-"),
        f"{h['open_price']:.2f}" if h["open_price"] is not None else "-",
        f"{h['close_price']:.2f}" if h["close_price"] is not None else "-",
        _pct(h["day_return_pct"]), _pct(h["holding_return_pct"]),
        reason(h["sell_reason"] if h["sell_time"] else None),
        reason(h["buy_reason"]),
    ]


def _col_colors(columns: List[str], cells: List[str]) -> List[Optional[tuple]]:
    """当日收益率/持有收益率两列按正负着色（其余不着色；缺列时跳过）。"""
    colors: List[Optional[tuple]] = [None] * len(cells)
    for name in ("当日收益率", "持有收益率"):
        if name in columns:
            idx = columns.index(name)
            v = cells[idx]
            if v != "-":
                colors[idx] = _POS if not v.startswith("-") else _NEG
    return colors


def render_report_image(report: dict, path: str,
                        ops_rows: Optional[List[tuple]] = None,
                        ops_date: Optional[str] = None) -> Optional[str]:
    """渲染报告 PNG 到 path（返回 path；PIL/字体不可用返回 None）。

    内容：操作清单（可选）+ 持仓明细表 + 本周期卖出表 + 模拟账户概览。
    """
    fonts = _load_fonts()
    if fonts is None:
        return None
    font, font_title = fonts
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    def text_w(s: str, f=font) -> int:
        box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), s, font=f)
        return box[2] - box[0]

    sections: List[dict] = []
    if ops_rows:
        sections.append({
            "title": f"{ops_date} 股票买入卖出操作",
            "columns": ["股票", "股票代码", "操作"],
            "rows": [[str(n), str(c), str(a)] for n, c, a in ops_rows],
        })
    holdings = report.get("holdings") or []
    sections.append({
        "title": f"持仓明细（{len(holdings)} 只）",
        "columns": list(HOLDING_COLUMNS),
        "rows": [_holding_cells(h) for h in holdings] or None,
        "empty": "当前无持仓",
    })
    sold = report.get("sold") or []
    sections.append({
        "title": f"本周期卖出（{len(sold)} 只）",
        "columns": list(HOLDING_COLUMNS),
        "rows": [_holding_cells(h) for h in sold] or None,
        "empty": "本周期无卖出",
    })
    s = report.get("stats") or {}
    account_lines = [
        f"本金 {report['initial_capital']:,.2f}    "
        f"总资产 {report['total_assets']:,.2f}    "
        f"现金 {report['cash']:,.2f}    "
        f"持仓市值 {report['position_value']:,.2f}",
        f"总交易次数 {s.get('总交易次数', 0)}    盈利次数 {s.get('盈利次数', 0)}    "
        f"亏损次数 {s.get('亏损次数', 0)}    胜率 {s.get('胜率', 0) * 100:.2f}%",
        f"总收益额 {s.get('总收益额', 0):+,.2f}    "
        f"总收益率 {s.get('总收益率', 0) * 100:+.2f}%",
    ]

    # ---- 布局：先测宽再逐段绘制 ----
    tables = []
    content_w = 0
    for sec in sections:
        cols = sec["columns"]
        widths = [max(_COL_W.get(c, 90), text_w(c) + 2 * _PAD_X)
                  for c in cols]
        for row in sec["rows"] or []:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], text_w(cell) + 2 * _PAD_X)
        table_w = sum(widths) + len(widths) + 1
        content_w = max(content_w, table_w, *(text_w(l) for l in account_lines))
        tables.append((sec, cols, widths, table_w))

    width = max(content_w + 2 * _MARGIN, 560)
    x0 = _MARGIN

    # 逐段计算高度
    line_h = font.size + _LINE_H_EXTRA
    title_h = font_title.size + 14
    blocks = []
    for sec, cols, widths, table_w in tables:
        h = title_h
        h += line_h + 2  # 表头
        if sec["rows"]:
            h += (line_h + 1) * len(sec["rows"])
        else:
            h += line_h + 1
        blocks.append(h + 14)  # 段间距

    height = _MARGIN + sum(blocks) + (line_h + 10) * len(account_lines) \
        + title_h + _MARGIN + 10

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    y = _MARGIN

    def draw_table(sec, cols, widths, table_w):
        nonlocal y
        draw.text((x0, y), sec["title"], font=font_title, fill=_TITLE)
        y += title_h
        # 表头
        x = x0
        for c, w in zip(cols, widths):
            draw.rectangle([x, y, x + w, y + line_h], fill=_HEADER_BG,
                           outline=_BORDER)
            draw.text((x + _PAD_X, y + _PAD_Y - 1), c, font=font, fill=_MUTED)
            x += w + 1
        y += line_h + 1
        # 数据行
        if not sec["rows"]:
            draw.rectangle([x0, y, x0 + table_w, y + line_h],
                           fill=_BG, outline=_BORDER)
            draw.text((x0 + _PAD_X, y + _PAD_Y - 1), sec["empty"],
                      font=font, fill=_MUTED)
            y += line_h + 1
        else:
            for row in sec["rows"]:
                colors = _col_colors(cols, row)
                x = x0
                for cell, w, color in zip(row, widths, colors):
                    draw.rectangle([x, y, x + w, y + line_h],
                                   fill=_BG, outline=_BORDER)
                    draw.text((x + _PAD_X, y + _PAD_Y - 1), cell, font=font,
                              fill=color or _TEXT)
                    x += w + 1
                y += line_h + 1
        y += 14

    for sec, cols, widths, table_w in tables:
        draw_table(sec, cols, widths, table_w)

    draw.text((x0, y), "模拟账户", font=font_title, fill=_TITLE)
    y += title_h
    for line in account_lines:
        color = _TEXT
        if line.startswith("总收益额"):
            color = _POS if s.get("总收益额", 0) > 0 else (
                _NEG if s.get("总收益额", 0) < 0 else _TEXT)
        draw.text((x0, y), line, font=font, fill=color)
        y += line_h + 10

    # 实际内容可能比预估矮，裁剪掉底部空白
    img = img.crop((0, 0, width, min(height, y + _MARGIN)))
    img.save(path, "PNG")
    return path
