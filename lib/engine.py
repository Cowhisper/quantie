"""回测评估框架：运行策略并生成 HTML 可视化报告。"""

import html
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from yacrs import _C, cli, configurable, register

import lib  # noqa: F401  import lib 即触发所有策略/条件的 yacrs 注册
from lib.strategies.base import BaseStrategy
from lib.trade import Trade

DEFAULT_DB = "a_stock_daily.db"

OVERVIEW_METRICS = [
    "回测股票数", "总交易次数", "盈利次数", "亏损次数",
    "胜率", "总收益率", "最大单笔盈利", "最大单笔亏损", "平均收益率",
]

TRADE_COLUMNS = [
    "股票代码", "名称", "买入日期", "卖出日期", "买入价格",
    "卖出价格", "收益率", "卖出原因", "买入原因",
]

# 配置文件中可调的策略注册表：短名 -> 类名
# （类用 yacrs @register 注册在类上，经 lib/__init__.py 导入完成注册；
#  实例通过 configurable()(类名) 创建，构造参数自动绑定同名配置段）
STRATEGY_REGISTRY: Dict[str, str] = {
    "ma": "MovingAverageCrossStrategy",
    "ross": "RossCameronGapAndGoStrategy",
}

CONDITION_NAMES = (
    "DateRangeCondition",
    "GapUpCondition",
    "PriceRangeCondition",
    "RelativeVolumeCondition",
)


def _instantiate(cls_name: str, **kwargs):
    """按注册名实例化（yacrs：configurable()(类名)(**kwargs)）。"""
    return configurable()(cls_name)(**kwargs)


def build_strategies(strategy_names: List[str],
                     condition_map: Optional[Dict[str, List[str]]] = None
                     ) -> List[BaseStrategy]:
    """按配置的策略名/条件名实例化策略；条件与策略的构造参数自动绑定
    各自的同名配置段（如 PriceRangeCondition.min_price）。"""
    condition_map = condition_map or {}
    strategies: List[BaseStrategy] = []
    for name in strategy_names:
        if name not in STRATEGY_REGISTRY:
            raise SystemExit(f"未知策略: {name}（可选: {', '.join(STRATEGY_REGISTRY)}）")
        cls_name = STRATEGY_REGISTRY[name]
        kwargs: Dict[str, object] = {}
        # yacrs 登记的构造参数名；支持 conditions 的策略才装配条件
        if "conditions" in _C._configurables[cls_name]["kwargs"]:
            conds = []
            # 注意 yacrs Node 重写了 get()（点路径取值且缺失时报错），这里用 dict 语义
            cond_names = condition_map[name] if name in condition_map else []
            for cond_name in cond_names:
                if cond_name not in CONDITION_NAMES:
                    raise SystemExit(
                        f"未知条件: {cond_name}（可选: {', '.join(CONDITION_NAMES)}）")
                conds.append(_instantiate(cond_name))
            kwargs["conditions"] = conds
        strategies.append(_instantiate(cls_name, **kwargs))
    return strategies


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------

def load_data(db_path: str, start_time: str, end_time: str,
              warmup_days: int = 120,
              limit: int = 0) -> pd.DataFrame:
    """从 SQLite 加载日线数据，附带 start_time 之前的历史数据用于指标预热。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    load_start = (pd.Timestamp(start_time) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM stock_daily WHERE date >= ? AND date <= ? ORDER BY ticker, date",
        conn, params=(load_start, end_time),
    )
    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    if limit and limit > 0:
        tickers = sorted(df["ticker"].unique())[:limit]
        df = df[df["ticker"].isin(tickers)]
    return df


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------

def compute_overview(trades: List[Trade], stock_count: int) -> Dict[str, float]:
    closed = [t for t in trades if t.return_pct is not None]
    returns = [t.return_pct for t in closed]
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)

    return {
        "回测股票数": stock_count,
        "总交易次数": len(closed),
        "盈利次数": wins,
        "亏损次数": losses,
        "胜率": wins / len(closed) if closed else 0.0,
        # 每笔固定买入数量（等额定仓），累计收益率 = 各笔收益率之和；
        # 用复利连乘会随交易笔数指数放大，与实际资金占用不符
        "总收益率": float(np.sum(returns)) if returns else 0.0,
        "最大单笔盈利": max(returns) if returns else 0.0,
        "最大单笔亏损": min(returns) if returns else 0.0,
        "平均收益率": float(np.mean(returns)) if returns else 0.0,
    }


PCT_METRICS = {"胜率", "总收益率", "最大单笔盈利", "最大单笔亏损", "平均收益率"}


def format_metric(key: str, value) -> str:
    if key in PCT_METRICS:
        return f"{value * 100:.2f}%"
    return str(value)


# ----------------------------------------------------------------------
# K线图 (内联 SVG，无需外部依赖)
# ----------------------------------------------------------------------

def _candlestick_svg(window: pd.DataFrame, buy_time: str, sell_time: str,
                     width: int = 880, height: int = 300) -> str:
    """根据窗口内的日线数据绘制 K 线 SVG，标注买卖点。"""
    if window.empty:
        return "<p>无图表数据</p>"

    pad_left, pad_right, pad_top, pad_bottom = 50, 20, 15, 25
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    high_max = float(window["high"].max())
    low_min = float(window["low"].min())
    span = (high_max - low_min) or 1.0

    n = len(window)
    slot = plot_w / max(n, 1)
    body_w = max(min(slot * 0.6, 14.0), 2.0)

    def y(price: float) -> float:
        return pad_top + (high_max - price) / span * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']

    # 网格线 + 价格刻度
    for j in range(5):
        price = low_min + span * j / 4
        yy = y(price)
        parts.append(f'<line x1="{pad_left}" y1="{yy:.1f}" x2="{width - pad_right}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_left - 5}" y="{yy + 3:.1f}" font-size="9" text-anchor="end" fill="#999">{price:.2f}</text>')

    for i, row in enumerate(window.itertuples()):
        cx = pad_left + slot * i + slot / 2
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        color = "#e0342f" if c >= o else "#21a366"  # 红涨绿跌
        parts.append(f'<line x1="{cx:.1f}" y1="{y(h):.1f}" x2="{cx:.1f}" y2="{y(l):.1f}" stroke="{color}" stroke-width="1"/>')
        top, bh = (y(o), y(c) - y(o)) if c >= o else (y(c), y(o) - y(c))
        parts.append(f'<rect x="{cx - body_w / 2:.1f}" y="{top:.1f}" width="{body_w:.1f}" height="{max(bh, 1.0):.1f}" fill="{color}"/>')
        if i % max(n // 8, 1) == 0:
            parts.append(f'<text x="{cx:.1f}" y="{height - 8}" font-size="8" text-anchor="middle" fill="#999">{row.date.strftime("%m-%d")}</text>')

    # 标注买卖点
    for date_str, label, color in ((buy_time, "B", "#1f77b4"), (sell_time, "S", "#ff7f0e")):
        if not date_str:
            continue
        ts = pd.Timestamp(date_str)
        idx = int((window["date"] == ts).idxmax()) if (window["date"] == ts).any() else None
        if idx is None:
            continue
        pos = window.index.get_loc(idx)
        row = window.iloc[pos]
        cx = pad_left + slot * pos + slot / 2
        cy = y(float(row.low))
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy - 8:.1f}" font-size="10" font-weight="bold" text-anchor="middle" fill="{color}">{label}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# HTML 报告
# ----------------------------------------------------------------------

_PAGE_CSS = """
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #f5f6fa; color: #333; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 32px; }
.card { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.desc { color: #888; font-size: 13px; margin: 4px 0 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
th, td { border: 1px solid #eee; padding: 6px 10px; text-align: center; white-space: nowrap; }
th { background: #fafbfc; }
.table-wrap { overflow-x: auto; }
tr.trade-row { cursor: pointer; }
tr.trade-row:hover { background: #f0f7ff; }
tr.chart-row td { padding: 8px; background: #fcfcfd; text-align: left; }
tr.chart-row.hidden { display: none; }
.pos { color: #e0342f; } .neg { color: #21a366; }
.toggle-hint { color: #bbb; font-size: 11px; font-weight: normal; margin-left: 8px; }
"""

_PAGE_JS = """
document.addEventListener('click', function (e) {
  var row = e.target.closest('tr.trade-row');
  if (!row) return;
  var chart = row.nextElementSibling;
  if (chart && chart.classList.contains('chart-row')) {
    chart.classList.toggle('hidden');
  }
});
"""


def _render_overview(overview: Dict[str, float]) -> str:
    header = "".join(f"<th>{k}</th>" for k in OVERVIEW_METRICS)
    values = "".join(
        f"<td>{format_metric(k, overview[k])}</td>"
        for k in OVERVIEW_METRICS
    )
    table = f"<table><tr>{header}</tr><tr>{values}</tr></table>"
    return f"<div class='table-wrap'>{table}</div>"


def _render_trades(trades: List[Trade], stock_data: Dict[str, pd.DataFrame],
                   max_charts: int = 0) -> str:
    """渲染交易明细表；max_charts > 0 时只为前 max_charts 笔交易嵌入 K 线图。"""
    rows = []
    for idx, t in enumerate(trades):
        r = t.return_pct
        ret_str = f"{r * 100:+.2f}%" if r is not None else "-"
        ret_cls = "pos" if (r or 0) > 0 else "neg"
        esc = html.escape
        rows.append(
            f"<tr class='trade-row'>"
            f"<td>{esc(t.ticker)}</td><td>{esc(t.name)}</td>"
            f"<td>{esc(t.buy_time or '-')}</td><td>{esc(t.sell_time or '-')}</td>"
            f"<td>{t.buy_price:.2f}</td>"
            f"<td>{f'{t.sell_price:.2f}' if t.sell_price is not None else '-'}</td>"
            f"<td class='{ret_cls}'>{ret_str}</td>"
            f"<td>{esc(t.sell_reason or '-')}</td><td>{esc(t.buy_reason or '-')}</td>"
            f"</tr>"
        )

        # 折叠的 K 线图：买入前 ~15 天 ~ 卖出后 ~15 天
        chart = None
        if max_charts <= 0 or idx < max_charts:
            df_stock = stock_data.get(t.ticker)
            if df_stock is not None and t.buy_time:
                mask = df_stock["date"] >= pd.Timestamp(t.buy_time) - pd.Timedelta(days=15)
                if t.sell_time:
                    mask &= df_stock["date"] <= pd.Timestamp(t.sell_time) + pd.Timedelta(days=15)
                chart = _candlestick_svg(df_stock[mask].reset_index(drop=True),
                                         t.buy_time, t.sell_time)
        if chart is None:
            chart = "<span class='desc'>（图表已省略，交易笔数过多）</span>"
        rows.append(
            f"<tr class='chart-row hidden'><td colspan='{len(TRADE_COLUMNS)}'>{chart}</td></tr>"
        )

    header = "".join(f"<th>{c}</th>" for c in TRADE_COLUMNS)
    table = f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    return f"<div class='table-wrap'>{table}</div>"


def generate_report(results: Dict[str, dict], output_path: str,
                    max_charts: int = 0) -> str:
    """
    生成 HTML 报告。results 格式:
        {策略名: {"overview": {...}, "trades": [Trade...], "stock_data": {ticker: df}}}
    max_charts > 0 时，只为每个策略的前 max_charts 笔交易嵌入 K 线图，
    避免全市场回测时报告体积爆炸。
    """
    sections = []
    for name, data in results.items():
        overview_html = _render_overview(data["overview"])
        trades_html = _render_trades(data["trades"], data["stock_data"],
                                     max_charts=max_charts)
        hint = "<span class='toggle-hint'>(点击行展开 K 线图)</span>"
        sections.append(
            f"<div class='card'><h2>{html.escape(name)}{hint}</h2>"
            f"<p class='desc'>{html.escape(data.get('description', ''))}</p>"
            f"{overview_html}<h2>交易明细</h2>{trades_html}</div>"
        )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>策略回测报告</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<div class="container">
<h1>策略回测报告</h1>
{''.join(sections)}
</div>
<script>{_PAGE_JS}</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    return output_path


# ----------------------------------------------------------------------
# 回测引擎
# ----------------------------------------------------------------------

class Engine:
    """对一组策略在相同数据区间上执行回测。"""

    def __init__(self, strategies: Optional[List[BaseStrategy]] = None) -> None:
        default_name = STRATEGY_REGISTRY["ma"]
        self.strategies = strategies if strategies is not None else [_instantiate(default_name)]

    def evaluate(self, df: pd.DataFrame, start_time: str, end_time: str) -> Dict[str, dict]:
        start_ts, end_ts = pd.Timestamp(start_time), pd.Timestamp(end_time)
        stock_data: Dict[str, pd.DataFrame] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            stock_data[ticker] = group.reset_index(drop=True)

        # 回测股票数：区间内（含预热数据之外）有数据的股票数
        stock_count = sum(
            1 for g in stock_data.values() if (g["date"] >= start_ts).any()
        )

        results: Dict[str, dict] = {}
        for strategy in self.strategies:
            print(f"[{strategy.name}] 开始回测 ({stock_count} 只股票)...")
            trades: List[Trade] = []
            for ticker, g in stock_data.items():
                if not (g["date"] >= start_ts).any() or not (g["date"] <= end_ts).any():
                    continue
                trades.extend(strategy.run(g, start_ts, end_ts))
            trades.sort(key=lambda t: (t.buy_time or "", t.ticker))

            overview = compute_overview(trades, stock_count)
            print(f"[{strategy.name}] 交易 {overview['总交易次数']} 笔 | "
                  f"胜率 {format_metric('胜率', overview['胜率'])} | "
                  f"总收益率 {format_metric('总收益率', overview['总收益率'])}")

            results[strategy.name] = {
                "overview": overview,
                "trades": trades,
                "stock_data": stock_data,
                "description": strategy.description,
            }
        return results


# ----------------------------------------------------------------------
# CLI（yacrs：-c 加载 yaml/json/toml 配置，key=value 命令行覆盖）
# ----------------------------------------------------------------------

@cli("engine")
@register("engine")
def main(db=DEFAULT_DB, start=None, end=None, output="strategy_report.html",
         warmup=120, limit=0, max_charts=1000,
         strategies=None, conditions=None) -> None:
    """
    回测入口。所有参数在配置文件中设置（见 strategy_test.yaml），
    也可以用 key=value 在命令行覆盖：
        python -m lib.engine -c strategy_test.yaml engine.start=2025-01-01
    """
    if not start or not end:
        raise SystemExit("请在配置文件中设置 engine.start / engine.end（回测起止日期 YYYY-MM-DD）")

    # 命令行 key=value 覆盖进来的值是字符串，数值参数统一转换
    warmup, limit, max_charts = int(warmup), int(limit), int(max_charts)
    if isinstance(strategies, str):  # 命令行覆盖单值时按一个策略处理
        strategies = [strategies]
    strategy_names = list(strategies) if strategies else ["ma"]

    df = load_data(db, start, end, warmup_days=warmup, limit=limit)
    if df.empty:
        print("数据库中没有数据，请先运行 download_data.py")
        return

    engine = Engine(strategies=build_strategies(strategy_names, conditions))
    results = engine.evaluate(df, start, end)
    path = generate_report(results, output, max_charts=max_charts)
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
