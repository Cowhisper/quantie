#!/usr/bin/env python3
"""
A股日频数据可视化页面
- 多股票选择
- 自定义日期区间
- 数据表格 + K线/成交量图
"""

from flask import Flask, render_template_string, jsonify, request
import sqlite3
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
DB_PATH = "a_stock_daily.db"

# ==================== 数据查询 ====================

def get_db_info():
    """返回数据库中所有股票的元信息以及全局日期范围。"""
    if not _db_exists():
        return {"stocks": [], "min_date": "", "max_date": ""}

    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql(
        "SELECT ticker, name, sector, first_date, last_date, record_count "
        "FROM stock_meta ORDER BY record_count DESC, ticker ASC",
        conn,
    )
    range_df = pd.read_sql(
        "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM stock_daily",
        conn,
    )
    conn.close()

    stocks = meta.where(pd.notnull(meta), None).to_dict("records")
    return {
        "stocks": stocks,
        "min_date": range_df["min_date"].iloc[0] or "",
        "max_date": range_df["max_date"].iloc[0] or "",
    }


def _db_exists():
    import os
    return os.path.exists(DB_PATH)


def _calc_indicators(group):
    """为单只股票计算均线、涨跌幅、振幅。"""
    group = group.sort_values("date").reset_index(drop=True)
    group["ma5"] = group["close"].rolling(window=5).mean().round(2)
    group["ma10"] = group["close"].rolling(window=10).mean().round(2)
    group["ma20"] = group["close"].rolling(window=20).mean().round(2)
    group["pct_change"] = (group["close"].pct_change() * 100).round(2)
    group["amplitude"] = (
        (group["high"] - group["low"]) / group["close"].shift(1) * 100
    ).round(2)
    return group.where(pd.notnull(group), None)


def get_stock_data(tickers, start_date=None, end_date=None):
    """
    按股票列表与日期区间查询日线数据。
    返回 {ticker: {"name": ..., "sector": ..., "rows": [...]}, ...}
    """
    if not tickers:
        return {}

    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(tickers))
    query = f"""
        SELECT ticker, name, sector, date, open, high, low, close,
               volume, amount, turn, pctChg
        FROM stock_daily
        WHERE ticker IN ({placeholders})
    """
    params = list(tickers)
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY ticker, date ASC"

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
        return {}

    result = {}
    for ticker, group in df.groupby("ticker", sort=False):
        group = _calc_indicators(group)
        first = group.iloc[0]
        result[ticker] = {
            "name": (first.get("name") or ticker),
            "sector": (first.get("sector") or "-"),
            "rows": _records(group),
        }
    return result


def _records(df):
    """将 DataFrame 转为纯 Python 字典列表，NaN/None 统一为 JSON null。"""
    return df.replace({np.nan: None}).to_dict("records")


# ==================== 前端页面 ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>A股数据可视化</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117; color: #c9d1d9; padding-bottom: 40px;
        }
        .header {
            padding: 15px 30px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
        }
        .header h1 { font-size: 20px; color: #58a6ff; margin-bottom: 6px; }
        .header p { font-size: 13px; color: #8b949e; }

        .controls {
            display: flex; flex-wrap: wrap; align-items: flex-end;
            gap: 18px; padding: 20px 30px;
            background: #161b22; border-bottom: 1px solid #30363d;
        }
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .control-group label { font-size: 13px; color: #8b949e; }
        input[type="date"], select, button {
            background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
            border-radius: 6px; font-size: 14px;
        }
        input[type="date"] { padding: 7px 10px; }
        select[multiple] {
            min-width: 260px; min-height: 160px; padding: 8px;
        }
        select option { padding: 4px 6px; }
        select option:checked { background: #1f6feb; color: #fff; }
        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; }
        button {
            padding: 8px 16px; cursor: pointer; transition: .15s;
        }
        button:hover { border-color: #58a6ff; }
        button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
        button.primary:hover { background: #388bfd; }

        .summary {
            padding: 12px 30px; font-size: 13px; color: #8b949e;
            border-bottom: 1px solid #30363d; background: #0d1117;
        }
        .summary span { margin-right: 20px; }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
            gap: 20px; padding: 20px 30px;
        }
        .chart-card {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 10px; padding: 14px;
        }
        .chart-title {
            font-size: 14px; color: #c9d1d9; margin-bottom: 10px;
            display: flex; justify-content: space-between;
        }
        .chart-title .sector { color: #8b949e; font-size: 12px; }
        .kline-chart { width: 100%; height: 320px; }
        .vol-chart { width: 100%; height: 110px; margin-top: 6px; }

        .table-wrap {
            padding: 0 30px; margin-top: 10px;
        }
        .table-wrap h3 {
            font-size: 15px; color: #c9d1d9; margin-bottom: 10px;
        }
        table {
            width: 100%; border-collapse: collapse; font-size: 13px;
            background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            overflow: hidden;
        }
        thead { background: #21262d; }
        th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid #30363d; }
        th { color: #8b949e; font-weight: 500; }
        td { color: #c9d1d9; }
        tr:hover { background: #1c2128; }
        .text-left { text-align: left; }
        .up { color: #ff4d4f; }
        .down { color: #00b578; }
        .empty {
            padding: 60px 30px; text-align: center; color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 A股日频数据可视化</h1>
        <p>选择股票与日期区间，查看数据表格与K线走势</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>开始日期</label>
            <input type="date" id="start-date">
        </div>
        <div class="control-group">
            <label>结束日期</label>
            <input type="date" id="end-date">
        </div>
        <div class="control-group">
            <label>选择股票（可多选 / 按住 Ctrl/Cmd）</label>
            <select id="stock-select" multiple></select>
        </div>
        <div class="btn-group">
            <button class="primary" onclick="loadData()">加载数据</button>
            <button onclick="selectAll()">全选</button>
            <button onclick="clearSelection()">清空</button>
        </div>
    </div>

    <div class="summary" id="summary-bar" style="display:none;">
        <span id="sum-stocks">股票: -</span>
        <span id="sum-records">记录: -</span>
        <span id="sum-range">日期: -</span>
    </div>

    <div id="empty-tip" class="empty">请选择至少一只股票并点击“加载数据”</div>

    <div class="charts-grid" id="charts-grid"></div>

    <div class="table-wrap" id="table-section" style="display:none;">
        <h3>数据明细</h3>
        <div style="overflow-x:auto;">
            <table>
                <thead>
                    <tr>
                        <th class="text-left">日期</th>
                        <th class="text-left">代码</th>
                        <th class="text-left">名称</th>
                        <th>开盘</th>
                        <th>收盘</th>
                        <th>最高</th>
                        <th>最低</th>
                        <th>涨跌幅</th>
                        <th>振幅</th>
                        <th>成交量</th>
                        <th>成交额</th>
                        <th>换手率</th>
                    </tr>
                </thead>
                <tbody id="data-table-body"></tbody>
            </table>
        </div>
    </div>

    <script>
        let chartInstances = [];
        let stockList = [];

        async function init() {
            const res = await fetch('/api/stocks');
            const info = await res.json();
            stockList = info.stocks || [];

            const select = document.getElementById('stock-select');
            select.innerHTML = stockList.map(s =>
                `<option value="${s.ticker}">${s.name} (${s.ticker}) [${s.sector}]</option>`
            ).join('');

            if (info.min_date) document.getElementById('start-date').value = info.min_date;
            if (info.max_date) document.getElementById('end-date').value = info.max_date;
        }

        function selectAll() {
            const select = document.getElementById('stock-select');
            Array.from(select.options).forEach(o => o.selected = true);
        }

        function clearSelection() {
            const select = document.getElementById('stock-select');
            Array.from(select.options).forEach(o => o.selected = false);
        }

        function getSelectedTickers() {
            const select = document.getElementById('stock-select');
            return Array.from(select.selectedOptions).map(o => o.value);
        }

        async function loadData() {
            const tickers = getSelectedTickers();
            const start = document.getElementById('start-date').value;
            const end = document.getElementById('end-date').value;

            if (tickers.length === 0) {
                alert('请至少选择一只股票');
                return;
            }
            if (start && end && start > end) {
                alert('开始日期不能晚于结束日期');
                return;
            }

            const params = new URLSearchParams();
            params.set('tickers', tickers.join(','));
            if (start) params.set('start', start);
            if (end) params.set('end', end);

            const res = await fetch(`/api/data?${params.toString()}`);
            const data = await res.json();
            render(data, tickers, start, end);
        }

        function clearCharts() {
            chartInstances.forEach(c => c.dispose());
            chartInstances = [];
            document.getElementById('charts-grid').innerHTML = '';
        }

        function render(data, selectedTickers, start, end) {
            clearCharts();

            const grid = document.getElementById('charts-grid');
            const tableBody = document.getElementById('data-table-body');
            const tableSection = document.getElementById('table-section');
            const emptyTip = document.getElementById('empty-tip');
            const summary = document.getElementById('summary-bar');
            tableBody.innerHTML = '';

            const ordered = selectedTickers.filter(t => data[t]);
            if (ordered.length === 0) {
                emptyTip.style.display = 'block';
                tableSection.style.display = 'none';
                summary.style.display = 'none';
                return;
            }
            emptyTip.style.display = 'none';
            summary.style.display = 'block';
            tableSection.style.display = 'block';

            let totalRecords = 0;
            let minDate = null, maxDate = null;
            const tableRows = [];

            ordered.forEach(ticker => {
                const item = data[ticker];
                const rows = item.rows || [];
                totalRecords += rows.length;
                if (rows.length === 0) return;

                const dates = rows.map(d => d.date);
                if (!minDate || dates[0] < minDate) minDate = dates[0];
                if (!maxDate || dates[dates.length - 1] > maxDate) maxDate = dates[dates.length - 1];

                renderOneStock(ticker, item.name, item.sector, rows);

                rows.forEach(r => {
                    tableRows.push({
                        date: r.date, ticker, name: item.name, sector: item.sector,
                        open: r.open, close: r.close, high: r.high, low: r.low,
                        pct_change: r.pct_change, amplitude: r.amplitude,
                        volume: r.volume, amount: r.amount, turn: r.turn
                    });
                });
            });

            tableRows.sort((a, b) => (b.date === a.date)
                ? a.ticker.localeCompare(b.ticker)
                : b.date.localeCompare(a.date));

            tableBody.innerHTML = tableRows.map(r => {
                const chgClass = (r.pct_change || 0) >= 0 ? 'up' : 'down';
                const chgSign = (r.pct_change || 0) > 0 ? '+' : '';
                return `<tr>
                    <td class="text-left">${r.date}</td>
                    <td class="text-left">${r.ticker}</td>
                    <td class="text-left">${r.name}</td>
                    <td>${fmt(r.open)}</td>
                    <td class="${chgClass}">${fmt(r.close)}</td>
                    <td>${fmt(r.high)}</td>
                    <td>${fmt(r.low)}</td>
                    <td class="${chgClass}">${r.pct_change === null || r.pct_change === undefined ? '-' : chgSign + r.pct_change.toFixed(2) + '%'}</td>
                    <td>${r.amplitude === null || r.amplitude === undefined ? '-' : r.amplitude.toFixed(2) + '%'}</td>
                    <td>${fmtVol(r.volume)}</td>
                    <td>${fmtAmount(r.amount)}</td>
                    <td>${r.turn === null || r.turn === undefined ? '-' : r.turn.toFixed(2) + '%'}</td>
                </tr>`;
            }).join('');

            document.getElementById('sum-stocks').textContent = `股票: ${ordered.length}`;
            document.getElementById('sum-records').textContent = `记录: ${totalRecords}`;
            document.getElementById('sum-range').textContent =
                `日期: ${minDate || '-'} ~ ${maxDate || '-'}`;
        }

        function fmt(v) {
            return v === null || v === undefined ? '-' : Number(v).toFixed(2);
        }
        function fmtVol(v) {
            return v === null || v === undefined ? '-' : (Number(v) / 10000).toFixed(1) + '万';
        }
        function fmtAmount(v) {
            if (v === null || v === undefined) return '-';
            return (Number(v) / 100000000).toFixed(2) + '亿';
        }

        function renderOneStock(ticker, name, sector, rows) {
            const grid = document.getElementById('charts-grid');
            const card = document.createElement('div');
            card.className = 'chart-card';
            const kid = `kline-${ticker}`;
            const vid = `vol-${ticker}`;
            card.innerHTML = `
                <div class="chart-title">
                    <span>${name} <strong>${ticker}</strong></span>
                    <span class="sector">${sector} · ${rows.length} 个交易日</span>
                </div>
                <div id="${kid}" class="kline-chart"></div>
                <div id="${vid}" class="vol-chart"></div>
            `;
            grid.appendChild(card);

            const kChart = echarts.init(document.getElementById(kid));
            const vChart = echarts.init(document.getElementById(vid));
            chartInstances.push(kChart, vChart);

            const dates = rows.map(d => d.date);
            const klineData = rows.map(d => [d.open, d.close, d.low, d.high]);
            const volumes = rows.map((d, i) => {
                const prev = rows[i - 1] || d;
                return {
                    value: d.volume,
                    itemStyle: { color: d.close >= prev.close ? '#ff4d4f' : '#00b578' }
                };
            });

            kChart.setOption({
                backgroundColor: 'transparent', animation: false,
                tooltip: {
                    trigger: 'axis', axisPointer: { type: 'cross' },
                    backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#c9d1d9' },
                    formatter: params => {
                        const idx = params[0].dataIndex;
                        const r = rows[idx];
                        let html = `<div style="font-weight:bold;margin-bottom:5px">${r.date}</div>`;
                        html += `开: ${fmt(r.open)} 收: <b>${fmt(r.close)}</b><br>`;
                        html += `高: ${fmt(r.high)} 低: ${fmt(r.low)}<br>`;
                        html += `涨: ${r.pct_change === null ? '-' : r.pct_change.toFixed(2)}% 振: ${r.amplitude === null ? '-' : r.amplitude.toFixed(2)}%<br>`;
                        html += `量: ${fmtVol(r.volume)}`;
                        if (r.ma5) html += `<br>MA5: ${r.ma5.toFixed(2)}`;
                        if (r.ma10) html += ` MA10: ${r.ma10.toFixed(2)}`;
                        if (r.ma20) html += ` MA20: ${r.ma20.toFixed(2)}`;
                        return html;
                    }
                },
                grid: { left: '3%', right: '3%', top: '10%', bottom: '12%' },
                xAxis: {
                    type: 'category', data: dates,
                    axisLine: { lineStyle: { color: '#30363d' } },
                    axisLabel: { color: '#8b949e' }
                },
                yAxis: {
                    scale: true,
                    splitLine: { lineStyle: { color: '#21262d' } },
                    axisLabel: { color: '#8b949e' }
                },
                dataZoom: [
                    { type: 'inside', start: 0, end: 100 },
                    { type: 'slider', show: true, bottom: 0, height: 18,
                      borderColor: '#30363d', fillerColor: 'rgba(88,166,255,0.2)',
                      handleStyle: { color: '#58a6ff' } }
                ],
                series: [
                    {
                        type: 'candlestick', name: 'K线', data: klineData,
                        itemStyle: {
                            color: '#ff4d4f', color0: '#00b578',
                            borderColor: '#ff4d4f', borderColor0: '#00b578'
                        }
                    },
                    {
                        type: 'line', name: 'MA5', data: rows.map(d => d.ma5),
                        smooth: true, lineStyle: { color: '#f2c94c', width: 1 }, symbol: 'none'
                    },
                    {
                        type: 'line', name: 'MA10', data: rows.map(d => d.ma10),
                        smooth: true, lineStyle: { color: '#9b51e0', width: 1 }, symbol: 'none'
                    },
                    {
                        type: 'line', name: 'MA20', data: rows.map(d => d.ma20),
                        smooth: true, lineStyle: { color: '#2f80ed', width: 1.5 }, symbol: 'none'
                    }
                ]
            });

            vChart.setOption({
                backgroundColor: 'transparent', animation: false,
                tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#c9d1d9' } },
                grid: { left: '3%', right: '3%', top: '5%', bottom: '20%' },
                xAxis: {
                    type: 'category', data: dates,
                    axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { show: false }
                },
                yAxis: {
                    splitLine: { lineStyle: { color: '#21262d' } },
                    axisLabel: { color: '#8b949e', formatter: v => (v / 10000).toFixed(0) + '万' }
                },
                dataZoom: [
                    { type: 'inside', start: 0, end: 100 },
                    { type: 'slider', show: false }
                ],
                series: [{ type: 'bar', name: '成交量', data: volumes, barWidth: '60%' }]
            });

            kChart.on('dataZoom', () => syncZoom(kChart, vChart));
            vChart.on('dataZoom', () => syncZoom(vChart, kChart));
        }

        function syncZoom(source, target) {
            const opt = source.getOption();
            const dz = opt.dataZoom && opt.dataZoom[0];
            if (dz) {
                target.dispatchAction({ type: 'dataZoom', start: dz.start, end: dz.end });
            }
        }

        window.addEventListener('resize', () => {
            chartInstances.forEach(c => c.resize());
        });

        init();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stocks")
def api_stocks():
    return jsonify(get_db_info())


@app.route("/api/data")
def api_data():
    tickers_raw = request.args.get("tickers", "")
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    return jsonify(get_stock_data(tickers, start_date, end_date))


if __name__ == "__main__":
    print("=" * 50)
    print("A股数据可视化页面启动中...")
    print("访问地址: http://localhost:5001")
    print("按 Ctrl+C 停止")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False)
