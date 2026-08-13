#!/usr/bin/env python3
"""
A股K线可视化看板
- 支持股票切换
- K线 + MA5/MA10/MA20 + 成交量
- 自动计算涨跌幅、振幅
"""

from flask import Flask, render_template_string, jsonify
import sqlite3
import pandas as pd
import json

app = Flask(__name__)
DB_PATH = "a_stock_daily.db"

# ==================== 数据查询 ====================
def get_stock_list():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM stock_meta ORDER BY record_count DESC", conn)
    conn.close()
    return df.to_dict('records')

def get_stock_data(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM stock_daily WHERE ticker = ? ORDER BY date ASC",
        conn, params=(ticker,)
    )
    conn.close()

    if df.empty:
        return None

    # 计算均线
    df['ma5'] = df['close'].rolling(window=5).mean().round(2)
    df['ma10'] = df['close'].rolling(window=10).mean().round(2)
    df['ma20'] = df['close'].rolling(window=20).mean().round(2)

    # 计算涨跌幅
    df['pct_change'] = df['close'].pct_change() * 100
    df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1) * 100).round(2)

    # 处理NaN
    df = df.where(pd.notnull(df), None)

    return df

# ==================== 前端页面 ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>A股K线分析</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117; color: #c9d1d9;
        }
        .header {
            padding: 15px 30px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
            display: flex; align-items: center; gap: 20px;
        }
        .header h1 { font-size: 18px; color: #58a6ff; }
        select {
            padding: 8px 15px;
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            font-size: 14px; cursor: pointer;
        }
        select:hover { border-color: #58a6ff; }
        .stats {
            display: flex; gap: 25px;
            margin-left: auto; font-size: 13px;
        }
        .stat-item { text-align: center; }
        .stat-label { color: #8b949e; font-size: 11px; margin-bottom: 4px; }
        .stat-value { font-weight: 600; font-size: 15px; }
        .up { color: #ff4d4f; }
        .down { color: #00b578; }
        .container { padding: 20px 30px; }
        #kline-chart { width: 100%; height: 520px; }
        #volume-chart { width: 100%; height: 160px; margin-top: 10px; }
        .info-bar {
            display: flex; gap: 30px; padding: 12px 0;
            font-size: 13px; color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 A股日频K线分析</h1>
        <select id="stock-select" onchange="changeStock()">
            <option value="">加载中...</option>
        </select>
        <div class="stats" id="stats-bar">
            <div class="stat-item">
                <div class="stat-label">最新价</div>
                <div class="stat-value" id="stat-close">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">日涨跌</div>
                <div class="stat-value" id="stat-change">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">20日振幅</div>
                <div class="stat-value" id="stat-amp">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">成交量</div>
                <div class="stat-value" id="stat-vol">-</div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="info-bar">
            <span id="stock-name">-</span>
            <span id="stock-sector">-</span>
            <span id="date-range">-</span>
        </div>
        <div id="kline-chart"></div>
        <div id="volume-chart"></div>
    </div>

    <script>
        let klineChart, volumeChart;
        let stockList = [];

        async function init() {
            klineChart = echarts.init(document.getElementById('kline-chart'));
            volumeChart = echarts.init(document.getElementById('volume-chart'));
            window.addEventListener('resize', () => {
                klineChart.resize();
                volumeChart.resize();
            });

            const res = await fetch('/api/stocks');
            stockList = await res.json();
            const select = document.getElementById('stock-select');
            select.innerHTML = stockList.map(s =>
                `<option value="${s.ticker}">${s.name} (${s.ticker}) [${s.sector}]</option>`
            ).join('');

            if (stockList.length > 0) {
                select.value = stockList[0].ticker;
                await changeStock();
            }
        }

        async function changeStock() {
            const ticker = document.getElementById('stock-select').value;
            if (!ticker) return;

            const res = await fetch(`/api/data/${ticker}`);
            const data = await res.json();
            if (!data || data.length === 0) return;

            const meta = stockList.find(s => s.ticker === ticker);
            document.getElementById('stock-name').textContent = meta.name;
            document.getElementById('stock-sector').textContent = meta.sector;
            document.getElementById('date-range').textContent =
                `${data[0].date} ~ ${data[data.length-1].date}`;

            const last = data[data.length - 1];
            const prev = data[data.length - 2] || last;
            const change = ((last.close - prev.close) / prev.close * 100).toFixed(2);
            const changeEl = document.getElementById('stat-change');
            changeEl.textContent = (change > 0 ? '+' : '') + change + '%';
            changeEl.className = 'stat-value ' + (change >= 0 ? 'up' : 'down');

            document.getElementById('stat-close').textContent = last.close.toFixed(2);
            document.getElementById('stat-close').className = 'stat-value ' + (change >= 0 ? 'up' : 'down');

            const recent20 = data.slice(-20);
            const amp = ((Math.max(...recent20.map(d => d.high)) - Math.min(...recent20.map(d => d.low)))
                        / recent20[0].close * 100).toFixed(1);
            document.getElementById('stat-amp').textContent = amp + '%';

            document.getElementById('stat-vol').textContent = (last.volume / 10000).toFixed(1) + '万';

            renderCharts(data);
        }

        function renderCharts(data) {
            const dates = data.map(d => d.date);
            const klineData = data.map(d => [d.open, d.close, d.low, d.high]);
            const volumes = data.map((d, i) => {
                const prev = data[i-1] || d;
                return {
                    value: d.volume,
                    itemStyle: { color: d.close >= prev.close ? '#ff4d4f' : '#00b578' }
                };
            });

            klineChart.setOption({
                backgroundColor: 'transparent',
                animation: false,
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'cross' },
                    backgroundColor: '#161b22',
                    borderColor: '#30363d',
                    textStyle: { color: '#c9d1d9' },
                    formatter: function(params) {
                        const d = params[0];
                        const item = data[d.dataIndex];
                        let html = `<div style="font-weight:bold;margin-bottom:5px">${d.name}</div>`;
                        html += `开: ${item.open.toFixed(2)} 收: <b>${item.close.toFixed(2)}</b><br>`;
                        html += `高: ${item.high.toFixed(2)} 低: ${item.low.toFixed(2)}<br>`;
                        html += `涨: ${(item.pct_change || 0).toFixed(2)}% 振: ${item.amplitude || 0}%<br>`;
                        html += `量: ${(item.volume/10000).toFixed(1)}万`;
                        if (item.ma5) html += `<br>MA5: ${item.ma5.toFixed(2)}`;
                        if (item.ma10) html += ` MA10: ${item.ma10.toFixed(2)}`;
                        return html;
                    }
                },
                grid: { left: '3%', right: '3%', top: '10%', bottom: '10%' },
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
                    { type: 'inside', start: 50, end: 100 },
                    { type: 'slider', show: true, bottom: 0, height: 20,
                      borderColor: '#30363d', fillerColor: 'rgba(88,166,255,0.2)',
                      handleStyle: { color: '#58a6ff' } }
                ],
                series: [
                    {
                        type: 'candlestick',
                        name: 'K线',
                        data: klineData,
                        itemStyle: {
                            color: '#ff4d4f', color0: '#00b578',
                            borderColor: '#ff4d4f', borderColor0: '#00b578'
                        }
                    },
                    {
                        type: 'line', name: 'MA5', data: data.map(d => d.ma5),
                        smooth: true, lineStyle: { color: '#f2c94c', width: 1 },
                        symbol: 'none'
                    },
                    {
                        type: 'line', name: 'MA10', data: data.map(d => d.ma10),
                        smooth: true, lineStyle: { color: '#9b51e0', width: 1 },
                        symbol: 'none'
                    },
                    {
                        type: 'line', name: 'MA20', data: data.map(d => d.ma20),
                        smooth: true, lineStyle: { color: '#2f80ed', width: 1.5 },
                        symbol: 'none'
                    }
                ]
            });

            volumeChart.setOption({
                backgroundColor: 'transparent',
                animation: false,
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: '#161b22', borderColor: '#30363d',
                    textStyle: { color: '#c9d1d9' }
                },
                grid: { left: '3%', right: '3%', top: '5%', bottom: '20%' },
                xAxis: {
                    type: 'category', data: dates,
                    axisLine: { lineStyle: { color: '#30363d' } },
                    axisLabel: { show: false }
                },
                yAxis: {
                    splitLine: { lineStyle: { color: '#21262d' } },
                    axisLabel: {
                        color: '#8b949e',
                        formatter: v => (v/10000).toFixed(0) + '万'
                    }
                },
                dataZoom: [
                    { type: 'inside', start: 50, end: 100 },
                    { type: 'slider', show: false }
                ],
                series: [{
                    type: 'bar', name: '成交量',
                    data: volumes,
                    barWidth: '60%'
                }]
            });

            klineChart.on('dataZoom', function(params) {
                const opt = klineChart.getOption();
                const start = opt.dataZoom[0].start;
                const end = opt.dataZoom[0].end;
                volumeChart.dispatchAction({
                    type: 'dataZoom', start: start, end: end
                });
            });
            volumeChart.on('dataZoom', function(params) {
                const opt = volumeChart.getOption();
                const start = opt.dataZoom[0].start;
                const end = opt.dataZoom[0].end;
                klineChart.dispatchAction({
                    type: 'dataZoom', start: start, end: end
                });
            });
        }

        init();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stocks')
def api_stocks():
    return jsonify(get_stock_list())

@app.route('/api/data/<ticker>')
def api_data(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return jsonify([])
    return jsonify(df.to_dict('records'))

if __name__ == '__main__':
    print("=" * 50)
    print("A股K线看板启动中...")
    print("访问地址: http://localhost:5001")
    print("按 Ctrl+C 停止")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=False)
