"""回测结果 Web 服务：Flask JSON API + Vue 前端（lib/web）。"""

import os
from typing import Dict

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from lib.engine import (_candlestick_svg, compute_daily_account, compute_equity,
                        compute_total_assets, format_metric, _holding_window)

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VENDOR_DIR = os.path.join(WEB_DIR, "vendor")


def create_app(results: Dict[str, dict], start_time, end_time,
               initial_capital: float) -> Flask:
    """根据回测结果创建 Flask app：每个策略一个标签页的数据接口。"""
    app = Flask(__name__, static_folder=None)

    # 逐日账户表只需计算一次，缓存进 results 供 /api/daily 按月分页读取
    for name, data in results.items():
        data["daily_account"] = compute_daily_account(
            data["trades"], data["stock_data"], start_time, end_time,
            initial_capital=initial_capital)
        # 交易日按时间顺序排列，提取出现过的月份（有序、去重）供月份跳转
        months = []
        for d in data["daily_account"]:
            m = d["date"][:7]
            if not months or months[-1] != m:
                months.append(m)
        data["months"] = months

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/vendor/<path:filename>")
    def vendor(filename):
        return send_from_directory(VENDOR_DIR, filename)

    @app.get("/api/overview")
    def overview():
        return jsonify([
            {
                "name": name,
                "description": data.get("description", ""),
                "metrics": [[k, format_metric(k, v)] for k, v in data["overview"].items()],
            }
            for name, data in results.items()
        ])

    @app.get("/api/equity")
    def equity():
        name = request.args["name"]
        data = results[name]
        payload = compute_equity(data["trades"], data["stock_data"],
                                 start_time, end_time)
        payload.update(compute_total_assets(data["daily_account"]))
        return jsonify(payload)

    @app.get("/api/daily")
    def daily():
        """按月分页的账户明细：返回全部可选月份、当前月及其每个交易日的
        账户快照（总资产/现金/持仓金额/收益率/持仓列表）。
        month 缺省为第一个月，非法月份回退到第一个月。"""
        name = request.args["name"]
        days = results[name]["daily_account"]
        months = results[name]["months"]
        month = request.args.get("month") or (months[0] if months else None)
        if month not in months:
            month = months[0] if months else None
        sel = [d for d in days if d["date"][:7] == month] if month else []
        return jsonify({"months": months, "month": month, "days": sel})

    @app.get("/api/chart")
    def chart():
        name = request.args["name"]
        idx = int(request.args["trade"])
        t = results[name]["trades"][idx]
        df = results[name]["stock_data"][t.ticker]
        window = _holding_window(df, t.buy_time, t.sell_time)
        svg = _candlestick_svg(window, t.buy_time, t.sell_time)
        return Response(svg, mimetype="image/svg+xml")

    return app


def serve(results: Dict[str, dict], start_time, end_time,
          host: str, port: int, initial_capital: float) -> None:
    app = create_app(results, start_time, end_time, initial_capital)
    print(f"报告服务已启动: http://{host}:{port} （Ctrl+C 停止）")
    app.run(host=host, port=port)
