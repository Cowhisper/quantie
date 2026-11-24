#!/usr/bin/env python3
"""quantie 统一命令行工具：A 股数据下载 / 策略回测 / 明日买入预测 / 每日任务 / 数据导出。

用法:
      python quantie.py <命令> [-c 配置] [key=value ...]

命令:
      download    增量下载行情数据（Baostock 日线，前复权；dry_run=true 预览）
      backtest    策略回测 + 报告（静态 HTML 或 Web 服务）
      forecast    预测下一交易日买入清单（每策略一个 CSV）
      daily       每日任务：数据更新 → 预测 → 模拟账户 → 企业微信推送
      export      数据库导出 CSV

配置默认 configs/quantie.yaml：一个文件包含 download/engine/forecast/
daily/export 五个任务段与策略/条件参数段，任何任务都用它；
key=value 可覆盖任意配置项，数字建议显式声明类型（key:float=50）:

      python quantie.py download download.end=2026-11-15 download.all=true
      python quantie.py backtest engine.limit=5 engine.start=2025-01-01
      python quantie.py forecast forecast.strategies=[ma,ross] forecast.date=2026-11-18
      python quantie.py daily daily.mode=push
      python quantie.py export export.mode=by_stock

定时任务（crontab 每天 02:00）:
      0 2 * * * cd /Users/kurt/Repo/quantie && /Users/kurt/miniconda3/bin/python quantie.py daily -c configs/quantie.yaml >> daily.log 2>&1
"""

import argparse
import json
import sys

import lib  # noqa: F401  import lib 即触发所有策略/条件的 yacrs 注册
from lib.tasks import (backtest_task, daily_task, download_task, export_task,
                       forecast_task)
from yacrs import _C

TASKS = {
    "download": download_task,
    "backtest": backtest_task,
    "forecast": forecast_task,
    "daily": daily_task,
    "export": export_task,
}

# 子命令 -> 配置文件中的配置段名（backtest 任务沿用 engine 段命名）
SECTION_NAMES = {"backtest": "engine"}

TASK_HELP = {
    "download": "增量下载行情数据（Baostock 日线，前复权）",
    "backtest": "策略回测 + 报告（静态 HTML 或 Web 服务）",
    "forecast": "预测下一交易日买入清单（每策略一个 CSV）",
    "daily": "每日任务：数据更新 → 预测 → 模拟账户 → 企业微信推送",
    "export": "数据库导出 CSV",
}


def _plain(value):
    """yacrs Node 递归转回普通 dict/list（任务函数按普通 dict 语义使用）。"""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _load_config(path: str) -> None:
    if not path:
        return
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            _C.update(json.load(f))
    elif path.endswith((".yaml", ".yml")):
        import yaml
        with open(path, encoding="utf-8") as f:
            _C.update(yaml.load(f, Loader=yaml.FullLoader))
    elif path.endswith(".toml"):
        import toml
        with open(path, encoding="utf-8") as f:
            _C.update(toml.load(f))
    else:
        raise SystemExit(f"不支持的配置文件类型: {path}（yaml/json/toml）")


def _apply_overrides(options) -> None:
    """key=value 覆盖任意配置项（与原先各脚本一致，可带类型声明 key:float=50）。"""
    for opt in options:
        if "=" not in opt:
            raise SystemExit(f"无法解析参数 {opt!r}，应为 key=value 形式")
        k, v = opt.split("=", 1)
        if ":" in k:
            k, dtype = k.split(":", 1)
            try:
                v = eval(dtype)(v)  # noqa: S307 与 yacrs 一致的类型声明语法
            except Exception:
                raise SystemExit(f"无法把 {v!r} 转换为 {dtype}")
        _C.set(k, v)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantie", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/quantie.yaml",
                        help="配置文件路径（默认 configs/quantie.yaml）")
    # 子命令后也接受 -c（默认 SUPPRESS，未传时不覆盖主解析器的默认值）
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", "-c", default=argparse.SUPPRESS,
                             help="配置文件路径（默认 configs/quantie.yaml）")
    subparsers = parser.add_subparsers(dest="command", metavar="<命令>")
    for name in TASKS:
        p = subparsers.add_parser(name, help=TASK_HELP[name],
                                  description=TASK_HELP[name],
                                  parents=[config_parent])
        p.add_argument("options", nargs="*", metavar="key=value",
                       help="覆盖任意配置项，如 download.end=2026-11-15")
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    _load_config(args.config)
    _apply_overrides(args.options)

    # 任务参数 = 配置文件中该命令对应配置段的内容（覆盖已并入 _C）
    section_name = SECTION_NAMES.get(args.command, args.command)
    section = _plain(_C[section_name]) if section_name in _C else {}
    TASKS[args.command](**section)


if __name__ == "__main__":
    main()
