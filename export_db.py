#!/usr/bin/env python3
"""
A股日频数据导出工具
从 SQLite 数据库导出到 CSV，支持多种模式
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime

# ==================== 配置 ====================
DB_PATH = "a_stock_daily.db"           # 数据库路径
OUTPUT_DIR = "csv_export"              # 导出目录


def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 创建目录: {path}")


def export_all(db_path, output_dir):
    """模式1: 导出全部数据到一个CSV"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM stock_daily ORDER BY ticker, date", conn)
    conn.close()

    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, "all_stocks_daily.csv")
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

    print(f"✅ 全量导出: {filepath}")
    print(f"   记录数: {len(df)} | 股票数: {df['ticker'].nunique()} | 列: {list(df.columns)}")
    return filepath


def export_by_stock(db_path, output_dir):
    """模式2: 每只股票单独一个CSV"""
    conn = sqlite3.connect(db_path)
    meta = pd.read_sql("SELECT ticker, name, sector FROM stock_meta", conn)

    ensure_dir(output_dir)
    exported = []

    for _, row in meta.iterrows():
        ticker = row['ticker']
        name = row['name']

        df = pd.read_sql(
            "SELECT * FROM stock_daily WHERE ticker = ? ORDER BY date",
            conn, params=(ticker,)
        )

        # 文件名: 600519_贵州茅台.csv
        safe_name = name.replace('/', '_').replace('\\', '_')
        filename = f"{ticker.replace('.', '_')}_{safe_name}.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        exported.append((ticker, name, len(df), filepath))
        print(f"  ✅ {ticker} {name}: {len(df)}条 -> {filename}")

    conn.close()
    print(f"📊 共导出 {len(exported)} 只股票到 {output_dir}/")
    return exported


def export_by_sector(db_path, output_dir):
    """模式3: 按板块合并导出"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM stock_daily ORDER BY sector, ticker, date", conn)
    sectors = df['sector'].unique()

    ensure_dir(output_dir)
    exported = []

    for sector in sectors:
        sector_df = df[df['sector'] == sector].copy()
        safe_sector = sector.replace('/', '_').replace('\\', '_')
        filename = f"sector_{safe_sector}.csv"
        filepath = os.path.join(output_dir, filename)

        sector_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        exported.append((sector, len(sector_df), filepath))
        print(f"  ✅ [{sector}]: {len(sector_df)}条 ({sector_df['ticker'].nunique()}只) -> {filename}")

    conn.close()
    print(f"📊 共导出 {len(exported)} 个板块到 {output_dir}/")
    return exported


def export_meta(db_path, output_dir):
    """模式4: 导出元信息表"""
    conn = sqlite3.connect(db_path)
    meta = pd.read_sql("SELECT * FROM stock_meta", conn)
    conn.close()

    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, "stock_meta.csv")
    meta.to_csv(filepath, index=False, encoding='utf-8-sig')

    print(f"✅ 元信息导出: {filepath}")
    print(f"   股票数: {len(meta)}")
    return filepath


def export_summary(db_path, output_dir):
    """模式5: 导出统计摘要"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM stock_daily", conn)
    conn.close()

    summary = df.groupby('ticker').agg({
        'name': 'first',
        'sector': 'first',
        'date': ['min', 'max', 'count'],
        'close': ['min', 'max', 'mean'],
        'volume': 'mean'
    }).round(2)

    summary.columns = ['name', 'sector', 'first_date', 'last_date', 'days',
                       'close_min', 'close_max', 'close_avg', 'volume_avg']
    summary = summary.reset_index()

    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, "summary.csv")
    summary.to_csv(filepath, index=False, encoding='utf-8-sig')

    print(f"✅ 统计摘要导出: {filepath}")
    print(summary.to_string(index=False))
    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description='A股日频数据导出工具')
    parser.add_argument('--db', default=DB_PATH, help=f'数据库路径 (默认: {DB_PATH})')
    parser.add_argument('--out', default=OUTPUT_DIR, help=f'输出目录 (默认: {OUTPUT_DIR})')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'by_stock', 'by_sector', 'meta', 'summary', 'everything'],
                        help='导出模式: all=全量合并, by_stock=按股票拆分, by_sector=按板块拆分, meta=元信息, summary=统计摘要, everything=全部')

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"❌ 数据库不存在: {args.db}")
        print(f"💡 请确认路径正确，或先运行数据下载脚本")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"📦 数据库: {args.db}")
    print(f"📁 输出目录: {args.out}")
    print(f"🔄 导出模式: {args.mode}")
    print(f"{'='*60}")

    if args.mode == 'all':
        export_all(args.db, args.out)

    elif args.mode == 'by_stock':
        export_by_stock(args.db, args.out)

    elif args.mode == 'by_sector':
        export_by_sector(args.db, args.out)

    elif args.mode == 'meta':
        export_meta(args.db, args.out)

    elif args.mode == 'summary':
        export_summary(args.db, args.out)

    elif args.mode == 'everything':
        print(">>> 模式1: 全量合并")
        export_all(args.db, args.out)
        print()
        print(">>> 模式2: 按股票拆分")
        export_by_stock(args.db, os.path.join(args.out, "by_stock"))
        print()
        print(">>> 模式3: 按板块拆分")
        export_by_sector(args.db, os.path.join(args.out, "by_sector"))
        print()
        print(">>> 模式4: 元信息")
        export_meta(args.db, args.out)
        print()
        print(">>> 模式5: 统计摘要")
        export_summary(args.db, args.out)

    print(f"{'='*60}")
    print(f"✅ 导出完成！文件保存在: {args.out}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
