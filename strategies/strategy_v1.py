import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import warnings

# 支持直接运行脚本（python strategies/strategy_v1.py）和作为模块导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from strategies.base import BaseStrategy

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：数据准备（兼容旧版直接调用）
# ============================================================

def load_data(file_path):
    """
    加载日线数据
    数据格式要求：包含 open, high, low, close, volume, amount, turn, pctChg
    """
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def load_all_data(db_path='a_stock_daily.db'):
    """
    从 SQLite 数据库加载全部日线数据
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM stock_daily ORDER BY ticker, date", conn)
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df


# ============================================================
# 第二部分：技术指标计算
# ============================================================

def calc_ma(df, windows=[5, 10, 20, 60]):
    """计算移动平均线"""
    for w in windows:
        df[f'MA{w}'] = df['close'].rolling(w).mean()
    return df


def calc_volume_ma(df, window=5):
    """计算均量线"""
    df['VOL_MA5'] = df['volume'].rolling(window).mean()
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    返回: DIF, DEA, MACD_BAR
    """
    df['EMA12'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD_BAR'] = 2 * (df['DIF'] - df['DEA'])
    return df


def calc_all_indicators(df):
    """一次性计算所有技术指标"""
    df = calc_ma(df)
    df = calc_volume_ma(df)
    df = calc_macd(df)
    return df


# ============================================================
# 第三部分：共振买入信号生成
# ============================================================

def generate_signals(df):
    """
    生成三种共振买入信号
    """
    df['Buy_1'] = False
    df['Buy_2'] = False
    df['Buy_3'] = False
    df['Buy_Signal'] = False

    if len(df) < 60:
        print("警告：数据长度不足60天，部分信号可能无法计算")
        return df

    df['MA20_high'] = df['MA20'] * 1.03
    df['MA20_low'] = df['MA20'] * 0.97
    df['HIGH_20'] = df['high'].rolling(20).max().shift(1)

    # 信号1：底部金叉+突破
    cond1_1 = (df['close'] > df['MA20']) & (df['volume'] > 1.5 * df['VOL_MA5'])
    max_ma = df[['MA5', 'MA10', 'MA20']].max(axis=1)
    min_ma = df[['MA5', 'MA10', 'MA20']].min(axis=1)
    cond1_2 = (max_ma - min_ma) < 0.02 * df['MA20']
    cond1_3 = (df['DIF'].shift(1) <= df['DEA'].shift(1)) & (df['DIF'] > df['DEA']) & (df['DIF'] < 0.1 * df['close'])
    cond1_4 = (df['close'] - df['open']) / df['open'] > 0.03
    df['Buy_1'] = cond1_1 & cond1_2 & cond1_3 & cond1_4

    # 信号2：空中加油（回踩+零轴上金叉）
    cond2_1 = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) & (df['MA20'] > df['MA20'].shift(10))
    cond2_2 = (df['close'] >= df['MA20_low']) & (df['close'] <= df['MA20_high'])
    cond2_3 = df['volume'] < 0.7 * df['VOL_MA5']
    cond2_4 = (df['close'] > df['close'].shift(1)) & (df['volume'] > 1.2 * df['VOL_MA5'])
    cond2_5 = (df['DIF'].shift(1) <= df['DEA'].shift(1)) & (df['DIF'] > df['DEA']) & (df['DIF'] > 0) & (df['DEA'] > 0)
    df['Buy_2'] = cond2_1 & cond2_2 & cond2_3 & cond2_4 & cond2_5

    # 信号3：突破前高
    cond3_1 = (df['close'] > df['HIGH_20']) & (df['volume'] > 1.8 * df['VOL_MA5'])
    cond3_2 = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
    cond3_3 = (df['DIF'] > 0) & (df['DEA'] > 0)
    cond3_4 = (df['MACD_BAR'] > df['MACD_BAR'].shift(1)) & (df['MACD_BAR'].shift(1) > df['MACD_BAR'].shift(2)) & (df['MACD_BAR'] > 0)
    pct_change = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    cond3_5 = (pct_change > 0.03) & (pct_change < 0.07)
    df['Buy_3'] = cond3_1 & cond3_2 & cond3_3 & cond3_4 & cond3_5

    df['Buy_Signal'] = df['Buy_1'] | df['Buy_2'] | df['Buy_3']
    return df


def analyze_stock(df_stock, min_days=60):
    """
    对单只股票计算指标并生成买入信号（兼容旧版）
    """
    df_stock = df_stock.sort_values('date').reset_index(drop=True)
    if len(df_stock) < min_days:
        return None
    df_stock = calc_all_indicators(df_stock)
    df_stock = generate_signals(df_stock)
    return df_stock


def scan_market(db_path='a_stock_daily.db', min_days=60):
    """
    扫描全市场，返回最新交易日触发买入信号的股票列表（兼容旧版）
    """
    df = load_all_data(db_path)
    if df.empty:
        print("数据库中没有数据")
        return pd.DataFrame()

    selected = []
    for ticker, group in df.groupby('ticker', sort=False):
        analyzed = analyze_stock(group.copy(), min_days)
        if analyzed is None or analyzed.empty:
            continue

        latest = analyzed.iloc[-1]
        if latest['Buy_Signal']:
            reasons = []
            if latest['Buy_1']:
                reasons.append('底部金叉+突破')
            if latest['Buy_2']:
                reasons.append('空中加油')
            if latest['Buy_3']:
                reasons.append('突破前高')

            selected.append({
                'ticker': ticker,
                'name': latest.get('name', ticker),
                'sector': latest.get('sector', '-'),
                'date': latest['date'],
                'close': latest['close'],
                'reason': ' + '.join(reasons)
            })

    return pd.DataFrame(selected)


def run_market_scan(db_path='a_stock_daily.db', min_days=60):
    """
    运行全市场扫描并输出选股结果（兼容旧版）
    """
    print("=" * 60)
    print("共振买入法 —— 全市场选股")
    print("=" * 60)
    print(f"\n[1] 加载数据库: {db_path}")

    df_all = load_all_data(db_path)
    print(f"    总记录数: {len(df_all)}")
    print(f"    股票数量: {df_all['ticker'].nunique()}")
    print(f"    日期范围: {df_all['date'].min().date()} 至 {df_all['date'].max().date()}")

    print(f"\n[2] 扫描全市场 (最少 {min_days} 天数据)...")
    selected = scan_market(db_path, min_days)

    print(f"\n[3] 选股结果: 共 {len(selected)} 只股票触发买入信号")
    print("-" * 60)
    if selected.empty:
        print("    暂无股票触发买入信号")
    else:
        selected = selected.sort_values('ticker').reset_index(drop=True)
        for _, row in selected.iterrows():
            print(f"    {row['ticker']} | {row['name']} | {row['sector']} | "
                  f"收盘价: {row['close']:.2f} | 日期: {row['date'].date()} | 原因: {row['reason']}")

    print("-" * 60)
    return selected


# ============================================================
# 第四部分：面向对象的策略类
# ============================================================

class ResonanceStrategy(BaseStrategy):
    """
    共振买入策略。

    同时监控三种买入形态：
    1. 底部金叉+突破
    2. 空中加油
    3. 突破前高
    """

    def __init__(self):
        super().__init__()
        self.name = "ResonanceStrategy"
        self.description = "基于均线、成交量与MACD共振的买入策略"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线、均量线与MACD指标"""
        return calc_all_indicators(df)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成三种共振买入信号"""
        return generate_signals(df)

    def get_signal_reason(self, row) -> str:
        """返回触发买入信号的具体原因"""
        reasons = []
        if row.get('Buy_1'):
            reasons.append('底部金叉+突破')
        if row.get('Buy_2'):
            reasons.append('空中加油')
        if row.get('Buy_3'):
            reasons.append('突破前高')
        return ' + '.join(reasons) if reasons else '信号触发'


# ============================================================
# 第五部分：旧版回测与可视化（保留直接调用入口）
# ============================================================

def backtest(df, initial_capital=100000, stop_loss_pct=0.05, trailing_stop_pct=0.08):
    """
    回测执行（兼容旧版）
    """
    strategy = ResonanceStrategy()
    return strategy.backtest(df, initial_capital, stop_loss_pct, trailing_stop_pct)


def get_signal_reason(row):
    """获取触发买入信号的具体原因（兼容旧版）"""
    return ResonanceStrategy().get_signal_reason(row)


def calculate_performance(trades, initial_capital):
    """计算回测绩效指标（兼容旧版）"""
    return ResonanceStrategy()._calculate_performance(trades, initial_capital)


def plot_results(df, trades):
    """绘制回测结果图（需安装matplotlib）"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='收盘价', linewidth=1.5, color='black')
        ax1.plot(df.index, df['MA5'], label='MA5', linewidth=1, alpha=0.7)
        ax1.plot(df.index, df['MA10'], label='MA10', linewidth=1, alpha=0.7)
        ax1.plot(df.index, df['MA20'], label='MA20', linewidth=1.5, alpha=0.8)

        buy_dates = [t['date'] for t in trades if t['action'] == 'BUY']
        buy_prices = [t['price'] for t in trades if t['action'] == 'BUY']
        sell_dates = [t['date'] for t in trades if t['action'] == 'SELL']
        sell_prices = [t['price'] for t in trades if t['action'] == 'SELL']

        ax1.scatter(buy_dates, buy_prices, marker='^', color='red', s=100, label='买入', zorder=5)
        ax1.scatter(sell_dates, sell_prices, marker='v', color='green', s=100, label='卖出', zorder=5)

        ax1.set_title('股价走势与交易信号')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.bar(df.index, df['volume'], alpha=0.5, label='成交量')
        ax2.set_title('成交量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(df.index, df['DIF'], label='DIF', color='blue', linewidth=1)
        ax3.plot(df.index, df['DEA'], label='DEA', color='orange', linewidth=1)
        bar_colors = ['red' if x > 0 else 'green' for x in df['MACD_BAR']]
        ax3.bar(df.index, df['MACD_BAR'], color=bar_colors, alpha=0.5, label='MACD柱')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib未安装，跳过图表绘制")


def run_backtest(data_path, initial_capital=100000):
    """
    运行完整回测（兼容旧版）
    """
    print("=" * 60)
    print("共振买入法回测系统")
    print("=" * 60)

    print(f"\n[1] 加载数据: {data_path}")
    df = load_data(data_path)
    print(f"    数据条数: {len(df)}")
    print(f"    时间范围: {df['date'].min()} 至 {df['date'].max()}")

    strategy = ResonanceStrategy()

    print("\n[2] 计算技术指标...")
    df = strategy.prepare_data(df)
    print("    指标计算完成")

    print("\n[3] 生成买入信号...")
    df = strategy.generate_signals(df)
    signal_count = df['Buy_Signal'].sum()
    print(f"    共生成 {signal_count} 个买入信号")

    print(f"\n[4] 执行回测 (初始资金: {initial_capital:,})...")
    trades, stats = strategy.backtest(df, initial_capital)

    print("\n[5] 回测结果:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            if key in ['胜率', '总收益率', '平均收益率']:
                print(f"    {key}: {value*100:.2f}%")
            elif key in ['总资金']:
                print(f"    {key}: {value:,.2f}")
            else:
                print(f"    {key}: {value:.2f}")
        else:
            print(f"    {key}: {value}")

    print("\n" + "-" * 40)
    print("交易明细:")
    for t in trades:
        if t['action'] == 'BUY':
            print(f"    {t['date']} | 买入 | 价格: {t['price']:.2f} | 数量: {t['shares']:.0f} | 原因: {t.get('reason','-')}")
        else:
            profit_pct = t.get('profit_pct', 0)
            print(f"    {t['date']} | 卖出 | 价格: {t['price']:.2f} | 收益: {profit_pct*100:.2f}% | 原因: {t.get('reason','-')}")

    print("\n[6] 绘制回测图表...")
    plot_results(df, trades)

    return df, trades, stats


# ============================================================
# 运行示例
# ============================================================

if __name__ == "__main__":
    DB_PATH = 'a_stock_daily.db'

    if os.path.exists(DB_PATH):
        selected = run_market_scan(DB_PATH, min_days=60)
    else:
        print("数据库不存在，使用模拟数据运行回测...\n")

        def generate_sample_data(days=300):
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=days)
            close = 100 + np.cumsum(np.random.randn(days) * 0.5)
            close = np.maximum(close, 10)
            open_price = close + np.random.randn(days) * 0.2
            high = np.maximum(close, open_price) + np.random.rand(days) * 0.5
            low = np.minimum(close, open_price) - np.random.rand(days) * 0.5
            volume = np.random.randint(1000, 10000, days)
            amount = volume * close
            turn = np.random.uniform(1, 5, days)
            pctChg = np.diff(close, prepend=close[0]) / close[0] * 100

            df = pd.DataFrame({
                'date': dates,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'amount': amount,
                'turn': turn,
                'pctChg': pctChg
            })
            return df

        df_sample = generate_sample_data(300)
        df_sample.to_csv('sample_data.csv', index=False)
        df_result, trades, stats = run_backtest('sample_data.csv', initial_capital=100000)
