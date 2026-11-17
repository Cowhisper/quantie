"""Trade: 描述一笔完整交易的基类。"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    """
    一笔交易：某只股票的一次买入 + 卖出。

    包含股票代码、买卖时间 / 数量 / 价格 / 原因。
    A 股 T+1：sell_time 应晚于 buy_time（当日买入次日才能卖出）；
    账户层（lib/account.py）会把卖出日早于/等于买入日的信号顺延至
    买入日后下一交易日并按其开盘价重定价。
    """

    ticker: str                      # 股票代码
    name: str = ""                   # 股票名称
    buy_time: Optional[str] = None   # 买入日期 (YYYY-MM-DD)
    buy_amount: float = 0.0          # 买入数量（股）
    buy_price: float = 0.0           # 买入价格
    buy_reason: str = ""             # 买入原因
    sell_time: Optional[str] = None  # 卖出日期 (YYYY-MM-DD)
    sell_amount: Optional[float] = None  # 卖出数量（股）
    sell_price: Optional[float] = None   # 卖出价格
    sell_reason: Optional[str] = None    # 卖出原因
    score: float = 0.0               # 买入倾向评分（越高越优先成交，见 lib/account.py）

    @property
    def return_pct(self) -> Optional[float]:
        """单笔收益率（未平仓返回 None）"""
        if self.sell_price is None or self.buy_price == 0:
            return None
        return (self.sell_price - self.buy_price) / self.buy_price
