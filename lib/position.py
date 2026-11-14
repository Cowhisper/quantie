"""仓位/资金管理：决定每笔交易的买入/卖出数量。"""

from abc import ABC, abstractmethod

from yacrs import register


class BasePosition(ABC):
    """
    仓位管理基类，所有仓位/资金管理必须继承此类。

    子类必须实现:
        buy_amount(self, price) -> int

    返回给定成交价格下本笔交易应买入的股数（0 表示放弃本次买入）。
    """

    @abstractmethod
    def buy_amount(self, price: float) -> int:
        """给定成交价格，返回买入股数；0 表示资金/仓位不足，放弃买入。"""
        raise NotImplementedError

    def sell_amount(self, buy_amount: float) -> float:
        """卖出数量，默认全部卖出。"""
        return buy_amount


@register()
class FixedCapitalPosition(BasePosition):
    """
    固定本金仓位：每笔交易投入固定金额的本金。

    买入股数 = capital // price，向下取整到 lot_size 的整数倍
    （A 股整手，默认 100 股/手）；不足一手返回 0（放弃买入）。
    卖出默认全部卖出。
    """

    def __init__(self, capital: float = 10000.0, lot_size: int = 100) -> None:
        super().__init__()
        if capital <= 0:
            raise ValueError(f"capital 必须为正数，got {capital}")
        if lot_size < 1:
            raise ValueError(f"lot_size 必须 >= 1，got {lot_size}")
        self.capital = capital
        self.lot_size = lot_size

    def buy_amount(self, price: float) -> int:
        if price <= 0:
            return 0
        shares = int(self.capital // price)
        return shares - shares % self.lot_size
