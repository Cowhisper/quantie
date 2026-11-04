from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List

import pandas as pd

from ..trade import Trade


class BaseStrategy(ABC):
    """
    策略基类，所有策略必须继承此类。

    子类必须实现:
        run(self, df, start_time, end_time) -> List[Trade]

    其中 df 为单只股票的日线数据（按日期升序），包含列:
        ticker, name, sector, date, open, high, low, close,
        volume, amount, turn, pctChg

    start_time / end_time 为回测区间（datetime 或字符串），
    策略应只在该区间内产生买入信号，
    但可以使用区间之前的数据计算技术指标（df 中已包含前置历史数据）。
    """

    name: str = ""
    description: str = ""

    # 买入 / 卖出原因定义，仅供说明与展示
    BUY_REASONS: ClassVar[Dict[str, str]] = {}
    SELL_REASONS: ClassVar[Dict[str, str]] = {}

    def __init__(self) -> None:
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    def run(self, df: pd.DataFrame, start_time, end_time) -> List[Trade]:
        """
        在回测区间 [start_time, end_time] 内运行策略，
        返回选中的交易列表。
        """
        raise NotImplementedError
