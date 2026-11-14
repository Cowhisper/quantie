from abc import ABC, abstractmethod

import pandas as pd


class BaseCondition(ABC):
    """
    条件基类，所有市场过滤条件必须继承此类。

    子类必须实现:
        __call__(self, df) -> pd.DataFrame

    其中 df 为日线数据（单只股票、按日期升序），包含列:
        ticker, name, sector, date, open, high, low, close,
        volume, amount, turn, pctChg
    返回值为其中满足条件的行（保持原列结构）。
    无状态条件（如价格/日期区间）可链式调用依次过滤；依赖历史数据的条件
    （如相对放量）应在完整数据上计算，组合多个条件时建议对每个条件的
    结果取交集（策略即按此方式使用）。
    """

    description: str = ""

    def __init__(self) -> None:
        if not self.description:
            self.description = self.__class__.__name__

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤出满足条件的条目"""
        raise NotImplementedError
