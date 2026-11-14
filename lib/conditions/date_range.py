import pandas as pd
from yacrs import register

from .base import BaseCondition


@register()
class DateRangeCondition(BaseCondition):
    """日期区间过滤：保留 date 落在 [start_time, end_time] 内的行（含端点）。"""

    def __init__(self, start_time, end_time) -> None:
        super().__init__()
        self.start_ts = pd.Timestamp(start_time)
        self.end_ts = pd.Timestamp(end_time)
        self.description = f"日期区间 {self.start_ts:%Y-%m-%d} ~ {self.end_ts:%Y-%m-%d}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        dates = pd.to_datetime(df["date"])
        return df[(dates >= self.start_ts) & (dates <= self.end_ts)]
