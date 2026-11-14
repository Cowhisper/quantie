import pandas as pd
from yacrs import register

from .base import BaseCondition


@register()
class GapUpCondition(BaseCondition):
    """跳空高开过滤：保留开盘价相对前一日收盘价高开 >= gap_min_pct 的行。"""

    def __init__(self, gap_min_pct: float = 0.04) -> None:
        super().__init__()
        self.gap_min_pct = gap_min_pct
        self.description = f"跳空高开 ≥ {gap_min_pct:.0%}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        gap_pct = df["open"] / df["close"].shift(1) - 1
        return df[gap_pct >= self.gap_min_pct]
