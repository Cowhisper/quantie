import pandas as pd
from yacrs import register

from .base import BaseCondition


@register()
class RelativeVolumeCondition(BaseCondition):
    """相对放量过滤：保留成交量 >= rel_vol_min 倍过去 vol_window 日均量（不含当日）的行。"""

    def __init__(self, rel_vol_min: float = 2.0, vol_window: int = 20) -> None:
        super().__init__()
        self.rel_vol_min = rel_vol_min
        self.vol_window = vol_window
        self.description = f"相对放量 ≥ {rel_vol_min}x（{vol_window} 日均量）"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        prev_avg_volume = df["volume"].shift(1).rolling(self.vol_window).mean()
        rel_volume = df["volume"] / prev_avg_volume
        return df[rel_volume >= self.rel_vol_min]
