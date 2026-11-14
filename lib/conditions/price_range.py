import pandas as pd
from yacrs import register

from .base import BaseCondition


@register()
class PriceRangeCondition(BaseCondition):
    """价格区间过滤：保留 column 列（默认收盘价）落在 [min_price, max_price] 内的行（含端点）。"""

    def __init__(self, min_price: float = 0.0, max_price: float = float("inf"),
                 column: str = "close") -> None:
        super().__init__()
        if min_price > max_price:
            raise ValueError(f"min_price ({min_price}) 不能大于 max_price ({max_price})")
        self.min_price = min_price
        self.max_price = max_price
        self.column = column
        self.description = f"{column} 价格区间 {min_price} ~ {max_price}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        col = pd.to_numeric(df[self.column], errors="coerce")
        return df[(col >= self.min_price) & (col <= self.max_price)]
