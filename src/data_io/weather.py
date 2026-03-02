"""
气象站数据读取与时间序列。
Load weather station time series (temperature, etc.).
"""
from __future__ import annotations
from typing import Any


def load_weather(path: str | None = None, **kwargs: Any) -> Any:
    """
    读取气象站数据（如逐时温度）。
    Load weather station data (e.g. hourly temperature).
    占位：返回 None 或空 DataFrame。
    """
    if path is None:
        return None
    # TODO: implement CSV/NetCDF load
    return None
