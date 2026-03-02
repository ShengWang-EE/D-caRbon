"""
Kriging 插值：由气象站点到建筑位置的温度估计。
Spatial interpolation from weather stations to building locations.
"""
from __future__ import annotations
from typing import Any


def interpolate_temperature(
    station_locs: Any,
    station_temps: Any,
    target_locs: Any,
    **kwargs: Any,
) -> Any:
    """
    在目标位置（如建筑中心）插值得到温度。
    Interpolate temperature at target locations (e.g. building centroids).
    占位：返回 None 或与 target_locs 同长的数组。
    """
    if station_locs is None or station_temps is None or target_locs is None:
        return None
    # TODO: implement Kriging (e.g. scipy or pykrige)
    return None
