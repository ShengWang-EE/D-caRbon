"""
微气象与热岛：冠层接口、温度修正。
Micro-meteorology and urban heat island: canopy interface, temperature correction.
"""
from __future__ import annotations
from typing import Any


def canopy_temperature_delta(geometry: Any, weather: Any, **kwargs: Any) -> Any:
    """
    冠层接口模型得到的区域温度增量（热岛等）。占位。
    """
    return None


def correct_temperature_for_zone(T_background: Any, delta: Any, **kwargs: Any) -> Any:
    """按热岛等修正后的区域温度。占位。"""
    return T_background
