"""
用户行为模型：舒适温度区间、最适温度、内部产热等。
User behavior: comfort band, setpoint, internal gains.
"""
from __future__ import annotations
from typing import Dict, Any


def get_comfort_band(
    T_min: float = 24.0,
    T_max: float = 28.0,
    T_star: float = 26.0,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    舒适温度区间与最适温度。占位。
    """
    return {"T_min": T_min, "T_max": T_max, "T_star": T_star, **kwargs}


def get_internal_gains(area_floor: float, **kwargs: Any) -> float:
    """
    内部产热（人员、设备等）W/m² 或总功率。占位。
    """
    return 0.0
