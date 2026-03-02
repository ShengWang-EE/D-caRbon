"""
温控负荷模型：额定功率、COP、制冷/制热功率与能效修正。
TCL (thermostatically controlled load) model: rated power, COP, cooling/heating power.
"""
from __future__ import annotations
from typing import Dict, Any


def get_tcl_params(
    rated_power: float = 1.0,
    cop: float = 2.5,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    温控负荷基本参数。占位。
    """
    return {"rated_power": rated_power, "COP": cop, **kwargs}


def correct_cop_by_temperature(cop: float, T_ambient: float, **kwargs: Any) -> float:
    """
    按环境温度修正 COP。占位。
    """
    return cop
