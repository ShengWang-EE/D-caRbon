"""
TCL (thermostatically controlled load) helpers:
- base TCL parameters
- COP temperature correction
- building-level COP estimation from attributes
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def get_tcl_params(
    rated_power: float = 1.0,
    cop: float = 2.5,
    **kwargs: Any,
) -> Dict[str, float]:
    """Return base TCL parameters."""
    return {"rated_power": rated_power, "COP": cop, **kwargs}


def correct_cop_by_temperature(
    cop: float,
    T_ambient: float,
    *,
    T_ref: float = 30.0,
    k: float = 0.05,
    cop_min: float = 2.5,
    cop_max: float = 4.0,
    **kwargs: Any,
) -> float:
    """
    Apply linear ambient-temperature correction:
    COP(T_out) = COP_ref - k * (T_out - T_ref)
    """
    corrected = float(cop) - float(k) * (float(T_ambient) - float(T_ref))
    return float(np.clip(corrected, cop_min, cop_max))


def _to_temperature_scalar(value: object, default: float = 30.0) -> float:
    """Normalize scalar/array-like temperature to one numeric ambient value in degC."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
        if arr.empty:
            return float(default)
        return float(arr.mean())
    scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(scalar):
        return float(default)
    return float(scalar)


def _to_temperature_array(value: object, default: float = 30.0) -> np.ndarray:
    """Normalize scalar/array-like temperature to a 1D numeric array in degC."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
    else:
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        arr = np.array([scalar], dtype=float)

    if arr.size == 0:
        return np.array([float(default)], dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return arr


def calculate_COP(
    buildings: pd.DataFrame,
    *,
    reference_temp_c: float = 30.0,
    temp_sensitivity_per_c: float = 0.05,
    cop_min: float = 2.5,
    cop_max: float = 4.0,
) -> pd.Series:
    """
    Estimate COP for each building using:
    1) base COP by use type
    2) linear correction by ambient temperature
    """
    base_cop_by_use = {
        "residential": 3.2,
        "commercial": 3.6,
        "public": 3.5,
        "industrial": 3.0,
        "unknown": 3.5,
    }

    if "use_category" in buildings.columns:
        use_key = (
            buildings["use_category"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .where(lambda s: s.isin(base_cop_by_use.keys()), "unknown")
        )
    else:
        use_key = pd.Series("unknown", index=buildings.index)

    base_cop = use_key.map(base_cop_by_use).astype(float)
    if "COP_base" in buildings.columns:
        cop_base_override = pd.to_numeric(buildings["COP_base"], errors="coerce")
        base_cop = cop_base_override.fillna(base_cop)

    if "temperature" in buildings.columns:
        temp_arrays = buildings["temperature"].apply(
            lambda v: _to_temperature_array(v, default=reference_temp_c)
        )
    else:
        temp_arrays = pd.Series(
            [np.array([reference_temp_c], dtype=float) for _ in range(len(buildings))],
            index=buildings.index,
            dtype=object,
        )

    cop_out: list[np.ndarray] = []
    for idx in buildings.index:
        base = float(base_cop.loc[idx])
        temp_arr = temp_arrays.loc[idx]
        corrected = base - float(temp_sensitivity_per_c) * (temp_arr - float(reference_temp_c))
        cop_out.append(np.clip(corrected, float(cop_min), float(cop_max)).astype(float))
    return pd.Series(cop_out, index=buildings.index, dtype=object)
