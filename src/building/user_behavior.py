"""
User behavior helpers: comfort setpoint, comfort band, internal gains.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def get_comfort_band(
    T_min: float = 24.0,
    T_max: float = 28.0,
    T_star: float = 26.0,
    **kwargs: Any,
) -> Dict[str, float]:
    """Return a simple comfort-band dictionary."""
    return {"T_min": T_min, "T_max": T_max, "T_star": T_star, **kwargs}


def get_internal_gains(area_floor: float, **kwargs: Any) -> float:
    """Placeholder for internal gains (people/equipment)."""
    return 0.0


def _to_temperature_array(value: object, default: float = 30.0) -> np.ndarray:
    """Normalize scalar/array-like temperature to a 1D numeric ambient array (degC)."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
        if arr.size == 0:
            return np.array([float(default)], dtype=float)
    else:
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        arr = np.array([scalar], dtype=float)
    return np.where(np.isfinite(arr), arr, float(default)).astype(float)


def _to_numeric_array(value: object, default: float = 0.0) -> np.ndarray:
    """Normalize scalar/array-like input to a 1D numeric array."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
        if arr.size == 0:
            return np.array([float(default)], dtype=float)
    else:
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        arr = np.array([scalar], dtype=float)
    return np.where(np.isfinite(arr), arr, float(default)).astype(float)


def _to_length_array(value: object, length: int, default: float) -> np.ndarray:
    arr = _to_numeric_array(value, default=default)
    if arr.size == length:
        return arr
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=float)
    if arr.size > length:
        return arr[:length]
    out = np.empty(length, dtype=float)
    out[: arr.size] = arr
    out[arr.size :] = float(arr[-1])
    return out


def set_comfort_zone(
    buildings: pd.DataFrame,
    *,
    seed: int = 42,
    user_std_c: float = 1.0,
    adaptive_weight: float = 0.20,
    setpoint_min_c: float = 21.0,
    setpoint_max_c: float = 28.0,
    zone_min_c: float = 20.0,
    zone_max_c: float = 30.0,
    temp_sensitivity: float = 0.35,
    flow_sensitivity: float = -0.25,
    band_factor_min: float = 0.65,
    band_factor_max: float = 1.80,
    neutral_outdoor_temp_c: float = 26.0,
    outdoor_scale_c: float = 10.0,
    flow_col_candidates: tuple[str, ...] = (
        "people_flow",
        "footfall",
        "occupancy_rate",
        "occupancy",
        "crowd_index",
    ),
) -> pd.DataFrame:
    """
    Build comfort setpoint and comfort band for each building.

    Algorithm:
    1) type-based base setpoint
    2) user heterogeneity: N(0, user_std_c)
    3) adaptive comfort correction with outdoor temperature:
       T_adapt = 17.8 + 0.31 * T_out
       T_set = (1-adaptive_weight) * (T_base + noise) + adaptive_weight * T_adapt
    4) time-varying band-width factor from outdoor temperature and people-flow
    5) use-type band deltas to construct [T_min, T_max]
    """
    if buildings.empty:
        return pd.DataFrame(
            {
                "T_star": pd.Series(dtype=float),
                "T_min": pd.Series(dtype=float),
                "T_max": pd.Series(dtype=float),
                "comfort_zone": pd.Series(dtype=object),
            },
            index=buildings.index,
        )

    base_setpoint_by_use = {
        "residential": 26.0,
        "commercial": 23.0,
        "public": 24.0,
        "industrial": 24.0,
        "unknown": 24.0,
    }
    band_delta_by_use: Dict[str, Tuple[float, float]] = {
        "residential": (2.0, 1.0),  # 26 -> [24, 27]
        "commercial": (1.0, 1.0),   # 23 -> [22, 24]
        "public": (1.0, 2.0),       # 24 -> [23, 26]
        "industrial": (1.0, 2.0),   # 24 -> [23, 26]
        "unknown": (1.0, 2.0),
    }

    if "use_category" in buildings.columns:
        use_key = (
            buildings["use_category"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        valid_use = use_key.isin(list(base_setpoint_by_use.keys()))
        use_key = use_key.where(valid_use, "unknown")
    else:
        use_key = pd.Series("unknown", index=buildings.index)

    base_setpoint = use_key.map(base_setpoint_by_use).astype(float)
    if "comfort_base" in buildings.columns:
        comfort_base_override = pd.to_numeric(buildings["comfort_base"], errors="coerce")
        base_setpoint = comfort_base_override.fillna(base_setpoint)

    if "temperature" in buildings.columns:
        outdoor_temp = pd.Series(
            [_to_temperature_array(v) for v in buildings["temperature"].tolist()],
            index=buildings.index,
            dtype=object,
        )
    else:
        outdoor_temp = pd.Series(
            [np.array([30.0], dtype=float) for _ in buildings.index],
            index=buildings.index,
            dtype=object,
        )

    rng = np.random.default_rng(seed)
    user_noise = pd.Series(
        rng.normal(loc=0.0, scale=user_std_c, size=len(buildings)),
        index=buildings.index,
        dtype=float,
    )

    minus_delta = use_key.map(lambda k: band_delta_by_use[k][0]).astype(float)
    plus_delta = use_key.map(lambda k: band_delta_by_use[k][1]).astype(float)

    t_star_out: list[np.ndarray] = []
    t_min_out: list[np.ndarray] = []
    t_max_out: list[np.ndarray] = []
    comfort_zone_out: list[tuple[np.ndarray, np.ndarray]] = []

    flow_col = next((c for c in flow_col_candidates if c in buildings.columns), None)

    for idx in buildings.index:
        t_out = _to_temperature_array(outdoor_temp.loc[idx], default=30.0)
        n_t = max(1, int(t_out.size))
        t_adapt = 17.8 + 0.31 * t_out

        base = float(base_setpoint.loc[idx])
        noise = float(user_noise.loc[idx])
        setpoint = (1.0 - adaptive_weight) * (base + noise) + adaptive_weight * t_adapt
        setpoint = np.clip(setpoint, setpoint_min_c, setpoint_max_c)

        # Time-varying comfort-band width:
        # - farther from neutral outdoor temperature -> wider/narrower by temp_sensitivity
        # - higher flow/occupancy typically tightens comfort (negative default flow_sensitivity)
        temp_intensity = np.clip(
            np.abs(t_out - float(neutral_outdoor_temp_c)) / max(float(outdoor_scale_c), 1e-6),
            0.0,
            1.0,
        )

        if flow_col is None:
            flow_norm = np.full(n_t, 0.5, dtype=float)
        else:
            flow_raw = _to_length_array(buildings.at[idx, flow_col], n_t, default=0.5)
            flow_min = float(np.nanmin(flow_raw))
            flow_max = float(np.nanmax(flow_raw))
            if np.isfinite(flow_min) and np.isfinite(flow_max) and flow_max > flow_min:
                flow_norm = (flow_raw - flow_min) / (flow_max - flow_min)
            else:
                flow_norm = np.full(n_t, 0.5, dtype=float)

        band_factor = (
            1.0
            + float(temp_sensitivity) * temp_intensity
            + float(flow_sensitivity) * (flow_norm - 0.5)
        )
        band_factor = np.clip(band_factor, float(band_factor_min), float(band_factor_max))

        minus_arr = float(minus_delta.loc[idx]) * band_factor
        plus_arr = float(plus_delta.loc[idx]) * band_factor

        t_min = np.clip(setpoint - minus_arr, zone_min_c, zone_max_c)
        t_max = np.clip(setpoint + plus_arr, zone_min_c, zone_max_c)

        invalid = t_min >= t_max
        if np.any(invalid):
            t_min = np.where(invalid, setpoint - 1.0, t_min)
            t_max = np.where(invalid, setpoint + 1.0, t_max)

        t_star_arr = np.round(setpoint, 3).astype(float)
        t_min_arr = np.round(t_min, 3).astype(float)
        t_max_arr = np.round(t_max, 3).astype(float)

        t_star_out.append(t_star_arr)
        t_min_out.append(t_min_arr)
        t_max_out.append(t_max_arr)
        comfort_zone_out.append((t_min_arr, t_max_arr))

    return pd.DataFrame(
        {
            "T_star": pd.Series(t_star_out, index=buildings.index, dtype=object),
            "T_min": pd.Series(t_min_out, index=buildings.index, dtype=object),
            "T_max": pd.Series(t_max_out, index=buildings.index, dtype=object),
            "comfort_zone": pd.Series(comfort_zone_out, index=buildings.index, dtype=object),
        },
        index=buildings.index,
    )
