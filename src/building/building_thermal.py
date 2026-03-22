"""
Building thermal helpers: temperature estimation and U-value assignment.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from src.building.tcl_model import calculate_COP
from src.building.user_behavior import set_comfort_zone

_YEAR_FIELDS = (
    "start_date",
    "opening_date",
    "year_of_construction",
    "construction",
)

_U_MIDPOINT = {
    "A": {"U_wall": 2.0, "U_roof": 1.6, "U_win": 5.5},
    "B": {"U_wall": 1.5, "U_roof": 1.2, "U_win": 4.0},
    "C": {"U_wall": 1.1, "U_roof": 0.8, "U_win": 2.7},
}

_WWR_BY_USE = {
    "residential": 0.25,
    "commercial": 0.40,
    "public": 0.35,
    "industrial": 0.20,
    "unknown": 0.30,
}

_TIE_PRIORITY = {"B": 0, "C": 1, "A": 2}

_AIR_EX_BASE_BY_USE = {
    "residential": 0.50,
    "commercial": 1.00,
    "public": 0.90,
    "industrial": 0.80,
    "unknown": 0.60,
}

_AIR_EX_FACTOR_BY_VINTAGE = {
    "A": 1.20,
    "B": 1.00,
    "C": 0.85,
}


def thermal_constants() -> Dict[str, float]:
    return {"Ca": 1.005*1000, "rhoa": 1.205}


def estimate_building_temp(
    macao_temperature: np.ndarray,
    macao_weather_station_location: np.ndarray,
    building_location: np.ndarray | list[float] | tuple[float, float],
) -> np.ndarray:
    """
    Port of Matlab `estimateBuildingTemp.m`.
    building_temp = macao_temperature @ weight, weight = d^2 / sum(d^2).
    """
    temperature = np.asarray(macao_temperature, dtype=float)
    station_location = np.asarray(macao_weather_station_location, dtype=float)
    location = np.asarray(building_location, dtype=float).reshape(2, 1)

    if station_location.shape[0] != 2:
        raise ValueError("macao_weather_station_location must have shape (2, n_stations).")
    if temperature.shape[1] != station_location.shape[1]:
        raise ValueError("macao_temperature columns must match number of weather stations.")

    diff_coordinate = location - station_location
    building_to_station_distance = np.sqrt(
        diff_coordinate[0, :] ** 2 + diff_coordinate[1, :] ** 2
    )
    distance_sq = building_to_station_distance**2
    distance_sq_sum = float(distance_sq.sum())
    if distance_sq_sum == 0:
        weight = np.full(distance_sq.shape, 1.0 / distance_sq.size)
    else:
        weight = distance_sq / distance_sq_sum
    return temperature @ weight


def estimate_building_temperature_series(
    macao_temperature: np.ndarray,
    macao_weather_station_location: np.ndarray,
    building_locations: np.ndarray | list[list[float]] | list[tuple[float, float]],
    buildings: gpd.GeoDataFrame,
    building_property: pd.DataFrame,
    *,
    output_col: str = "building_temperature_series",
) -> tuple[list[np.ndarray], gpd.GeoDataFrame, pd.DataFrame]:
    """
    Vectorized building-temperature interpolation for many buildings at once.

    Returns one temperature series per building, and writes it to both tables.
    """
    temperature = np.asarray(macao_temperature, dtype=float)
    station_location = np.asarray(macao_weather_station_location, dtype=float)
    locations = np.asarray(building_locations, dtype=float)

    if station_location.shape[0] != 2:
        raise ValueError("macao_weather_station_location must have shape (2, n_stations).")
    if temperature.shape[1] != station_location.shape[1]:
        raise ValueError("macao_temperature columns must match number of weather stations.")
    if locations.ndim != 2 or locations.shape[1] != 2:
        raise ValueError("building_locations must have shape (n_buildings, 2).")
    if int(locations.shape[0]) != int(len(buildings.index)):
        raise ValueError("building_locations rows must match number of buildings.")

    dx = locations[:, [0]] - station_location[[0], :]
    dy = locations[:, [1]] - station_location[[1], :]
    distance_sq = dx * dx + dy * dy
    distance_sq_sum = distance_sq.sum(axis=1, keepdims=True)

    weights = np.divide(
        distance_sq,
        distance_sq_sum,
        out=np.full_like(distance_sq, 1.0 / distance_sq.shape[1]),
        where=distance_sq_sum > 0,
    )
    building_temp_matrix = temperature @ weights.T
    building_temperature_series = [
        building_temp_matrix[:, i].copy() for i in range(building_temp_matrix.shape[1])
    ]

    bld = buildings.copy()
    prop = building_property.copy()

    temperature_series_obj = pd.Series(
        building_temperature_series,
        index=bld.index,
        dtype=object,
    )
    bld[output_col] = temperature_series_obj.to_numpy()
    prop[output_col] = temperature_series_obj.reindex(prop.index).to_numpy()

    return building_temperature_series, bld, prop


def to_temperature_scalar(value: object, default: float = 0.0) -> float:
    """Reduce scalar/array-like temperature input to one numeric value."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
        if arr.empty:
            return float(default)
        return float(arr.mean())
    scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(scalar):
        return float(default)
    return float(scalar)


def _to_temperature_array(value: object, default: float = 0.0) -> np.ndarray:
    """Normalize scalar/array-like input to a 1D numeric array."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
    else:
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        arr = np.array([scalar], dtype=float)
    if arr.size == 0:
        return np.array([float(default)], dtype=float)
    return np.where(np.isfinite(arr), arr, float(default)).astype(float)


def estimate_building_height_from_context(
    buildings: gpd.GeoDataFrame,
    *,
    area_col: str = "footprint_area_m2",
    k: int = 5,
    default_height_m: float = 10.0,
) -> pd.Series:
    """
    Estimate missing building heights from nearby buildings and footprint area.

    The estimator uses nearby buildings with known heights as references and
    combines spatial proximity with footprint-area similarity.
    """
    df = buildings.copy()

    if area_col in df.columns:
        area_series = pd.to_numeric(df[area_col], errors="coerce")
    else:
        metric_gdf = df
        if metric_gdf.crs is not None and metric_gdf.crs.is_geographic:
            metric_crs = metric_gdf.estimate_utm_crs()
            metric_gdf = metric_gdf.to_crs(metric_crs if metric_crs is not None else "EPSG:3857")
        area_series = pd.Series(metric_gdf.geometry.area, index=df.index, dtype=float)

    height_series = pd.Series(np.nan, index=df.index, dtype=float)
    for col in ["height_m", "Elevation", "height"]:
        if col in df.columns:
            height_series = height_series.fillna(pd.to_numeric(df[col], errors="coerce"))
    if "building:levels" in df.columns:
        levels = pd.to_numeric(df["building:levels"], errors="coerce")
        height_series = height_series.fillna(levels * 3.0)

    metric_gdf = df
    if metric_gdf.crs is not None and metric_gdf.crs.is_geographic:
        metric_crs = metric_gdf.estimate_utm_crs()
        metric_gdf = metric_gdf.to_crs(metric_crs if metric_crs is not None else "EPSG:3857")
    centroids = metric_gdf.geometry.centroid
    x = pd.to_numeric(centroids.x, errors="coerce")
    y = pd.to_numeric(centroids.y, errors="coerce")

    out = height_series.copy()
    valid_mask = np.isfinite(out) & (out > 0)
    missing_mask = ~(np.isfinite(out) & (out > 0))
    if not missing_mask.any():
        return out.fillna(default_height_m)
    if valid_mask.sum() == 0:
        return out.fillna(default_height_m)

    train_xy = np.column_stack([x[valid_mask].to_numpy(dtype=float), y[valid_mask].to_numpy(dtype=float)])
    train_h = out[valid_mask].to_numpy(dtype=float)
    train_area = area_series[valid_mask].to_numpy(dtype=float)
    train_log_area = np.log1p(np.clip(train_area, a_min=0.0, a_max=None))

    target_idx = out.index[missing_mask]
    target_xy = np.column_stack([x[missing_mask].to_numpy(dtype=float), y[missing_mask].to_numpy(dtype=float)])
    target_area = area_series[missing_mask].to_numpy(dtype=float)
    target_log_area = np.log1p(np.clip(target_area, a_min=0.0, a_max=None))

    eps = 1e-6
    neighbor_k = int(max(1, min(k, train_xy.shape[0])))
    predicted = np.full(target_xy.shape[0], float(default_height_m), dtype=float)

    # Batch KNN search to avoid Python loops over each target building.
    valid_target_xy = np.isfinite(target_xy).all(axis=1)
    if valid_target_xy.any():
        valid_rows = np.where(valid_target_xy)[0]
        chunk_size = 2048
        for chunk_start in range(0, valid_rows.size, chunk_size):
            chunk_rows = valid_rows[chunk_start : chunk_start + chunk_size]
            chunk_xy = target_xy[chunk_rows]

            dx = chunk_xy[:, [0]] - train_xy[None, :, 0]
            dy = chunk_xy[:, [1]] - train_xy[None, :, 1]
            dist = np.sqrt(dx * dx + dy * dy)

            if neighbor_k < train_xy.shape[0]:
                nbr_idx = np.argpartition(dist, kth=neighbor_k - 1, axis=1)[:, :neighbor_k]
                nbr_dist = np.take_along_axis(dist, nbr_idx, axis=1)
            else:
                nbr_idx = np.broadcast_to(
                    np.arange(train_xy.shape[0], dtype=int),
                    (chunk_xy.shape[0], train_xy.shape[0]),
                )
                nbr_dist = dist

            nbr_height = train_h[nbr_idx]
            nbr_log_area = train_log_area[nbr_idx]

            target_log = target_log_area[chunk_rows]
            target_log_col = target_log[:, None]
            area_gap = np.where(
                np.isfinite(target_log_col),
                np.abs(nbr_log_area - target_log_col),
                0.0,
            )

            weights = 1.0 / np.maximum(nbr_dist, eps)
            weights *= 1.0 / (1.0 + area_gap)
            weight_sum = weights.sum(axis=1)

            pred_chunk = np.divide(
                (weights * nbr_height).sum(axis=1),
                weight_sum,
                out=np.full(chunk_rows.shape[0], float(default_height_m), dtype=float),
                where=weight_sum > 0,
            )
            predicted[chunk_rows] = pred_chunk

    out.loc[target_idx] = predicted
    out = pd.to_numeric(out, errors="coerce").fillna(default_height_m)
    return out.clip(lower=1.0)


def prepare_building_thermal_state(
    macao_building: gpd.GeoDataFrame,
    building_temperature: pd.Series | np.ndarray | Sequence[Any],
    *,
    building_temperature_col: str = "temperature",
    osm_raw_columns: set[str] | None = None,
    height_knn_k: int = 5,
    height_default_m: float = 10.0,
    uvalue_knn_k: int = 7,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Build building-level thermal state and flexibility inputs.

    Returns the updated building GeoDataFrame together with a pure property
    DataFrame for debugging and downstream inspection.
    """
    bld = macao_building.copy()

    # 2.1 read precomputed building temperature
    building_temperature_series = pd.Series(building_temperature, index=bld.index)
    if len(building_temperature_series) != len(bld):
        raise ValueError("Length of building_temperature must match number of buildings.")
    bld[building_temperature_col] = building_temperature_series
    if building_temperature_col != "temperature":
        bld["temperature"] = bld[building_temperature_col]

    # Stage 1: geometry/U-value/air-exchange thermal parameters.
    bld, _ = prepare_building_thermal_parameters(
        bld,
        osm_raw_columns=osm_raw_columns,
        height_knn_k=height_knn_k,
        height_default_m=height_default_m,
        uvalue_knn_k=uvalue_knn_k,
    )

    # Stage 2: virtual storage related quantities from temperature.
    bld, property_table = compute_virtual_storage_related_state(
        bld,
        building_temperature_series,
        building_temperature_col=building_temperature_col,
        osm_raw_columns=osm_raw_columns,
    )
    return bld, property_table


def prepare_building_thermal_parameters(
    buildings: gpd.GeoDataFrame,
    *,
    osm_raw_columns: set[str] | None = None,
    height_knn_k: int = 5,
    height_default_m: float = 10.0,
    uvalue_knn_k: int = 7,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Prepare static thermal parameters before virtual-storage calculations."""
    bld = buildings.copy()

    # calculate volume: building height x footprint area
    if "Shape_Area" in bld.columns:
        footprint_area_series = pd.to_numeric(bld["Shape_Area"], errors="coerce")
    else:
        area_gdf = bld
        if area_gdf.crs is not None and area_gdf.crs.is_geographic:
            projected_crs = area_gdf.estimate_utm_crs()
            area_gdf = area_gdf.to_crs(projected_crs if projected_crs is not None else "EPSG:3857")
        footprint_area_series = area_gdf.geometry.area

    height_series = pd.Series(np.nan, index=bld.index, dtype=float)
    if "Elevation" in bld.columns:
        height_series = height_series.fillna(pd.to_numeric(bld["Elevation"], errors="coerce"))
    if "height" in bld.columns:
        height_series = height_series.fillna(pd.to_numeric(bld["height"], errors="coerce"))
    if "building:levels" in bld.columns:
        levels = pd.to_numeric(bld["building:levels"], errors="coerce")
        height_series = height_series.fillna(levels * 3.0)

    missing_height_mask = ~(np.isfinite(height_series) & (height_series > 0))
    if missing_height_mask.any():
        height_context_input = bld.copy()
        height_context_input["footprint_area_m2"] = footprint_area_series
        estimated_height = estimate_building_height_from_context(
            height_context_input,
            area_col="footprint_area_m2",
            k=height_knn_k,
            default_height_m=height_default_m,
        )
        height_series.loc[missing_height_mask] = estimated_height.loc[missing_height_mask]

    height_series = (
        pd.to_numeric(height_series, errors="coerce")
        .fillna(height_default_m)
        .clip(lower=1.0)
    )

    bld["footprint_area_m2"] = footprint_area_series
    bld["height_m"] = height_series
    bld["volume"] = height_series * footprint_area_series

    # estimate U value: envelope thermal transmittance U-value (W/m2K)
    bld = assign_uvalue_midpoint_knn(bld, k=uvalue_knn_k)
    bld = assign_air_exchange_rate(bld)

    property_table = _build_building_property_table(bld, osm_raw_columns=osm_raw_columns)
    return bld, property_table


def compute_virtual_storage_related_state(
    buildings: gpd.GeoDataFrame,
    building_temperature: pd.Series | np.ndarray | Sequence[Any],
    *,
    building_temperature_col: str = "temperature",
    osm_raw_columns: set[str] | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Recompute only temperature-dependent thermal quantities.

    This avoids rerunning height inference, U-value estimation, and air-exchange
    assignment when only the local building temperature changes.
    """
    bld = buildings.copy()
    building_temperature_series = pd.Series(building_temperature, index=bld.index)
    if len(building_temperature_series) != len(bld):
        raise ValueError("Length of building_temperature must match number of buildings.")

    # Keep per-building time series as vectors to avoid scalar collapse.
    building_temperature_vectors = pd.Series(
        [_to_temperature_array(v) for v in building_temperature_series.tolist()],
        index=bld.index,
        dtype=object,
    )
    bld[building_temperature_col] = building_temperature_vectors
    if building_temperature_col != "temperature":
        bld["temperature"] = building_temperature_vectors

    required_cols = ["volume", "Uvalue", "N_ex", "footprint_area_m2"]
    missing_cols = [col for col in required_cols if col not in bld.columns]
    if missing_cols:
        raise ValueError(
            "Missing precomputed thermal columns for fast update: "
            + ", ".join(missing_cols)
        )

    bld = _update_temperature_dependent_thermal_fields(
        bld,
        building_temperature_col=building_temperature_col,
    )
    property_table = _build_building_property_table(bld, osm_raw_columns=osm_raw_columns)
    return bld, property_table


def _update_temperature_dependent_thermal_fields(
    bld: gpd.GeoDataFrame,
    *,
    building_temperature_col: str = "temperature",
) -> gpd.GeoDataFrame:
    """Update COP, comfort zone, storage capacity, and charge/discharge power."""
    df = bld.copy()
    if building_temperature_col != "temperature" and building_temperature_col in df.columns:
        df["temperature"] = df[building_temperature_col]

    # 2.1 Correct COP for temperature-dependent performance (keep per-building arrays)
    df["COP"] = calculate_COP(df)

    # 2.2 Determine comfort zone
    comfort_df = set_comfort_zone(df)
    df[["T_star", "T_min", "T_max", "comfort_zone"]] = comfort_df[
        ["T_star", "T_min", "T_max", "comfort_zone"]
    ]

    # 2.3 virtual storage capacity
    constants = thermal_constants()
    ca = constants["Ca"]
    rhoa = constants["rhoa"]

    n_ex_series_h = pd.to_numeric(df["N_ex"], errors="coerce").fillna(0.0)
    n_ex_series_s = n_ex_series_h / 3600.0
    df["N_ex_h"] = n_ex_series_h
    df["N_ex_s"] = n_ex_series_s

    if {"A_wall", "A_roof", "A_win"}.issubset(df.columns):
        a_s_series = df[["A_wall", "A_roof", "A_win"]].sum(axis=1, min_count=1)
    else:
        a_s_series = pd.Series(np.nan, index=df.index, dtype=float)
    a_s_series = a_s_series.fillna(pd.to_numeric(df["footprint_area_m2"], errors="coerce"))
    df["A_s"] = a_s_series

    k_series = pd.to_numeric(df["Uvalue"], errors="coerce")
    v_series = pd.to_numeric(df["volume"], errors="coerce")

    energy_storage_capacity_list: list[np.ndarray] = []
    charging_power_list: list[np.ndarray] = []
    discharging_power_list: list[np.ndarray] = []
    duration_list: list[np.ndarray] = []
    t_star_list: list[np.ndarray] = []
    t_min_list: list[np.ndarray] = []
    t_max_list: list[np.ndarray] = []

    def _to_length_array(value: object, length: int, *, default: float) -> np.ndarray:
        arr = _to_temperature_array(value, default=default)
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

    for idx in df.index:
        temp_arr = _to_temperature_array(df.at[idx, "temperature"], default=30.0)
        n_t = max(1, int(temp_arr.size))

        cop_arr = _to_length_array(df.at[idx, "COP"], n_t, default=np.nan)
        denom = np.where(np.abs(cop_arr) > 0, cop_arr, np.nan)

        t_min_arr = _to_length_array(df.at[idx, "T_min"], n_t, default=22.0)
        t_max_arr = _to_length_array(df.at[idx, "T_max"], n_t, default=26.0)
        t_star_default = (t_min_arr + t_max_arr) / 2.0
        t_star_arr = _to_length_array(df.at[idx, "T_star"], n_t, default=float(np.nanmean(t_star_default)))

        delta_band = np.maximum(t_max_arr - t_min_arr, 0.0)
        delta_t_chr = np.maximum(t_max_arr - t_star_arr, 0.0)
        delta_t_dis = np.maximum(t_star_arr - t_min_arr, 0.0)

        k_val = float(pd.to_numeric(pd.Series([k_series.loc[idx]]), errors="coerce").iloc[0])
        v_val = float(pd.to_numeric(pd.Series([v_series.loc[idx]]), errors="coerce").iloc[0])
        a_s_val = float(pd.to_numeric(pd.Series([df.at[idx, "A_s"]]), errors="coerce").iloc[0])
        n_ex_s_val = float(pd.to_numeric(pd.Series([n_ex_series_s.loc[idx]]), errors="coerce").iloc[0])

        capacity_num = float(ca * rhoa) * v_val * delta_band
        ch_num = delta_t_chr * k_val * a_s_val + float(ca * rhoa) * v_val * delta_t_chr * n_ex_s_val
        dis_num = delta_t_dis * k_val * a_s_val + float(ca * rhoa) * v_val * delta_t_dis * n_ex_s_val

        energy_storage_arr = (capacity_num / denom).astype(float)
        charging_power_arr = (ch_num / denom).astype(float)
        discharging_power_arr = (dis_num / denom).astype(float)
        duration_arr = np.divide(
            energy_storage_arr,
            discharging_power_arr,
            out=np.zeros_like(energy_storage_arr, dtype=float),
            where=np.abs(discharging_power_arr) > 0,
        )
        duration_arr = duration_arr / 3600.0

        energy_storage_capacity_list.append(energy_storage_arr)
        charging_power_list.append(charging_power_arr)
        discharging_power_list.append(discharging_power_arr)
        duration_list.append(duration_arr.astype(float))
        t_star_list.append(t_star_arr.astype(float))
        t_min_list.append(t_min_arr.astype(float))
        t_max_list.append(t_max_arr.astype(float))

    df["T_star"] = pd.Series(t_star_list, index=df.index, dtype=object)
    df["T_min"] = pd.Series(t_min_list, index=df.index, dtype=object)
    df["T_max"] = pd.Series(t_max_list, index=df.index, dtype=object)

    df["energy_storage_capacity"] = pd.Series(energy_storage_capacity_list, index=df.index, dtype=object)
    df["charging_power"] = pd.Series(charging_power_list, index=df.index, dtype=object)

    # 2.5 discharging power
    df["discharging_power"] = pd.Series(discharging_power_list, index=df.index, dtype=object)
    df["duration"] = pd.Series(duration_list, index=df.index, dtype=object)
    return df


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _parse_year_from_row(row: pd.Series) -> int | None:
    for field in _YEAR_FIELDS:
        raw = row.get(field)
        if pd.isna(raw):
            continue
        match = re.search(r"(18|19|20)\d{2}", str(raw).strip())
        if not match:
            continue
        year = int(match.group(0))
        if 1800 <= year <= 2100:
            return year
    return None


def _year_to_band(year: int) -> str:
    if year < 1995:
        return "A"
    if year <= 2010:
        return "B"
    return "C"


def _use_to_wwr(use_category: Any) -> float:
    key = str(use_category).strip().lower() if pd.notna(use_category) else "unknown"
    return _WWR_BY_USE.get(key, _WWR_BY_USE["unknown"])


def _prepare_knn_feature_row(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            _to_float(row.get("height_m")),
            _to_float(row.get("building:levels")),
            _to_float(row.get("footprint_area_m2")),
            _to_float(row.get("volume")),
        ],
        dtype=float,
    )


def _compute_envelope_areas(
    row: pd.Series,
    metric_geometry: Any,
) -> tuple[float, float, float, float, float]:
    footprint_area = _to_float(row.get("footprint_area_m2"))
    if np.isnan(footprint_area) and metric_geometry is not None and not metric_geometry.is_empty:
        footprint_area = _to_float(metric_geometry.area)

    height_m = _to_float(row.get("height_m"))
    if np.isnan(height_m):
        levels = _to_float(row.get("building:levels"))
        if not np.isnan(levels):
            height_m = levels * 3.0
    if np.isnan(height_m):
        height_m = 10.0

    perimeter_m = np.nan
    if metric_geometry is not None and not metric_geometry.is_empty:
        perimeter_m = _to_float(metric_geometry.length)
    if np.isnan(perimeter_m) and np.isfinite(footprint_area) and footprint_area > 0:
        perimeter_m = 4.0 * np.sqrt(footprint_area)

    if np.isnan(footprint_area):
        return footprint_area, perimeter_m, np.nan, np.nan, np.nan

    a_roof = footprint_area
    if np.isnan(perimeter_m):
        return footprint_area, perimeter_m, np.nan, a_roof, np.nan

    a_facade = perimeter_m * height_m
    wwr = _use_to_wwr(row.get("use_category"))
    a_win = a_facade * wwr
    a_wall = a_facade * (1.0 - wwr)
    return footprint_area, perimeter_m, a_wall, a_roof, a_win


def _knn_predict_band(train_df: pd.DataFrame, target_row: pd.Series, k: int) -> str:
    if train_df.empty:
        return "B"

    target_use = str(target_row.get("use_category", "unknown")).strip().lower()
    same_use = train_df[
        train_df["use_category"].fillna("unknown").astype(str).str.lower() == target_use
    ]
    candidate_df = same_use if len(same_use) >= max(3, min(k, len(train_df))) else train_df

    feature_cols = ["height_m", "building:levels", "footprint_area_m2", "volume"]
    x_train = np.column_stack(
        [pd.to_numeric(candidate_df[col], errors="coerce").to_numpy(dtype=float) for col in feature_cols]
    )
    y_train = candidate_df["vintage_band"].astype(str).to_numpy()
    x_target = _prepare_knn_feature_row(target_row)

    x_train_scaled = np.full_like(x_train, np.nan, dtype=float)
    x_target_scaled = np.full_like(x_target, np.nan, dtype=float)
    has_feature = False
    for col in range(x_train.shape[1]):
        vals = x_train[:, col]
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue
        mean = float(np.mean(vals[finite]))
        std = float(np.std(vals[finite]))
        if std <= 0:
            continue
        has_feature = True
        x_train_scaled[finite, col] = (vals[finite] - mean) / std
        if np.isfinite(x_target[col]):
            x_target_scaled[col] = (x_target[col] - mean) / std
    if not has_feature:
        return "B"

    distances: list[tuple[float, str]] = []
    for i in range(x_train_scaled.shape[0]):
        cand = x_train_scaled[i, :]
        overlap = np.isfinite(cand) & np.isfinite(x_target_scaled)
        if not overlap.any():
            continue
        diff = cand[overlap] - x_target_scaled[overlap]
        dist = float(np.sqrt(np.mean(diff**2)))
        distances.append((dist, y_train[i]))
    if not distances:
        return "B"

    distances.sort(key=lambda x: x[0])
    top = distances[: min(k, len(distances))]
    counts: dict[str, int] = {}
    min_d: dict[str, float] = {}
    for dist, band in top:
        counts[band] = counts.get(band, 0) + 1
        min_d[band] = min(min_d.get(band, np.inf), dist)

    max_votes = max(counts.values())
    tied = [b for b, c in counts.items() if c == max_votes]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda b: (min_d.get(b, np.inf), _TIE_PRIORITY.get(b, 99)))
    return tied[0]


def assign_uvalue_midpoint_knn(
    buildings: gpd.GeoDataFrame,
    *,
    k: int = 7,
) -> gpd.GeoDataFrame:
    """
    Deterministic U-value assignment (no sampling):
    1) direct year -> vintage band
    2) missing year -> KNN vintage imputation
    3) band midpoint -> U_wall/U_roof/U_win
    4) area-weighted equivalent Uvalue
    """
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain 'geometry' column.")

    df = buildings.copy()
    if "use_category" not in df.columns:
        df["use_category"] = "unknown"

    metric_gdf = df
    if metric_gdf.crs is not None and metric_gdf.crs.is_geographic:
        utm = metric_gdf.estimate_utm_crs()
        metric_gdf = metric_gdf.to_crs(utm if utm is not None else "EPSG:3857")

    fallback_area = pd.to_numeric(metric_gdf.geometry.area, errors="coerce")
    if "footprint_area_m2" in df.columns:
        df["footprint_area_m2"] = pd.to_numeric(df["footprint_area_m2"], errors="coerce").fillna(
            fallback_area
        )
    else:
        df["footprint_area_m2"] = fallback_area

    year_candidates: list[pd.Series] = []
    for field in _YEAR_FIELDS:
        if field not in df.columns:
            continue
        extracted = (
            df[field]
            .astype(str)
            .str.extract(r"((?:18|19|20)\d{2})", expand=False)
        )
        year_num = pd.to_numeric(extracted, errors="coerce")
        year_num = year_num.where((year_num >= 1800) & (year_num <= 2100))
        year_candidates.append(year_num)

    if year_candidates:
        vintage_year = pd.concat(year_candidates, axis=1).bfill(axis=1).iloc[:, 0]
    else:
        vintage_year = pd.Series(np.nan, index=df.index, dtype=float)

    vintage_band = pd.Series("", index=df.index, dtype=object)
    vintage_band = vintage_band.where(~(vintage_year < 1995), "A")
    vintage_band = vintage_band.where(~((vintage_year >= 1995) & (vintage_year <= 2010)), "B")
    vintage_band = vintage_band.where(~(vintage_year > 2010), "C")
    vintage_source = pd.Series("", index=df.index, dtype=object)
    vintage_source.loc[vintage_year.notna()] = "direct_year"

    df["vintage_year"] = pd.to_numeric(vintage_year, errors="coerce").astype("Int64")
    df["vintage_band"] = vintage_band
    df["vintage_source"] = vintage_source

    train_df = df[df["vintage_band"].isin(["A", "B", "C"])].copy()
    for idx in df.index[df["vintage_band"] == ""]:
        if len(train_df) < max(3, min(k, len(df))):
            df.at[idx, "vintage_band"] = "B"
            df.at[idx, "vintage_source"] = "fallback_B"
        else:
            pred = _knn_predict_band(train_df, df.loc[idx], k)
            df.at[idx, "vintage_band"] = pred
            df.at[idx, "vintage_source"] = "knn_imputed"

    band_series = df["vintage_band"].where(df["vintage_band"].isin(["A", "B", "C"]), "B")
    df["U_wall"] = band_series.map(lambda b: float(_U_MIDPOINT[b]["U_wall"]))
    df["U_roof"] = band_series.map(lambda b: float(_U_MIDPOINT[b]["U_roof"]))
    df["U_win"] = band_series.map(lambda b: float(_U_MIDPOINT[b]["U_win"]))

    footprint = pd.to_numeric(df["footprint_area_m2"], errors="coerce")
    geom_area = pd.to_numeric(metric_gdf.geometry.area, errors="coerce")
    footprint = footprint.fillna(geom_area)
    df["footprint_area_m2"] = footprint

    if "height_m" in df.columns:
        height = pd.to_numeric(df["height_m"], errors="coerce")
    else:
        height = pd.Series(np.nan, index=df.index, dtype=float)
    if "building:levels" in df.columns:
        levels = pd.to_numeric(df["building:levels"], errors="coerce")
        height = height.fillna(levels * 3.0)
    height = height.fillna(10.0)

    perimeter = pd.to_numeric(metric_gdf.geometry.length, errors="coerce")
    perimeter_fallback = pd.Series(
        4.0 * np.sqrt(np.clip(footprint.to_numpy(dtype=float), a_min=0.0, a_max=None)),
        index=df.index,
        dtype=float,
    )
    perimeter = perimeter.where(np.isfinite(perimeter), perimeter_fallback)
    perimeter = pd.Series(perimeter, index=df.index, dtype=float)

    use_key = (
        df.get("use_category", pd.Series("unknown", index=df.index))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    wwr = use_key.map(_WWR_BY_USE).fillna(_WWR_BY_USE["unknown"]).to_numpy(dtype=float)

    fp_np = footprint.to_numpy(dtype=float)
    pm_np = perimeter.to_numpy(dtype=float)
    h_np = height.to_numpy(dtype=float)

    a_roof_np = fp_np.copy()
    a_facade_np = pm_np * h_np
    a_win_np = a_facade_np * wwr
    a_wall_np = a_facade_np * (1.0 - wwr)

    invalid_fp = ~np.isfinite(fp_np)
    invalid_pm = ~np.isfinite(pm_np)
    a_roof_np = np.where(invalid_fp, np.nan, a_roof_np)
    a_wall_np = np.where(invalid_fp | invalid_pm, np.nan, a_wall_np)
    a_win_np = np.where(invalid_fp | invalid_pm, np.nan, a_win_np)

    df["perimeter_m"] = pm_np
    df["A_wall"] = a_wall_np
    df["A_roof"] = a_roof_np
    df["A_win"] = a_win_np

    uw_np = pd.to_numeric(df["U_wall"], errors="coerce").to_numpy(dtype=float)
    ur_np = pd.to_numeric(df["U_roof"], errors="coerce").to_numpy(dtype=float)
    uwi_np = pd.to_numeric(df["U_win"], errors="coerce").to_numpy(dtype=float)
    total_np = a_wall_np + a_roof_np + a_win_np

    area_weighted = (uw_np * a_wall_np + ur_np * a_roof_np + uwi_np * a_win_np) / total_np
    fallback_weighted = 0.5 * uw_np + 0.2 * ur_np + 0.3 * uwi_np
    use_area_weighted = np.isfinite(total_np) & (total_np > 0)

    df["Uvalue"] = np.where(use_area_weighted, area_weighted, fallback_weighted)
    df["Uvalue_source"] = np.where(use_area_weighted, "area_weighted", "fallback_weighted")

    return df


def assign_air_exchange_rate(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Deterministic air-exchange assignment from building type and vintage band.
    N_ex = base(use_category) * factor(vintage_band)
    """
    df = buildings.copy()
    if "use_category" not in df.columns:
        df["use_category"] = "unknown"

    if "vintage_band" not in df.columns:
        bands = []
        for _, row in df.iterrows():
            year = _parse_year_from_row(row)
            bands.append(_year_to_band(year) if year is not None else "B")
        df["vintage_band"] = bands
    else:
        vintage_band = (
            df["vintage_band"]
            .astype(str)
            .str.strip()
            .replace({"": "B", "nan": "B", "None": "B"})
        )
        df["vintage_band"] = vintage_band.where(vintage_band.isin(["A", "B", "C"]), "B")

    use_key = (
        df["use_category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    use_key = use_key.where(use_key.isin(list(_AIR_EX_BASE_BY_USE.keys())), "unknown")
    base_series = use_key.map(_AIR_EX_BASE_BY_USE)
    factor_series = df["vintage_band"].map(_AIR_EX_FACTOR_BY_VINTAGE).fillna(1.0)

    df["N_ex"] = (base_series * factor_series).astype(float)
    df["N_ex_source"] = "type_vintage_rule"
    return df


def _build_building_property_table(
    buildings: pd.DataFrame,
    *,
    osm_raw_columns: set[str] | None = None,
) -> pd.DataFrame:
    if osm_raw_columns is None:
        derived_cols = [c for c in buildings.columns if c != "geometry"]
    else:
        derived_cols = [c for c in buildings.columns if c != "geometry" and c not in osm_raw_columns]
    return pd.DataFrame(buildings[derived_cols]).copy()


def get_thermal_params(
    volume: float,
    U_value: float,
    area: float,
    n_air_exchange: float = 0.5,
    **kwargs: Any,
) -> Dict[str, float]:
    c = thermal_constants()
    return {
        "Ca": c["Ca"],
        "rhoa": c["rhoa"],
        "U": U_value,
        "A": area,
        "V": volume,
        "N_ex": n_air_exchange,
        **kwargs,
    }
