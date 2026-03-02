"""
Building thermal helpers: temperature estimation and U-value assignment.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd

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
    return {"Ca": 1.005, "rhoa": 1.205}


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

    x_train = np.vstack(candidate_df["knn_features"].to_numpy())
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

    years: list[float] = []
    bands: list[str] = []
    sources: list[str] = []
    features: list[np.ndarray] = []
    for _, row in df.iterrows():
        y = _parse_year_from_row(row)
        years.append(np.nan if y is None else float(y))
        if y is None:
            bands.append("")
            sources.append("")
        else:
            bands.append(_year_to_band(y))
            sources.append("direct_year")
        features.append(_prepare_knn_feature_row(row))

    df["vintage_year"] = pd.array(years, dtype="Int64")
    df["vintage_band"] = bands
    df["vintage_source"] = sources
    df["knn_features"] = features

    train_df = df[df["vintage_band"].isin(["A", "B", "C"])].copy()
    for idx in df.index[df["vintage_band"] == ""]:
        if len(train_df) < max(3, min(k, len(df))):
            df.at[idx, "vintage_band"] = "B"
            df.at[idx, "vintage_source"] = "fallback_B"
        else:
            pred = _knn_predict_band(train_df, df.loc[idx], k)
            df.at[idx, "vintage_band"] = pred
            df.at[idx, "vintage_source"] = "knn_imputed"

    u_wall: list[float] = []
    u_roof: list[float] = []
    u_win: list[float] = []
    perim: list[float] = []
    a_wall: list[float] = []
    a_roof: list[float] = []
    a_win: list[float] = []
    ueq: list[float] = []
    ueq_source: list[str] = []

    for idx, row in df.iterrows():
        band = row.get("vintage_band", "B")
        if band not in _U_MIDPOINT:
            band = "B"
        u = _U_MIDPOINT[band]
        uw = float(u["U_wall"])
        ur = float(u["U_roof"])
        uwi = float(u["U_win"])
        u_wall.append(uw)
        u_roof.append(ur)
        u_win.append(uwi)

        mgeom = metric_gdf.loc[idx, "geometry"] if idx in metric_gdf.index else None
        fp, pm, aw, ar, awin = _compute_envelope_areas(row, mgeom)
        if np.isnan(df.at[idx, "footprint_area_m2"]) and np.isfinite(fp):
            df.at[idx, "footprint_area_m2"] = fp

        perim.append(pm)
        a_wall.append(aw)
        a_roof.append(ar)
        a_win.append(awin)

        total = aw + ar + awin
        if np.isfinite(total) and total > 0:
            val = (uw * aw + ur * ar + uwi * awin) / total
            src = "area_weighted"
        else:
            val = 0.5 * uw + 0.2 * ur + 0.3 * uwi
            src = "fallback_weighted"
        ueq.append(float(val))
        ueq_source.append(src)

    df["U_wall"] = u_wall
    df["U_roof"] = u_roof
    df["U_win"] = u_win
    df["perimeter_m"] = perim
    df["A_wall"] = a_wall
    df["A_roof"] = a_roof
    df["A_win"] = a_win
    df["Uvalue"] = ueq
    df["Uvalue_source"] = ueq_source

    return df.drop(columns=["knn_features"])


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
        df["vintage_band"] = (
            df["vintage_band"]
            .astype(str)
            .str.strip()
            .replace({"": "B", "nan": "B", "None": "B"})
            .where(lambda s: s.isin(["A", "B", "C"]), "B")
        )

    use_key = (
        df["use_category"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .where(
            lambda s: s.isin(_AIR_EX_BASE_BY_USE.keys()),
            "unknown",
        )
    )
    base_series = use_key.map(_AIR_EX_BASE_BY_USE)
    factor_series = df["vintage_band"].map(_AIR_EX_FACTOR_BY_VINTAGE).fillna(1.0)

    df["N_ex"] = (base_series * factor_series).astype(float)
    df["N_ex_source"] = "type_vintage_rule"
    return df


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
