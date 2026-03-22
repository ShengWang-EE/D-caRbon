from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


IMPORTANT_BUILDING_PROPERTY_COLUMNS: tuple[str, ...] = (
    "osmid",
    "osm_id",
    "name",
    "name:en",
    "building",
    "building:use",
    "amenity",
    "shop",
    "office",
    "landuse",
    "use_category",
    "footprint_area_m2",
    "height_m",
    "floors",
    "volume",
    "Uvalue",
    "A_s",
    "N_ex",
    "T_star",
    "temperature",
    "COP",
    "energy_storage_capacity",
    "charging_power",
    "discharging_power",
    "duration",
    "discharging_power_thermal",
    "ac_electric_power_comfort",
    "T_local_i",
    "delta_t_ac_i",
    "delta_t_ac_z",
    "delta_t_uhi",
    "grid_id",
)


def compact_building_property(
    property_table: pd.DataFrame,
    *,
    extra_keep: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Keep only key columns used by the pipeline/debugging views."""
    keep_cols = list(IMPORTANT_BUILDING_PROPERTY_COLUMNS)
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c not in keep_cols])
    selected = [c for c in keep_cols if c in property_table.columns]
    return property_table.loc[:, selected].copy()


def prepare_building_locations(
    buildings: gpd.GeoDataFrame,
    building_property: pd.DataFrame | None = None,
    *,
    x_col: str = "xBuilding",
    y_col: str = "yBuilding",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, np.ndarray]:
    """Build Nx2 building location array and persist numeric x/y to both tables."""
    bld = buildings.copy()

    has_x = x_col in bld.columns
    has_y = y_col in bld.columns

    x_raw = pd.to_numeric(bld[x_col], errors="coerce") if has_x else pd.Series(np.nan, index=bld.index)
    y_raw = pd.to_numeric(bld[y_col], errors="coerce") if has_y else pd.Series(np.nan, index=bld.index)

    need_centroid = x_raw.isna().any() or y_raw.isna().any() or (not has_x) or (not has_y)
    if need_centroid:
        if "geometry" not in bld.columns:
            raise KeyError("Cannot infer coordinates: missing x/y columns and geometry.")
        centroids = bld.geometry.centroid
        cx = pd.to_numeric(centroids.x, errors="coerce")
        cy = pd.to_numeric(centroids.y, errors="coerce")
        x_final = x_raw.fillna(cx)
        y_final = y_raw.fillna(cy)
    else:
        x_final = x_raw
        y_final = y_raw

    bld[x_col] = x_final.to_numpy(dtype=float)
    bld[y_col] = y_final.to_numpy(dtype=float)

    if building_property is None:
        cols = [c for c in bld.columns if c != "geometry"]
        prop = pd.DataFrame(bld[cols]).copy()
    else:
        prop = building_property.copy()

    prop[x_col] = x_final.reindex(prop.index).to_numpy(dtype=float)
    prop[y_col] = y_final.reindex(prop.index).to_numpy(dtype=float)

    locations = np.column_stack([x_final.to_numpy(dtype=float), y_final.to_numpy(dtype=float)])
    return bld, prop, locations


def to_numeric_scalar(value: object, default: float = 0.0) -> float:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
        if arr.empty:
            return float(default)
        return float(arr.mean())
    scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(scalar):
        return float(default)
    return float(scalar)


def to_numeric_array(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
    else:
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        arr = np.array([scalar], dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr


def has_finite_points(value: object) -> bool:
    arr = to_numeric_array(value)
    if arr.size == 0:
        return False
    return bool(np.isfinite(arr).any())


def sanitize_series_for_plot(value: object) -> np.ndarray:
    """Convert to numeric 1D array and fill gaps so lines are visible in plots."""
    arr = to_numeric_array(value).astype(float, copy=False)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if not np.isfinite(arr).any():
        return arr
    s = pd.Series(arr)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.fillna(s.mean())
    return s.to_numpy(dtype=float)


def to_length_array(value: object, length: int, default: float = 0.0) -> np.ndarray:
    arr = to_numeric_array(value).astype(float, copy=False)
    if arr.size == 0:
        return np.full(length, float(default), dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
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


def aggregate_power_series_mw(series: pd.Series) -> np.ndarray:
    """Aggregate per-building power vectors into a single MW time series."""
    vectors: list[np.ndarray] = []
    for value in series:
        arr = to_numeric_array(value).astype(float, copy=False)
        if arr.size == 0:
            arr = np.array([0.0], dtype=float)
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(float)
        vectors.append(arr)

    if not vectors:
        return np.array([], dtype=float)

    n_t = max(int(v.size) for v in vectors)
    stacked = np.vstack([to_length_array(v, n_t, default=0.0) for v in vectors])
    return stacked.sum(axis=0) / 1e6


def compute_ac_power_at_comfort_setpoint(
    buildings: gpd.GeoDataFrame,
    *,
    temperature_col: str = "temperature",
    comfort_col: str = "T_star",
    cop_col: str = "COP",
    uvalue_col: str = "Uvalue",
    envelope_area_col: str = "A_s",
    volume_col: str = "volume",
    air_exchange_col: str = "N_ex",
    footprint_area_col: str = "footprint_area_m2",
    internal_gain_density_col: str = "epsilon_b",
    solar_irradiance_col: str = "I_solar",
    solar_area_col: str = "A_sol",
    output_dir: Path | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Compute cooling/electric AC power at comfort setpoint for each building."""
    bld = buildings.copy()

    ca = 1005.0
    rhoa = 1.205

    q_cool_out: list[np.ndarray] = []
    p_ac_out: list[np.ndarray] = []

    for idx in bld.index:
        t_a = to_numeric_array(bld.at[idx, temperature_col]).astype(float)
        n_t = max(1, int(t_a.size))

        t_comf = to_numeric_scalar(bld.at[idx, comfort_col], default=24.0)
        delta_t = np.maximum(t_a - float(t_comf), 0.0)

        k_b = to_numeric_scalar(bld.at[idx, uvalue_col], default=0.0)
        a_s = to_numeric_scalar(bld.at[idx, envelope_area_col], default=0.0)
        v_b = to_numeric_scalar(bld.at[idx, volume_col], default=0.0)
        n_ex_h = to_numeric_scalar(bld.at[idx, air_exchange_col], default=0.0)
        n_ex_s = max(float(n_ex_h), 0.0) / 3600.0

        a_b = to_numeric_scalar(bld.at[idx, footprint_area_col], default=0.0)
        eps_b = (
            to_numeric_scalar(bld.at[idx, internal_gain_density_col], default=0.0)
            if internal_gain_density_col in bld.columns
            else 0.0
        )
        a_sol = (
            to_numeric_scalar(bld.at[idx, solar_area_col], default=a_b)
            if solar_area_col in bld.columns
            else a_b
        )
        i_solar = (
            to_length_array(bld.at[idx, solar_irradiance_col], n_t, default=0.0)
            if solar_irradiance_col in bld.columns
            else np.zeros(n_t, dtype=float)
        )

        envelope_load = delta_t * float(k_b) * float(a_s)
        ventilation_load = ca * rhoa * float(v_b) * delta_t * float(n_ex_s)
        internal_load = np.full(n_t, float(eps_b) * float(a_b), dtype=float)
        solar_load = i_solar * float(a_sol)

        q_cool = envelope_load + ventilation_load + internal_load + solar_load
        q_cool = np.maximum(np.where(np.isfinite(q_cool), q_cool, 0.0), 0.0)

        cop_arr = to_length_array(bld.at[idx, cop_col], n_t, default=np.nan)
        p_ac = np.divide(q_cool, cop_arr, out=np.zeros(n_t, dtype=float), where=cop_arr > 0)
        p_ac = np.maximum(np.where(np.isfinite(p_ac), p_ac, 0.0), 0.0)

        q_cool_out.append(q_cool.astype(float))
        p_ac_out.append(p_ac.astype(float))

    bld["discharging_power_thermal"] = pd.Series(q_cool_out, index=bld.index, dtype=object)
    bld["ac_electric_power_comfort"] = pd.Series(p_ac_out, index=bld.index, dtype=object)

    cols = [c for c in bld.columns if c != "geometry"]
    building_property = pd.DataFrame(bld[cols]).copy()

    if output_dir is not None:
        output_path = output_dir / "macao_building_property_ac_power.csv"
        building_property.to_csv(output_path, index=False)
        print(f"AC power building property saved to: {output_path}")

    return bld, building_property


def vector_delta_series(before: pd.Series, after: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    before_out: list[np.ndarray] = []
    after_out: list[np.ndarray] = []
    delta_out: list[np.ndarray] = []
    for idx in before.index:
        b = to_numeric_array(before.loc[idx]).astype(float)
        a = to_numeric_array(after.loc[idx]).astype(float)
        n = max(int(b.size), int(a.size), 1)
        b_vec = to_length_array(b, n, default=0.0)
        a_vec = to_length_array(a, n, default=0.0)
        before_out.append(b_vec)
        after_out.append(a_vec)
        delta_out.append(a_vec - b_vec)
    return (
        pd.Series(before_out, index=before.index, dtype=object),
        pd.Series(after_out, index=after.index, dtype=object),
        pd.Series(delta_out, index=after.index, dtype=object),
    )


def recompute_after_microclimate(
    baseline_building: gpd.GeoDataFrame,
    macao_building_updated: gpd.GeoDataFrame,
    macao_building_property_updated: pd.DataFrame,
    *,
    data_dir: Path,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    """Summarize before/after flexibility impacts using precomputed updated states."""

    # Keep before/after/delta as vectors for all key quantities.
    temperature_before, temperature_after, temperature_delta = vector_delta_series(
        baseline_building["temperature"],
        macao_building_updated["temperature"],
    )
    macao_building_property_updated["temperature_before"] = temperature_before
    macao_building_property_updated["temperature_after"] = temperature_after
    macao_building_property_updated["temperature_delta"] = temperature_delta

    cop_before, cop_after, cop_delta = vector_delta_series(
        baseline_building["COP"],
        macao_building_updated["COP"],
    )
    macao_building_property_updated["COP_before"] = cop_before
    macao_building_property_updated["COP_after"] = cop_after
    macao_building_property_updated["COP_delta"] = cop_delta

    cap_before, cap_after, cap_delta = vector_delta_series(
        baseline_building["energy_storage_capacity"],
        macao_building_updated["energy_storage_capacity"],
    )
    macao_building_property_updated["energy_storage_capacity_before"] = cap_before
    macao_building_property_updated["energy_storage_capacity_after"] = cap_after
    macao_building_property_updated["energy_storage_capacity_delta"] = cap_delta

    ch_before, ch_after, ch_delta = vector_delta_series(
        baseline_building["charging_power"],
        macao_building_updated["charging_power"],
    )
    macao_building_property_updated["charging_power_before"] = ch_before
    macao_building_property_updated["charging_power_after"] = ch_after
    macao_building_property_updated["charging_power_delta"] = ch_delta

    dis_before, dis_after, dis_delta = vector_delta_series(
        baseline_building["discharging_power"],
        macao_building_updated["discharging_power"],
    )
    macao_building_property_updated["discharging_power_before"] = dis_before
    macao_building_property_updated["discharging_power_after"] = dis_after
    macao_building_property_updated["discharging_power_delta"] = dis_delta

    for col in ["T_local_i", "delta_t_ac_z", "delta_t_uhi", "grid_id"]:
        if col in macao_building_updated.columns:
            macao_building_property_updated[col] = macao_building_updated[col].to_numpy()

    macao_building_property_comparison = macao_building_property_updated.copy()

    microclimate_impact_summary = {
        "temperature_delta_mean_c": macao_building_property_updated["temperature_delta"].apply(to_numeric_scalar).mean(),
        "cop_delta_mean": macao_building_property_updated["COP_delta"].apply(to_numeric_scalar).mean(),
        "energy_storage_capacity_delta_mwh": macao_building_property_updated["energy_storage_capacity_delta"].apply(to_numeric_scalar).sum() / 3.6e9,
        "charging_power_delta_mw": macao_building_property_updated["charging_power_delta"].apply(to_numeric_scalar).sum() / 1e6,
        "discharging_power_delta_mw": macao_building_property_updated["discharging_power_delta"].apply(to_numeric_scalar).sum() / 1e6,
    }

    def _aggregate_vector_series(series: pd.Series, scale: float) -> np.ndarray:
        vectors = [to_numeric_array(v).astype(float) for v in series]
        if not vectors:
            return np.array([], dtype=float)
        n_t = max((int(v.size) for v in vectors), default=0)
        if n_t <= 0:
            return np.array([], dtype=float)
        stacked = np.vstack([to_length_array(v, n_t, default=0.0) for v in vectors])
        return stacked.sum(axis=0) / float(scale)

    aggregated_capacity_mwh = _aggregate_vector_series(
        macao_building_property_updated["energy_storage_capacity"],
        3.6e9,
    )
    aggregated_discharging_mw = _aggregate_vector_series(
        macao_building_property_updated["discharging_power"],
        1e6,
    )

    aggregated_metrics_cache_path = data_dir / "macao_aggregated_flexibility_cache.csv"
    n_t_cache = max(int(aggregated_capacity_mwh.size), int(aggregated_discharging_mw.size), 0)
    if n_t_cache > 0:
        cap_vec = to_length_array(aggregated_capacity_mwh, n_t_cache, default=0.0)
        dis_vec = to_length_array(aggregated_discharging_mw, n_t_cache, default=0.0)
        pd.DataFrame(
            {
                "t_idx": np.arange(n_t_cache, dtype=int),
                "aggregated_capacity_mwh": cap_vec,
                "aggregated_discharging_mw": dis_vec,
            }
        ).to_csv(aggregated_metrics_cache_path, index=False)
    else:
        pd.DataFrame(columns=["t_idx", "aggregated_capacity_mwh", "aggregated_discharging_mw"]).to_csv(
            aggregated_metrics_cache_path,
            index=False,
        )

    return (
        macao_building_updated,
        macao_building_property_updated,
        macao_building_property_comparison,
        microclimate_impact_summary,
        aggregated_capacity_mwh,
        aggregated_discharging_mw,
    )
