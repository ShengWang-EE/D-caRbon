"""Micro-meteorology utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


def _to_numeric_scalar(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
        if arr.empty:
            return float(default)
        return float(arr.mean())
    scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(scalar):
        return float(default)
    return float(scalar)


def _to_numeric_array(value: Any, default: float = 0.0) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
    else:
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        arr = np.array([scalar], dtype=float)
    if arr.size == 0:
        return np.array([float(default)], dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return arr


def _infer_vector_length(series: pd.Series) -> int:
    for value in series:
        arr = _to_numeric_array(value, default=np.nan)
        if arr.size > 1:
            return int(arr.size)
    return 1


def _to_fixed_length_array(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    arr = _to_numeric_array(value, default=default)
    if length <= 1:
        return np.array([float(arr[0])], dtype=float)
    if arr.size == length:
        return arr.astype(float)
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=float)
    if arr.size > length:
        return arr[:length].astype(float)
    out = np.empty(length, dtype=float)
    out[: arr.size] = arr
    out[arr.size :] = float(arr[-1])
    return out


def canopy_temperature_delta(geometry: Any, weather: Any, **kwargs: Any) -> Any:
    """Placeholder canopy-interface model."""
    return None


def correct_temperature_for_zone(T_background: Any, delta: Any, **kwargs: Any) -> Any:
    """Placeholder corrected local temperature model."""
    return T_background


def run_ac_micro_meteorology_pipeline(
    buildings: gpd.GeoDataFrame,
    *,
    building_temperature_col: str = "temperature",
    osm_raw_columns: set[str] | None = None,
    output_dir: Path | None = None,
    grid_size_m: float = 250.0,
    grid_id_col: str = "grid_id",
    place_name: str = "Macau, China",
    clip_to_boundary: bool = True,
    window_start_hour: float = 14.0,
    window_end_hour: float = 18.0,
    write_step1_files: bool = True,
    write_step2_files: bool = False,
    reuse_cached_mapping: bool = True,
    k_by_zone: dict[str, float] | None = None,
    apply_building_idw: bool = True,
    idw_neighbors: int = 4,
    idw_power: float = 2.0,
    idw_chunk_size: int = 2000,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute micro-meteorology Step 1 to Step 9b in one function call.

    Returns:
    1) buildings with grid/micro-meteorology outputs
    2) grid_stats with grid-level outputs
    3) building_property table (non-geometry, optionally excluding raw OSM columns)
    """
    if k_by_zone is None:
        k_by_zone = {
            "open_area": 0.0020,
            "normal_urban": 0.0035,
            "dense_urban": 0.0050,
        }

    # Step 1. Discretize space and build building-to-grid mapping.
    # Reuse cached mapping files when available to skip repeated spatial joins.
    bld, grid_stats, _ = build_spatial_grid_mapping(
        buildings,
        grid_size_m=grid_size_m,
        grid_id_col=grid_id_col,
        output_dir=output_dir,
        write_csv=write_step1_files,
        place_name=place_name,
        clip_to_boundary=clip_to_boundary,
        reuse_cached_mapping=reuse_cached_mapping,
    )

    # Step 2. Prepare background meteorology and static UHI correction.
    grid_stats, _, _ = prepare_background_meteorology_uhi(
        grid_stats,
        bld,
        building_temperature_col=building_temperature_col,
        grid_id_col=grid_id_col,
        zone_type_col="zone_type",
        delta_t_uhi_col="delta_t_uhi",
        t_met_grid_col="T_met_z",
        t_base_col="T_base_z",
        window_start_hour=window_start_hour,
        window_end_hour=window_end_hour,
        output_dir=output_dir,
        write_csv=write_step2_files,
    )

    # Write grid-level background temperature back to buildings for downstream use.
    grid_uhi_map = grid_stats.set_index(grid_id_col)["delta_t_uhi"]
    grid_t_met_map = grid_stats.set_index(grid_id_col)["T_met_z"]
    grid_t_base_map = grid_stats.set_index(grid_id_col)["T_base_z"]
    bld["delta_t_uhi"] = bld[grid_id_col].map(grid_uhi_map)
    bld["T_met_z"] = bld[grid_id_col].map(grid_t_met_map)
    bld["T_base_z"] = bld[grid_id_col].map(grid_t_base_map)

    # Step 3. Read building cooling power as the module input (vector-aware).
    # Prefer an explicit thermal-cooling column to avoid double-dividing by COP.
    q_cool_col = "discharging_power_thermal" if "discharging_power_thermal" in bld.columns else "discharging_power"
    n_t_candidates = [_infer_vector_length(bld[q_cool_col])]
    if "COP" in bld.columns:
        n_t_candidates.append(_infer_vector_length(bld["COP"]))
    if building_temperature_col in bld.columns:
        n_t_candidates.append(_infer_vector_length(bld[building_temperature_col]))
    n_t = max(1, int(max(n_t_candidates)))

    q_cool_input = bld[q_cool_col].apply(lambda v: _to_fixed_length_array(v, n_t, default=0.0))
    q_cool_input = q_cool_input.apply(lambda arr: np.maximum(arr, 0.0))
    bld["Q_cool_input"] = pd.Series(q_cool_input, index=bld.index, dtype=object)
    bld["Q_cool_input_source"] = q_cool_col

    # Step 4. Convert cooling power to AC electric power and rejected heat (vector-aware).
    cop_input = bld["COP"].apply(lambda v: _to_fixed_length_array(v, n_t, default=np.nan))

    p_ac_input: list[np.ndarray] = []
    q_rej_input: list[np.ndarray] = []
    for idx in bld.index:
        q_arr = q_cool_input.loc[idx]
        cop_arr = cop_input.loc[idx]
        denom = np.where(cop_arr > 0, cop_arr, np.nan)
        p_arr = np.divide(q_arr, denom)
        p_arr = np.where(np.isfinite(p_arr), p_arr, 0.0)
        p_arr = np.maximum(p_arr, 0.0)
        qrej_arr = q_arr + p_arr
        qrej_arr = np.where(np.isfinite(qrej_arr), qrej_arr, 0.0)
        p_ac_input.append(p_arr.astype(float))
        q_rej_input.append(qrej_arr.astype(float))

    bld["P_AC_input"] = pd.Series(p_ac_input, index=bld.index, dtype=object)
    bld["Q_rej_input"] = pd.Series(q_rej_input, index=bld.index, dtype=object)

    # Step 5. Aggregate to grid level and compute rejected-heat flux density (vector-aware).
    gid_str = bld[grid_id_col].astype(str)
    p_total_map: dict[str, np.ndarray] = {}
    p_mean_map: dict[str, np.ndarray] = {}
    q_total_map: dict[str, np.ndarray] = {}
    q_mean_map: dict[str, np.ndarray] = {}

    for gid in grid_stats[grid_id_col].astype(str):
        mask = gid_str == gid
        if not mask.any():
            zeros = np.zeros(n_t, dtype=float)
            p_total_map[gid] = zeros
            p_mean_map[gid] = zeros
            q_total_map[gid] = zeros
            q_mean_map[gid] = zeros
            continue

        p_stack = np.vstack(bld.loc[mask, "P_AC_input"].to_numpy())
        q_stack = np.vstack(bld.loc[mask, "Q_rej_input"].to_numpy())
        p_total_map[gid] = p_stack.sum(axis=0).astype(float)
        p_mean_map[gid] = p_stack.mean(axis=0).astype(float)
        q_total_map[gid] = q_stack.sum(axis=0).astype(float)
        q_mean_map[gid] = q_stack.mean(axis=0).astype(float)

    grid_gid = grid_stats[grid_id_col].astype(str)
    grid_stats["P_AC_grid_total_w"] = grid_gid.map(p_total_map)
    grid_stats["P_AC_grid_mean_w"] = grid_gid.map(p_mean_map)
    grid_stats["Q_rej_grid_total_w"] = grid_gid.map(q_total_map)
    grid_stats["Q_rej_grid_mean_w"] = grid_gid.map(q_mean_map)

    area_series = pd.to_numeric(grid_stats["A_z"], errors="coerce").fillna(0.0)
    q_rej_flux: list[np.ndarray] = []
    for idx in grid_stats.index:
        total_arr = _to_fixed_length_array(grid_stats.at[idx, "Q_rej_grid_total_w"], n_t, default=0.0)
        area = float(area_series.loc[idx])
        if area > 0:
            q_rej_flux.append((total_arr / area).astype(float))
        else:
            q_rej_flux.append(np.zeros(n_t, dtype=float))
    grid_stats["q_rej_w_m2"] = pd.Series(q_rej_flux, index=grid_stats.index, dtype=object)

    q_rej_grid_mean_map = grid_stats.set_index(grid_id_col)["Q_rej_grid_mean_w"]
    q_rej_flux_map = grid_stats.set_index(grid_id_col)["q_rej_w_m2"]
    bld["Q_rej_grid_mean_w"] = bld[grid_id_col].map(q_rej_grid_mean_map)
    bld["q_rej_w_m2"] = bld[grid_id_col].map(q_rej_flux_map)

    # Step 6. Set the grid ventilation / diffusion parameter k_z.
    zone_key = (
        grid_stats.get("zone_type", pd.Series("normal_urban", index=grid_stats.index))
        .fillna("normal_urban")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    grid_stats["k_z"] = zone_key.map(k_by_zone).fillna(k_by_zone["normal_urban"])

    if {"h_mix_z", "u_eff_z"}.issubset(grid_stats.columns):
        rho_a = 1.2
        cp_air = 1005.0
        h_mix = pd.to_numeric(grid_stats["h_mix_z"], errors="coerce")
        u_eff = pd.to_numeric(grid_stats["u_eff_z"], errors="coerce")
        k_physical = 1.0 / (rho_a * cp_air * h_mix * u_eff)
        k_physical = k_physical.where((h_mix > 0) & (u_eff > 0) & np.isfinite(k_physical))
        grid_stats["k_z"] = pd.to_numeric(k_physical, errors="coerce").fillna(grid_stats["k_z"])

    grid_stats["k_z"] = pd.to_numeric(grid_stats["k_z"], errors="coerce").fillna(k_by_zone["normal_urban"])
    grid_stats["k_z"] = grid_stats["k_z"].clip(lower=0.0)
    k_z_map = grid_stats.set_index(grid_id_col)["k_z"]
    bld["k_z"] = bld[grid_id_col].map(k_z_map)

    # Step 7. Compute the grid-level temperature rise caused by AC rejected heat.
    delta_t_ac_list: list[np.ndarray] = []
    for idx in grid_stats.index:
        k_val = float(pd.to_numeric(pd.Series([grid_stats.at[idx, "k_z"]]), errors="coerce").iloc[0])
        if not np.isfinite(k_val) or k_val < 0:
            k_val = 0.0
        q_arr = _to_fixed_length_array(grid_stats.at[idx, "q_rej_w_m2"], n_t, default=0.0)
        delta = np.maximum(k_val * q_arr, 0.0)
        delta_t_ac_list.append(delta.astype(float))
    grid_stats["delta_t_ac_z"] = pd.Series(delta_t_ac_list, index=grid_stats.index, dtype=object)
    delta_t_ac_map = grid_stats.set_index(grid_id_col)["delta_t_ac_z"]
    bld["delta_t_ac_z"] = bld[grid_id_col].map(delta_t_ac_map)

    # Step 8. Compute the final local air temperature for each grid.
    t_local_list: list[np.ndarray] = []
    for idx in grid_stats.index:
        t_met = _to_fixed_length_array(grid_stats.at[idx, "T_met_z"], n_t, default=np.nan)
        d_uhi = _to_fixed_length_array(grid_stats.at[idx, "delta_t_uhi"], n_t, default=0.0)
        d_ac = _to_fixed_length_array(grid_stats.at[idx, "delta_t_ac_z"], n_t, default=0.0)
        base = _to_fixed_length_array(grid_stats.at[idx, "T_base_z"], n_t, default=np.nan)
        t_met = np.where(np.isfinite(t_met), t_met, base)
        d_uhi = np.where(np.isfinite(d_uhi), d_uhi, 0.0)
        t_local = t_met + d_uhi + d_ac
        t_local_list.append(np.where(np.isfinite(t_local), t_local, base).astype(float))
    grid_stats["T_local_z"] = pd.Series(t_local_list, index=grid_stats.index, dtype=object)

    # Step 9. Write grid temperature back to the building level.
    grid_t_local_map = grid_stats.set_index(grid_id_col)["T_local_z"]
    bld["T_local_i_grid"] = bld[grid_id_col].map(grid_t_local_map)
    bld["T_local_z"] = bld["T_local_i_grid"]
    bld["T_local_i"] = bld["T_local_i_grid"]

    # Step 9b. Refine building temperature rise using location-weighted nearby grids.
    # Grid only provides rise terms; local temperature is updated from each building's own baseline.
    if n_t > 1:
        bld["delta_t_ac_i"] = bld["delta_t_ac_z"].apply(lambda v: _to_fixed_length_array(v, n_t, default=0.0))
        bld["T_local_i"] = bld.apply(
            lambda row: (
                _to_fixed_length_array(row.get(building_temperature_col, np.nan), n_t, default=30.0)
                + _to_fixed_length_array(row.get("delta_t_uhi", 0.0), n_t, default=0.0)
                + _to_fixed_length_array(row.get("delta_t_ac_i", 0.0), n_t, default=0.0)
            ).astype(float),
            axis=1,
        )
        building_property = _build_building_property_table(bld, osm_raw_columns=osm_raw_columns)
        return bld, grid_stats, building_property

    delta_t_default_arrays = bld["delta_t_ac_z"].apply(lambda v: _to_fixed_length_array(v, n_t, default=0.0))
    delta_t_ac_i_arrays = [arr.copy() for arr in delta_t_default_arrays.to_list()]

    if apply_building_idw:
        grid_xy_m = grid_stats[[grid_id_col, "x_ref", "y_ref", "delta_t_ac_z"]].rename(
            columns={"x_ref": "grid_x_m", "y_ref": "grid_y_m"}
        )
        grid_xy_m["grid_x_m"] = pd.to_numeric(grid_xy_m["grid_x_m"], errors="coerce")
        grid_xy_m["grid_y_m"] = pd.to_numeric(grid_xy_m["grid_y_m"], errors="coerce")
        grid_xy_m["delta_t_ac_z"] = grid_xy_m["delta_t_ac_z"].apply(
            lambda v: _to_fixed_length_array(v, n_t, default=0.0)
        )
        grid_xy_m = grid_xy_m.dropna(subset=["grid_x_m", "grid_y_m", "delta_t_ac_z"]).reset_index(drop=True)
    else:
        grid_xy_m = pd.DataFrame()

    if apply_building_idw and not grid_xy_m.empty:
        grid_points = grid_xy_m[["grid_x_m", "grid_y_m"]].to_numpy(dtype=float)
        grid_delta = np.vstack(grid_xy_m["delta_t_ac_z"].to_numpy())
        building_xy = bld[["x_m", "y_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        valid_building = np.isfinite(building_xy).all(axis=1)
        if valid_building.any():
            k = int(max(1, min(idw_neighbors, grid_points.shape[0])))
            b_idx = np.where(valid_building)[0]
            bxy = building_xy[valid_building]
            chunk_size = int(max(1, idw_chunk_size))
            eps = 1e-6

            for start in range(0, bxy.shape[0], chunk_size):
                end = min(start + chunk_size, bxy.shape[0])
                sub_xy = bxy[start:end]
                dx = sub_xy[:, [0]] - grid_points[:, 0][None, :]
                dy = sub_xy[:, [1]] - grid_points[:, 1][None, :]
                dist = np.sqrt(dx * dx + dy * dy)

                if k < grid_points.shape[0]:
                    nbr_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
                else:
                    nbr_idx = np.tile(np.arange(grid_points.shape[0]), (dist.shape[0], 1))

                nbr_dist = np.take_along_axis(dist, nbr_idx, axis=1)
                nbr_delta = grid_delta[nbr_idx]

                zero_mask = nbr_dist < eps
                has_zero = zero_mask.any(axis=1)
                weighted = np.zeros((nbr_dist.shape[0], n_t), dtype=float)

                if has_zero.any():
                    first_zero = zero_mask.argmax(axis=1)
                    rows = np.where(has_zero)[0]
                    weighted[rows] = nbr_delta[rows, first_zero[rows], :]

                nonzero_rows = np.where(~has_zero)[0]
                if nonzero_rows.size > 0:
                    nz_dist = np.maximum(nbr_dist[nonzero_rows], eps)
                    w = 1.0 / np.power(nz_dist, idw_power)
                    w_sum = w.sum(axis=1)
                    weighted[nonzero_rows] = (
                        (w[:, :, None] * nbr_delta[nonzero_rows]).sum(axis=1)
                        / np.maximum(w_sum[:, None], eps)
                    )

                for local_i, b_global_idx in enumerate(b_idx[start:end]):
                    delta_t_ac_i_arrays[b_global_idx] = weighted[local_i].astype(float)

    bld["delta_t_ac_i"] = pd.Series(
        [np.maximum(_to_fixed_length_array(v, n_t, default=0.0), 0.0).astype(float) for v in delta_t_ac_i_arrays],
        index=bld.index,
        dtype=object,
    )
    bld["T_local_i"] = bld.apply(
        lambda row: (
            _to_fixed_length_array(row.get(building_temperature_col, np.nan), n_t, default=30.0)
            + _to_fixed_length_array(row.get("delta_t_uhi", 0.0), n_t, default=0.0)
            + _to_fixed_length_array(row.get("delta_t_ac_i", 0.0), n_t, default=0.0)
        ).astype(float),
        axis=1,
    )

    # Build a non-geometry property table for inspection in the debugger.
    building_property = _build_building_property_table(bld, osm_raw_columns=osm_raw_columns)
    return bld, grid_stats, building_property


def build_spatial_grid_mapping(
    buildings: gpd.GeoDataFrame,
    *,
    grid_size_m: float = 250.0,
    grid_id_col: str = "grid_id",
    output_dir: Path | None = None,
    write_csv: bool = True,
    place_name: str = "Macau, China",
    clip_to_boundary: bool = True,
    reuse_cached_mapping: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Path]]:
    """
    Step 1: Discretize space and build building-to-grid mapping.

    Returns updated building table, grid statistics, and output file paths.
    """
    gdf = buildings.copy()
    outputs: dict[str, Path] = {}

    if output_dir is not None:
        outputs["grid_stats_path"] = output_dir / "macao_grid_stats_step1.csv"
        outputs["building_grid_map_path"] = output_dir / "macao_building_grid_mapping_step1.csv"
        outputs["grid_cells_path"] = output_dir / "macao_grid_cells_step1.gpkg"
        boundary_path = output_dir / "osm_macao_boundary.gpkg"
        if boundary_path.exists():
            outputs["boundary_path"] = boundary_path

        if reuse_cached_mapping:
            cached_map_path = outputs["building_grid_map_path"]
            cached_grid_path = outputs["grid_stats_path"]
            if cached_map_path.exists() and cached_grid_path.exists():
                cached_map = pd.read_csv(cached_map_path)
                cached_grid = pd.read_csv(cached_grid_path)
                required_map_cols = {
                    "building_index",
                    grid_id_col,
                    "grid_col",
                    "grid_row",
                    "x_m",
                    "y_m",
                    "A_z",
                    "zone_type",
                }
                if required_map_cols.issubset(cached_map.columns) and len(cached_map) == len(gdf):
                    cached_map = cached_map.set_index("building_index").reindex(gdf.index)
                    if not cached_map.index.isna().any():
                        gdf["x_m"] = pd.to_numeric(cached_map["x_m"], errors="coerce")
                        gdf["y_m"] = pd.to_numeric(cached_map["y_m"], errors="coerce")
                        gdf["grid_col"] = pd.to_numeric(cached_map["grid_col"], errors="coerce").astype("Int64")
                        gdf["grid_row"] = pd.to_numeric(cached_map["grid_row"], errors="coerce").astype("Int64")
                        gdf[grid_id_col] = cached_map[grid_id_col].astype(str)
                        gdf["A_z"] = pd.to_numeric(cached_map["A_z"], errors="coerce")
                        gdf["zone_type"] = cached_map["zone_type"].astype(str)
                        return gdf, cached_grid, outputs

    metric_gdf = gdf
    if metric_gdf.crs is not None and metric_gdf.crs.is_geographic:
        grid_crs = metric_gdf.estimate_utm_crs()
        metric_gdf = metric_gdf.to_crs(grid_crs if grid_crs is not None else "EPSG:3857")

    centroids = metric_gdf.geometry.centroid
    x_m = pd.to_numeric(centroids.x, errors="coerce")
    y_m = pd.to_numeric(centroids.y, errors="coerce")
    if x_m.isna().all() or y_m.isna().all():
        raise ValueError("Failed to build building-grid mapping: invalid building centroid coordinates.")
    x_m = x_m.fillna(x_m.median())
    y_m = y_m.fillna(y_m.median())

    boundary_gdf: gpd.GeoDataFrame | None = None
    boundary_union = None
    if clip_to_boundary:
        boundary_path = output_dir / "osm_macao_boundary.gpkg" if output_dir is not None else None
        if boundary_path is not None and boundary_path.exists():
            boundary_gdf = gpd.read_file(boundary_path)
        else:
            import osmnx as ox

            boundary_gdf = ox.geocode_to_gdf(place_name)
            if boundary_gdf.empty:
                raise ValueError(f"OSM boundary download failed for place='{place_name}'.")
            boundary_gdf = boundary_gdf[["geometry"]].copy()
            if boundary_path is not None:
                boundary_gdf.to_file(boundary_path, driver="GPKG")

        if boundary_path is not None and boundary_path.exists():
            outputs["boundary_path"] = boundary_path

        if boundary_gdf.crs is None:
            boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
        if boundary_gdf.crs != metric_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(metric_gdf.crs)

        boundary_union = (
            boundary_gdf.geometry.union_all()
            if hasattr(boundary_gdf.geometry, "union_all")
            else boundary_gdf.unary_union
        )
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
    else:
        minx, miny, maxx, maxy = metric_gdf.total_bounds

    x_edges = np.arange(minx, maxx + grid_size_m, grid_size_m)
    y_edges = np.arange(miny, maxy + grid_size_m, grid_size_m)
    if x_edges.size < 2:
        x_edges = np.array([minx, minx + grid_size_m])
    if y_edges.size < 2:
        y_edges = np.array([miny, miny + grid_size_m])

    grid_records: list[dict[str, object]] = []
    for col_idx, x_left in enumerate(x_edges[:-1]):
        for row_idx, y_bottom in enumerate(y_edges[:-1]):
            cell_geom = box(x_left, y_bottom, x_left + grid_size_m, y_bottom + grid_size_m)
            if boundary_union is not None:
                cell_geom = cell_geom.intersection(boundary_union)
            if cell_geom.is_empty:
                continue
            cell_area = float(cell_geom.area)
            if cell_area <= 0:
                continue
            grid_records.append(
                {
                    grid_id_col: f"{col_idx}_{row_idx}",
                    "grid_col": int(col_idx),
                    "grid_row": int(row_idx),
                    "A_z": cell_area if clip_to_boundary else float(grid_size_m * grid_size_m),
                    "geometry": cell_geom,
                }
            )

    if not grid_records:
        raise ValueError("No grid cells generated. Check boundary or grid_size_m.")

    grid_cells = gpd.GeoDataFrame(grid_records, geometry="geometry", crs=metric_gdf.crs)

    point_gdf = gpd.GeoDataFrame(index=gdf.index, geometry=centroids, crs=metric_gdf.crs)
    join_cols = [grid_id_col, "grid_col", "grid_row", "A_z", "geometry"]
    joined = gpd.sjoin(point_gdf, grid_cells[join_cols], how="left", predicate="within")

    missing_idx = joined.index[joined[grid_id_col].isna()]
    if len(missing_idx) > 0:
        nearest = gpd.sjoin_nearest(
            point_gdf.loc[missing_idx],
            grid_cells[join_cols],
            how="left",
            distance_col="_dist_to_grid",
        )
        joined.loc[missing_idx, [grid_id_col, "grid_col", "grid_row", "A_z"]] = nearest[
            [grid_id_col, "grid_col", "grid_row", "A_z"]
        ].to_numpy()

    gdf["x_m"] = x_m
    gdf["y_m"] = y_m
    gdf["grid_col"] = pd.to_numeric(joined["grid_col"], errors="coerce").astype("Int64")
    gdf["grid_row"] = pd.to_numeric(joined["grid_row"], errors="coerce").astype("Int64")
    gdf[grid_id_col] = joined[grid_id_col].astype(str)
    gdf["A_z"] = pd.to_numeric(joined["A_z"], errors="coerce")

    occupied_stats = (
        gdf.groupby(grid_id_col, as_index=False)
        .agg(
            building_count=("geometry", "size"),
            mean_height_m=("height_m", "mean"),
        )
        .reset_index(drop=True)
    )
    grid_stats = (
        grid_cells[[grid_id_col, "grid_col", "grid_row", "A_z"]]
        .merge(occupied_stats, on=grid_id_col, how="left")
        .fillna({"building_count": 0})
        .sort_values(by=grid_id_col)
        .reset_index(drop=True)
    )
    grid_stats["building_count"] = grid_stats["building_count"].astype(int)
    grid_stats["building_density_per_km2"] = grid_stats["building_count"] / grid_stats["A_z"] * 1e6

    density_ref = grid_stats.loc[grid_stats["building_density_per_km2"] > 0, "building_density_per_km2"]
    if density_ref.size >= 3:
        q1 = density_ref.quantile(1 / 3)
        q2 = density_ref.quantile(2 / 3)
    else:
        q1 = grid_stats["building_density_per_km2"].quantile(1 / 3)
        q2 = grid_stats["building_density_per_km2"].quantile(2 / 3)

    grid_stats["zone_type"] = "normal_urban"
    grid_stats.loc[grid_stats["building_density_per_km2"] <= q1, "zone_type"] = "open_area"
    grid_stats.loc[grid_stats["building_density_per_km2"] >= q2, "zone_type"] = "dense_urban"

    zone_type_map = grid_stats.set_index(grid_id_col)["zone_type"]
    gdf["zone_type"] = gdf[grid_id_col].map(zone_type_map)

    if write_csv and output_dir is not None:
        grid_stats.to_csv(outputs["grid_stats_path"], index=False)
        gdf.reset_index().rename(columns={"index": "building_index"})[
            ["building_index", grid_id_col, "grid_col", "grid_row", "x_m", "y_m", "A_z", "zone_type"]
        ].to_csv(outputs["building_grid_map_path"], index=False)
        grid_cells.merge(grid_stats[[grid_id_col, "zone_type"]], on=grid_id_col, how="left").to_file(
            outputs["grid_cells_path"],
            driver="GPKG",
        )

    return gdf, grid_stats, outputs


def prepare_background_meteorology_uhi(
    grid_stats: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    building_temperature_col: str = "temperature",
    grid_id_col: str = "grid_id",
    zone_type_col: str = "zone_type",
    delta_t_uhi_col: str = "delta_t_uhi",
    t_met_grid_col: str = "T_met_z",
    t_base_col: str = "T_base_z",
    window_start_hour: float = 14.0,
    window_end_hour: float = 18.0,
    uhi_by_zone: dict[str, float] | None = None,
    output_dir: Path | None = None,
    write_csv: bool = False,
) -> tuple[pd.DataFrame, float, dict[str, Path]]:
    """
    Step 2: Prepare background meteorology and solar-modulated UHI correction.

    Returns:
    1) grid table with grid-aggregated T_met_z, delta_t_uhi and T_base_z
    2) building-temperature mean background temperature T_met
    3) optional output paths
    """
    if uhi_by_zone is None:
        uhi_by_zone = {
            "open_area": 0.4,
            "normal_urban": 0.9,
            "dense_urban": 1.4,
        }

    gs = grid_stats.copy()
    bld = buildings.copy()

    if building_temperature_col not in bld.columns:
        raise ValueError(f"Missing building temperature column: '{building_temperature_col}'.")

    # Preserve time dimension for grid background temperature.
    n_t = _infer_vector_length(bld[building_temperature_col])
    bld["_building_temperature_vector"] = bld[building_temperature_col].apply(
        lambda v: _to_fixed_length_array(v, n_t, default=30.0)
    )
    bld["_building_temperature_scalar"] = bld["_building_temperature_vector"].apply(_to_temperature_scalar)
    temp_series = pd.to_numeric(bld["_building_temperature_scalar"], errors="coerce").dropna()
    if temp_series.empty:
        raise ValueError(f"Building temperature column '{building_temperature_col}' is empty or non-numeric.")

    t_met_vector = np.vstack(bld["_building_temperature_vector"].to_numpy()).mean(axis=0).astype(float)
    t_met = float(np.nanmean(t_met_vector))

    if "xBuilding" in bld.columns and "yBuilding" in bld.columns:
        x_ref = pd.to_numeric(bld["xBuilding"], errors="coerce")
        y_ref = pd.to_numeric(bld["yBuilding"], errors="coerce")
    else:
        centroid_ref = bld.geometry.centroid
        x_ref = pd.to_numeric(centroid_ref.x, errors="coerce")
        y_ref = pd.to_numeric(centroid_ref.y, errors="coerce")

    bld["_x_ref"] = x_ref
    bld["_y_ref"] = y_ref
    grid_xy = (
        bld.groupby(grid_id_col, as_index=False)
        .agg(_x_ref=("_x_ref", "mean"), _y_ref=("_y_ref", "mean"))
        .rename(columns={"_x_ref": "x_ref", "_y_ref": "y_ref"})
    )
    gs = gs.merge(grid_xy, on=grid_id_col, how="left")

    gid_str = bld[grid_id_col].astype(str)
    grid_temp_map: dict[str, np.ndarray] = {}
    for gid in gs[grid_id_col].astype(str):
        mask = gid_str == gid
        if not mask.any():
            grid_temp_map[gid] = t_met_vector.copy()
            continue
        grid_temp_map[gid] = np.vstack(bld.loc[mask, "_building_temperature_vector"].to_numpy()).mean(axis=0).astype(float)
    gs[t_met_grid_col] = gs[grid_id_col].astype(str).map(grid_temp_map)

    gs = estimate_continuous_uhi_from_density(
        gs,
        density_col="building_density_per_km2",
        uhi_by_zone=uhi_by_zone,
        output_col="delta_t_uhi_density",
    )

    # Build a time-varying UHI weight from solar irradiance (if available).
    # Spatial baseline remains density-driven, while temporal profile follows radiation.
    solar_weight = np.ones(n_t, dtype=float)
    if "I_solar" in bld.columns:
        solar_vectors = [
            _to_fixed_length_array(v, n_t, default=0.0)
            for v in bld["I_solar"].to_list()
        ]
        if solar_vectors:
            solar_stack = np.vstack(solar_vectors)
            solar_profile = np.nanmean(solar_stack, axis=0)
            solar_profile = np.where(np.isfinite(solar_profile), solar_profile, 0.0)
            solar_max = float(np.nanmax(solar_profile)) if solar_profile.size > 0 else 0.0
            if np.isfinite(solar_max) and solar_max > 0:
                solar_norm = np.clip(solar_profile / solar_max, 0.0, 1.0)
                # Keep nonzero nighttime UHI, but amplify during daytime with solar forcing.
                solar_weight = 0.35 + 0.65 * solar_norm

    if delta_t_uhi_col in gs.columns:
        uhi_base_series = pd.to_numeric(gs[delta_t_uhi_col], errors="coerce").fillna(gs["delta_t_uhi_density"])
    else:
        uhi_base_series = pd.to_numeric(gs["delta_t_uhi_density"], errors="coerce").fillna(0.0)

    delta_t_uhi_values: list[np.ndarray] = []
    for uhi_base in uhi_base_series.to_numpy(dtype=float):
        delta_t_uhi_values.append((float(uhi_base) * solar_weight).astype(float))
    gs[delta_t_uhi_col] = pd.Series(delta_t_uhi_values, index=gs.index, dtype=object)

    t_base_values: list[np.ndarray] = []
    for idx in gs.index:
        t_met_arr = _to_fixed_length_array(gs.at[idx, t_met_grid_col], n_t, default=t_met)
        d_uhi_arr = _to_fixed_length_array(gs.at[idx, delta_t_uhi_col], n_t, default=0.0)
        t_base_values.append((t_met_arr + d_uhi_arr).astype(float))
    gs[t_base_col] = pd.Series(t_base_values, index=gs.index, dtype=object)

    outputs: dict[str, Path] = {}
    if output_dir is not None:
        outputs["grid_background_path"] = output_dir / "macao_grid_background_step2.csv"

    if write_csv and output_dir is not None:
        keep_cols = [
            c
            for c in [
                grid_id_col,
                zone_type_col,
                "A_z",
                "building_count",
                "building_density_per_km2",
                "x_ref",
                "y_ref",
                t_met_grid_col,
                delta_t_uhi_col,
                t_base_col,
            ]
            if c in gs.columns
        ]
        gs[keep_cols].to_csv(outputs["grid_background_path"], index=False)

    return gs, t_met, outputs


def _to_temperature_scalar(value: object, default: float = 30.0) -> float:
    """Reduce scalar/array-like building temperature input to one numeric value."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
        if arr.empty:
            return float(default)
        return float(arr.mean())
    scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(scalar):
        return float(default)
    return float(scalar)


def _build_building_property_table(
    buildings: gpd.GeoDataFrame,
    *,
    osm_raw_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Build a debug-friendly non-geometry property table."""
    if osm_raw_columns is None:
        cols = [c for c in buildings.columns if c != "geometry"]
    else:
        cols = [c for c in buildings.columns if c != "geometry" and c not in osm_raw_columns]
    return pd.DataFrame(buildings[cols]).copy()


def estimate_continuous_uhi_from_density(
    grid_stats: pd.DataFrame,
    *,
    density_col: str = "building_density_per_km2",
    uhi_by_zone: dict[str, float] | None = None,
    output_col: str = "delta_t_uhi_density",
) -> pd.DataFrame:
    """
    Map building density to a continuous UHI increment using piecewise-linear interpolation.

    The three UHI anchor values are taken from `uhi_by_zone`, while the density
    anchors are inferred from the current grid-density distribution.
    """
    if uhi_by_zone is None:
        uhi_by_zone = {
            "open_area": 0.4,
            "normal_urban": 0.9,
            "dense_urban": 1.4,
        }

    gs = grid_stats.copy()
    density = pd.to_numeric(gs.get(density_col, pd.Series(0.0, index=gs.index)), errors="coerce")
    density = density.fillna(0.0).clip(lower=0.0)
    positive_density = density[density > 0]

    if positive_density.empty:
        anchor_open = 0.0
        anchor_normal = 1.0
        anchor_dense = 2.0
    else:
        anchor_open = float(positive_density.quantile(1 / 3))
        anchor_normal = float(positive_density.quantile(0.5))
        anchor_dense = float(positive_density.quantile(2 / 3))

    eps = 1e-6
    anchor_open = max(anchor_open, 0.0)
    anchor_normal = max(anchor_normal, anchor_open + eps)
    anchor_dense = max(anchor_dense, anchor_normal + eps)

    x_points = np.array([0.0, anchor_open, anchor_normal, anchor_dense], dtype=float)
    y_points = np.array(
        [
            float(uhi_by_zone["open_area"]),
            float(uhi_by_zone["open_area"]),
            float(uhi_by_zone["normal_urban"]),
            float(uhi_by_zone["dense_urban"]),
        ],
        dtype=float,
    )

    gs["uhi_density_anchor_open"] = anchor_open
    gs["uhi_density_anchor_normal"] = anchor_normal
    gs["uhi_density_anchor_dense"] = anchor_dense
    gs["uhi_density_norm"] = np.interp(
        density.to_numpy(dtype=float),
        np.array([anchor_open, anchor_dense], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        left=0.0,
        right=1.0,
    )
    gs[output_col] = np.interp(
        density.to_numpy(dtype=float),
        x_points,
        y_points,
        left=float(uhi_by_zone["open_area"]),
        right=float(uhi_by_zone["dense_urban"]),
    )
    gs[output_col] = pd.to_numeric(gs[output_col], errors="coerce").fillna(float(uhi_by_zone["normal_urban"]))
    return gs
