from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from src.building.building_thermal import prepare_building_thermal_state


def _file_signature(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _array_signature(arr: np.ndarray) -> dict[str, object]:
    arr_np = np.asarray(arr)
    if arr_np.size == 0:
        payload = b""
    else:
        payload = np.ascontiguousarray(arr_np, dtype=np.float32).tobytes()
    digest = hashlib.sha1(payload).hexdigest()
    return {
        "shape": tuple(int(x) for x in arr_np.shape),
        "dtype": str(arr_np.dtype),
        "sha1": digest,
    }


def _buildings_signature(buildings: gpd.GeoDataFrame) -> dict[str, object]:
    hasher = hashlib.sha1()

    idx_hash = pd.util.hash_pandas_object(pd.Index(buildings.index), index=False)
    hasher.update(idx_hash.to_numpy(dtype=np.uint64).tobytes())

    key_cols = [
        "use_category",
        "Shape_Area",
        "Elevation",
        "height",
        "building:levels",
        "xBuilding",
        "yBuilding",
    ]
    present_cols = [c for c in key_cols if c in buildings.columns]
    for col in present_cols:
        col_hash = pd.util.hash_pandas_object(buildings[col], index=False)
        hasher.update(col_hash.to_numpy(dtype=np.uint64).tobytes())

    if "geometry" in buildings.columns:
        bounds = buildings.geometry.bounds.to_numpy(dtype=float)
        bounds = np.nan_to_num(bounds, nan=0.0, posinf=0.0, neginf=0.0)
        hasher.update(np.round(bounds, 3).tobytes())

    return {
        "n_rows": int(len(buildings)),
        "n_cols": int(len(buildings.columns)),
        "present_key_cols": present_cols,
        "sha1": hasher.hexdigest(),
    }


def load_or_prepare_thermal_state(
    *,
    base_dir: Path,
    data_dir: Path,
    osm_buildings_path: Path,
    macao_building: gpd.GeoDataFrame,
    building_temperature_series: pd.Series | np.ndarray | Sequence[Any],
    macao_temperature: np.ndarray,
    macao_start_date: str,
    macao_end_date: str,
    osm_raw_columns: set[str],
    height_knn_k: int = 5,
    height_default_m: float = 10.0,
    uvalue_knn_k: int = 7,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, bool]:
    """Load prepared thermal state from cache, or compute and save cache."""
    prepare_cache_meta_path = data_dir / "macao_prepare_cache_meta.json"
    prepare_cache_building_path = data_dir / "macao_building_prepared.pkl"
    prepare_cache_property_path = data_dir / "macao_building_property_prepared.pkl"

    prepare_cache_key = {
        "start_date": macao_start_date,
        "end_date": macao_end_date,
        "cache_schema_version": 2,
        "params": {
            "height_knn_k": height_knn_k,
            "height_default_m": height_default_m,
            "uvalue_knn_k": uvalue_knn_k,
        },
        "input_sizes": {
            "n_buildings": int(len(macao_building)),
            "n_stations": int(macao_temperature.shape[1]) if getattr(macao_temperature, "ndim", 0) == 2 else 0,
            "n_hours": int(macao_temperature.shape[0]) if getattr(macao_temperature, "ndim", 0) >= 1 else 0,
        },
        "input_signatures": {
            "osm_buildings": _file_signature(osm_buildings_path),
            "weather_table": _file_signature(data_dir / "macao_weather_filled.csv"),
            "building_thermal_py": _file_signature(base_dir / "src" / "building" / "building_thermal.py"),
            "classify_buildings_py": _file_signature(base_dir / "src" / "data_io" / "buildings.py"),
            "temperature_array": _array_signature(np.asarray(macao_temperature)),
            "prepared_input_buildings": _buildings_signature(macao_building),
        },
    }

    if (
        prepare_cache_meta_path.exists()
        and prepare_cache_building_path.exists()
        and prepare_cache_property_path.exists()
    ):
        try:
            cached_meta = json.loads(prepare_cache_meta_path.read_text(encoding="utf-8"))
            if cached_meta == prepare_cache_key:
                cached_building = pd.read_pickle(prepare_cache_building_path)
                cached_property = pd.read_pickle(prepare_cache_property_path)
                print(f"Loaded prepared thermal state from cache: {prepare_cache_building_path}")
                return cached_building, cached_property, True
        except Exception:
            pass

    t_prepare_start = time.perf_counter()
    prepared_building, prepared_property = prepare_building_thermal_state(
        macao_building,
        building_temperature_series,
        building_temperature_col="temperature",
        osm_raw_columns=osm_raw_columns,
        height_knn_k=height_knn_k,
        height_default_m=height_default_m,
        uvalue_knn_k=uvalue_knn_k,
    )
    t_prepare_end = time.perf_counter()
    print(f"prepare_building_thermal_state runtime: {t_prepare_end - t_prepare_start:.2f}s")

    prepared_building.to_pickle(prepare_cache_building_path)
    prepared_property.to_pickle(prepare_cache_property_path)
    prepare_cache_meta_path.write_text(
        json.dumps(prepare_cache_key, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return prepared_building, prepared_property, False
