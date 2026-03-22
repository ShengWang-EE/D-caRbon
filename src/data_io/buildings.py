"""
建筑几何与属性数据读取与预处理。
Load building geometry and attributes (Shapefile / GeoJSON / CSV / OSM) and
provide helpers for classifying building use.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd


def load_buildings(path: str | None = None, **kwargs: Any) -> Any:
    """
    读取建筑几何与属性。
    Load building footprints and attributes (volume, type, etc.).
    占位：若 path 为空返回 None；否则简单用 geopandas 读取矢量文件。
    """
    if path is None:
        return None
    return gpd.read_file(path, **kwargs)


def classify_building_use(
    buildings: gpd.GeoDataFrame,
    place: str,
    *,
    download_landuse: bool = True,
    landuse_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    为建筑数据添加一列 use_category（residential/commercial/industrial/public/unknown）。

    规则：
    1) 首先基于建筑自身的 OSM tag（building/building:use/amenity/shop/office/landuse）分类；
    2) 对仍为 unknown 的建筑，可选地下载同区域 landuse 多边形，并通过空间叠加补充分类。
    """

    df = buildings.copy()

    def _build_property_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        cols = [c for c in gdf.columns if c != "geometry"]
        return pd.DataFrame(gdf[cols]).copy()

    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")

    def _normalize_landuse_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf
        if "landuse" in gdf.columns:
            return gdf[["geometry", "landuse"]].rename(columns={"landuse": "landuse_zone"})
        if "landuse_zone" in gdf.columns:
            return gdf[["geometry", "landuse_zone"]]
        return gpd.GeoDataFrame(columns=["geometry", "landuse_zone"], geometry="geometry", crs=gdf.crs)

    def _load_landuse_source() -> gpd.GeoDataFrame:
        cache_path: Path | None = None
        if cache_dir is not None:
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / f"landuse_{_slug(place)}.gpkg"

        if landuse_path is not None:
            local_path = Path(landuse_path)
            if local_path.exists():
                try:
                    return gpd.read_file(local_path)
                except Exception:
                    pass

        if cache_path is not None and cache_path.exists():
            try:
                return gpd.read_file(cache_path)
            except Exception:
                pass

        if not download_landuse:
            return gpd.GeoDataFrame(columns=["geometry", "landuse"], geometry="geometry", crs=df.crs)

        downloaded = ox.features_from_place(place, tags={"landuse": True})
        if downloaded.empty:
            return downloaded

        if cache_path is not None:
            try:
                cache_df = downloaded[[c for c in ["geometry", "landuse"] if c in downloaded.columns]].copy()
                if "landuse" in cache_df.columns and "geometry" in cache_df.columns:
                    cache_df.to_file(cache_path, driver="GPKG")
            except Exception:
                pass
        return downloaded

    def _norm_col(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series("", index=df.index, dtype="object")
        return df[col].fillna("").astype(str).str.strip().str.lower()

    b = _norm_col("building")
    bu = _norm_col("building:use")
    amenity = _norm_col("amenity")
    shop = _norm_col("shop")
    office = _norm_col("office")
    lu_tag = _norm_col("landuse")

    # 住宅类
    residential_keys = {
        "residential",
        "apartments",
        "terrace",
        "detached",
        "semidetached_house",
        "house",
        "dormitory",
        "bungalow",
    }

    # 工业类
    industrial_keys = {
        "industrial",
        "factory",
        "manufacture",
        "warehouse",
        "plant",
    }

    # 商业/办公类
    commercial_keys = {
        "retail",
        "commercial",
        "hotel",
        "supermarket",
        "mall",
        "shop",
    }

    # 公共服务类（学校、医院、政府等）
    public_amenities = {
        "school",
        "university",
        "college",
        "kindergarten",
        "hospital",
        "clinic",
        "library",
        "police",
        "fire_station",
        "townhall",
        "courthouse",
        "government",
        "public_building",
    }

    df["use_category"] = "unknown"

    residential_mask = b.isin(residential_keys) | bu.isin(residential_keys) | lu_tag.eq("residential")
    df.loc[residential_mask, "use_category"] = "residential"

    industrial_mask = b.isin(industrial_keys) | bu.isin(industrial_keys) | lu_tag.eq("industrial")
    df.loc[df["use_category"].eq("unknown") & industrial_mask, "use_category"] = "industrial"

    commercial_mask = (
        b.isin(commercial_keys)
        | bu.isin(commercial_keys)
        | shop.ne("")
        | office.ne("")
        | lu_tag.isin({"commercial", "retail"})
    )
    df.loc[df["use_category"].eq("unknown") & commercial_mask, "use_category"] = "commercial"

    public_mask = amenity.isin(public_amenities) | b.isin({"public", "civic"})
    df.loc[df["use_category"].eq("unknown") & public_mask, "use_category"] = "public"

    # 若不需要 landuse 兜底，直接返回
    if not download_landuse:
        return df, _build_property_table(df)

    n_unknown_before = int((df["use_category"] == "unknown").sum())
    if n_unknown_before == 0:
        return df, _build_property_table(df)

    # 按“本地优先 + 缓存 + 在线兜底”加载 landuse 多边形。
    landuse = _load_landuse_source()
    if landuse.empty or "landuse" not in landuse.columns:
        landuse = _normalize_landuse_columns(landuse)
        if landuse.empty or "landuse_zone" not in landuse.columns:
            return df, _build_property_table(df)

    # Avoid column-name collision in spatial join with buildings' own "landuse".
    landuse = _normalize_landuse_columns(landuse)
    if df.crs and landuse.crs and df.crs != landuse.crs:
        landuse = landuse.to_crs(df.crs)

    unknown_mask = df["use_category"] == "unknown"
    unknown_buildings = df[unknown_mask]

    joined = gpd.sjoin(
        unknown_buildings,
        landuse,
        how="left",
        predicate="within",
    )

    # Prefer the explicit renamed column; keep fallbacks for legacy schemas.
    if "landuse_zone" in joined.columns:
        landuse_series = joined["landuse_zone"]
    elif "landuse_right" in joined.columns:
        landuse_series = joined["landuse_right"]
    elif "landuse" in joined.columns:
        landuse_series = joined["landuse"]
    else:
        return df, _build_property_table(df)

    lu_norm = landuse_series.fillna("").astype(str).str.strip().str.lower()
    inferred_use = lu_norm.map(
        {
            "residential": "residential",
            "commercial": "commercial",
            "retail": "commercial",
            "mixed_use": "commercial",
            "industrial": "industrial",
            "institutional": "public",
            "education": "public",
            "recreation_ground": "public",
            "cemetery": "public",
            "religious": "public",
        }
    ).fillna("unknown")

    valid_inferred = inferred_use[inferred_use.ne("unknown")]
    if not valid_inferred.empty:
        # Spatial join can produce duplicate building indexes; keep one deterministic label.
        inferred_by_building = valid_inferred.groupby(level=0).first()
        df.loc[inferred_by_building.index, "use_category"] = inferred_by_building.to_numpy()

    return df, _build_property_table(df)

