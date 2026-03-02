"""
建筑几何与属性数据读取与预处理。
Load building geometry and attributes (Shapefile / GeoJSON / CSV / OSM) and
provide helpers for classifying building use.
"""
from __future__ import annotations

from pathlib import Path
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
) -> gpd.GeoDataFrame:
    """
    为建筑数据添加一列 use_category（residential/commercial/industrial/public/unknown）。

    规则：
    1) 首先基于建筑自身的 OSM tag（building/building:use/amenity/shop/office/landuse）分类；
    2) 对仍为 unknown 的建筑，可选地下载同区域 landuse 多边形，并通过空间叠加补充分类。
    """

    df = buildings.copy()
    n_total = len(df)

    def _classify_use_from_tags(row: pd.Series) -> str:
        def norm(x: object) -> str:
            return str(x).strip().lower() if pd.notna(x) and str(x).strip() != "" else ""

        b = norm(row.get("building"))
        bu = norm(row.get("building:use"))
        amenity = norm(row.get("amenity"))
        shop = norm(row.get("shop"))
        office = norm(row.get("office"))
        lu_tag = norm(row.get("landuse"))

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
        if b in residential_keys or bu in residential_keys or lu_tag == "residential":
            return "residential"

        # 工业类
        industrial_keys = {
            "industrial",
            "factory",
            "manufacture",
            "warehouse",
            "plant",
        }
        if b in industrial_keys or bu in industrial_keys or lu_tag == "industrial":
            return "industrial"

        # 商业/办公类
        commercial_keys = {
            "retail",
            "commercial",
            "hotel",
            "supermarket",
            "mall",
            "shop",
        }
        if b in commercial_keys or bu in commercial_keys:
            return "commercial"
        if shop or office:
            return "commercial"
        if lu_tag in {"commercial", "retail"}:
            return "commercial"

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
        if amenity in public_amenities or b in {"public", "civic"}:
            return "public"

        # 其他暂标记为 unknown，后面用 landuse 再补一轮
        return "unknown"

    df["use_category"] = df.apply(_classify_use_from_tags, axis=1)

    # 若不需要 landuse 兜底，直接返回
    if not download_landuse:
        return df

    n_unknown_before = int((df["use_category"] == "unknown").sum())
    if n_unknown_before == 0:
        return df

    # 下载同一 place 范围的 landuse 多边形，作为兜底分类依据
    landuse = ox.features_from_place(place, tags={"landuse": True})
    if landuse.empty or "landuse" not in landuse.columns:
        return df

    # Avoid column-name collision in spatial join with buildings' own "landuse".
    landuse = landuse[["geometry", "landuse"]].rename(
        columns={"landuse": "landuse_zone"}
    )
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

    def _map_landuse_to_use(lu: object) -> str:
        lu_str = str(lu).strip().lower()
        if lu_str == "":
            return "unknown"
        if lu_str == "residential":
            return "residential"
        if lu_str in {"commercial", "retail", "mixed_use"}:
            return "commercial"
        if lu_str == "industrial":
            return "industrial"
        if lu_str in {
            "institutional",
            "education",
            "recreation_ground",
            "cemetery",
            "religious",
        }:
            return "public"
        return "unknown"

    # Prefer the explicit renamed column; keep fallbacks for legacy schemas.
    if "landuse_zone" in joined.columns:
        landuse_series = joined["landuse_zone"]
    elif "landuse_right" in joined.columns:
        landuse_series = joined["landuse_right"]
    elif "landuse" in joined.columns:
        landuse_series = joined["landuse"]
    else:
        return df

    inferred_use = landuse_series.map(_map_landuse_to_use)
    for idx, use_cat in zip(joined.index, inferred_use):
        if use_cat != "unknown":
            df.at[idx, "use_category"] = use_cat

    return df

