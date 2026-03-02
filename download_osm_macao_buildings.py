from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd


def main() -> None:
    """
    从 OpenStreetMap 下载澳门范围的建筑数据（building=*），
    统计带有高度相关信息的建筑数量，
    并保存为 data/osm_macao_buildings.gpkg，方便后续查看字段与建模使用。
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    out_path = data_dir / "osm_macao_buildings.gpkg"

    data_dir.mkdir(parents=True, exist_ok=True)

    place = "Macau, China"
    print(f"Downloading OSM buildings for: {place} ...")

    # 获取澳门范围内所有 building=* 的要素（多边形/多线等）
    buildings: gpd.GeoDataFrame = ox.features_from_place(place, tags={"building": True})

    n_total = len(buildings)
    print(f"Downloaded {n_total} building features.")
    print("Columns / fields:")
    print(buildings.columns.tolist())

    # 统计具有高度信息的建筑数量
    # 常见高度相关字段：height, building:levels, levels, roof:height, min_height 等
    candidate_height_cols = [
        "height",
        "building:height",
        "building:levels",
        "levels",
        "roof:height",
        "min_height",
        "building:levels:underground",
    ]
    height_cols = [c for c in candidate_height_cols if c in buildings.columns]

    if height_cols:
        def _has_height_info(row: pd.Series) -> bool:
            for col in height_cols:
                val = row.get(col, None)
                if pd.notna(val) and str(val).strip() != "":
                    return True
            return False

        has_height_mask = buildings.apply(_has_height_info, axis=1)
        n_with_height = int(has_height_mask.sum())

        print(f"Height-related columns found: {height_cols}")
        print(f"Buildings with ANY height-related info: {n_with_height} / {n_total} "
              f"({n_with_height / n_total * 100:.1f}% )")

        # 可选：为后续分析添加一列标记
        buildings["has_height_info"] = has_height_mask.astype(int)
    else:
        print("No typical height-related columns found in OSM data.")

    # 保存为 GeoPackage，便于在 QGIS / geopandas 中查看与使用
    buildings.to_file(out_path, layer="buildings", driver="GPKG")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


