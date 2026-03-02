"""
Simple visualization helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk


def plot_timeseries(x: Any, y: Any, **kwargs: Any) -> Any:
    """Placeholder for time-series plotting."""
    return None


def plot_map_points(lons: Any, lats: Any, values: Any = None, **kwargs: Any) -> Any:
    """Placeholder for map point plotting."""
    return None


def plot_building_heatmap(
    buildings: gpd.GeoDataFrame,
    *,
    value_column: str = "volume",
    cmap: str = "YlOrRd",
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    edgecolor: str = "black",
    linewidth: float = 0.1,
    show: bool = True,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """
    Plot building polygons as a heatmap by one attribute column.
    Example: value_column="volume".
    """
    if value_column not in buildings.columns:
        raise KeyError(f"Column '{value_column}' not found in buildings GeoDataFrame.")
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")

    plot_df = buildings.copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
    plot_df = plot_df[plot_df.geometry.notna()]

    fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(
        column=value_column,
        cmap=cmap,
        legend=True,
        ax=ax,
        edgecolor=edgecolor,
        linewidth=linewidth,
        missing_kwds={"color": "lightgray", "label": "No data"},
    )

    ax.set_title(title or f"Building Heatmap by {value_column}")
    ax.set_axis_off()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        backend = (plt.get_backend() or "").lower()
        # Skip show() on non-interactive backends (e.g., FigureCanvasAgg).
        if "agg" not in backend:
            plt.show()

    return fig, ax


def export_building_heatmap_3d_web(
    buildings: gpd.GeoDataFrame,
    *,
    value_column: str = "volume",
    height_column: str | None = None,
    output_html: str = "data/volume_heatmap_3d.html",
    cmap: str = "YlOrRd",
    opacity: float = 0.85,
    elevation_scale: float = 1.0,
) -> str:
    """
    Export an interactive 3D building heatmap to an HTML file.
    Color encodes `value_column`; extrusion uses `height_column` if provided.
    """
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if value_column not in buildings.columns:
        raise KeyError(f"Column '{value_column}' not found in buildings GeoDataFrame.")
    if height_column is not None and height_column not in buildings.columns:
        raise KeyError(
            f"Column '{height_column}' not found in buildings GeoDataFrame."
        )

    gdf = buildings.copy()
    gdf[value_column] = pd.to_numeric(gdf[value_column], errors="coerce")
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No valid geometry available for plotting.")

    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf_wgs84 = gdf.to_crs(epsg=4326)
    else:
        gdf_wgs84 = gdf

    if height_column is not None:
        height_series = pd.to_numeric(gdf[height_column], errors="coerce")
    else:
        # If height is not provided, estimate from volume / footprint area.
        area_gdf = gdf
        if area_gdf.crs is not None and area_gdf.crs.is_geographic:
            projected_crs = area_gdf.estimate_utm_crs()
            area_gdf = area_gdf.to_crs(
                projected_crs if projected_crs is not None else "EPSG:3857"
            )
        area_series = area_gdf.geometry.area.replace(0, np.nan)
        height_series = gdf[value_column] / area_series

    finite_values = gdf[value_column].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(finite_values.min()), float(finite_values.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

    cmap_obj = plt.get_cmap(cmap)

    def to_color(value: float) -> list[int]:
        if pd.isna(value):
            return [180, 180, 180, 180]
        norm = (float(value) - vmin) / (vmax - vmin)
        norm = min(1.0, max(0.0, norm))
        r, g, b, _ = cmap_obj(norm)
        return [int(r * 255), int(g * 255), int(b * 255), 210]

    def polygon_exteriors(geom: Any) -> list[list[list[float]]]:
        if geom is None or geom.is_empty:
            return []
        gtype = geom.geom_type
        if gtype == "Polygon":
            return [[list(coord) for coord in geom.exterior.coords]]
        if gtype == "MultiPolygon":
            polys = []
            for poly in geom.geoms:
                polys.append([list(coord) for coord in poly.exterior.coords])
            return polys
        return []

    records: list[dict[str, Any]] = []
    for idx, row in gdf_wgs84.iterrows():
        value = gdf.at[idx, value_column]
        elev = height_series.at[idx]
        if pd.isna(elev) or elev < 0:
            elev = 0.0
        color = to_color(value)
        for exterior in polygon_exteriors(row.geometry):
            records.append(
                {
                    "polygon": exterior,
                    "value": None if pd.isna(value) else float(value),
                    "height_m": float(elev),
                    "fill_color": color,
                }
            )

    if not records:
        raise ValueError("No polygon geometry available for 3D rendering.")

    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    view_state = pdk.ViewState(
        latitude=(miny + maxy) / 2.0,
        longitude=(minx + maxx) / 2.0,
        zoom=14,
        pitch=50,
        bearing=0,
    )

    layer = pdk.Layer(
        "PolygonLayer",
        records,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_elevation="height_m",
        extruded=True,
        elevation_scale=elevation_scale,
        opacity=opacity,
        pickable=True,
        stroked=False,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"html": "<b>Value:</b> {value}<br/><b>Height (m):</b> {height_m}"},
    )

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out_path), open_browser=False)
    return str(out_path)
