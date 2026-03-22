"""Centralized plotting utilities for the project."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, cast

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pydeck as pdk
from matplotlib import font_manager as fm
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


STATION_NAME_EN_MAP: dict[str, str] = {
    "路環市區": "Coloane Urban Area",
    "大炮台山": "Monte Fort",
    "紀念孫中山市政公園": "Dr. Sun Yat Sen Municipal Park",
    "澳門大學": "University of Macau",
    "海事博物館": "Maritime Museum",
    "外港碼頭": "Outer Harbour Ferry Terminal",
    "九澳": "Ka Ho",
    "東亞運站": "East Asian Games",
    "大潭山": "Taipa Grande",
}


def _is_empty_like(value: object) -> bool:
    """Return True when value should be treated as missing textual metadata."""
    if value is None:
        return True
    if isinstance(value, (pd.Series, np.ndarray, list, tuple, dict, set)):
        return len(value) == 0
    if value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, (pd.Timestamp, pd.Timedelta, np.datetime64, np.timedelta64)):
        return bool(pd.isna(value))
    return str(value).strip() == ""


def _pick_cjk_font_name() -> str | None:
    """Pick an installed CJK-capable font name when available."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return None


def plot_weather_station_violin_all_years(
    data_dir: Path,
    *,
    weather_file: str = "macao_weather_filled.csv",
    date_col: str = "Date",
    output_name: str = "macao_weather_station_violin_all_years.png",
    station_name_map: dict[str, str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Plot per-station temperature violin distributions using all historical records."""
    weather_path = data_dir / weather_file
    weather_df = pd.read_csv(weather_path)
    if date_col not in weather_df.columns:
        raise KeyError(f"Missing date column '{date_col}' in {weather_path}.")

    weather_df[date_col] = pd.to_datetime(weather_df[date_col], errors="coerce")
    weather_df = weather_df.dropna(subset=[date_col]).sort_values(date_col)

    station_cols = [c for c in weather_df.columns if c != date_col]
    if not station_cols:
        raise ValueError(f"No weather station columns found in {weather_path}.")

    long_df = weather_df.melt(
        id_vars=[date_col],
        value_vars=station_cols,
        var_name="station",
        value_name="temperature_c",
    )
    long_df["temperature_c"] = pd.to_numeric(long_df["temperature_c"], errors="coerce")
    long_df = long_df.dropna(subset=["temperature_c"])

    if long_df.empty:
        raise ValueError(f"No valid station temperature values found in {weather_path}.")

    station_order = (
        long_df.groupby("station")["temperature_c"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    label_map = dict(STATION_NAME_EN_MAP)
    if station_name_map:
        label_map.update(station_name_map)
    station_labels = [label_map.get(str(name), str(name)) for name in station_order]

    # Ensure Chinese station names can render on systems with limited default fonts.
    font_name = _pick_cjk_font_name()
    plt.rcParams["axes.unicode_minus"] = False

    station_arrays = [
        np.asarray(
            pd.to_numeric(long_df.loc[long_df["station"] == station, "temperature_c"], errors="coerce"),
            dtype=float,
        )
        for station in station_order
    ]

    finite_all = np.concatenate([arr[np.isfinite(arr)] for arr in station_arrays if arr.size > 0])
    if finite_all.size == 0:
        raise ValueError("No finite temperature values available for density plotting.")

    x_min = float(np.nanquantile(finite_all, 0.005))
    x_max = float(np.nanquantile(finite_all, 0.995))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min = float(np.nanmin(finite_all))
        x_max = float(np.nanmax(finite_all))
        if x_max <= x_min:
            x_max = x_min + 1.0

    n_bins = 120
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    density_matrix = np.zeros((len(station_order), n_bins), dtype=float)
    station_stats: list[dict[str, float]] = []

    for i, arr in enumerate(station_arrays):
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            station_stats.append(
                {"mean": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "p05": np.nan, "p95": np.nan}
            )
            continue
        hist, _ = np.histogram(valid, bins=bin_edges, density=True)
        density_matrix[i, :] = hist.astype(float)
        station_stats.append(
            {
                "mean": float(np.nanmean(valid)),
                "median": float(np.nanmedian(valid)),
                "q1": float(np.nanquantile(valid, 0.25)),
                "q3": float(np.nanquantile(valid, 0.75)),
                "p05": float(np.nanquantile(valid, 0.05)),
                "p95": float(np.nanquantile(valid, 0.95)),
            }
        )

    fig_h = max(5.0, 0.55 * len(station_order) + 1.5)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    cmap_obj = plt.get_cmap("YlOrRd")
    max_density = float(np.nanmax(density_matrix)) if density_matrix.size > 0 else 1.0
    if not np.isfinite(max_density) or max_density <= 0:
        max_density = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max_density)

    # Draw one long horizontal bar per station, split into small temperature bins.
    bin_left = bin_edges[:-1]
    bin_width = np.diff(bin_edges)
    bar_h = 0.52
    for i in range(len(station_order)):
        y_center = i + 1
        dens = density_matrix[i]
        colors = cmap_obj(norm(dens))
        ax.barh(
            np.full(n_bins, y_center, dtype=float),
            bin_width,
            left=bin_left,
            height=bar_h,
            color=colors,
            edgecolor="none",
            align="center",
        )

        # Add boxplot-like typical markers on each density strip.
        s = station_stats[i]
        if np.isfinite(s["p05"]) and np.isfinite(s["p95"]):
            ax.hlines(y_center, s["p05"], s["p95"], color="black", linewidth=1.1, alpha=0.95, zorder=3)
        if np.isfinite(s["q1"]) and np.isfinite(s["q3"]):
            ax.hlines(y_center, s["q1"], s["q3"], color="black", linewidth=3.6, alpha=0.95, zorder=4)
        if np.isfinite(s["median"]):
            ax.vlines(
                s["median"],
                y_center - bar_h / 2.0,
                y_center + bar_h / 2.0,
                color="black",
                linewidth=1.3,
                alpha=0.95,
                zorder=5,
            )
        if np.isfinite(s["mean"]):
            ax.plot(
                s["mean"],
                y_center,
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.9,
                zorder=6,
            )

    ax.set_yticks(np.arange(1, len(station_order) + 1))
    ax.set_yticklabels(station_labels)
    if font_name is not None:
        for label in ax.get_yticklabels():
            label.set_fontname(font_name)
    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("Weather Station")
    ax.set_title("Macau Weather Stations: Horizontal Density Strips (All Years)")
    ax.grid(axis="x", alpha=0.15)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("Probability density")

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markersize=5, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.3, label="Median"),
        Line2D([0], [0], color="black", linewidth=3.6, label="Q1-Q3"),
        Line2D([0], [0], color="black", linewidth=1.1, label="P05-P95"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()

    output_path = data_dir / output_name
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Weather-station density-strip figure saved to: {output_path}")

    summary = (
        long_df.groupby("station")["temperature_c"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .sort_values("median")
    )
    return fig, summary


def plot_individual_building_flexibility_time_series(
    macao_building_updated: gpd.GeoDataFrame,
    macao_building_property_updated: pd.DataFrame,
    representative_requests: list[dict[str, object]],
    time_index: pd.DatetimeIndex,
    include_duration: bool = False,
) -> tuple[Figure, pd.DataFrame]:
    """Plot flexibility time-series for representative buildings and return metadata."""
    required_cols = ["charging_power", "discharging_power", "energy_storage_capacity"]
    missing_cols = [c for c in required_cols if c not in macao_building_property_updated.columns]
    if missing_cols:
        raise ValueError("Missing required columns in macao_building_property_updated: " + ", ".join(missing_cols))
    has_duration = include_duration and ("duration" in macao_building_property_updated.columns)

    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)

    prop = macao_building_property_updated
    bld_meta = pd.DataFrame(macao_building_updated.drop(columns=["geometry"], errors="ignore")).copy()

    def _as_array(v: object) -> np.ndarray:
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v, dtype=float).reshape(-1)
        else:
            scalar = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            arr = np.array([scalar], dtype=float)
        return arr

    selected: list[dict[str, object]] = []

    address_cols = [
        col
        for col in (
            "addr:full",
            "addr:housenumber",
            "addr:street",
            "addr:street_1",
            "addr:housenumber_1",
            "addr:suburb",
            "addr:city",
            "addr:district",
            "addr:province",
            "addr:postcode",
            "addr:country",
            "addr:housename",
        )
        if col in prop.columns or col in bld_meta.columns
    ]
    for req in representative_requests:
        label = str(req.get("label", "representative"))
        use_category = req.get("use_category", None)
        building_id = req.get("building_id", None)

        if building_id is not None:
            if building_id not in prop.index:
                raise ValueError(f"building_id={building_id} not found in macao_building_property_updated index.")
            idx = int(pd.to_numeric(pd.Series([building_id]), errors="raise").iloc[0])
        else:
            candidates = prop
            if use_category is not None and "use_category" in candidates.columns:
                key = str(use_category).strip().lower()
                candidates = candidates[
                    candidates["use_category"].fillna("").astype(str).str.lower() == key
                ]
            if candidates.empty:
                raise ValueError(f"No candidate building for label='{label}', use_category='{use_category}'.")

            if "footprint_area_m2" in candidates.columns:
                area = pd.to_numeric(candidates["footprint_area_m2"], errors="coerce")
                idx = int(area.idxmax()) if area.notna().any() else int(candidates.index[0])
            else:
                idx = int(candidates.index[0])

        row = prop.loc[idx]
        meta_row = bld_meta.loc[idx] if idx in bld_meta.index else pd.Series(dtype=object)
        ch = _as_array(row["charging_power"])
        dis = _as_array(row["discharging_power"])
        cap = _as_array(row["energy_storage_capacity"])
        dur = _as_array(row["duration"]) if has_duration else None

        name = row.get("name", row.get("name:en", ""))
        if _is_empty_like(name):
            name = meta_row.get("name", meta_row.get("name:en", ""))
        if _is_empty_like(name):
            name = row.get("name:en", "")
        if _is_empty_like(name):
            name = meta_row.get("name:en", "")

        address_parts: list[str] = []
        for col in address_cols:
            val = row.get(col, "")
            if _is_empty_like(val):
                val = meta_row.get(col, "")
            if _is_empty_like(val):
                continue
            text = str(val).strip()
            if text:
                address_parts.append(text)
        address = ", ".join(dict.fromkeys(address_parts))

        if address == "" and "osmid" in bld_meta.columns:
            osmid_val = row.get("osmid", meta_row.get("osmid", np.nan))
            if not _is_empty_like(osmid_val):
                same_osmid = bld_meta[bld_meta["osmid"] == osmid_val]
                group_parts: list[str] = []
                for col in address_cols:
                    if col not in same_osmid.columns:
                        continue
                    vals = same_osmid[col].dropna().astype(str).str.strip()
                    vals = vals[vals != ""]
                    if not vals.empty:
                        group_parts.extend(vals.tolist())
                address = ", ".join(dict.fromkeys(group_parts))

        if address == "":
            address = "N/A (no OSM address tags for selected building)"

        n_candidates = [len(ch), len(dis), len(cap), len(time_index)]
        if dur is not None:
            n_candidates.append(len(dur))
        n = min(n_candidates)
        if n <= 0:
            raise ValueError(f"Selected building #{idx} has empty flexibility series.")

        selected.append(
            {
                "label": label,
                "building_id": idx,
                "use_category": row.get("use_category", use_category),
                "name": name,
                "address": address,
                "time": time_index[:n],
                "charging_kw": ch[:n] / 1e3,
                "discharging_kw": dis[:n] / 1e3,
                "capacity_kwh": cap[:n] / 3.6e6,
                "duration_h": dur[:n] if dur is not None else None,
            }
        )

    n_rows = len(selected)
    n_cols = 4 if has_duration else 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, max(3.2 * n_rows, 4.0)), sharex=False)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r, item in enumerate(selected):
        ax0, ax1, ax2 = axes[r, 0], axes[r, 1], axes[r, 2]
        ax3 = axes[r, 3] if has_duration else None
        t = item["time"]
        if isinstance(t, pd.DatetimeIndex):
            x_hour = t.hour.to_numpy(dtype=int)
        else:
            x_hour = np.arange(np.asarray(item["charging_kw"], dtype=float).size, dtype=int)

        ax0.plot(x_hour, item["charging_kw"], color="tab:blue", linewidth=1.5)
        ax1.plot(x_hour, item["discharging_kw"], color="tab:orange", linewidth=1.5)
        ax2.plot(x_hour, item["capacity_kwh"], color="tab:green", linewidth=1.5)
        if has_duration and ax3 is not None:
            ax3.plot(x_hour, item["duration_h"], color="tab:red", linewidth=1.5)

        ax0.set_ylabel("kW")
        ax1.set_ylabel("kW")
        ax2.set_ylabel("kWh")
        if has_duration and ax3 is not None:
            ax3.set_ylabel("h")

        ax0.set_title(f"Charging | {item['label']} | id={item['building_id']}")
        ax1.set_title(f"Discharging | {item['label']} | id={item['building_id']}")
        ax2.set_title(f"Capacity | {item['label']} | id={item['building_id']}")
        if has_duration and ax3 is not None:
            ax3.set_title(f"Duration | {item['label']} | id={item['building_id']}")

        plot_axes = [ax0, ax1, ax2]
        if has_duration and ax3 is not None:
            plot_axes.append(ax3)

        xticks = np.arange(int(np.nanmin(x_hour)), int(np.nanmax(x_hour)) + 1, 2)
        if xticks.size == 0:
            xticks = np.unique(x_hour)
        for ax in plot_axes:
            ax.grid(alpha=0.25)
            ax.set_xlabel("Hour")
            ax.set_xticks(xticks)

    fig.suptitle("Individual Building Flexibility Time Series")
    fig.tight_layout()
    representative_building_info = pd.DataFrame(
        [
            {
                "label": item["label"],
                "building_id": item["building_id"],
                "use_category": item["use_category"],
                "name": item["name"],
                "address": item["address"],
            }
            for item in selected
        ]
    )
    return fig, representative_building_info


def plot_timeseries(x: Any, y: Any, **kwargs: Any) -> Any:
    """Placeholder for time-series plotting."""
    return None


def plot_aggregated_discharging_power_timeseries(
    baseline_discharging_power: pd.Series,
    aggregated_discharging_after_mw: object,
    *,
    start_datetime: str | pd.Timestamp,
    output_path: str | Path = "data/aggregated_discharging_power_timeseries.png",
    time_step_hours: float = 1.0,
    dpi: int = 180,
) -> tuple[str, float]:
    """Plot city-level aggregated discharging power before/after microclimate correction."""

    def _aggregate_power_series_mw(series: pd.Series) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for value in series:
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
            else:
                arr = pd.to_numeric(pd.Series([value]), errors="coerce").to_numpy(dtype=float)
            if arr.size == 0:
                arr = np.array([0.0], dtype=float)
            arr = np.where(np.isfinite(arr), arr, 0.0).astype(float)
            vectors.append(arr)

        if not vectors:
            return np.array([], dtype=float)

        n_t = max(int(v.size) for v in vectors)
        stacked = np.vstack(
            [np.pad(v, (0, n_t - int(v.size)), mode="constant", constant_values=0.0) for v in vectors]
        )
        return stacked.sum(axis=0) / 1e6

    aggregated_discharging_before_mw = _aggregate_power_series_mw(baseline_discharging_power)
    aggregated_discharging_after_mw_arr = np.asarray(aggregated_discharging_after_mw, dtype=float).reshape(-1)
    n_t_dis = max(int(aggregated_discharging_before_mw.size), int(aggregated_discharging_after_mw_arr.size), 1)

    if int(aggregated_discharging_before_mw.size) != n_t_dis:
        aggregated_discharging_before_mw = np.pad(
            aggregated_discharging_before_mw,
            (0, n_t_dis - int(aggregated_discharging_before_mw.size)),
            mode="constant",
            constant_values=0.0,
        )
    if int(aggregated_discharging_after_mw_arr.size) != n_t_dis:
        aggregated_discharging_after_mw_arr = np.pad(
            aggregated_discharging_after_mw_arr,
            (0, n_t_dis - int(aggregated_discharging_after_mw_arr.size)),
            mode="constant",
            constant_values=0.0,
        )

    hour_index = np.arange(n_t_dis, dtype=int)
    power_diff_mw = aggregated_discharging_after_mw_arr - aggregated_discharging_before_mw
    abs_diff_mw = np.abs(power_diff_mw)
    diff_energy_mwh = float(np.trapezoid(abs_diff_mw, dx=float(time_step_hours)))

    fig_dis, ax_dis = plt.subplots(figsize=(9.0, 4.6))
    ax_dis.plot(hour_index, aggregated_discharging_before_mw, label="Before microclimate correction", linewidth=2.0)
    ax_dis.plot(hour_index, aggregated_discharging_after_mw_arr, label="After microclimate correction", linewidth=2.0)
    ax_dis.fill_between(
        hour_index,
        aggregated_discharging_before_mw,
        aggregated_discharging_after_mw_arr,
        where=(aggregated_discharging_after_mw_arr >= aggregated_discharging_before_mw),
        interpolate=True,
        color="tab:orange",
        alpha=0.18,
        label="Difference area (after >= before)",
    )
    ax_dis.fill_between(
        hour_index,
        aggregated_discharging_before_mw,
        aggregated_discharging_after_mw_arr,
        where=(aggregated_discharging_after_mw_arr < aggregated_discharging_before_mw),
        interpolate=True,
        color="tab:blue",
        alpha=0.15,
        label="Difference area (after < before)",
    )
    ax_dis.set_xlabel("Hour")
    ax_dis.set_ylabel("Aggregated discharging power (MW)")
    ax_dis.set_title("Aggregated Building Discharging Power (Macau)")
    if n_t_dis > 0:
        # Place the energy-difference label on the shaded area near the max gap.
        idx_peak = int(np.nanargmax(abs_diff_mw))
        x_label = float(hour_index[idx_peak])
        y_low = float(min(aggregated_discharging_before_mw[idx_peak], aggregated_discharging_after_mw_arr[idx_peak]))
        y_high = float(max(aggregated_discharging_before_mw[idx_peak], aggregated_discharging_after_mw_arr[idx_peak]))
        y_label = y_low + 0.5 * (y_high - y_low)
        ax_dis.text(
            x_label,
            y_label,
            f"{diff_energy_mwh:.3f} MWh",
            va="center",
            ha="center",
            fontsize=10,
            color="#303030",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.78, "edgecolor": "#888"},
            zorder=6,
        )
    ax_dis.legend(loc="best")
    ax_dis.grid(alpha=0.25)
    if n_t_dis <= 24:
        ax_dis.set_xticks(hour_index)
    else:
        tick_step = max(1, n_t_dis // 12)
        ax_dis.set_xticks(hour_index[::tick_step])
    fig_dis.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_dis.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig_dis)
    return str(out), diff_energy_mwh


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
    """Plot building polygons as a heatmap by one attribute column."""
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
        if "agg" not in backend:
            plt.show()

    return fig, ax


def plot_discharging_power_max_histogram(
    baseline_discharging_power: pd.Series,
    corrected_discharging_power: pd.Series,
    *,
    output_path: str | Path = "data/discharging_power_max_histogram.png",
    x_limits_mw: tuple[float, float] | None = None,
    dpi: int = 180,
) -> str:
    """Plot max discharging-power histograms with empty ranges skipped and first cluster expanded."""

    def extract_max_power(series: pd.Series) -> np.ndarray:
        """Extract maximum power value from each building's time series."""
        max_values: list[float] = []
        for value in series:
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                arr = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
                if arr.size > 0:
                    arr = np.where(np.isfinite(arr), arr, 0.0)
                    max_values.append(float(np.max(arr)))
                else:
                    max_values.append(0.0)
            else:
                scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                max_values.append(float(scalar) if pd.notna(scalar) else 0.0)
        return np.array(max_values, dtype=float) / 1e6

    max_before_mw = extract_max_power(baseline_discharging_power)
    max_after_mw = extract_max_power(corrected_discharging_power)

    if x_limits_mw is not None:
        x_min, x_max = float(x_limits_mw[0]), float(x_limits_mw[1])
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            raise ValueError("x_limits_mw must be a finite increasing tuple, e.g. (0.0, 0.1).")
        max_before_mw = max_before_mw[(max_before_mw >= x_min) & (max_before_mw <= x_max)]
        max_after_mw = max_after_mw[(max_after_mw >= x_min) & (max_after_mw <= x_max)]
        if max_before_mw.size == 0 and max_after_mw.size == 0:
            raise ValueError("No discharging-power maxima fall within x_limits_mw.")

    combined = np.concatenate([max_before_mw, max_after_mw])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        raise ValueError("No finite discharging-power maxima available for histogram plotting.")

    if x_limits_mw is not None:
        left_global = float(x_limits_mw[0])
        right = float(x_limits_mw[1])
    else:
        left_global = 0.0
        data_max = float(np.max(finite))
        right = max(data_max * 1.02, 1e-6)
    probe_bins = np.linspace(left_global, right, 140)
    total_counts, probe_edges = np.histogram(finite, bins=probe_bins)
    occupied = np.where(total_counts > 0)[0]

    if occupied.size == 0:
        segments = [(left_global, right)]
    else:
        # Merge nearby occupied bins so tiny single-bin gaps do not fragment the axis.
        groups: list[list[int]] = [[int(occupied[0])]]
        for idx in occupied[1:]:
            if int(idx) - groups[-1][-1] <= 2:
                groups[-1].append(int(idx))
            else:
                groups.append([int(idx)])

        segments = []
        for g in groups:
            i0, i1 = g[0], g[-1]
            left_edge = float(probe_edges[i0])
            right_edge = float(probe_edges[i1 + 1])
            span = max(right_edge - left_edge, right * 0.006)
            margin = 0.08 * span
            a = max(left_global, left_edge - margin)
            b = min(right, right_edge + margin)
            if b <= a:
                b = min(right, a + right * 0.01)
            segments.append((a, b))

        # Keep up to three meaningful segments for readability.
        if len(segments) > 3:
            segments = segments[:2] + [segments[-1]]

    width_ratios = []
    for i, (a, b) in enumerate(segments):
        span = max(b - a, right * 0.01)
        weight = span
        if i == 0:
            weight *= 3.8  # Expand the first dense interval.
        width_ratios.append(weight)

    fig, axes = plt.subplots(
        1,
        len(segments),
        figsize=(12.8, 5.8),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.06},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, ((left, right_seg), ax) in enumerate(zip(segments, axes)):
        in_seg_before = max_before_mw[(max_before_mw >= left) & (max_before_mw <= right_seg)]
        in_seg_after = max_after_mw[(max_after_mw >= left) & (max_after_mw <= right_seg)]

        local_span = max(right_seg - left, 1e-6)
        n_bins = 46 if i == 0 else 24
        local_bins = np.linspace(left, right_seg, n_bins)

        ax.hist(
            in_seg_before,
            bins=local_bins,
            alpha=0.62,
            label="Before microclimate correction" if i == 0 else None,
            color="tab:blue",
            edgecolor="black",
            linewidth=0.45,
        )
        ax.hist(
            in_seg_after,
            bins=local_bins,
            alpha=0.62,
            label="After microclimate correction" if i == 0 else None,
            color="tab:orange",
            edgecolor="black",
            linewidth=0.45,
        )

        ax.set_xlim(left, right_seg)
        ax.grid(axis="y", alpha=0.22)
        if local_span < 0.12:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
        else:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

        if i == 0:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=5))
        else:
            # Right-side broken-axis panels are narrow; hide labels to avoid unavoidable overlap.
            ax.set_xticks([])
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # Draw diagonal break marks between segments.
        if i < len(axes) - 1:
            ax.spines["right"].set_visible(False)
            d = 0.01
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft=False, left=False)
            d = 0.01
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    axes[0].set_ylabel("Number of Buildings")
    fig.supxlabel("Max Discharging Power (MW)")
    fig.suptitle("Distribution of Building Max Discharging Power (Macau)")
    axes[0].legend(loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_building_area_height_carbon_scatter(
    buildings: gpd.GeoDataFrame,
    *,
    area_column: str = "footprint_area_m2",
    height_column: str = "height_m",
    size_column: str = "carbon_emission_scalar",
    category_column: str = "use_category",
    output_path: str | Path = "data/building_area_height_carbon_scatter.png",
    dpi: int = 180,
) -> str:
    """Plot building footprint area vs height, sized by carbon emissions and colored by use type."""
    required_columns = [area_column, height_column, size_column, category_column]
    missing_columns = [col for col in required_columns if col not in buildings.columns]
    if missing_columns:
        raise KeyError("Missing required columns for scatter plot: " + ", ".join(missing_columns))

    plot_df = pd.DataFrame(buildings.drop(columns=["geometry"], errors="ignore")).copy()
    plot_df[area_column] = pd.to_numeric(plot_df[area_column], errors="coerce")
    plot_df[height_column] = pd.to_numeric(plot_df[height_column], errors="coerce")
    plot_df[size_column] = pd.to_numeric(plot_df[size_column], errors="coerce")
    plot_df[category_column] = plot_df[category_column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    numeric_columns = [area_column, height_column, size_column]
    plot_df.loc[:, numeric_columns] = plot_df.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[area_column, height_column, size_column])
    plot_df = plot_df[plot_df[category_column].str.lower().ne("unknown")].copy()
    plot_df = plot_df[(plot_df[area_column] > 0) & (plot_df[height_column] > 0) & (plot_df[size_column] >= 0)].copy()
    if plot_df.empty:
        raise ValueError("No valid building records available for area-height-carbon scatter plotting.")

    categories = sorted(plot_df[category_column].unique().tolist())
    cmap_obj = plt.get_cmap("tab10")
    color_lookup = {cat: cmap_obj(i % max(1, len(categories))) for i, cat in enumerate(categories)}

    carbon_values = plot_df[size_column].to_numpy(dtype=float)
    finite_carbon = carbon_values[np.isfinite(carbon_values)]
    if finite_carbon.size == 0:
        raise ValueError("No finite carbon-emission values available for scatter marker sizing.")

    # Use 6 discrete size levels to improve visual grouping of carbon emissions.
    level_count = 6
    size_levels = np.array([28.0, 48.0, 76.0, 112.0, 158.0, 220.0], dtype=float)
    q_edges = np.nanquantile(finite_carbon, np.linspace(0.0, 1.0, level_count + 1))
    q_edges = np.maximum.accumulate(q_edges)
    if np.allclose(q_edges[0], q_edges[-1]):
        marker_sizes = np.full(plot_df.shape[0], size_levels[2], dtype=float)
        size_bins = np.zeros(plot_df.shape[0], dtype=int)
        q_edges = np.array([q_edges[0]] * (level_count + 1), dtype=float)
    else:
        # digitize returns bins in [1, level_count], convert to [0, level_count-1]
        size_bins = np.digitize(carbon_values, q_edges[1:-1], right=True)
        size_bins = np.clip(size_bins, 0, level_count - 1)
        marker_sizes = size_levels[size_bins]

    plot_df["_marker_size"] = marker_sizes

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    for category in categories:
        group = plot_df[plot_df[category_column] == category]
        ax.scatter(
            group[area_column],
            group[height_column],
            s=group["_marker_size"],
            c=[color_lookup[category]],
            alpha=0.62,
            edgecolors="white",
            linewidths=0.35,
            label=category.title(),
        )

    ax.set_xlabel("Building Footprint Area (m$^2$, log scale)")
    ax.set_ylabel("Building Height (m)")
    ax.set_title("Macau Buildings: Footprint Area vs Height")
    ax.set_xscale("log")
    ax.grid(alpha=0.2)

    category_legend = ax.legend(title="Building Use Type", loc="upper left", frameon=True, framealpha=0.92)
    ax.add_artist(category_legend)

    size_handles: list[Line2D] = []
    size_labels: list[str] = []
    for i in range(level_count):
        left = float(q_edges[i])
        right = float(q_edges[i + 1])
        size_pts2 = float(size_levels[i])
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#7a7a7a",
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=np.sqrt(max(size_pts2, 1.0)),
            )
        )
        if i == 0:
            size_labels.append(f"<= {right:.2f} kg")
        elif i == level_count - 1:
            size_labels.append(f"> {left:.2f} kg")
        else:
            size_labels.append(f"{left:.2f} - {right:.2f} kg")

    if size_handles:
        ax.legend(size_handles, size_labels, title="Carbon Emission", loc="lower right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def export_building_heatmap_3d_web(
    buildings: gpd.GeoDataFrame,
    *,
    value_column: str = "volume",
    height_column: str | None = None,
    output_html: str = "data/volume_heatmap_3d.html",
    cmap: str = "turbo",
    opacity: float = 0.85,
    elevation_scale: float = 1.0,
    color_scale: str = "linear",
    color_clip_quantiles: tuple[float, float] | None = None,
    legend_unit: str | None = None,
) -> str:
    """Export an interactive 3D building heatmap to an HTML file."""
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if value_column not in buildings.columns:
        raise KeyError(f"Column '{value_column}' not found in buildings GeoDataFrame.")
    if height_column is not None and height_column not in buildings.columns:
        raise KeyError(f"Column '{height_column}' not found in buildings GeoDataFrame.")

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
        area_gdf = gdf
        if area_gdf.crs is not None and area_gdf.crs.is_geographic:
            projected_crs = area_gdf.estimate_utm_crs()
            area_gdf = area_gdf.to_crs(projected_crs if projected_crs is not None else "EPSG:3857")
        area_series = area_gdf.geometry.area.replace(0, np.nan)
        height_series = gdf[value_column] / area_series

    finite_values = gdf[value_column].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        clip_min, clip_max = 0.0, 1.0
    else:
        if color_clip_quantiles is not None:
            q_low, q_high = color_clip_quantiles
            clip_min = float(finite_values.quantile(q_low))
            clip_max = float(finite_values.quantile(q_high))
        else:
            clip_min = float(finite_values.min())
            clip_max = float(finite_values.max())
        if np.isclose(clip_min, clip_max):
            clip_max = clip_min + 1.0

    cmap_obj = plt.get_cmap(cmap)

    def transform_color_value(value: float) -> float:
        clipped = float(np.clip(value, clip_min, clip_max))
        if color_scale == "log":
            shift = max(0.0, -clip_min)
            return float(np.log1p(clipped + shift))
        return clipped

    transformed_min = transform_color_value(clip_min)
    transformed_max = transform_color_value(clip_max)
    if np.isclose(transformed_min, transformed_max):
        transformed_max = transformed_min + 1.0

    def to_color(value: float) -> list[int]:
        if pd.isna(value):
            return [180, 180, 180, 180]
        transformed = transform_color_value(float(value))
        norm = (transformed - transformed_min) / (transformed_max - transformed_min)
        norm = min(1.0, max(0.0, norm))
        r, g, b, _ = cmap_obj(norm)
        return [int(r * 255), int(g * 255), int(b * 255), 210]

    def format_sig(value: float | None, digits: int = 4) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}g}"

    def to_numeric_scalar(value: Any, default: float = np.nan) -> float:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
            if arr.empty:
                return float(default)
            return float(arr.mean())
        scalar = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(scalar):
            return float(default)
        return float(scalar)

    def to_english_label(value: Any) -> str:
        """Keep only English/ASCII part for tooltip labels."""
        if _is_empty_like(value):
            return ""
        text = str(value)
        ascii_text = text.encode("ascii", "ignore").decode("ascii")
        ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
        # Remove leading/trailing separators that may remain after dropping non-ASCII chars.
        ascii_text = ascii_text.strip("-_/|,:;[](){}")
        return ascii_text

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
        value = to_numeric_scalar(gdf.at[idx, value_column])
        elev_raw = pd.to_numeric(pd.Series([height_series.at[idx]]), errors="coerce").iloc[0]
        elev = float(elev_raw) if pd.notna(elev_raw) else 0.0
        if elev < 0:
            elev = 0.0
        energy_storage_kwh = np.nan
        if "energy_storage_capacity" in gdf.columns:
            energy_storage_raw = to_numeric_scalar(gdf.at[idx, "energy_storage_capacity"])
            if pd.notna(energy_storage_raw):
                energy_storage_kwh = float(energy_storage_raw) / 3.6e6
        discharging_power_kw = np.nan
        if "discharging_power" in gdf.columns:
            discharging_power_raw = to_numeric_scalar(gdf.at[idx, "discharging_power"])
            if pd.notna(discharging_power_raw):
                discharging_power_kw = float(discharging_power_raw) / 1e3

        building_name = "N/A"
        for name_col in ("name", "name:en", "addr:housename", "building", "amenity", "shop", "tourism", "office"):
            if name_col not in gdf.columns:
                continue
            candidate = gdf.at[idx, name_col]
            name_en = to_english_label(candidate)
            if name_en != "":
                building_name = name_en
                break

        color = to_color(value)
        for exterior in polygon_exteriors(row.geometry):
            records.append(
                {
                    "polygon": exterior,
                    "building name": building_name,
                    "carbon emission (kg)": None if pd.isna(value) else float(value),
                    "carbon emission display": format_sig(value),
                    "equivalent energy storage capacity (kWh)": None if pd.isna(energy_storage_kwh) else energy_storage_kwh,
                    "equivalent energy storage capacity display": format_sig(energy_storage_kwh),
                    "equivalent discharging power (kW)": None if pd.isna(discharging_power_kw) else discharging_power_kw,
                    "equivalent discharging power display": format_sig(discharging_power_kw),
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
        tooltip=cast(Any, {
            "html": (
                "<b>Building name:</b> {building name}"
                "<br/><b>Carbon emission (kg):</b> {carbon emission display}"
                "<br/><b>Equivalent energy storage capacity (kWh):</b> {equivalent energy storage capacity display}"
                "<br/><b>Equivalent discharging power (kW):</b> {equivalent discharging power display}"
            )
        }),
    )

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out_path), open_browser=False)

    legend_stops = np.linspace(0.0, 1.0, 9)
    legend_colors: list[str] = []
    for t in legend_stops:
        r, g, b, _ = cmap_obj(float(t))
        legend_colors.append(f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})")
    gradient_css = ", ".join(legend_colors)
    scale_label = "log scale" if color_scale == "log" else "linear scale"
    unit_suffix = f" ({legend_unit})" if legend_unit else ""
    unit_line = f"<div style=\"margin-top: 2px; color: #555;\">Unit: {legend_unit}</div>" if legend_unit else ""
    legend_html = (
        "<div style=\"position: fixed; right: 14px; bottom: 20px; z-index: 9999; "
        "width: 260px; background: rgba(255, 255, 255, 0.95); border: 1px solid #bcbcbc; "
        "border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.18); padding: 10px 12px; "
        "font-family: Arial, sans-serif; color: #1f1f1f; font-size: 12px;\">"
        f"<div style=\"font-weight: 600; margin-bottom: 6px;\">{value_column}{unit_suffix}</div>"
        f"<div style=\"height: 12px; border-radius: 5px; border: 1px solid #9a9a9a; background: linear-gradient(to right, {gradient_css});\"></div>"
        "<div style=\"display: flex; justify-content: space-between; margin-top: 4px;\">"
        f"<span>{format_sig(clip_min, 5)}</span><span>{format_sig(clip_max, 5)}</span>"
        "</div>"
        f"<div style=\"margin-top: 4px; color: #555;\">{scale_label}, clipped by quantiles</div>"
        f"{unit_line}"
        "</div>"
    )

    html_text = out_path.read_text(encoding="utf-8")
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", legend_html + "</body>")
    else:
        html_text = html_text + legend_html
    out_path.write_text(html_text, encoding="utf-8")
    return str(out_path)


def export_building_density_3d_web(
    buildings: gpd.GeoDataFrame,
    *,
    output_html: str = "data/macao_building_density_3d.html",
    grid_size_m: float = 250.0,
    density_column: str = "building_density_per_km2",
    height_column: str | None = "height_m",
    cmap: str = "YlOrRd",
    color_scale: str = "log",
) -> str:
    """Compute island-wide building density on a regular grid and export a 3D web map."""
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be > 0.")

    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No valid geometry available for density plotting.")

    metric = gdf
    if metric.crs is not None and metric.crs.is_geographic:
        projected_crs = metric.estimate_utm_crs()
        metric = metric.to_crs(projected_crs if projected_crs is not None else "EPSG:3857")

    centroids = metric.geometry.centroid
    x = pd.to_numeric(centroids.x, errors="coerce")
    y = pd.to_numeric(centroids.y, errors="coerce")
    if x.isna().all() or y.isna().all():
        raise ValueError("Failed to compute centroid coordinates for density plotting.")

    x = x.fillna(x.median())
    y = y.fillna(y.median())
    minx = float(x.min())
    miny = float(y.min())

    col_idx = np.floor((x.to_numpy(dtype=float) - minx) / float(grid_size_m)).astype(int)
    row_idx = np.floor((y.to_numpy(dtype=float) - miny) / float(grid_size_m)).astype(int)
    grid_key = pd.Series([f"{c}_{r}" for c, r in zip(col_idx, row_idx)], index=gdf.index, dtype=object)

    grid_counts = grid_key.value_counts(dropna=False)
    cell_area_km2 = (float(grid_size_m) * float(grid_size_m)) / 1e6
    density = grid_key.map(grid_counts).astype(float) / max(cell_area_km2, 1e-12)

    gdf[density_column] = density.to_numpy(dtype=float)

    return export_building_heatmap_3d_web(
        gdf,
        value_column=density_column,
        height_column=height_column if (height_column in gdf.columns if height_column else False) else None,
        output_html=output_html,
        cmap=cmap,
        color_scale=color_scale,
        color_clip_quantiles=(0.02, 0.98),
    )


def plot_building_density_surface_3d(
    buildings: gpd.GeoDataFrame,
    *,
    output_png: str = "data/macao_building_density_surface_3d.png",
    grid_size_m: float = 200.0,
    smooth_passes: int = 2,
    cmap: str = "jet",
    elev: float = 38.0,
    azim: float = -58.0,
    use_log_scale: bool = True,
) -> str:
    """Plot island-wide building density as a continuous 3D surface with contours."""
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be > 0.")

    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No valid geometry available for density surface plotting.")

    metric = gdf
    if metric.crs is not None and metric.crs.is_geographic:
        projected_crs = metric.estimate_utm_crs()
        metric = metric.to_crs(projected_crs if projected_crs is not None else "EPSG:3857")

    centroids = metric.geometry.centroid
    x = pd.to_numeric(centroids.x, errors="coerce")
    y = pd.to_numeric(centroids.y, errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        raise ValueError("Failed to compute centroid coordinates for density surface plotting.")

    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)

    x_edges = np.arange(float(np.min(xv)), float(np.max(xv)) + float(grid_size_m), float(grid_size_m))
    y_edges = np.arange(float(np.min(yv)), float(np.max(yv)) + float(grid_size_m), float(grid_size_m))
    if x_edges.size < 2:
        x_edges = np.array([float(np.min(xv)), float(np.min(xv)) + float(grid_size_m)])
    if y_edges.size < 2:
        y_edges = np.array([float(np.min(yv)), float(np.min(yv)) + float(grid_size_m)])

    hist, x_bins, y_bins = np.histogram2d(xv, yv, bins=[x_edges, y_edges])
    cell_area_km2 = (float(grid_size_m) * float(grid_size_m)) / 1e6
    density = hist / max(cell_area_km2, 1e-12)

    def _smooth2d(arr: np.ndarray, passes: int) -> np.ndarray:
        out = arr.astype(float, copy=True)
        k = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float) / 16.0
        for _ in range(max(0, int(passes))):
            p = np.pad(out, pad_width=1, mode="edge")
            out = (
                k[0, 0] * p[:-2, :-2]
                + k[0, 1] * p[:-2, 1:-1]
                + k[0, 2] * p[:-2, 2:]
                + k[1, 0] * p[1:-1, :-2]
                + k[1, 1] * p[1:-1, 1:-1]
                + k[1, 2] * p[1:-1, 2:]
                + k[2, 0] * p[2:, :-2]
                + k[2, 1] * p[2:, 1:-1]
                + k[2, 2] * p[2:, 2:]
            )
        return out

    density_smooth = _smooth2d(density, passes=smooth_passes)
    z = np.log1p(density_smooth) if use_log_scale else density_smooth

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2.0
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)
    Z = z.T

    fig = plt.figure(figsize=(10.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0.0,
        antialiased=True,
        alpha=0.95,
        rstride=1,
        cstride=1,
    )

    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    if np.isclose(z_min, z_max):
        z_max = z_min + 1.0

    ax.contour(X, Y, Z, zdir="z", offset=z_min, levels=24, cmap=cmap, linewidths=0.5)
    ax.set_zlim(z_min, z_max * 1.05)
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Density (log scale)" if use_log_scale else "Density")
    ax.set_title("Macau Building Density 3D Surface")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.06)
    cbar.set_label("Density (log scale)" if use_log_scale else "Density")

    fig.tight_layout()
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def export_building_density_3d_map_web(
    buildings: gpd.GeoDataFrame,
    *,
    output_html: str = "data/macao_building_density_3d_map.html",
    radius_m: float = 120.0,
    elevation_scale: float = 3.0,
    coverage: float = 1.0,
    upper_percentile: float = 99.0,
) -> str:
    """Render island-wide building density as a 3D map-based surface-like layer."""
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0.")

    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No valid geometry available for map-based density plotting.")

    # HexagonLayer expects geographic coordinates.
    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    cent = gdf.geometry.centroid
    lons = pd.to_numeric(cent.x, errors="coerce")
    lats = pd.to_numeric(cent.y, errors="coerce")
    valid = lons.notna() & lats.notna()
    if not valid.any():
        raise ValueError("Failed to compute centroid lon/lat for map-based density plotting.")

    pts = pd.DataFrame(
        {
            "lon": lons[valid].to_numpy(dtype=float),
            "lat": lats[valid].to_numpy(dtype=float),
            # uniform weight: density comes from point aggregation count
            "weight": np.ones(int(valid.sum()), dtype=float),
        }
    )

    min_lon, min_lat, max_lon, max_lat = float(pts["lon"].min()), float(pts["lat"].min()), float(pts["lon"].max()), float(pts["lat"].max())
    view_state = pdk.ViewState(
        longitude=(min_lon + max_lon) / 2.0,
        latitude=(min_lat + max_lat) / 2.0,
        zoom=12.2,
        pitch=58,
        bearing=18,
    )

    layer = pdk.Layer(
        "HexagonLayer",
        pts.to_dict(orient="records"),
        get_position="[lon, lat]",
        get_weight="weight",
        radius=float(radius_m),
        elevation_scale=float(elevation_scale),
        extruded=True,
        coverage=float(coverage),
        upper_percentile=float(upper_percentile),
        color_range=[
            [49, 54, 149],
            [69, 117, 180],
            [116, 173, 209],
            [171, 217, 233],
            [224, 243, 248],
            [254, 224, 144],
            [253, 174, 97],
            [244, 109, 67],
            [215, 48, 39],
            [165, 0, 38],
        ],
        pickable=True,
        auto_highlight=True,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip=cast(Any, {"html": "<b>Building density proxy:</b> {elevationValue}"}),
    )

    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out), open_browser=False)
    return str(out)


def export_microclimate_rise_point_heatmap_web(
    buildings: gpd.GeoDataFrame,
    *,
    value_column: str = "delta_t_ac_i",
    output_html: str = "data/macao_microclimate_rise_points.html",
    point_radius_m: float = 10.0,
    color_clip_quantiles: tuple[float, float] = (0.02, 0.98),
    cmap: str = "YlOrRd",
    color_by_rank: bool = True,
    opacity: float = 0.55,
) -> str:
    """Export building-level microclimate temperature-rise map as colored points."""
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if value_column not in buildings.columns:
        raise KeyError(f"Column '{value_column}' not found in buildings GeoDataFrame.")
    if point_radius_m <= 0:
        raise ValueError("point_radius_m must be > 0.")
    if not (0.0 < opacity <= 1.0):
        raise ValueError("opacity must be in (0, 1].")

    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No valid geometry available for microclimate point plotting.")

    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    cent = gdf.geometry.centroid
    lons = pd.to_numeric(cent.x, errors="coerce")
    lats = pd.to_numeric(cent.y, errors="coerce")

    def _to_scalar(value: Any) -> float:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
            if arr.empty:
                return float("nan")
            return float(arr.mean())
        s = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(s) if pd.notna(s) else float("nan")

    rise = gdf[value_column].apply(_to_scalar)
    valid = lons.notna() & lats.notna() & rise.notna()
    if not valid.any():
        raise ValueError("No valid lon/lat/value points available for microclimate point plotting.")

    pts = pd.DataFrame(
        {
            "lon": lons[valid].to_numpy(dtype=float),
            "lat": lats[valid].to_numpy(dtype=float),
            "rise_c": rise[valid].to_numpy(dtype=float),
        }
    )

    q_low, q_high = color_clip_quantiles
    v_min = float(pts["rise_c"].quantile(q_low))
    v_max = float(pts["rise_c"].quantile(q_high))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or np.isclose(v_min, v_max):
        v_min = float(pts["rise_c"].min())
        v_max = float(pts["rise_c"].max())
        if np.isclose(v_min, v_max):
            v_max = v_min + 1.0

    cmap_obj = plt.get_cmap(cmap)

    if color_by_rank:
        # Rank-based normalization avoids one-color saturation on skewed data.
        pts["rise_norm"] = pts["rise_c"].rank(method="average", pct=True)
    else:
        pts["rise_norm"] = pts["rise_c"].clip(lower=v_min, upper=v_max).map(
            lambda x: (float(x) - v_min) / (v_max - v_min)
        )

    def _to_color(norm_value: float) -> list[int]:
        t = float(norm_value)
        t = min(1.0, max(0.0, t))
        r, g, b, _ = cmap_obj(t)
        return [int(r * 255), int(g * 255), int(b * 255), 220]

    pts["fill_color"] = pts["rise_norm"].apply(_to_color)
    pts["rise_display"] = pts["rise_c"].map(lambda v: f"{v:.3f}")

    min_lon, min_lat = float(pts["lon"].min()), float(pts["lat"].min())
    max_lon, max_lat = float(pts["lon"].max()), float(pts["lat"].max())
    view_state = pdk.ViewState(
        longitude=(min_lon + max_lon) / 2.0,
        latitude=(min_lat + max_lat) / 2.0,
        zoom=12.4,
        pitch=0,
        bearing=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        pts.to_dict(orient="records"),
        get_position="[lon, lat]",
        get_fill_color="fill_color",
        get_radius=float(point_radius_m),
        pickable=True,
        stroked=False,
        radius_units="meters",
        opacity=float(opacity),
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip=cast(Any, {"html": "<b>Microclimate rise (degC):</b> {rise_display}"}),
    )

    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out), open_browser=False)
    return str(out)


def export_microclimate_rise_heatmap2d_web(
    buildings: gpd.GeoDataFrame,
    *,
    value_column: str = "delta_t_ac_i",
    output_html: str = "data/macao_microclimate_rise_heatmap2d.html",
    output_png: str | None = "data/macao_microclimate_rise_heatmap2d.png",
    radius_m: float = 70.0,
    intensity: float = 1.0,
    threshold: float = 0.0,
    heatmap_opacity: float = 0.55,
    min_weight_floor: float = 0.10,
    color_clip_quantiles: tuple[float, float] = (0.02, 0.98),
    map_style: str = "osm",
) -> tuple[str, str | None]:
    """Export building-level microclimate rise as a 2D map heatmap (no points)."""
    if "geometry" not in buildings.columns:
        raise KeyError("GeoDataFrame must contain a 'geometry' column.")
    if value_column not in buildings.columns:
        raise KeyError(f"Column '{value_column}' not found in buildings GeoDataFrame.")
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0.")
    if not (0.0 < heatmap_opacity <= 1.0):
        raise ValueError("heatmap_opacity must be in (0, 1].")
    if not (0.0 <= min_weight_floor < 1.0):
        raise ValueError("min_weight_floor must be in [0, 1).")

    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No valid geometry available for microclimate heatmap plotting.")

    if gdf.crs is not None and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    cent = gdf.geometry.centroid
    lons = pd.to_numeric(cent.x, errors="coerce")
    lats = pd.to_numeric(cent.y, errors="coerce")

    def _to_scalar(value: Any) -> float:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
            if arr.empty:
                return float("nan")
            return float(arr.mean())
        s = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(s) if pd.notna(s) else float("nan")

    rise = gdf[value_column].apply(_to_scalar)
    valid = lons.notna() & lats.notna() & rise.notna()
    if not valid.any():
        raise ValueError("No valid lon/lat/value points available for microclimate heatmap plotting.")

    pts = pd.DataFrame(
        {
            "lon": lons[valid].to_numpy(dtype=float),
            "lat": lats[valid].to_numpy(dtype=float),
            "rise_c": rise[valid].to_numpy(dtype=float),
        }
    )

    q_low, q_high = color_clip_quantiles
    v_min = float(pts["rise_c"].quantile(q_low))
    v_max = float(pts["rise_c"].quantile(q_high))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or np.isclose(v_min, v_max):
        v_min = float(pts["rise_c"].min())
        v_max = float(pts["rise_c"].max())
        if np.isclose(v_min, v_max):
            v_max = v_min + 1.0

    pts["rise_plot"] = pts["rise_c"].clip(lower=v_min, upper=v_max)
    rise_norm = (pts["rise_plot"] - v_min) / (v_max - v_min)
    rise_norm = rise_norm.clip(lower=0.0, upper=1.0)

    # Rank-based remapping increases contrast; floor keeps low-rise regions visible.
    rise_rank = rise_norm.rank(method="average", pct=True).to_numpy(dtype=float)
    pts["rise_weight"] = min_weight_floor + (1.0 - min_weight_floor) * rise_rank

    min_lon, min_lat = float(pts["lon"].min()), float(pts["lat"].min())
    max_lon, max_lat = float(pts["lon"].max()), float(pts["lat"].max())
    view_state = pdk.ViewState(
        longitude=(min_lon + max_lon) / 2.0,
        latitude=(min_lat + max_lat) / 2.0,
        zoom=12.4,
        pitch=0,
        bearing=0,
    )

    layer = pdk.Layer(
        "HeatmapLayer",
        pts.to_dict(orient="records"),
        get_position="[lon, lat]",
        get_weight="rise_weight",
        radius_pixels=float(radius_m),
        intensity=float(intensity),
        threshold=float(threshold),
        opacity=float(heatmap_opacity),
        aggregation="MEAN",
        color_range=[
            [33, 102, 172],
            [67, 147, 195],
            [146, 197, 222],
            [209, 229, 240],
            [247, 247, 247],
            [253, 219, 199],
            [244, 165, 130],
            [214, 96, 77],
            [178, 24, 43],
            [103, 0, 31],
        ],
    )

    # Build a robust basemap stack:
    # 1) OSM tile layer (online) for full cartographic context.
    # 2) Local building outlines (offline fallback) when remote tiles are unavailable.
    basemap_osm = pdk.Layer(
        "TileLayer",
        data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
        render_sub_layers="""
            new deck.BitmapLayer(props, {
                data: null,
                image: props.data,
                bounds: [
                    props.tile.bbox.west,
                    props.tile.bbox.south,
                    props.tile.bbox.east,
                    props.tile.bbox.north
                ]
            })
        """,
    )

    gdf_outline = gdf[["geometry"]].copy()
    gdf_outline = gdf_outline[gdf_outline.geometry.notna()]
    outlines_geojson = cast(dict[str, Any], gdf_outline.__geo_interface__)
    building_outline_layer = pdk.Layer(
        "GeoJsonLayer",
        outlines_geojson,
        stroked=True,
        filled=True,
        get_fill_color=[245, 245, 245, 18],
        get_line_color=[120, 120, 120, 90],
        line_width_min_pixels=0.5,
        pickable=False,
    )

    map_style_normalized = str(map_style).strip().lower()
    deck_map_style = None if map_style_normalized in {"osm", "openstreetmap"} else "light"
    layers: list[Any] = [building_outline_layer, layer]
    if deck_map_style is None:
        layers = [basemap_osm, building_outline_layer, layer]

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=deck_map_style,
        tooltip=cast(Any, {"html": "<b>Microclimate rise heatmap</b>"}),
    )

    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out), open_browser=False)

    # Add a lightweight HTML legend overlay for quick interpretation.
    legend_html = f"""
<div style="position: fixed; z-index: 10000; right: 16px; bottom: 18px; width: 260px;
            background: rgba(255,255,255,0.92); border: 1px solid #999; border-radius: 6px;
            padding: 10px 12px; font-family: Arial, sans-serif; font-size: 12px; color: #222;">
  <div style="font-weight: 600; margin-bottom: 6px;">Microclimate Rise (degC)</div>
  <div style="height: 12px; border-radius: 2px;
              background: linear-gradient(to right,
              rgb(49,54,149), rgb(69,117,180), rgb(116,173,209), rgb(171,217,233),
              rgb(224,243,248), rgb(254,224,144), rgb(253,174,97), rgb(244,109,67),
              rgb(215,48,39), rgb(165,0,38));"></div>
  <div style="display:flex; justify-content:space-between; margin-top:4px;">
    <span>{v_min:.3f}</span><span>{v_max:.3f}</span>
  </div>
  <div style="margin-top: 6px; color:#555;">Clipped to {int(q_low*100)}%-{int(q_high*100)}% quantiles</div>
</div>
"""
    html_text = out.read_text(encoding="utf-8")
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", legend_html + "\n</body>")
        out.write_text(html_text, encoding="utf-8")

    png_path_str: str | None = None
    if output_png:
        # Paper-ready static figure (2D heatmap + colorbar).
        n_x = 280
        n_y = 280
        x_edges = np.linspace(min_lon, max_lon, n_x + 1)
        y_edges = np.linspace(min_lat, max_lat, n_y + 1)
        H, _, _ = np.histogram2d(
            pts["lon"].to_numpy(dtype=float),
            pts["lat"].to_numpy(dtype=float),
            bins=[x_edges, y_edges],
            weights=pts["rise_plot"].to_numpy(dtype=float),
        )
        C, _, _ = np.histogram2d(
            pts["lon"].to_numpy(dtype=float),
            pts["lat"].to_numpy(dtype=float),
            bins=[x_edges, y_edges],
        )
        Z = np.divide(H, np.maximum(C, 1.0))
        Z = np.where(C > 0, Z, np.nan)

        fig, ax = plt.subplots(figsize=(7.8, 6.6))

        # Draw local OSM building footprints as static basemap for PNG output.
        # This guarantees map context even when online tile services are unavailable.
        try:
            gdf_bg = gdf[["geometry"]].copy()
            gdf_bg = gdf_bg[gdf_bg.geometry.notna()]
            if not gdf_bg.empty:
                gdf_bg.plot(
                    ax=ax,
                    color="#f2f4f5",
                    edgecolor="#c7cdd1",
                    linewidth=0.18,
                    alpha=0.95,
                    zorder=1,
                )
        except Exception:
            # Keep PNG export robust; heatmap rendering should proceed even if basemap plotting fails.
            pass

        im = ax.imshow(
            Z.T,
            origin="lower",
            extent=(min_lon, max_lon, min_lat, max_lat),
            cmap="YlOrRd",
            aspect="equal",
            vmin=v_min,
            vmax=v_max,
            alpha=0.86,
            zorder=3,
        )
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Microclimate Temperature Rise Heatmap (Macau)")
        cb = fig.colorbar(im, ax=ax, pad=0.015)
        cb.set_label("Microclimate rise (degC)")
        fig.tight_layout()

        out_png = Path(output_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        png_path_str = str(out_png)

    return str(out), png_path_str
