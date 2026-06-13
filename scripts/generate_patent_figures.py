from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from PIL import Image
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOC_DIR = ROOT / "Documents"
FIG_DIR = DOC_DIR / "patent_figures"

SOURCE_DOCX = DOC_DIR / "一种考虑微气候反馈的建筑-电网需求响应碳减排潜力评估方法_专利完善稿_论文扩写版.docx"
OUTPUT_DOCX = DOC_DIR / "一种考虑微气候反馈的建筑-电网需求响应碳减排潜力评估方法_专利完善稿_论文案例附图版.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

OLD_DESCRIPTION_TO_NEW = {
    "图1为本发明整体评估流程示意图；": "图1为澳门气象站温度统计分布图；",
    "图2为本发明建筑用途识别与热工参数补全流程示意图；": "图2为代表性建筑柔性时序图；",
    "图3为本发明单建筑温控负荷等效储能建模流程示意图；": "图3为澳门建筑密度空间分布图；",
    "图4为本发明城市网格化微气候反馈建模流程示意图；": "图4为澳门微气候温升空间分布图；",
    "图5为本发明建筑群聚合需求响应能力时间序列示意图；": "图5为澳门建筑群聚合需求响应能力时序图；",
    "图6为本发明建筑群碳减排潜力空间分布示意图。": "图6为澳门建筑碳减排潜力空间分布图。",
}

OLD_CAPTION_TO_NEW = {
    "图1  整体评估流程示意图": "图1  澳门气象站温度统计分布图",
    "图2  建筑用途识别与热工参数补全流程示意图": "图2  代表性建筑柔性时序图",
    "图3  单建筑温控负荷等效储能建模流程示意图": "图3  澳门建筑密度空间分布图",
    "图4  城市网格化微气候反馈建模流程示意图": "图4  澳门微气候温升空间分布图",
    "图5  建筑群聚合需求响应能力时间序列示意图": "图5  澳门建筑群聚合需求响应能力时序图",
    "图6  建筑群碳减排潜力空间分布示意图": "图6  澳门建筑碳减排潜力空间分布图",
}

USE_LABEL_MAP = {
    "commercial": "商业建筑",
    "public": "公共建筑",
    "residential": "住宅建筑",
    "industrial": "工业建筑",
}

STATION_LABELS = {
    "路環市區": "路环市区",
    "大炮台山": "大炮台山",
    "紀念孫中山市政公園": "纪念孙中山市政公园",
    "澳門大學": "澳门大学",
    "海事博物館": "海事博物馆",
    "外港碼頭": "外港码头",
    "九澳": "九澳",
    "東亞運站": "东亚运动站",
    "大潭山": "大潭山",
}


def pick_chinese_font() -> str:
    preferred = [
        "SimSun",
        "FangSong",
        "STSong",
        "Microsoft YaHei",
        "SimHei",
        "KaiTi",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    raise RuntimeError("No suitable Chinese font was found on this machine.")


FONT_NAME = pick_chinese_font()

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_NAME],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_text(element: ET.Element) -> str:
    texts = []
    for node in element.iter():
        if node.tag == f"{{{W_NS}}}t" and node.text:
            texts.append(node.text)
    return "".join(texts).strip()


def set_paragraph_text(paragraph: ET.Element, new_text: str) -> None:
    text_nodes = [node for node in paragraph.iter() if node.tag == f"{{{W_NS}}}t"]
    if text_nodes:
        text_nodes[0].text = new_text
        for node in text_nodes[1:]:
            node.text = ""
        return

    run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
    t = ET.SubElement(run, f"{{{W_NS}}}t")
    t.text = new_text


def cm_to_emu(cm: float) -> int:
    return int(cm / 2.54 * 914400)


def build_image_paragraph(
    *,
    rel_id: str,
    image_name: str,
    image_width_px: int,
    image_height_px: int,
    width_cm: float,
    docpr_id: int,
) -> ET.Element:
    max_height_cm = 18.0
    height_cm = width_cm * image_height_px / max(image_width_px, 1)
    if height_cm > max_height_cm:
        height_cm = max_height_cm
        width_cm = height_cm * image_width_px / max(image_height_px, 1)

    cx = cm_to_emu(width_cm)
    cy = cm_to_emu(height_cm)

    p = ET.Element(f"{{{W_NS}}}p")
    p_pr = ET.SubElement(p, f"{{{W_NS}}}pPr")
    jc = ET.SubElement(p_pr, f"{{{W_NS}}}jc")
    jc.set(f"{{{W_NS}}}val", "center")

    r = ET.SubElement(p, f"{{{W_NS}}}r")
    drawing = ET.SubElement(r, f"{{{W_NS}}}drawing")
    inline = ET.SubElement(
        drawing,
        f"{{{WP_NS}}}inline",
        {"distT": "0", "distB": "0", "distL": "0", "distR": "0"},
    )
    extent = ET.SubElement(inline, f"{{{WP_NS}}}extent")
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    effect = ET.SubElement(inline, f"{{{WP_NS}}}effectExtent")
    effect.set("l", "0")
    effect.set("t", "0")
    effect.set("r", "0")
    effect.set("b", "0")
    doc_pr = ET.SubElement(inline, f"{{{WP_NS}}}docPr")
    doc_pr.set("id", str(docpr_id))
    doc_pr.set("name", image_name)
    c_nv = ET.SubElement(inline, f"{{{WP_NS}}}cNvGraphicFramePr")
    locks = ET.SubElement(c_nv, f"{{{A_NS}}}graphicFrameLocks")
    locks.set("noChangeAspect", "1")

    graphic = ET.SubElement(inline, f"{{{A_NS}}}graphic")
    graphic_data = ET.SubElement(graphic, f"{{{A_NS}}}graphicData")
    graphic_data.set("uri", "http://schemas.openxmlformats.org/drawingml/2006/picture")

    pic = ET.SubElement(graphic_data, f"{{{PIC_NS}}}pic")
    nv_pic_pr = ET.SubElement(pic, f"{{{PIC_NS}}}nvPicPr")
    c_nv_pr = ET.SubElement(nv_pic_pr, f"{{{PIC_NS}}}cNvPr")
    c_nv_pr.set("id", "0")
    c_nv_pr.set("name", image_name)
    ET.SubElement(nv_pic_pr, f"{{{PIC_NS}}}cNvPicPr")

    blip_fill = ET.SubElement(pic, f"{{{PIC_NS}}}blipFill")
    blip = ET.SubElement(blip_fill, f"{{{A_NS}}}blip")
    blip.set(f"{{{R_NS}}}embed", rel_id)
    stretch = ET.SubElement(blip_fill, f"{{{A_NS}}}stretch")
    ET.SubElement(stretch, f"{{{A_NS}}}fillRect")

    sp_pr = ET.SubElement(pic, f"{{{PIC_NS}}}spPr")
    xfrm = ET.SubElement(sp_pr, f"{{{A_NS}}}xfrm")
    off = ET.SubElement(xfrm, f"{{{A_NS}}}off")
    off.set("x", "0")
    off.set("y", "0")
    ext = ET.SubElement(xfrm, f"{{{A_NS}}}ext")
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geom = ET.SubElement(sp_pr, f"{{{A_NS}}}prstGeom")
    geom.set("prst", "rect")
    ET.SubElement(geom, f"{{{A_NS}}}avLst")
    return p


def parse_deck_json_from_html(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")
    match = re.search(r"const jsonInput = (\{.*?\})\s*;\s*const tooltip", html, re.S)
    if match is None:
        raise RuntimeError(f"Unable to find jsonInput in {html_path}.")
    return json.loads(match.group(1))


def parse_array_text(value: object) -> np.ndarray:
    text = str(value).strip()
    if not text:
        return np.array([], dtype=float)
    cleaned = text.strip("[]").replace(",", " ")
    arr = np.fromstring(cleaned, sep=" ", dtype=float)
    return arr[np.isfinite(arr)]


def generate_fig1(output_path: Path) -> None:
    weather_df = pd.read_csv(DATA_DIR / "macao_weather_filled.csv")
    date_col = "Date"
    weather_df[date_col] = pd.to_datetime(weather_df[date_col], errors="coerce")
    weather_df = weather_df.dropna(subset=[date_col]).sort_values(date_col)
    station_cols = [c for c in weather_df.columns if c != date_col]

    long_df = weather_df.melt(
        id_vars=[date_col],
        value_vars=station_cols,
        var_name="station",
        value_name="temperature_c",
    )
    long_df["temperature_c"] = pd.to_numeric(long_df["temperature_c"], errors="coerce")
    long_df = long_df.dropna(subset=["temperature_c"])

    station_order = (
        long_df.groupby("station")["temperature_c"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    station_labels = [STATION_LABELS.get(name, name) for name in station_order]

    station_arrays = [
        np.asarray(
            pd.to_numeric(long_df.loc[long_df["station"] == station, "temperature_c"], errors="coerce"),
            dtype=float,
        )
        for station in station_order
    ]

    finite_all = np.concatenate([arr[np.isfinite(arr)] for arr in station_arrays if arr.size > 0])
    x_min = float(np.nanquantile(finite_all, 0.005))
    x_max = float(np.nanquantile(finite_all, 0.995))
    n_bins = 120
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    density_matrix = np.zeros((len(station_order), n_bins), dtype=float)
    station_stats: list[dict[str, float]] = []

    for i, arr in enumerate(station_arrays):
        valid = arr[np.isfinite(arr)]
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
    fig, ax = plt.subplots(figsize=(12.0, fig_h), dpi=300)
    cmap_obj = plt.get_cmap("Greys")
    norm = mcolors.Normalize(vmin=0.0, vmax=float(np.nanmax(density_matrix)) if density_matrix.size > 0 else 1.0)

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
        s = station_stats[i]
        ax.hlines(y_center, s["p05"], s["p95"], color="black", linewidth=1.0, zorder=3)
        ax.hlines(y_center, s["q1"], s["q3"], color="black", linewidth=3.2, zorder=4)
        ax.vlines(s["median"], y_center - bar_h / 2.0, y_center + bar_h / 2.0, color="black", linewidth=1.1, zorder=5)
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
    ax.set_xlabel("气温 / ℃")
    ax.set_ylabel("气象站")
    ax.set_title("图1  澳门气象站温度统计分布图", fontsize=14, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.15)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("概率密度")

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markersize=5, label="均值"),
        Line2D([0], [0], color="black", linewidth=1.1, label="中位数"),
        Line2D([0], [0], color="black", linewidth=3.2, label="四分位区间"),
        Line2D([0], [0], color="black", linewidth=1.0, label="5%-95%区间"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_fig2(output_path: Path) -> None:
    prop = pd.read_csv(DATA_DIR / "macao_building_property_ac_power.csv", low_memory=False)
    representative_indices = [165, 58, 618, 102]
    metric_labels = [
        ("charging_power", "充电功率", "kW"),
        ("discharging_power", "放电功率", "kW"),
        ("energy_storage_capacity", "等效储能容量", "kWh"),
    ]

    fig, axes = plt.subplots(len(representative_indices), len(metric_labels), figsize=(12.4, 10.8), dpi=300)

    for row_idx, building_idx in enumerate(representative_indices):
        row = prop.loc[building_idx]
        use_text = USE_LABEL_MAP.get(str(row["use_category"]).strip().lower(), "代表性建筑")
        hours = np.arange(24)

        for col_idx, (column, metric_name, unit) in enumerate(metric_labels):
            ax = axes[row_idx, col_idx]
            arr = parse_array_text(row[column])
            if arr.size == 0:
                arr = np.zeros(24, dtype=float)
            if column == "energy_storage_capacity":
                series = arr / 3.6e6
            else:
                series = arr / 1e3

            ax.plot(hours[: series.size], series, color="black", linewidth=1.5)
            ax.set_xlim(0, max(series.size - 1, 23))
            ax.set_xticks(np.arange(0, 24, 2))
            ax.grid(alpha=0.25, linestyle=":")
            ax.set_xlabel("时刻 / h")
            ax.set_ylabel(unit)
            ax.set_title(f"{use_text}  {metric_name}（ID={building_idx}）", fontsize=10, pad=6)

    fig.suptitle("图2  代表性建筑柔性时序图", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_fig3(output_path: Path) -> None:
    grid_cells = gpd.read_file(DATA_DIR / "macao_grid_cells_step1.gpkg")
    grid_stats = pd.read_csv(DATA_DIR / "macao_grid_stats_step1.csv")

    plot_gdf = grid_cells.merge(grid_stats, on="grid_id", how="left")
    plot_gdf["building_density_per_km2"] = pd.to_numeric(plot_gdf["building_density_per_km2"], errors="coerce")
    q_low = float(plot_gdf["building_density_per_km2"].quantile(0.10))
    q_high = float(plot_gdf["building_density_per_km2"].quantile(0.95))
    plot_gdf["_plot_value"] = plot_gdf["building_density_per_km2"].clip(lower=q_low, upper=q_high)
    if plot_gdf.crs is not None and not plot_gdf.crs.is_geographic:
        plot_gdf = plot_gdf.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(9.2, 8.0), dpi=300)
    plot_gdf.plot(
        ax=ax,
        column="_plot_value",
        cmap="Greys",
        linewidth=0.15,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "建筑密度 / 栋·km$^{-2}$", "shrink": 0.78},
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("图3  澳门建筑密度空间分布图", fontsize=14, fontweight="bold", pad=10)
    ax.text(0.03, 0.03, "注：色阶越深表示建筑密度越高", transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
    fig.savefig(output_path)
    plt.close(fig)


def generate_fig4(output_path: Path) -> None:
    deck_json = parse_deck_json_from_html(DATA_DIR / "macao_microclimate_rise_heatmap2d.html")
    points = pd.DataFrame(deck_json["layers"][2]["data"])

    points["rise_plot"] = pd.to_numeric(points["rise_plot"], errors="coerce")
    points = points.dropna(subset=["lon", "lat", "rise_plot"])
    q_low = float(points["rise_plot"].quantile(0.02))
    q_high = float(points["rise_plot"].quantile(0.98))
    plot_values = points["rise_plot"].clip(lower=q_low, upper=q_high)

    fig, ax = plt.subplots(figsize=(8.8, 8.8), dpi=300)
    hb = ax.hexbin(
        points["lon"],
        points["lat"],
        C=plot_values,
        reduce_C_function=np.mean,
        gridsize=130,
        mincnt=1,
        cmap="Greys",
        linewidths=0.0,
    )
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_xlim(float(points["lon"].min()), float(points["lon"].max()))
    ax.set_ylim(float(points["lat"].min()), float(points["lat"].max()))
    ax.set_aspect("equal")
    x_formatter = ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.set_title("图4  澳门微气候温升空间分布图", fontsize=14, fontweight="bold", pad=10)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.82)
    cbar.set_label("微气候温升 / ℃")
    ax.text(0.03, 0.03, "注：色阶越深表示局地温升越高", transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
    fig.savefig(output_path)
    plt.close(fig)


def generate_fig5(output_path: Path) -> None:
    baseline_prop = pd.read_csv(DATA_DIR / "macao_building_property_ac_power.csv", low_memory=False)
    baseline_series = np.vstack(baseline_prop["discharging_power"].map(parse_array_text).tolist()).sum(axis=0) / 1e6
    cache = pd.read_csv(DATA_DIR / "macao_aggregated_flexibility_cache.csv")
    corrected_series = cache["aggregated_discharging_mw"].to_numpy(dtype=float)
    hours = np.arange(corrected_series.size)
    diff_energy = float(np.trapezoid(np.abs(corrected_series - baseline_series), dx=1.0))

    fig, ax = plt.subplots(figsize=(11.2, 5.8), dpi=300)
    ax.plot(hours, baseline_series, color="black", linewidth=1.8, linestyle="--", label="微气候修正前")
    ax.plot(hours, corrected_series, color="black", linewidth=2.0, linestyle="-", label="微气候修正后")
    ax.fill_between(
        hours,
        baseline_series,
        corrected_series,
        where=(corrected_series >= baseline_series),
        color="#bfbfbf",
        alpha=0.45,
        interpolate=True,
        label="差值区域",
    )
    idx_peak = int(np.nanargmax(np.abs(corrected_series - baseline_series)))
    y_mid = (baseline_series[idx_peak] + corrected_series[idx_peak]) / 2.0
    ax.text(
        hours[idx_peak],
        y_mid,
        f"{diff_energy:.3f} MWh",
        va="center",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.82, "edgecolor": "#888888"},
    )
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("聚合等效放电功率 / MW")
    ax.set_xticks(hours)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=True, loc="upper left")
    ax.set_title("图5  澳门建筑群聚合需求响应能力时序图", fontsize=14, fontweight="bold", pad=10)
    fig.savefig(output_path)
    plt.close(fig)


def generate_fig6(output_path: Path) -> None:
    deck_json = parse_deck_json_from_html(DATA_DIR / "carbon_emission_heatmap_3d.html")
    building_items = deck_json["layers"][0]["data"]

    polygons = []
    values = []
    for item in building_items:
        polygon_coords = item.get("polygon", [])
        if len(polygon_coords) < 4:
            continue
        try:
            polygons.append(Polygon(polygon_coords))
            values.append(float(item.get("carbon emission (kg)", 0.0)))
        except Exception:
            continue

    plot_gdf = gpd.GeoDataFrame({"carbon_kg": values}, geometry=polygons, crs="EPSG:4326")
    q_low = float(plot_gdf["carbon_kg"].quantile(0.02))
    q_high = float(plot_gdf["carbon_kg"].quantile(0.98))
    plot_gdf["_plot_value"] = plot_gdf["carbon_kg"].clip(lower=q_low, upper=q_high)

    fig, ax = plt.subplots(figsize=(9.2, 8.0), dpi=300)
    plot_gdf.plot(
        ax=ax,
        column="_plot_value",
        cmap="Greys",
        linewidth=0.04,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "碳减排潜力 / kg", "shrink": 0.78},
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("图6  澳门建筑碳减排潜力空间分布图", fontsize=14, fontweight="bold", pad=10)
    ax.text(0.03, 0.03, "注：色阶越深表示建筑减排潜力越高", transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
    fig.savefig(output_path)
    plt.close(fig)


def iter_figure_specs() -> Iterable[tuple[str, str, Path]]:
    return [
        ("图1  澳门气象站温度统计分布图", "fig1_weather_station_cn_bw.png", FIG_DIR / "fig1_weather_station_cn_bw.png"),
        ("图2  代表性建筑柔性时序图", "fig2_representative_building_cn_bw.png", FIG_DIR / "fig2_representative_building_cn_bw.png"),
        ("图3  澳门建筑密度空间分布图", "fig3_building_density_cn_bw.png", FIG_DIR / "fig3_building_density_cn_bw.png"),
        ("图4  澳门微气候温升空间分布图", "fig4_microclimate_rise_cn_bw.png", FIG_DIR / "fig4_microclimate_rise_cn_bw.png"),
        ("图5  澳门建筑群聚合需求响应能力时序图", "fig5_aggregated_dr_cn_bw.png", FIG_DIR / "fig5_aggregated_dr_cn_bw.png"),
        ("图6  澳门建筑碳减排潜力空间分布图", "fig6_carbon_distribution_cn_bw.png", FIG_DIR / "fig6_carbon_distribution_cn_bw.png"),
    ]


def generate_all_figures() -> dict[str, Path]:
    ensure_dir(FIG_DIR)
    generate_fig1(FIG_DIR / "fig1_weather_station_cn_bw.png")
    generate_fig2(FIG_DIR / "fig2_representative_building_cn_bw.png")
    generate_fig3(FIG_DIR / "fig3_building_density_cn_bw.png")
    generate_fig4(FIG_DIR / "fig4_microclimate_rise_cn_bw.png")
    generate_fig5(FIG_DIR / "fig5_aggregated_dr_cn_bw.png")
    generate_fig6(FIG_DIR / "fig6_carbon_distribution_cn_bw.png")
    return {caption: path for caption, _, path in iter_figure_specs()}


def update_docx_captions_and_embed(source_docx: Path, output_docx: Path, figure_map: dict[str, Path]) -> None:
    with zipfile.ZipFile(source_docx, "r") as zf:
        archive_data = {info.filename: zf.read(info.filename) for info in zf.infolist()}

    doc_root = ET.fromstring(archive_data["word/document.xml"])
    rel_root = ET.fromstring(archive_data["word/_rels/document.xml.rels"])
    content_root = ET.fromstring(archive_data["[Content_Types].xml"])

    body = doc_root.find(f".//{{{W_NS}}}body")
    if body is None:
        raise RuntimeError("Word document body was not found.")

    for child in list(body):
        text = flatten_text(child)
        if text in OLD_DESCRIPTION_TO_NEW:
            set_paragraph_text(child, OLD_DESCRIPTION_TO_NEW[text])
        elif text in OLD_CAPTION_TO_NEW:
            set_paragraph_text(child, OLD_CAPTION_TO_NEW[text])

    existing_rel_nums = []
    for rel in rel_root.findall(f"{{{REL_NS}}}Relationship"):
        rel_id = rel.get("Id", "")
        match = re.fullmatch(r"rId(\d+)", rel_id)
        if match:
            existing_rel_nums.append(int(match.group(1)))
    next_rel_num = max(existing_rel_nums, default=0) + 1
    docpr_id = 100

    defaults = {node.get("Extension") for node in content_root.findall(f"{{{CT_NS}}}Default")}
    if "png" not in defaults:
        ET.SubElement(
            content_root,
            f"{{{CT_NS}}}Default",
            {"Extension": "png", "ContentType": "image/png"},
        )

    children = list(body)
    insert_offset = 0
    for caption, image_path in figure_map.items():
        image_name = image_path.name
        media_target = f"word/media/{image_name}"
        archive_data[media_target] = image_path.read_bytes()

        rel_id = f"rId{next_rel_num}"
        next_rel_num += 1
        ET.SubElement(
            rel_root,
            f"{{{REL_NS}}}Relationship",
            {
                "Id": rel_id,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"media/{image_name}",
            },
        )

        image_width_px, image_height_px = Image.open(image_path).size
        paragraph = build_image_paragraph(
            rel_id=rel_id,
            image_name=image_name,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            width_cm=15.0,
            docpr_id=docpr_id,
        )
        docpr_id += 1

        target_index = None
        for idx, child in enumerate(children):
            if flatten_text(child) == caption:
                target_index = idx
                break
        if target_index is None:
            continue
        body.insert(target_index + 1 + insert_offset, paragraph)
        insert_offset += 1

    ET.register_namespace("w", W_NS)
    ET.register_namespace("r", R_NS)
    ET.register_namespace("wp", WP_NS)
    ET.register_namespace("a", A_NS)
    ET.register_namespace("pic", PIC_NS)
    ET.register_namespace("", REL_NS)

    archive_data["word/document.xml"] = ET.tostring(doc_root, encoding="utf-8", xml_declaration=True)
    archive_data["word/_rels/document.xml.rels"] = ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)
    archive_data["[Content_Types].xml"] = ET.tostring(content_root, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(output_docx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in archive_data.items():
            zf.writestr(name, payload)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    figure_map = generate_all_figures()
    update_docx_captions_and_embed(SOURCE_DOCX, OUTPUT_DOCX, figure_map)
    print(f"Generated {len(figure_map)} case-study patent figures.")
    print(f"Standalone figure folder created at: {FIG_DIR}")
    print(f"Patent docx with case-study figures created at: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
