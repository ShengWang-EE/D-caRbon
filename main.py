from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_osm_macao_buildings import main as download_osm_macao_buildings
from src.building.building_thermal import (
    compute_virtual_storage_related_state,
    estimate_building_temperature_series,
    prepare_building_thermal_parameters,
    to_temperature_scalar,
)
from src.cluster.micro_meteorology import run_ac_micro_meteorology_pipeline
from src.data_io.buildings import classify_building_use
from src.data_io.weather import load_macao_temperature_window, load_nasa_power_ghi_window
from src.utils.plotting import (
    export_building_density_3d_map_web,
    export_building_heatmap_3d_web,
    export_microclimate_rise_heatmap2d_web,
    plot_aggregated_discharging_power_timeseries,
    plot_building_area_height_carbon_scatter,
    plot_discharging_power_max_histogram,
    plot_individual_building_flexibility_time_series,
    plot_weather_station_violin_all_years,
)
from src.utils.pipeline_helpers import (
    compact_building_property,
    compute_ac_power_at_comfort_setpoint,
    prepare_building_locations,
    recompute_after_microclimate,
    to_numeric_scalar,
    vector_delta_series,
)
from UK_power_system_generator import data_reader_writer

# %% 1 preparation
# 1.1 Read baseline input data: temperature series, weather stations, buildings
base_dir = Path(__file__).parent  # Assume main.py is in the project root
data_dir = base_dir / "data"

# Load Macau temperature data by date window from cleaned weather table.
macao_start_date = "2025-07-28 00:00:00+08:00"
macao_end_date = "2025-07-28 23:00:00+08:00"
macao_temperature = load_macao_temperature_window(data_dir, macao_start_date, macao_end_date)

# Keep using station locations from the legacy metadata CSV.
station_meta_path = data_dir / "Macao temperature 20230905.csv"
station_meta = pd.read_csv(station_meta_path, header=None)
macao_weather_station_location = station_meta.iloc[1:3, 13:22].to_numpy()

# 1.2 Load/obtain Macau building data (prefer local OSM GPKG; download if missing)
osm_buildings_path = data_dir / "osm_macao_buildings.gpkg"
if not osm_buildings_path.exists():
    download_osm_macao_buildings()

macao_building = gpd.read_file(osm_buildings_path, layer="buildings")
osm_raw_columns = set(macao_building.columns)

# Add building use category (residential/commercial/industrial/public/unknown)
macao_building, macao_building_property = classify_building_use(
    macao_building,
    place="Macau, China",
    download_landuse=True,
    cache_dir=data_dir / "cache",
)

# fig: plot weather station (all historical years)
weather_violin_fig, weather_station_temp_summary = plot_weather_station_violin_all_years(data_dir)

# %% 2 Building-level temperature, geometry, thermal params, and flexibility
# 2.1 estimate temperature: infer building temperature from station temperature + location
macao_building, macao_building_property, building_locations = prepare_building_locations(
    macao_building,
    macao_building_property,
)

building_temperature_series, macao_building, macao_building_property = estimate_building_temperature_series(
    macao_temperature,
    macao_weather_station_location,
    building_locations,
    macao_building,
    macao_building_property,
)

# 2.2 Prepare thermal parameters (geometry/U-value/air-exchange)
macao_building, macao_building_property = prepare_building_thermal_parameters(
    macao_building,
    osm_raw_columns=osm_raw_columns,
    height_knn_k=5,
    height_default_m=10.0,
    uvalue_knn_k=7,
)

# 2.3 Compute virtual storage related quantities from building temperature
macao_building, macao_building_property = compute_virtual_storage_related_state(
    macao_building,
    building_temperature_series,
    building_temperature_col="temperature",
    osm_raw_columns=osm_raw_columns,
)

# fig: 画出澳门全岛的建筑密度，用3D图展示
density_3d_map_path = export_building_density_3d_map_web(
    macao_building,
    output_html=str(data_dir / "macao_building_density_3d_map.html"),
    radius_m=20.0,
    elevation_scale=3.2,
    coverage=1.0,
    upper_percentile=99.0,
)
print(f"3D building-density map (web) saved to: {density_3d_map_path}")


# %% 3 计算微气象效应
baseline_building = macao_building.copy()

# 从 NASA POWER 读取逐小时全球水平辐射，供负荷和UHI时间权重使用。
if macao_building.crs is not None and macao_building.crs.is_geographic:
    bld_wgs84 = macao_building
else:
    bld_wgs84 = macao_building.to_crs("EPSG:4326")
centroids_wgs84 = bld_wgs84.geometry.centroid
site_lat = float(pd.to_numeric(centroids_wgs84.y, errors="coerce").mean())
site_lon = float(pd.to_numeric(centroids_wgs84.x, errors="coerce").mean())

ghi_series = load_nasa_power_ghi_window(
    start_date=macao_start_date,
    end_date=macao_end_date,
    latitude=site_lat,
    longitude=site_lon,
)
macao_building["I_solar"] = pd.Series([ghi_series.copy() for _ in macao_building.index], index=macao_building.index, dtype=object)

# 计算空调功率
macao_building, macao_building_property = compute_ac_power_at_comfort_setpoint(
    macao_building,
    output_dir=data_dir,
)

# 计算微气象温升.
macao_building, grid_stats, macao_building_property = run_ac_micro_meteorology_pipeline(
    macao_building,
    building_temperature_col="temperature",
    osm_raw_columns=osm_raw_columns,
    output_dir=data_dir,
    grid_size_m=250.0,
    grid_id_col="grid_id",
    place_name="Macau, China",
    clip_to_boundary=True,
    window_start_hour=14.0,
    window_end_hour=18.0,
    write_step1_files=True,
    write_step2_files=False,
    reuse_cached_mapping=True,
    apply_building_idw=False,
    idw_neighbors=4,
    idw_power=2.0,
)

# fig: 画出微气候温升的空间分布2D热力图（地图热力层，不绘制离散点）
# 总温升 = 空调排热温升(delta_t_ac_i, 时间向量均值) + UHI温升(delta_t_uhi, 标量)
macao_building["delta_t_total"] = (
    macao_building["delta_t_ac_i"].apply(to_numeric_scalar)
    + pd.to_numeric(macao_building["delta_t_uhi"], errors="coerce").fillna(0.0)
)
microclimate_rise_map_path, microclimate_rise_png_path = export_microclimate_rise_heatmap2d_web(
    macao_building,
    value_column="delta_t_total",
    output_html=str(data_dir / "macao_microclimate_rise_heatmap2d.html"),
    output_png=str(data_dir / "macao_microclimate_rise_heatmap2d.png"),
    radius_m=65.0,
    intensity=1.0,
    threshold=0.08,
    heatmap_opacity=0.72,
    min_weight_floor=0.0,
    color_clip_quantiles=(0.005, 0.995),
    map_style="osm",
)
print(f"Microclimate-rise 2D heatmap saved to: {microclimate_rise_map_path}")
if microclimate_rise_png_path is not None:
    print(f"Microclimate-rise 2D heatmap PNG saved to: {microclimate_rise_png_path}")

# Recompute virtual energy-storage related states under local microclimate temperatures.
macao_building_updated, macao_building_property_updated = compute_virtual_storage_related_state(
    macao_building,
    macao_building["T_local_i"],
    building_temperature_col="temperature",
    osm_raw_columns=osm_raw_columns,
)

macao_building_updated, macao_building_property_updated, macao_building_property_comparison, microclimate_impact_summary, aggregated_capacity_mwh, aggregated_discharging_mw = recompute_after_microclimate(
    baseline_building,
    macao_building_updated,
    macao_building_property_updated,
    data_dir=data_dir,
)

# plot fig 1: 挑选一个澳门具有代表性的建筑（大商场/赌场），画出
# charging/discharging power、equivalent energy storage capacity 的时间序列
# 横坐标仅显示小时（Hour），不显示完整日期，避免标签拥挤。
# 这里选择多个代表性建筑，覆盖多种类型（商业/公共/居民/工业）。
# 若已知某类型代表建筑id，可直接填写；设为 None 时自动在该类型中选取面积最大的建筑。
representative_requests = [
    {"label": "commercial building", "use_category": "commercial", "building_id": 165},
    {"label": "public building", "use_category": "public", "building_id": None},
    {"label": "residential building", "use_category": "residential", "building_id": None},
    {"label": "industrial building", "use_category": "industrial", "building_id": None},
]

fig1, representative_building_info = plot_individual_building_flexibility_time_series(
    macao_building_updated,
    macao_building_property_updated,
    representative_requests,
    time_index=pd.date_range(macao_start_date, macao_end_date, freq="h"),
)
fig1_output_path = data_dir / "individual_building_flexibility_timeseries.png"
fig1.savefig(fig1_output_path, dpi=180, bbox_inches="tight")
print(f"Representative buildings flexibility figure saved to: {fig1_output_path}")

# fig: 澳门全部建筑的discharging power的总和的时间序列，画出两条曲线，分别是微气候修正前和修正后的。
fig_dis_output_path, fig_dis_integral_mwh = plot_aggregated_discharging_power_timeseries(
    baseline_building["discharging_power"],
    aggregated_discharging_mw,
    start_datetime=macao_start_date,
    output_path=data_dir / "aggregated_discharging_power_timeseries.png",
    time_step_hours=1.0,
    dpi=180,
)
print(f"Aggregated discharging-power timeseries figure saved to: {fig_dis_output_path}")

# fig: 画出澳门建筑discharging power每天最大值的histogram，区分微气象修正前和修正后
fig_hist_output_path = plot_discharging_power_max_histogram(
    baseline_building["discharging_power"],
    macao_building_property_updated["discharging_power"],
    output_path=data_dir / "discharging_power_max_histogram.png",
    x_limits_mw=(0.0, 0.1),
    dpi=180,
)
print(f"Building max discharging power histogram saved to: {fig_hist_output_path}")


# %% 4 计算碳排放及画图
# 中国南方电网碳排放系数, 2023年，约0.67 kgCO₂/kWh
carbon_emission_factor = 0.67

# 计算建筑碳排放 (kg): keep array-valued results, and derive a scalar view for map rendering.
macao_building_property_updated["carbon_emission"] = macao_building_property_updated["discharging_power"].apply(
    lambda arr: np.asarray(arr, dtype=float) / 1e3 / 3600 * carbon_emission_factor * 8760
)
macao_building_updated["carbon_emission"] = macao_building_property_updated["carbon_emission"].to_numpy()
macao_building_updated["carbon_emission_scalar"] = (
    macao_building_property_updated["carbon_emission"].apply(to_numeric_scalar).to_numpy()
)

# 按当前年化碳指标口径，汇总所有建筑的降碳潜力并换算为日平均值。
total_daily_avg_carbon_reduction_kg = (
    pd.to_numeric(macao_building_updated["carbon_emission_scalar"], errors="coerce").fillna(0.0).sum()
)
print(f"Daily-average total carbon-reduction potential across all buildings: {total_daily_avg_carbon_reduction_kg:.2f} kgCO2")

# fig：碳排放热力图
web_map_path = export_building_heatmap_3d_web(
    macao_building_updated,
    value_column="carbon_emission_scalar",
    height_column="height_m",
    cmap="turbo",
    color_scale="log",
    color_clip_quantiles=(0.02, 0.98),
    legend_unit="kg",
    output_html=str(data_dir / "carbon_emission_heatmap_3d.html"),
)
print(f"3D web map saved to: {web_map_path}")

# fig: 再画一个散点图，横坐标是建筑footprint_area_m2，纵坐标是height_m，点的大小是碳排放量，点的颜色是建筑用途类型
scatter_output_path = plot_building_area_height_carbon_scatter(
    macao_building_updated,
    area_column="footprint_area_m2",
    height_column="height_m",
    size_column="carbon_emission_scalar",
    category_column="use_category",
    output_path=data_dir / "building_area_height_carbon_scatter.png",
    dpi=180,
)
print(f"Building area-height-carbon scatter figure saved to: {scatter_output_path}")

flag = 1