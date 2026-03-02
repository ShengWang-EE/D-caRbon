from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from download_osm_macao_buildings import main as download_osm_macao_buildings
from src.building.building_thermal import (
    assign_air_exchange_rate,
    assign_uvalue_midpoint_knn,
    estimate_building_temp,
    thermal_constants,
)
from src.data_io.buildings import classify_building_use
from src.utils.viz import export_building_heatmap_3d_web

# %% 1 读取基础输入数据：温度序列、气象站位置、建筑数据
base_dir = Path(__file__).parent  # 假设 main.py 在 C DRcarbon 根目录
data_dir = base_dir / "data"

# 读取澳门温度数据和气象站位置（来自 CSV）
csv_path = data_dir / "Macao temperature 20230905.csv"
xls_all = pd.read_csv(csv_path, header=None)

macao_temperature = xls_all.iloc[1:289, 1:10].to_numpy()
macao_weather_station_location = xls_all.iloc[1:3, 13:22].to_numpy()

# 读取/获取澳门建筑数据（优先使用已下载的 OSM GPKG，如果不存在则先下载）
osm_buildings_path = data_dir / "osm_macao_buildings.gpkg"
if not osm_buildings_path.exists():
    download_osm_macao_buildings()

macao_building = gpd.read_file(osm_buildings_path, layer="buildings")

# 为建筑添加使用性质类别（residential/commercial/industrial/public/unknown）
macao_building = classify_building_use(
    macao_building,
    place="Macau, China",
    download_landuse=True,
)




# %% 2 针对每一栋建筑，计算温度、体积、热工参数以及对电网的灵活性指标
## 2.1 estimate temperature：根据气象站温度与建筑位置估计建筑温度序列或等效温度

building_temperature_series = []
for _, row in macao_building.iterrows():
    if "xBuilding" in macao_building.columns and "yBuilding" in macao_building.columns:
        building_location = [row["xBuilding"], row["yBuilding"]]
    else:
        centroid = row.geometry.centroid
        building_location = [centroid.x, centroid.y]

    building_temp = estimate_building_temp(
        macao_temperature,
        macao_weather_station_location,
        building_location,
    )
    building_temperature_series.append(building_temp)

macao_building["temperature"] = building_temperature_series

## 2.2 calcualte volume: building height x footprint area
if "Elevation" in macao_building.columns:
    height_series = pd.to_numeric(macao_building["Elevation"], errors="coerce")
elif "height" in macao_building.columns:
    height_series = pd.to_numeric(macao_building["height"], errors="coerce")
elif "building:levels" in macao_building.columns:
    levels = pd.to_numeric(macao_building["building:levels"], errors="coerce")
    height_series = levels * 3.0  # meters, assumed floor height
else:
    height_series = pd.Series(10.0, index=macao_building.index)  # default height

if "Shape_Area" in macao_building.columns:
    footprint_area_series = pd.to_numeric(macao_building["Shape_Area"], errors="coerce")
else:
    area_gdf = macao_building
    if area_gdf.crs is not None and area_gdf.crs.is_geographic:
        projected_crs = area_gdf.estimate_utm_crs()
        area_gdf = area_gdf.to_crs(projected_crs if projected_crs is not None else "EPSG:3857")
    footprint_area_series = area_gdf.geometry.area

macao_building["footprint_area_m2"] = footprint_area_series
macao_building["volume"] = height_series * footprint_area_series

macao_building["height_m"] = height_series

## 2.3 virtual storage capacity: 
constants = thermal_constants()
ca = constants["Ca"]
rhoa = constants["rhoa"]

if "T_max" in macao_building.columns:
    t_max_series = pd.to_numeric(macao_building["T_max"], errors="coerce")
else:
    t_max_series = pd.Series(26.0, index=macao_building.index)

if "T_min" in macao_building.columns:
    t_min_series = pd.to_numeric(macao_building["T_min"], errors="coerce")
else:
    t_min_series = pd.Series(22.0, index=macao_building.index)

macao_building["T_max"] = t_max_series
macao_building["T_min"] = t_min_series
delta_t_series = (t_max_series - t_min_series).clip(lower=0)
macao_building["energy_storage_capacity"] = ca * rhoa * macao_building["volume"] * delta_t_series



web_map_path = export_building_heatmap_3d_web(
    macao_building,
    value_column="volume",
    height_column="height_m",
    cmap="YlOrRd",
    output_html=str(data_dir / "volume_heatmap_3d.html"),
)
print(f"3D web map saved to: {web_map_path}")



## 2.3 estimate U value：估算建筑围护结构的传热系数 U 值（W/m²K）
macao_building = assign_uvalue_midpoint_knn(macao_building, k=7)
macao_building = assign_air_exchange_rate(macao_building)

## 2.4 charging power:
## f^{c,chr,max}_{12,j} = (T_max - T_star) * K_j * A_s_j
##                       + C_a * rho_a * V_j * (T_max - T_star) * N_ex_j
if "T_star" in macao_building.columns:
    t_star_series = pd.to_numeric(macao_building["T_star"], errors="coerce")
else:
    t_star_series = (macao_building["T_max"] + macao_building["T_min"]) / 2.0

n_ex_series = pd.to_numeric(macao_building["N_ex"], errors="coerce")

macao_building["T_star"] = t_star_series
macao_building["N_ex"] = n_ex_series

a_s_series = macao_building[["A_wall", "A_roof", "A_win"]].sum(axis=1, min_count=1)
a_s_series = a_s_series.fillna(macao_building["footprint_area_m2"])
macao_building["A_s"] = a_s_series

delta_t_chr = (macao_building["T_max"] - macao_building["T_star"]).clip(lower=0)
k_series = pd.to_numeric(macao_building["Uvalue"], errors="coerce")
v_series = pd.to_numeric(macao_building["volume"], errors="coerce")

conduction_term = delta_t_chr * k_series * macao_building["A_s"]
ventilation_term = ca * rhoa * v_series * delta_t_chr * macao_building["N_ex"]
macao_building["charging_power"] = conduction_term + ventilation_term

## 2.5 discharging power:
## f^{c,dis,max}_{12,j} = (T_star - T_min) * K_j * A_s_j
##                       + C_a * rho_a * V_j * (T_star - T_min) * N_ex_j
delta_t_dis = (macao_building["T_star"] - macao_building["T_min"]).clip(lower=0)
conduction_term_dis = delta_t_dis * k_series * macao_building["A_s"]
ventilation_term_dis = ca * rhoa * v_series * delta_t_dis * macao_building["N_ex"]
macao_building["discharging_power"] = conduction_term_dis + ventilation_term_dis



## 2.3 计算热工参数：U/COP/内部得热/热容

#     % calculate COP：估算建筑对应空调/热泵系统的能效系数
#     macaoBuilding(i).COP = estimateCOP();
#     % estimate inner heat：估算人员与设备带来的内部产热
#     macaoBuilding(i).varepsilon = estimateInnerHeat();
#     % estimate U value：估算建筑围护结构的传热系数 U 值
#     macaoBuilding(i).Uvalue = estimateUvalue();
#     % calculate flexibility：根据热惯性与舒适区计算建筑对电网的等效储能与充放电能力
#     [capacity, selfDissipation, charging, discharging, duration] = calculateFlexibility(macaoBuilding(i));
# end

# %% 3 进行电力系统UC，建立稳态值


# %% 4 在稳态值的基础上进行DR，看看边际碳排放削减


flag = 1




