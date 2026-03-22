from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "energy system"
PYPSA_ROOT = DATA_ROOT / "PyPSA-GB_data"


def _read_excel(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """读取 Excel 文件，缺少 openpyxl 时给出明确提示。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {p}")
    try:
        return pd.read_excel(p, **kwargs)
    except ImportError as exc:
        raise ImportError("读取 Excel 需要 openpyxl，请执行: python -m pip install openpyxl") from exc


def _normalize_name(name: Any) -> str:
    """将文本归一化为仅包含小写字母数字，便于列名匹配。"""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """从候选列名中选出 DataFrame 内最匹配的一列。"""
    cols = list(df.columns)
    norm_map = {_normalize_name(c): c for c in cols}

    for c in candidates:
        if c in cols:
            return c

    for c in candidates:
        n = _normalize_name(c)
        if n in norm_map:
            return norm_map[n]

    for c in candidates:
        n = _normalize_name(c)
        for col in cols:
            if n and n in _normalize_name(col):
                return col

    raise KeyError(f"无法匹配列名，候选={candidates}，已有={cols}")


def _haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """计算单点到多点的球面距离（单位：公里）。"""
    r = 6371.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lats.astype(float))
    lon2 = np.radians(lons.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _nearest_bus_ids(point_lats: np.ndarray, point_lons: np.ndarray, bus_lats: np.ndarray, bus_lons: np.ndarray) -> np.ndarray:
    """将坐标映射到最近母线编号（从 1 开始）。"""
    if len(bus_lats) == 0:
        return np.zeros(len(point_lats), dtype=int)

    out = np.zeros(len(point_lats), dtype=int)
    for i, (plat, plon) in enumerate(zip(point_lats, point_lons)):
        d = _haversine_km(float(plat), float(plon), bus_lats, bus_lons)
        out[i] = int(np.argmin(d)) + 1
    return out


def _extract_last_cost(cost_df: pd.DataFrame, keywords: list[str]) -> float:
    """提取匹配关键词的最新燃料价格，并转换到 GBP/MWh。"""
    target = None
    for col in cost_df.columns:
        norm = _normalize_name(col)
        if all(k in norm for k in keywords):
            target = col
            break

    if target is None:
        return np.nan

    series = pd.to_numeric(cost_df[target], errors="coerce").dropna()
    if series.empty:
        return np.nan

    return float(series.iloc[-1] / 100.0 * 1000.0)


def _df_to_numeric_array(df: pd.DataFrame) -> np.ndarray:
    """将 DataFrame 转换为纯数值二维数组。"""
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def preprocess_UK_power_demand_data(input_files: list[str | Path] | None = None, output_file: str | Path | None = None) -> pd.DataFrame:
    """预处理英国电力需求原始数据，统一时间戳并排序。"""
    if input_files is None:
        input_files = [DATA_ROOT / "power demand (from gridwatch)" / "power demand 2011-2025 (halfhour).csv"]

    frames: list[pd.DataFrame] = []
    for f in input_files:
        p = Path(f)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"电力需求文件不存在: {p}")
        frames.append(pd.read_csv(p))

    data = pd.concat(frames, ignore_index=True)
    ts_col = _pick_col(data, ["timestamp", "Timestamp"])
    demand_col = _pick_col(data, ["demand", "Demand"])

    data[ts_col] = pd.to_datetime(data[ts_col], errors="coerce")
    data[demand_col] = pd.to_numeric(data[demand_col], errors="coerce")
    data = data.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)

    if output_file is not None:
        out = Path(output_file)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out, index=False)

    return data


def preprocess_UK_gas_demand_data(filename: str | Path, data_root: str | Path = DATA_ROOT) -> pd.DataFrame:
    """预处理英国燃气需求 CSV，输出按日期聚合的宽表。"""
    base = Path(data_root)
    p = Path(filename)
    if not p.is_absolute():
        p = base / "gas demand" / "raw" / p

    if not p.exists():
        raise FileNotFoundError(f"燃气需求文件不存在: {p}")

    raw = pd.read_csv(p)
    date_col = _pick_col(raw, ["Applicable For", "date"])
    item_col = _pick_col(raw, ["Data Item", "item"])
    value_col = _pick_col(raw, ["Value", "value"])

    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")

    table = (
        raw.pivot_table(index=date_col, columns=item_col, values=value_col, aggfunc="mean")
        .sort_index()
        .reset_index()
    )

    rename_map = {
        "NTS Energy Offtaken, Industrial Offtake Total": "NTSEnergyOfftaken_IndustrialOfftakeTotal",
        "NTS Energy Offtaken, Interconnector Exports Total": "NTSEnergyOfftaken_InterconnectorExportsTotal",
        "NTS Energy Offtaken, LDZ Offtake Total": "NTSEnergyOfftaken_LDZOfftakeTotal",
        "NTS Energy Offtaken, Powerstations Total": "NTSEnergyOfftaken_PowerstationsTotal",
        "NTS Energy Offtaken, Storage Injection Total": "NTSEnergyOfftaken_StorageInjectionTotal",
    }
    return table.rename(columns=rename_map)


def data_reader_writer(option: int = 1) -> dict[str, Any]:
    """按 MATLAB 原逻辑构建英国电-气系统 mpc1 数据结构。"""
    # electricity demand
    gridwatch_data = preprocess_UK_power_demand_data()
    ts_col = _pick_col(gridwatch_data, ["timestamp"])
    demand_col = _pick_col(gridwatch_data, ["demand"])

    selected = gridwatch_data[ts_col].between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-30") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    electricity_system_data = gridwatch_data.loc[selected].copy()
    electricity_system_data[demand_col] = pd.to_numeric(electricity_system_data[demand_col], errors="coerce").interpolate(method="linear", limit_direction="both")

    demand_distribution_info = pd.read_csv(PYPSA_ROOT / "demand" / "Demand_Distribution.csv")
    dist_col = "2024" if "2024" in demand_distribution_info.columns else demand_distribution_info.columns[1]
    distribution = pd.to_numeric(demand_distribution_info[dist_col], errors="coerce").fillna(0.0).to_numpy()
    distribution = distribution / distribution.sum()

    demand_series = electricity_system_data[demand_col].to_numpy(dtype=float)
    electricity_nodal_demand = np.outer(distribution, demand_series)
    mean_electric_demand = electricity_nodal_demand.mean(axis=1)

    # bus
    bus_ref = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="bus")
    bus_info = pd.read_csv(PYPSA_ROOT / "network" / "BusesBasedGBsystem" / "buses.csv")
    carrier_col = _pick_col(bus_info, ["carrier"])
    bus_ac = bus_info.loc[bus_info[carrier_col].astype(str) == "AC"].reset_index(drop=True)
    nb = len(bus_ac)

    # bus.Type：优先用参考表，长度不匹配时自动截断或补默认值
    try:
        type_col = _pick_col(bus_ref, ["Type"])
        bus_type = pd.to_numeric(bus_ref[type_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    except KeyError:
        bus_type = np.full(nb, 1.0, dtype=float)
    if bus_type.size >= nb:
        bus_type = bus_type[:nb]
    else:
        bus_type = np.pad(bus_type, (0, nb - bus_type.size), constant_values=1.0)

    # bus.Pd：长度与交流母线数对齐
    bus_pd = np.asarray(mean_electric_demand, dtype=float).reshape(-1)
    if bus_pd.size >= nb:
        bus_pd = bus_pd[:nb]
    else:
        fill = float(np.nanmean(bus_pd)) if bus_pd.size > 0 else 0.0
        bus_pd = np.pad(bus_pd, (0, nb - bus_pd.size), constant_values=fill)

    bus = pd.DataFrame(
        {
            "BusID": np.arange(1, nb + 1, dtype=float),
            "Type": bus_type,
            "Pd": bus_pd,
            "Qd": np.zeros(nb, dtype=float),
            "Gs": np.zeros(nb, dtype=float),
            "Bs": np.zeros(nb, dtype=float),
            "ESystemID": np.ones(nb, dtype=float),
            "Vm": np.ones(nb, dtype=float),
            "Va": np.zeros(nb, dtype=float),
            "BaseKV": pd.to_numeric(bus_ac[_pick_col(bus_ac, ["v_nom"])], errors="coerce").fillna(0.0),
            "Zone": np.ones(nb, dtype=float),
            "Vmax": 1.1 * np.ones(nb, dtype=float),
            "Vmin": 0.9 * np.ones(nb, dtype=float),
        }
    )
    if nb > 0 and (bus["Type"] == 1).all():
        bus.loc[0, "Type"] = 3

    bus_extra = pd.DataFrame(
        {
            "name": bus_ac[_pick_col(bus_ac, ["name"])].astype(str).to_numpy(),
            "lon": pd.to_numeric(bus_ac[_pick_col(bus_ac, ["x", "lon", "longitude"])], errors="coerce").to_numpy(),
            "lat": pd.to_numeric(bus_ac[_pick_col(bus_ac, ["y", "lat", "latitude"])], errors="coerce").to_numpy(),
        }
    )

    mpc: dict[str, Any] = {"bus": bus, "bus_extra": bus_extra}

    # branch
    branch_ref = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="branch")
    branch_info = pd.read_csv(PYPSA_ROOT / "network" / "BusesBasedGBsystem" / "lines.csv")
    n_branch = len(branch_info)
    name_to_id = dict(zip(bus_extra["name"], bus["BusID"].astype(int)))

    def branch_ref_col(name: str, default: float) -> np.ndarray:
        try:
            col = _pick_col(branch_ref, [name])
            arr = pd.to_numeric(branch_ref[col], errors="coerce").fillna(default).to_numpy(dtype=float)
        except KeyError:
            arr = np.full(n_branch, default, dtype=float)
        if arr.size >= n_branch:
            return arr[:n_branch]
        return np.pad(arr, (0, n_branch - arr.size), constant_values=default)

    s_nom = pd.to_numeric(branch_info[_pick_col(branch_info, ["s_nom"])], errors="coerce").fillna(0.0)
    branch = pd.DataFrame(
        {
            "FromBus": branch_info[_pick_col(branch_info, ["bus0"])].astype(str).map(name_to_id).fillna(0).astype(float),
            "ToBus": branch_info[_pick_col(branch_info, ["bus1"])].astype(str).map(name_to_id).fillna(0).astype(float),
            "r": pd.to_numeric(branch_info[_pick_col(branch_info, ["r"])], errors="coerce").fillna(0.0),
            "x": pd.to_numeric(branch_info[_pick_col(branch_info, ["x"])], errors="coerce").fillna(0.0),
            "b": pd.to_numeric(branch_info[_pick_col(branch_info, ["b"])], errors="coerce").fillna(0.0),
            "RateA": s_nom,
            "RateB": s_nom,
            "RateC": s_nom,
            "K": branch_ref_col("K", 1.0),
            "Angle": branch_ref_col("Angle", 0.0),
            "Status": branch_ref_col("Status", 1.0),
            "ang_min": -360 * np.ones(n_branch, dtype=float),
            "ang_max": 360 * np.ones(n_branch, dtype=float),
        }
    )
    mpc["branch"] = branch
    mpc["branch_extra"] = pd.DataFrame(
        {
            "fbName": branch_info[_pick_col(branch_info, ["bus0"])].astype(str),
            "tbName": branch_info[_pick_col(branch_info, ["bus1"])].astype(str),
        }
    )

    # gen + gencost
    gen_info = pd.read_csv(PYPSA_ROOT / "power stations" / "power_stations_locations_2020.csv")
    n_gen = len(gen_info)
    gen_cols = [
        "BusID", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "Status", "Pmax", "Pmin",
        "Pc1", "Pc2", "Qc1min", "Qc1max", "Qc2min", "Qc2max", "RampRate", "ramp_10", "ramp_30", "ramp_q", "apf",
    ]
    gen = pd.DataFrame(np.zeros((n_gen, 21), dtype=float), columns=gen_cols)

    gen_lon = pd.to_numeric(gen_info[_pick_col(gen_info, ["x", "lon", "longitude"])], errors="coerce").fillna(0.0).to_numpy()
    gen_lat = pd.to_numeric(gen_info[_pick_col(gen_info, ["y", "lat", "latitude"])], errors="coerce").fillna(0.0).to_numpy()
    gen["BusID"] = _nearest_bus_ids(gen_lat, gen_lon, bus_extra["lat"].to_numpy(), bus_extra["lon"].to_numpy())
    gen["Pmax"] = pd.to_numeric(gen_info[_pick_col(gen_info, ["Installed Capacity (MW)", "InstalledCapacity_MW_", "InstalledCapacityMW"])], errors="coerce").fillna(0.0)
    gen["Pmin"] = 0.0
    gen["Vg"] = 1.0
    gen["mBase"] = 100.0
    gen["Status"] = 1.0

    gen_fuel_info = pd.read_csv(PYPSA_ROOT / "generator_data_by_fuel.csv")
    cost_path = PYPSA_ROOT / "marginal_cost_data.xlsx"
    fuel_cost_info = _read_excel(cost_path) if cost_path.exists() else pd.read_csv(PYPSA_ROOT / "marginal_cost_data")

    fuel_col = _pick_col(gen_info, ["Fuel"])
    tech_col = _pick_col(gen_info, ["Technology"])

    gen_extra = pd.DataFrame(
        {
            "lon": gen_lon,
            "lat": gen_lat,
            "min_up_time": np.zeros(n_gen, dtype=float),
            "min_down_time": np.zeros(n_gen, dtype=float),
            "ramp_up": np.zeros(n_gen, dtype=float),
            "ramp_down": np.zeros(n_gen, dtype=float),
            "fuel": gen_info[fuel_col].astype(str).to_numpy(),
            "tech": gen_info[tech_col].astype(str).to_numpy(),
        }
    )

    fuel_lookup = {_normalize_name(v): i for i, v in enumerate(gen_fuel_info[_pick_col(gen_fuel_info, ["fuel"])].astype(str))}
    min_up_col = _pick_col(gen_fuel_info, ["min_up_time"])
    min_down_col = _pick_col(gen_fuel_info, ["min_down_time"])
    ramp_up_col = _pick_col(gen_fuel_info, ["ramp_limit_up"])
    ramp_down_col = _pick_col(gen_fuel_info, ["ramp_limit_down"])
    startup_col = _pick_col(gen_fuel_info, ["start_up_cost"])
    marginal_col = _pick_col(gen_fuel_info, ["marginal_costs"])

    coal_price = _extract_last_cost(fuel_cost_info, ["coal", "kwh"])
    oil_price = _extract_last_cost(fuel_cost_info, ["oil", "kwh"])
    gas_price = _extract_last_cost(fuel_cost_info, ["gas", "kwh"])

    gencost_start = np.zeros(n_gen, dtype=float)
    gencost_marginal = np.zeros(n_gen, dtype=float)

    for i in range(n_gen):
        idx = fuel_lookup.get(_normalize_name(gen_info.at[i, fuel_col]))
        if idx is None:
            idx = fuel_lookup.get(_normalize_name(gen_info.at[i, tech_col]))
        if idx is None:
            continue

        gen_extra.at[i, "min_up_time"] = float(gen_fuel_info.at[idx, min_up_col])
        gen_extra.at[i, "min_down_time"] = float(gen_fuel_info.at[idx, min_down_col])
        gen_extra.at[i, "ramp_up"] = float(gen_fuel_info.at[idx, ramp_up_col])
        gen_extra.at[i, "ramp_down"] = float(gen_fuel_info.at[idx, ramp_down_col])
        gencost_start[i] = float(gen_fuel_info.at[idx, startup_col])
        gencost_marginal[i] = float(gen_fuel_info.at[idx, marginal_col])

        fuel_norm = _normalize_name(gen_fuel_info.at[idx, _pick_col(gen_fuel_info, ["fuel"])])
        if "coal" in fuel_norm and not np.isnan(coal_price):
            gencost_marginal[i] = coal_price
        elif "oil" in fuel_norm and not np.isnan(oil_price):
            gencost_marginal[i] = oil_price
        elif ("ccgt" in fuel_norm or "ocgt" in fuel_norm or "gas" in fuel_norm) and not np.isnan(gas_price):
            gencost_marginal[i] = 0.0 if option == 2 else gas_price

    gencost = pd.DataFrame(
        {
            "CostType": 2.0 * np.ones(n_gen, dtype=float),
            "StartCost": gencost_start,
            "ShutCost": np.zeros(n_gen, dtype=float),
            "Order": 3.0 * np.ones(n_gen, dtype=float),
            "CostA": np.zeros(n_gen, dtype=float),
            "CostB": gencost_marginal,
            "CostC": np.zeros(n_gen, dtype=float),
        }
    )

    # renewable
    solar_info = pd.read_csv(PYPSA_ROOT / "renewables" / "atlite" / "inputs" / "Solar_Photovoltaics" / "Solar_Photovoltaics_2020.csv")
    onshore_info = pd.read_csv(PYPSA_ROOT / "renewables" / "atlite" / "inputs" / "Wind_Onshore" / "Wind_Onshore_2020.csv")
    offshore_info = pd.read_csv(PYPSA_ROOT / "renewables" / "atlite" / "inputs" / "Wind_Offshore" / "Wind_Offshore_2020.csv")

    solar_lon = pd.to_numeric(solar_info[_pick_col(solar_info, ["x"])], errors="coerce").fillna(0.0).to_numpy()
    solar_lat = pd.to_numeric(solar_info[_pick_col(solar_info, ["y"])], errors="coerce").fillna(0.0).to_numpy()
    onshore_lon = pd.to_numeric(onshore_info[_pick_col(onshore_info, ["x"])], errors="coerce").fillna(0.0).to_numpy()
    onshore_lat = pd.to_numeric(onshore_info[_pick_col(onshore_info, ["y"])], errors="coerce").fillna(0.0).to_numpy()
    offshore_lon = pd.to_numeric(offshore_info[_pick_col(offshore_info, ["x"])], errors="coerce").fillna(0.0).to_numpy()
    offshore_lat = pd.to_numeric(offshore_info[_pick_col(offshore_info, ["y"])], errors="coerce").fillna(0.0).to_numpy()

    solar_bus = _nearest_bus_ids(solar_lat, solar_lon, bus_extra["lat"].to_numpy(), bus_extra["lon"].to_numpy())
    onshore_bus = _nearest_bus_ids(onshore_lat, onshore_lon, bus_extra["lat"].to_numpy(), bus_extra["lon"].to_numpy())
    offshore_bus = _nearest_bus_ids(offshore_lat, offshore_lon, bus_extra["lat"].to_numpy(), bus_extra["lon"].to_numpy())

    solar_capacity = pd.to_numeric(solar_info[_pick_col(solar_info, ["Installed Capacity (MWelec)", "InstalledCapacity_MWelec_", "InstalledCapacityMWelec"])], errors="coerce").fillna(0.0).to_numpy()
    onshore_capacity = (
        pd.to_numeric(onshore_info[_pick_col(onshore_info, ["Turbine Capacity (MW)", "TurbineCapacity_MW_", "TurbineCapacityMW"])], errors="coerce").fillna(0.0).to_numpy()
        * pd.to_numeric(onshore_info[_pick_col(onshore_info, ["No. of Turbines", "No_OfTurbines"])], errors="coerce").fillna(0.0).to_numpy()
    )
    offshore_capacity = (
        pd.to_numeric(offshore_info[_pick_col(offshore_info, ["Turbine Capacity (MW)", "TurbineCapacity_MW_", "TurbineCapacityMW"])], errors="coerce").fillna(0.0).to_numpy()
        * pd.to_numeric(offshore_info[_pick_col(offshore_info, ["No. of Turbines", "No_OfTurbines"])], errors="coerce").fillna(0.0).to_numpy()
    )

    renewable = pd.DataFrame(
        {
            "lon": np.concatenate([solar_lon, onshore_lon, offshore_lon]),
            "lat": np.concatenate([solar_lat, onshore_lat, offshore_lat]),
            "bus": np.concatenate([solar_bus, onshore_bus, offshore_bus]),
            "Pmax": np.concatenate([solar_capacity, onshore_capacity, offshore_capacity]),
            "Pmin": 0.0,
            "min_up_time": 0.0,
            "min_down_time": 0.0,
            "ramp_up": np.concatenate([solar_capacity, onshore_capacity, offshore_capacity]),
            "ramp_down": np.concatenate([solar_capacity, onshore_capacity, offshore_capacity]),
            "fuel": (["Solar"] * len(solar_info)) + (["Onshore wind"] * len(onshore_info)) + (["Offshore wind"] * len(offshore_info)),
            "tech": (["Solar"] * len(solar_info)) + (["Onshore wind"] * len(onshore_info)) + (["Offshore wind"] * len(offshore_info)),
        }
    )

    year = 2020
    solar_output_data = pd.concat([pd.read_csv(PYPSA_ROOT / "renewables" / "atlite" / "outputs" / "PV" / f"PV_{year}_{i}.csv") for i in [1, 2, 3, 4]], ignore_index=True)
    onshore_output_data = pd.read_csv(PYPSA_ROOT / "renewables" / "atlite" / "outputs" / "Wind_Onshore" / f"Wind_Onshore_{year}.csv")
    offshore_output_data = pd.read_csv(PYPSA_ROOT / "renewables" / "atlite" / "outputs" / "Wind_Offshore" / f"Wind_Offshore_{year}.csv")

    mean_renewable_capacity = np.concatenate(
        [
            pd.to_numeric(solar_output_data.iloc[:, 1:].stack(), errors="coerce").unstack().mean(axis=0).to_numpy(dtype=float),
            pd.to_numeric(onshore_output_data.iloc[:, 1:].stack(), errors="coerce").unstack().mean(axis=0).to_numpy(dtype=float),
            pd.to_numeric(offshore_output_data.iloc[:, 1:].stack(), errors="coerce").unstack().mean(axis=0).to_numpy(dtype=float),
        ]
    )
    renewable.loc[: min(len(renewable), len(mean_renewable_capacity)) - 1, "Pmax"] = mean_renewable_capacity[: min(len(renewable), len(mean_renewable_capacity))]

    renewable_group = (
        renewable.groupby(["bus", "fuel", "tech"], as_index=False)
        .agg(
            sum_Pmax=("Pmax", "sum"),
            sum_Pmin=("Pmin", "sum"),
            max_min_up_time=("min_up_time", "max"),
            max_min_down_time=("min_down_time", "max"),
            mean_lon=("lon", "mean"),
            mean_lat=("lat", "mean"),
            sum_ramp_up=("ramp_up", "sum"),
            sum_ramp_down=("ramp_down", "sum"),
        )
        .copy()
    )
    order = {"Solar": 0, "Onshore wind": 1, "Offshore wind": 2}
    renewable_group["_ord"] = renewable_group["fuel"].map(order).fillna(99)
    renewable_group = renewable_group.sort_values(["_ord", "bus"]).drop(columns=["_ord"]).reset_index(drop=True)

    n_renew = len(renewable_group)
    renewable_agg = pd.DataFrame(np.zeros((n_renew, 21), dtype=float), columns=gen_cols)
    renewable_agg["BusID"] = renewable_group["bus"].to_numpy(dtype=float)
    renewable_agg["Pmax"] = renewable_group["sum_Pmax"].to_numpy(dtype=float)
    renewable_agg["Pmin"] = renewable_group["sum_Pmin"].to_numpy(dtype=float)
    renewable_agg["Vg"] = 1.0
    renewable_agg["mBase"] = 100.0
    renewable_agg["Status"] = 1.0

    renewable_cost = pd.DataFrame(
        {
            "CostType": 2.0 * np.ones(n_renew, dtype=float),
            "StartCost": np.zeros(n_renew, dtype=float),
            "ShutCost": np.zeros(n_renew, dtype=float),
            "Order": 3.0 * np.ones(n_renew, dtype=float),
            "CostA": np.zeros(n_renew, dtype=float),
            "CostB": np.zeros(n_renew, dtype=float),
            "CostC": np.zeros(n_renew, dtype=float),
        }
    )

    renewable_extra = pd.DataFrame(
        {
            "lon": renewable_group["mean_lon"].to_numpy(dtype=float),
            "lat": renewable_group["mean_lat"].to_numpy(dtype=float),
            "bus": renewable_group["bus"].to_numpy(dtype=float),
            "Pmax": renewable_group["sum_Pmax"].to_numpy(dtype=float),
            "Pmin": renewable_group["sum_Pmin"].to_numpy(dtype=float),
            "min_up_time": renewable_group["max_min_up_time"].to_numpy(dtype=float),
            "min_down_time": renewable_group["max_min_down_time"].to_numpy(dtype=float),
            "ramp_up": renewable_group["sum_ramp_up"].to_numpy(dtype=float),
            "ramp_down": renewable_group["sum_ramp_down"].to_numpy(dtype=float),
            "fuel": renewable_group["fuel"].astype(str).to_numpy(),
            "tech": renewable_group["tech"].astype(str).to_numpy(),
        }
    )

    mpc["gen"] = pd.concat([gen, renewable_agg], ignore_index=True)
    mpc["gencost"] = pd.concat([gencost, renewable_cost], ignore_index=True)
    mpc["gen_extra"] = pd.concat([gen_extra, renewable_extra], ignore_index=True, sort=False)

    # gas system
    gbus = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="Gbus")
    mpc["Gbus"] = gbus.iloc[:, :6].copy()
    mpc["Gbus_extra"] = gbus.iloc[:, 6:].copy()

    gas_demand_table = preprocess_UK_gas_demand_data("2022 gas demand.csv")
    industrial_col = _pick_col(gas_demand_table, ["NTSEnergyOfftaken_IndustrialOfftakeTotal"])
    ldz_col = _pick_col(gas_demand_table, ["NTSEnergyOfftaken_LDZOfftakeTotal"])
    storage_col = _pick_col(gas_demand_table, ["NTSEnergyOfftaken_StorageInjectionTotal"])
    gas_demand_pure = (
        pd.to_numeric(gas_demand_table[industrial_col], errors="coerce").fillna(0.0)
        + pd.to_numeric(gas_demand_table[ldz_col], errors="coerce").fillna(0.0)
        + pd.to_numeric(gas_demand_table[storage_col], errors="coerce").fillna(0.0)
    ) / 1e6
    gas_demand_mean = float(gas_demand_pure.mean())

    gbus_demand_col = _pick_col(mpc["Gbus"], ["Demand"])
    mpc["Gbus"][gbus_demand_col] = gas_demand_mean * (
        pd.to_numeric(mpc["Gbus"][gbus_demand_col], errors="coerce").fillna(0.0) / 100.0
    )

    gline = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="Gline")
    diameter_col = _pick_col(gline, ["Diameter"])
    length_col = _pick_col(gline, ["Length"])
    topology_col = _pick_col(gline, ["Topology"])

    gline["C"] = 2.85 * pd.to_numeric(gline[diameter_col], errors="coerce").fillna(0.0) ** (8.0 / 3.0) / np.sqrt(
        pd.to_numeric(gline[length_col], errors="coerce").replace(0, np.nan)
    )
    gline["C"] = gline["C"].replace([np.inf, -np.inf], np.nan).fillna(999.0)
    gline.loc[gline[topology_col].astype(str) != "Pipeline", "C"] = 999.0

    mpc["Gline"] = gline.iloc[:, :5].copy()
    mpc["Gline_extra"] = gline.iloc[:, 5:].copy()

    gsou = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="Gsou")
    mpc["Gsou"] = gsou.iloc[:, :4].copy()
    mpc["Gsou_extra"] = gsou.iloc[:, 4:].copy()

    gcost_info = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="Gcost")
    price_col = _pick_col(gcost_info, ["Price"])
    mpc["Gcost"] = pd.DataFrame({"Price": [float(pd.to_numeric(gcost_info[price_col], errors="coerce").dropna().iloc[0])]})

    mpc["Gstore"] = _read_excel(DATA_ROOT / "GB energy network v1.xlsx", sheet_name="Gstore")
    mpc["ptg"] = pd.DataFrame()

    # gas bus for gas-fired generators
    ng = len(mpc["gen"])
    mpc["gen_extra"]["gas_bus"] = np.zeros(ng, dtype=float)

    gbus_lat_col = _pick_col(gbus, ["Lat", "Latitude", "y"])
    gbus_lon_col = _pick_col(gbus, ["Lon", "Longitude", "x"])
    gbus_lats = pd.to_numeric(gbus[gbus_lat_col], errors="coerce").fillna(0.0).to_numpy()
    gbus_lons = pd.to_numeric(gbus[gbus_lon_col], errors="coerce").fillna(0.0).to_numpy()

    fuel_series = mpc["gen_extra"].get("fuel", pd.Series([""] * ng)).astype(str)
    gen_lats = pd.to_numeric(mpc["gen_extra"].get("lat", pd.Series(np.nan, index=range(ng))), errors="coerce").to_numpy()
    gen_lons = pd.to_numeric(mpc["gen_extra"].get("lon", pd.Series(np.nan, index=range(ng))), errors="coerce").to_numpy()

    idx_gpp = fuel_series.str.lower().str.contains("natural gas|ccgt|ocgt|gas", regex=True, na=False)
    for i in np.where(idx_gpp.to_numpy())[0]:
        if np.isnan(gen_lats[i]) or np.isnan(gen_lons[i]):
            continue
        mpc["gen_extra"].at[i, "gas_bus"] = _nearest_bus_ids(np.array([gen_lats[i]]), np.array([gen_lons[i]]), gbus_lats, gbus_lons)[0]

    mpc1: dict[str, Any] = {
        "baseMVA": 100.0,
        "version": "2",
        "bus": _df_to_numeric_array(mpc["bus"]),
        "gen": _df_to_numeric_array(mpc["gen"]),
        "branch": _df_to_numeric_array(mpc["branch"]),
        "gencost": _df_to_numeric_array(mpc["gencost"]),
        "areas": np.array([[1.0, 1.0]], dtype=float),
        "Gbus": _df_to_numeric_array(mpc["Gbus"]),
        "Gline": _df_to_numeric_array(mpc["Gline"]),
        "Gsou": _df_to_numeric_array(mpc["Gsou"]),
        "Gcost": _df_to_numeric_array(mpc["Gcost"]),
        "Gstore": _df_to_numeric_array(mpc["Gstore"]),
        "bus_extra": mpc["bus_extra"],
        "branch_extra": mpc["branch_extra"],
        "gen_extra": mpc["gen_extra"],
        "Gbus_extra": mpc["Gbus_extra"],
        "Gline_extra": mpc["Gline_extra"],
        "Gsou_extra": mpc["Gsou_extra"],
        "ptg": mpc["ptg"],
    }
    return mpc1
if __name__ == "__main__":
    case = data_reader_writer(option=1)
    print("bus", case["bus"].shape)
    print("gen", case["gen"].shape)
    print("branch", case["branch"].shape)
