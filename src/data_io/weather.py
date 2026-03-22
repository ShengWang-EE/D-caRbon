"""
气象站数据读取与时间序列。
Load weather station time series (temperature, etc.).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import requests

import numpy as np
import pandas as pd


def load_weather(path: str | None = None, **kwargs: Any) -> Any:
    """
    读取气象站数据（如逐时温度）。
    Load weather station data (e.g. hourly temperature).
    占位：返回 None 或空 DataFrame。
    """
    if path is None:
        return None
    # TODO: implement CSV/NetCDF load
    return None


def load_macao_temperature_window(data_dir: Path, start_date: str, end_date: str) -> np.ndarray:
    """Load station temperature matrix for a time window from macao_weather_filled.csv."""
    weather_path = data_dir / "macao_weather_filled.csv"
    weather_df = pd.read_csv(weather_path)
    weather_df["Date"] = pd.to_datetime(weather_df["Date"], errors="coerce")
    weather_df = weather_df.dropna(subset=["Date"]).sort_values("Date")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if end_ts < start_ts:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    window_df = weather_df[(weather_df["Date"] >= start_ts) & (weather_df["Date"] <= end_ts)]
    if window_df.empty:
        raise ValueError(f"No temperature records found in [{start_date}, {end_date}]")

    station_cols = [c for c in window_df.columns if c != "Date"]
    return window_df[station_cols].apply(pd.to_numeric, errors="coerce").to_numpy()


def load_nasa_power_ghi_window(
    *,
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timeout_s: float = 30.0,
) -> np.ndarray:
    """Load hourly global horizontal irradiance (W/m2) from NASA POWER.

    Uses ALLSKY_SFC_SW_DWN and returns one value per hour in [start_date, end_date].
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if end_ts < start_ts:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    start_ymd = start_ts.strftime("%Y%m%d")
    end_ymd = end_ts.strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={float(longitude):.6f}"
        f"&latitude={float(latitude):.6f}&start={start_ymd}&end={end_ymd}&format=JSON&time-standard=LST"
    )

    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()

    params = payload.get("properties", {}).get("parameter", {})
    ghi_dict = params.get("ALLSKY_SFC_SW_DWN", {})
    if not isinstance(ghi_dict, dict) or not ghi_dict:
        raise ValueError("NASA POWER response missing ALLSKY_SFC_SW_DWN hourly data.")

    ghi_series = pd.Series(ghi_dict, dtype=float)
    ghi_series.index = pd.to_datetime(ghi_series.index.astype(str), format="%Y%m%d%H", errors="coerce")
    ghi_series = ghi_series.dropna()
    ghi_series = pd.to_numeric(ghi_series, errors="coerce")
    ghi_series = ghi_series.where(ghi_series >= 0, 0.0)

    local_start = pd.to_datetime(start_date)
    local_end = pd.to_datetime(end_date)
    if local_start.tzinfo is not None:
        local_start = local_start.tz_localize(None)
    if local_end.tzinfo is not None:
        local_end = local_end.tz_localize(None)
    target_index = pd.date_range(local_start, local_end, freq="h")

    ghi_series = ghi_series.reindex(target_index)
    ghi_series = ghi_series.interpolate(method="linear", limit_direction="both").fillna(0.0)
    return ghi_series.to_numpy(dtype=float)
