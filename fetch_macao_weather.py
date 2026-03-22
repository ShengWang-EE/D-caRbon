"""
Fetch Macao weather data from the provided URL and export to CSV.

Usage:
  python fetch_macao_weather.py

The script tries multiple parsing strategies (JSON, CSV, HTML tables,
simple regex) to handle different response types.
"""
from __future__ import annotations

import html
import io
from pathlib import Path
import re
import sys
from typing import Optional

import pandas as pd
import requests


DEFAULT_URL = (
    "https://std.puiching.edu.mo/~pcmsams/pages/weatherData/weatherData/"
    "multiple2.php?startDate=2021-01-01&endDate=2025-12-31&interval=60&element=temp&draw=0"
)

IGNORED_FILL_COLUMNS = {"澳門污水處理廠", "澳门污水处理厂"}


def try_parse_json(resp: requests.Response) -> Optional[pd.DataFrame]:
    try:
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, (list, dict)):
        try:
            return pd.json_normalize(data)
        except Exception:
            try:
                return pd.DataFrame(data)
            except Exception:
                return None
    return None


def try_parse_csv(text: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return None


def try_parse_embedded_pre_csv(text: str) -> Optional[pd.DataFrame]:
    match = re.search(
        r"<pre[^>]*id=['\"]tempData['\"][^>]*>(.*?)</pre>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    csv_text = html.unescape(match.group(1)).strip()
    if not csv_text.startswith("Date,"):
        return None

    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


def try_parse_html_tables(text: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(text)
    except Exception:
        return None
    if not tables:
        return None
    # Return the largest table (most columns * rows)
    tables.sort(key=lambda df: df.shape[0] * max(1, df.shape[1]), reverse=True)
    return tables[0]


def try_parse_regex(text: str) -> Optional[pd.DataFrame]:
    # Try to find timestamp and numeric value pairs like: 2021-01-01 00:00, 23.4
    # This is a best-effort fallback for poorly structured pages.
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} ?\d{2}:?\d{2}).*?([-+]?[0-9]*\.?[0-9]+)")
    rows = pattern.findall(text)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["datetime", "value"])
    try:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    except Exception:
        pass
    return df


def clean_weather_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Date"])

    value_cols = [col for col in cleaned.columns if col != "Date"]
    fill_cols = [col for col in value_cols if col not in IGNORED_FILL_COLUMNS]
    ignored_cols = [col for col in value_cols if col in IGNORED_FILL_COLUMNS]
    for col in value_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    duplicate_row_count = int(cleaned.duplicated(subset=["Date"]).sum())
    unique_timestamp_count = int(cleaned["Date"].nunique())

    aggregation = {col: "mean" for col in fill_cols}
    for col in ignored_cols:
        aggregation[col] = "first"

    cleaned = cleaned.groupby("Date", as_index=False).agg(aggregation).sort_values("Date")

    full_index = pd.date_range(
        cleaned["Date"].min(),
        cleaned["Date"].max(),
        freq="h",
        tz=cleaned["Date"].dt.tz,
    )
    cleaned = cleaned.set_index("Date").reindex(full_index)
    cleaned.index.name = "Date"

    missing_value_count_before = int(cleaned[fill_cols].isna().sum().sum())
    cleaned[fill_cols] = cleaned[fill_cols].interpolate(
        method="time", limit_direction="both"
    )
    missing_value_count_after = int(cleaned[fill_cols].isna().sum().sum())
    ignored_missing_value_count = int(cleaned[ignored_cols].isna().sum().sum()) if ignored_cols else 0

    report = {
        "duplicate_row_count": duplicate_row_count,
        "inserted_timestamp_count": len(full_index) - unique_timestamp_count,
        "missing_value_count_before": missing_value_count_before,
        "missing_value_count_after": missing_value_count_after,
        "ignored_missing_value_count": ignored_missing_value_count,
    }
    return cleaned.reset_index(), report


def derive_filled_output_path(out_csv: str) -> str:
    path = Path(out_csv)
    return str(path.with_name(f"{path.stem}_filled{path.suffix}"))


def fetch_and_save(url: str, out_csv: str = "data/macao_weather.csv") -> None:
    def save_outputs(parsed_df: pd.DataFrame, source_label: str) -> None:
        print(f"{source_label} -> saving raw CSV")
        parsed_df.to_csv(out_csv, index=False)

        filled_df, report = clean_weather_dataframe(parsed_df)
        filled_csv = derive_filled_output_path(out_csv)
        filled_df.to_csv(filled_csv, index=False)
        print(f"Saved cleaned CSV: {filled_csv}")
        print(
            "Cleaning summary:",
            f"duplicate rows={report['duplicate_row_count']},",
            f"inserted timestamps={report['inserted_timestamp_count']},",
            f"missing values before={report['missing_value_count_before']},",
            f"missing values after={report['missing_value_count_after']},",
            f"ignored-column missing values={report['ignored_missing_value_count']}",
        )

    print(f"Requesting: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Try JSON
    df = try_parse_json(resp)
    if df is not None:
        save_outputs(df, "Parsed response as JSON")
        return

    text = resp.text

    # Try the embedded CSV payload used by this page.
    df = try_parse_embedded_pre_csv(text)
    if df is not None:
        save_outputs(df, "Parsed embedded CSV payload")
        return

    # Try CSV
    df = try_parse_csv(text)
    if df is not None:
        save_outputs(df, "Parsed response as CSV")
        return

    # Try HTML tables
    df = try_parse_html_tables(text)
    if df is not None:
        save_outputs(df, "Parsed response as HTML table")
        return

    # Fallback regex
    df = try_parse_regex(text)
    if df is not None:
        save_outputs(df, "Parsed response with regex fallback")
        return

    # If nothing worked, save raw response for inspection
    alt_path = out_csv + ".raw.html"
    print("Unable to parse response reliably — saving raw content to:", alt_path)
    with open(alt_path, "w", encoding="utf-8") as f:
        f.write(text)


def main(argv=None):
    argv = argv or sys.argv[1:]
    url = DEFAULT_URL
    out = "data/macao_weather.csv"
    if len(argv) >= 1:
        url = argv[0]
    if len(argv) >= 2:
        out = argv[1]
    fetch_and_save(url, out)


if __name__ == "__main__":
    main()
