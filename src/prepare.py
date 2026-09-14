"""
src/prepare.py - Task 1: Load and prepare the data

Loads Plant 1 generation and sensor data, converts timestamps,
sums power across inverters, merges with weather sensors,
resamples to hourly means, handles missing values, and saves
data/plant1_hourly.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so imports work seamlessly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.load_data import load_raw


def prepare_data(data_dir="data", save_csv=True):
    """
    Execute Task 1 data preparation pipeline.
    Returns:
        hourly_df: Prepared hourly DataFrame
        stats: Dictionary of reporting statistics for Table 1
    """
    data_dir = Path(data_dir)
    print("=" * 60)
    print("TASK 1: LOAD AND PREPARE THE DATA")
    print("=" * 60)

    # 1.1 Load raw CSVs using the provided loader
    raw = load_raw(data_dir)
    gen1 = raw["gen1"].copy()
    sensor1 = raw["sensor1"].copy()

    raw_gen_rows = len(gen1)
    raw_sensor_rows = len(sensor1)
    print(f"1.1 Loaded raw data:")
    print(f"    Raw generation rows (Plant 1): {raw_gen_rows:,}")
    print(f"    Raw sensor rows (Plant 1):     {raw_sensor_rows:,}")

    # 1.2 Convert DATE_TIME in both Plant 1 files to datetime
    # Inspect raw strings: gen1 uses DD-MM-YYYY HH:MM, sensor1 uses YYYY-MM-DD HH:MM:SS
    print(f"\n1.2 Converting timestamps...")
    print(f"    Sample raw gen1 DATE_TIME:   '{gen1['DATE_TIME'].iloc[0]}'")
    print(f"    Sample raw sensor1 DATE_TIME: '{sensor1['DATE_TIME'].iloc[0]}'")

    gen1["datetime"] = pd.to_datetime(gen1["DATE_TIME"], format="%d-%m-%Y %H:%M")
    sensor1["datetime"] = pd.to_datetime(sensor1["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

    print(f"    Gen1 start:   {gen1['datetime'].min()} | end: {gen1['datetime'].max()}")
    print(f"    Sensor1 start: {sensor1['datetime'].min()} | end: {sensor1['datetime'].max()}")

    # 1.3 Sum AC_POWER and DC_POWER over all inverters at each timestamp
    print(f"\n1.3 Aggregating power over inverters...")
    inverters_count = gen1["SOURCE_KEY"].nunique()
    print(f"    Found {inverters_count} unique inverters in Plant 1")
    plant1_power = gen1.groupby("datetime")[["AC_POWER", "DC_POWER"]].sum().reset_index()
    print(f"    Plant-level power timestamps: {len(plant1_power):,}")

    # 1.4 Merge plant-level power with sensor file on timestamp
    # Report how many timestamps exist in only one of the two files
    gen_times = set(plant1_power["datetime"])
    sensor_times = set(sensor1["datetime"])
    only_in_gen = len(gen_times - sensor_times)
    only_in_sensor = len(sensor_times - gen_times)
    only_one_file = only_in_gen + only_in_sensor

    print(f"\n1.4 Timestamp alignment & merge:")
    print(f"    Timestamps in gen only:    {only_in_gen}")
    print(f"    Timestamps in sensor only: {only_in_sensor}")
    print(f"    Total in only one file:    {only_one_file}")

    # Merge on timestamp using outer join to keep full temporal extent
    sensor_cols = ["datetime", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    merged = pd.merge(plant1_power, sensor1[sensor_cols], on="datetime", how="outer")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    print(f"    Merged 15-min rows: {len(merged):,}")

    # 1.5 Resample to hourly means (gives about 34 days * 24 hours = 816 rows)
    print(f"\n1.5 Resampling to hourly means...")
    merged.set_index("datetime", inplace=True)
    hourly = merged.resample("1h").mean()
    hourly_rows = len(hourly)
    print(f"    Hourly rows after resampling: {hourly_rows} (34 days * 24h = {34 * 24})")

    # Rename columns to exact lowercase specification
    hourly.rename(
        columns={
            "AC_POWER": "ac_power",
            "DC_POWER": "dc_power",
            "AMBIENT_TEMPERATURE": "ambient_temp",
            "MODULE_TEMPERATURE": "module_temp",
            "IRRADIATION": "irradiation",
        },
        inplace=True,
    )

    # 1.6 Report missing values and handle them
    print(f"\n1.6 Missing values audit and handling:")
    rows_with_na = int(hourly.isnull().any(axis=1).sum())
    print(f"    Hourly rows with missing values before handling: {rows_with_na}")
    print("    Missing count by column:")
    for col, count in hourly.isnull().sum().items():
        print(f"      {col}: {count}")

    # Physics-based handling of missing values:
    # 1. For power and irradiation at night (hours 19:00 to 05:00), solar irradiance
    #    and generated power are physically zero.
    night_mask = (hourly.index.hour < 6) | (hourly.index.hour >= 19)
    hourly.loc[night_mask, "ac_power"] = hourly.loc[night_mask, "ac_power"].fillna(0.0)
    hourly.loc[night_mask, "dc_power"] = hourly.loc[night_mask, "dc_power"].fillna(0.0)
    hourly.loc[night_mask, "irradiation"] = hourly.loc[night_mask, "irradiation"].fillna(0.0)

    # 2. Smooth physical variables (ambient_temp, module_temp, and any remaining daytime gaps)
    #    are interpolated linearly across time.
    hourly = hourly.interpolate(method="time")

    # 3. Any edge NaNs filled with forward/backward fill
    hourly = hourly.bfill().ffill()

    # Ensure power and irradiation remain strictly non-negative
    hourly["ac_power"] = hourly["ac_power"].clip(lower=0.0)
    hourly["dc_power"] = hourly["dc_power"].clip(lower=0.0)
    hourly["irradiation"] = hourly["irradiation"].clip(lower=0.0)

    rows_with_na_after = int(hourly.isnull().any(axis=1).sum())
    print(f"    Hourly rows with missing values after handling: {rows_with_na_after}")

    # Reset index to include 'datetime' as column
    hourly = hourly.reset_index()

    # Ensure column order matches specification
    final_cols = ["datetime", "ac_power", "dc_power", "ambient_temp", "module_temp", "irradiation"]
    hourly = hourly[final_cols]

    # Save CSV to data/plant1_hourly.csv
    if save_csv:
        out_csv = data_dir / "plant1_hourly.csv"
        hourly.to_csv(out_csv, index=False)
        print(f"\n[OK] Saved hourly data to: {out_csv}")

    stats = {
        "raw_gen_rows": raw_gen_rows,
        "raw_sensor_rows": raw_sensor_rows,
        "only_one_file": only_one_file,
        "only_in_gen": only_in_gen,
        "only_in_sensor": only_in_sensor,
        "hourly_rows": hourly_rows,
        "rows_with_na": rows_with_na,
    }

    print("=" * 60)
    print("TASK 1 COMPLETE")
    print("=" * 60)
    return hourly, stats


if __name__ == "__main__":
    df, stats = prepare_data()
    print("\nPrepared Data Head (5 rows):")
    print(df.head())
    print("\nData Summary Statistics:")
    print(df.describe().to_string())
