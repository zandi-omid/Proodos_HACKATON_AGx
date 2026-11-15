"""
Download HRRR (3 km, hourly) two-height weather for any points in CONUS
from Open-Meteo - no API key, no GRIB files.

Two modes:
  Future (forecast, next ~2 days, plus a few past days):
      python science/fetch_hrrr.py forecast
  Past (retrospective, HRRR archive from ~Dec 2023):
      python science/fetch_hrrr.py past 2025-10-01 2025-10-07

Output: one CSV per point in data/, e.g. data/hrrr_fresno_forecast.csv
Columns: time, temp_2m_F, temp_80m_F, wind_10m_mph, wind_80m_mph,
         dir_10m_deg, dir_80m_deg, rh_pct, cloud_pct, elev_m, temp_80m_source,
         dT_80m_minus_2m_F, inversion,
         temp_80m_hrrr_F, hrrr_lowest_level_agl_m, dT_80m_hrrr_minus_2m_F, inversion_hrrr

Note: Open-Meteo serves no HRRR temperature at 80 m, so temp_80m_F is filled
from NOAA GFS (temp_80m_source = "gfs"). All other columns are HRRR.
The *_hrrr columns are an all-HRRR check: the HRRR temperature profile
(2 m + pressure levels above ground) interpolated to 80 m.

Requires: pip install requests pandas numpy
"""
import sys
from pathlib import Path
import requests
import numpy as np
import pandas as pd

# ---- 1. Your points (edit freely): "name": (latitude, longitude), US only ----
POINTS = {
    "fresno":     (36.74, -119.79),   # Central Valley orchards
    "modesto":    (37.64, -120.99),
    "maricopa_az": (33.07, -111.97),
    "kern_almonds": (35.60, -119.33),  # almond/pistachio belt between Wasco and Shafter, Kern Co.
}

# ---- 2. Variables (two heights: 2 m & 80 m temp, 10 m & 80 m wind) ----
HOURLY = [
    "temperature_2m", "temperature_80m",
    "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m",
    "relative_humidity_2m", "cloud_cover",
]

# ---- 3. All-HRRR alternative to the 80 m temperature ----
# HRRR also gives temperature on pressure levels. Levels that are ABOVE the
# ground (pressure lower than surface pressure) form a profile we interpolate
# to 80 m - HRRR only, no GFS mix. In the Central Valley the lowest level above
# ground is 1000 hPa (~40-60 m) on high-pressure days, otherwise 975 hPa (~200 m).
PRESSURE_LEVELS = [1000, 975, 950, 925]   # highest pressure (lowest height) first
HOURLY += ["surface_pressure"]
HOURLY += [f"temperature_{p}hPa" for p in PRESSURE_LEVELS]
HOURLY += [f"geopotential_height_{p}hPa" for p in PRESSURE_LEVELS]

# Output folder: <repo>/data, so the script works from any working directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch(lat, lon, mode, start=None, end=None, model="ncep_hrrr_conus"):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY),
        "models": model,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    if mode == "forecast":
        url = FORECAST_URL
        params.update({"forecast_days": 2, "past_days": 2})
    else:
        url = PAST_URL
        params.update({"start_date": start, "end_date": end})

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j["hourly"])
    df["elev_m"] = j["elevation"]  # ground height above sea level used by Open-Meteo
    df = df.rename(columns={
        "temperature_2m": "temp_2m_F", "temperature_80m": "temp_80m_F",
        "wind_speed_10m": "wind_10m_mph", "wind_speed_80m": "wind_80m_mph",
        "wind_direction_10m": "dir_10m_deg", "wind_direction_80m": "dir_80m_deg",
        "relative_humidity_2m": "rh_pct", "cloud_cover": "cloud_pct",
    })
    df["time"] = df["time"].str.replace("T", " ")
    return df


def add_hrrr_80m(df, target_m=80):
    """All-HRRR estimate of the temperature at 80 m above ground.

    Build each hour's HRRR temperature profile from the 2 m value plus every
    pressure level that is above ground, then interpolate linearly to 80 m.
    Adds: temp_80m_hrrr_F, dT_80m_hrrr_minus_2m_F, inversion_hrrr and
    hrrr_lowest_level_agl_m (height of the lowest level used; if it is ~200 m,
    the 80 m value is a straight-line guess between 2 m and 200 m).
    Then drops the raw pressure-level columns.
    """
    t80, lowest = [], []
    for _, row in df.iterrows():
        heights, temps = [2.0], [row["temp_2m_F"]]
        for p in PRESSURE_LEVELS:  # lowest level first, so heights go up
            agl = row[f"geopotential_height_{p}hPa"] - row["elev_m"]
            if row["surface_pressure"] > p and agl > 10:  # level is above ground
                heights.append(agl)
                temps.append(row[f"temperature_{p}hPa"])
        t80.append(np.interp(target_m, heights, temps))
        lowest.append(round(heights[1]) if len(heights) > 1 else np.nan)
    df["temp_80m_hrrr_F"] = np.round(t80, 1)
    df["hrrr_lowest_level_agl_m"] = lowest
    df["dT_80m_hrrr_minus_2m_F"] = (df["temp_80m_hrrr_F"] - df["temp_2m_F"]).round(1)
    df["inversion_hrrr"] = df["dT_80m_hrrr_minus_2m_F"] > 0
    raw = [c for c in df.columns if c.endswith("hPa")]
    return df.drop(columns=raw + ["surface_pressure"])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    if mode == "past" and not (start and end):
        sys.exit("Usage: python science/fetch_hrrr.py past YYYY-MM-DD YYYY-MM-DD")

    for name, (lat, lon) in POINTS.items():
        # Everything HRRR has (2 m temp, 10 m & 80 m wind, RH, cloud) comes from HRRR.
        df = fetch(lat, lon, mode, start, end)
        df["temp_80m_source"] = "hrrr"

        # Open-Meteo does NOT publish HRRR temperature at 80 m, so that column comes
        # back empty. Fill ONLY the missing 80 m temperatures from NOAA GFS (the same
        # source Open-Meteo's "best_match" uses for 80 m) and record that in the CSV.
        missing = df["temp_80m_F"].isna()
        if missing.any():
            gfs = fetch(lat, lon, mode, start, end, model="gfs_seamless")
            df.loc[missing, "temp_80m_F"] = gfs.loc[missing, "temp_80m_F"]
            df.loc[missing, "temp_80m_source"] = "gfs"
            print(f"  {name}: 80 m temperature filled from GFS for {int(missing.sum())} hours")

        df["temp_80m_F"] = df["temp_80m_F"].astype(float)  # empty HRRR column was text
        df["dT_80m_minus_2m_F"] = (df["temp_80m_F"] - df["temp_2m_F"]).round(1)
        df["inversion"] = df["dT_80m_minus_2m_F"] > 0
        df = add_hrrr_80m(df)
        out = DATA_DIR / f"hrrr_{name}_{mode}.csv"
        df.to_csv(out, index=False)
        n_inv = int(df["inversion"].sum())
        n_hrrr = int(df["inversion_hrrr"].sum())
        print(f"{name}: {len(df)} hours saved to data/{out.name}  |  inversion hours: "
              f"{n_inv} (GFS 80 m), {n_hrrr} (HRRR profile at 80 m)")


if __name__ == "__main__":
    main()
