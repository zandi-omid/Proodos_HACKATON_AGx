# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""
Download HRRR (3 km, hourly) two-height weather for any points in CONUS
from Open-Meteo - no API key, no GRIB files.

Two modes:
  Future (forecast, next ~2 days, plus a few past days):
      python science/fetch_hrrr.py forecast
  Past (HRRR archive: surface from ~2018, pressure levels from ~Mar 2021):
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
import time
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


def get_json(url, params, tries=3, timeout=60):
    """GET with retries: the free API sometimes times out, is busy (HTTP 5xx)
    or rate-limits us (HTTP 429: wait and try again)."""
    attempt, waits = 0, 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:  # rate limit: wait, don't count as a failed try
                try:
                    reason = r.json().get("reason", "")
                except ValueError:  # busy server may not send JSON
                    reason = r.text[:100]
                if "Daily" in reason or waits >= 20:
                    sys.exit(f"Open-Meteo limit reached: {reason} Re-run later; finished files are kept.")
                # too many parallel requests clears fast; per-minute/hour limits need longer
                pause = 600 if "Hourly" in reason else 10 if "concurrent" in reason else 60
                print(f"  rate limited ({reason}) - waiting {pause} s...")
                time.sleep(pause)
                waits += 1
                continue
            if r.status_code == 400:  # bad request (e.g. date out of range): retrying won't help
                sys.exit(f"Open-Meteo rejected the request: {r.json().get('reason', r.text)}")
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            attempt += 1
            if attempt == tries:
                sys.exit(f"Could not reach Open-Meteo after {tries} tries ({e.__class__.__name__}). "
                         "Check your internet connection and try again in a minute.")
            print(f"  Open-Meteo did not answer ({e.__class__.__name__}), retrying in {5 * attempt} s...")
            time.sleep(5 * attempt)


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

    j = get_json(url, params)
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
            z, sp = row[f"geopotential_height_{p}hPa"], row["surface_pressure"]
            if pd.isna(z) or pd.isna(sp) or pd.isna(row[f"temperature_{p}hPa"]):
                continue  # no pressure-level data this hour (archive starts ~Mar 2021)
            agl = z - row["elev_m"]
            if sp > p and agl > 10:  # level is above ground
                heights.append(agl)
                temps.append(row[f"temperature_{p}hPa"])
        # need the 2 m value and at least one level above 80 m to interpolate
        ok = not pd.isna(temps[0]) and max(heights) >= target_m
        t80.append(np.interp(target_m, heights, temps) if ok else np.nan)
        lowest.append(round(heights[1]) if len(heights) > 1 else np.nan)
    df["temp_80m_hrrr_F"] = np.round(t80, 1)
    df["hrrr_lowest_level_agl_m"] = lowest
    df["dT_80m_hrrr_minus_2m_F"] = (df["temp_80m_hrrr_F"] - df["temp_2m_F"]).round(1)
    df["inversion_hrrr"] = (df["dT_80m_hrrr_minus_2m_F"] > 0).where(df["dT_80m_hrrr_minus_2m_F"].notna())
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
        if df["temp_2m_F"].isna().all():
            sys.exit(f"HRRR has no data for {start} to {end}. "
                     "The archive starts around 2018 (HRRR 80 m estimate from ~Mar 2021).")
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
        # inversion = dT > 0; left empty (not False) when there is no 80 m data
        df["inversion"] = (df["dT_80m_minus_2m_F"] > 0).where(df["dT_80m_minus_2m_F"].notna())
        df = add_hrrr_80m(df)
        out = DATA_DIR / f"hrrr_{name}_{mode}.csv"
        df.to_csv(out, index=False)
        for col, label in (("temp_80m_F", "GFS"), ("temp_80m_hrrr_F", "HRRR")):
            if df[col].isna().all():
                print(f"  WARNING {name}: no {label} 80 m temperature for these dates -> "
                      f"its inversion column is empty (unknown), not 'no inversion'")
        n_inv = int(df["inversion"].sum())
        n_hrrr = int(df["inversion_hrrr"].sum())
        print(f"{name}: {len(df)} hours saved to data/{out.name}  |  inversion hours: "
              f"{n_inv} (GFS 80 m), {n_hrrr} (HRRR profile at 80 m)")


if __name__ == "__main__":
    main()
