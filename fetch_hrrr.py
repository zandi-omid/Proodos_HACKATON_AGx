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
         dir_10m_deg, dir_80m_deg, rh_pct, cloud_pct, dT_80m_minus_2m_F, inversion

Requires: pip install requests pandas
"""
import sys
import requests
import pandas as pd

# ---- 1. Your points (edit freely) ----
POINTS = {
    "fresno":     (36.74, -119.79),   # Central Valley orchards
    "modesto":    (37.64, -120.99),
    "maricopa_az": (33.07, -111.97),
}

# ---- 2. Variables (two heights: 2 m & 80 m temp, 10 m & 80 m wind) ----
HOURLY = [
    "temperature_2m", "temperature_80m",
    "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m",
    "relative_humidity_2m", "cloud_cover",
]

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
    h = r.json()["hourly"]
    df = pd.DataFrame(h).rename(columns={
        "temperature_2m": "temp_2m_F", "temperature_80m": "temp_80m_F",
        "wind_speed_10m": "wind_10m_mph", "wind_speed_80m": "wind_80m_mph",
        "wind_direction_10m": "dir_10m_deg", "wind_direction_80m": "dir_80m_deg",
        "relative_humidity_2m": "rh_pct", "cloud_cover": "cloud_pct",
    })
    df["time"] = df["time"].str.replace("T", " ")
    return df


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    if mode == "past" and not (start and end):
        sys.exit("Usage: python science/fetch_hrrr.py past YYYY-MM-DD YYYY-MM-DD")

    for name, (lat, lon) in POINTS.items():
        df = fetch(lat, lon, mode, start, end)
        # If HRRR has no 80 m temperature for this request, fall back to the blended model
        if df["temp_80m_F"].isna().all():
            print(f"  {name}: HRRR returned no 80 m temperature -> using blended GFS/HRRR (best_match)")
            df = fetch(lat, lon, mode, start, end, model="best_match")

        df["dT_80m_minus_2m_F"] = (df["temp_80m_F"] - df["temp_2m_F"]).round(1)
        df["inversion"] = df["dT_80m_minus_2m_F"] > 0
        out = f"data/hrrr_{name}_{mode}.csv"
        df.to_csv(out, index=False)
        n_inv = int(df["inversion"].sum())
        print(f"{name}: {len(df)} hours saved to {out}  |  inversion hours: {n_inv}")


if __name__ == "__main__":
    main()
