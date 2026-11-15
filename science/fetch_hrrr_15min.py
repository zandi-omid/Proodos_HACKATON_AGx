# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""
Download 15-minute HRRR weather from Open-Meteo.
Real 15-min model output (checked): temperature 2 m, wind speed + direction at 10 m and 80 m.
Relative humidity 2 m is also included, but it is HOURLY values filled in between
(checked: its 15-min wiggles do not follow temperature's), so use it as hourly.
Real 15-min data only exists from about 22 Sep 2023; before that ALL variables
are hourly values smoothed in between.

There is no 80 m temperature here, so no dT. Instead we compute signals that
hint at an inversion (stable, "decoupled" air near the ground):
  - wind_ratio_80_10   : wind 80 m / wind 10 m. At night in an inversion the ground
                         layer detaches: calm at 10 m while it's still windy at 80 m
  - dir_diff_deg       : how much the wind direction differs between 10 m and 80 m
  - cooling_F_per_hr   : change of 2 m temperature over the last hour
                         (fast cooling at sunset = inversion forming)

Usage (same points as fetch_hrrr.py):
    python science/fetch_hrrr_15min.py forecast
    python science/fetch_hrrr_15min.py past 2025-10-01 2025-10-07

Output: data/hrrr15_<point>_<mode>.csv, one row every 15 min, local time.
Requires: pip install requests pandas
"""
import sys
import pandas as pd

from fetch_hrrr import POINTS, DATA_DIR, FORECAST_URL, PAST_URL, get_json  # shared settings & helper

# 15-min variables: all real 15-min output except relative_humidity_2m (hourly, filled in)
MINUTELY_15 = [
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m",
]


def fetch_15min(lat, lon, mode, start=None, end=None):
    params = {
        "latitude": lat, "longitude": lon,
        "minutely_15": ",".join(MINUTELY_15),
        "models": "ncep_hrrr_conus",
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

    df = pd.DataFrame(get_json(url, params)["minutely_15"]).rename(columns={
        "temperature_2m": "temp_2m_F", "relative_humidity_2m": "rh_pct",
        "wind_speed_10m": "wind_10m_mph", "wind_speed_80m": "wind_80m_mph",
        "wind_direction_10m": "dir_10m_deg", "wind_direction_80m": "dir_80m_deg",
    })
    df["time"] = df["time"].str.replace("T", " ")
    return df.dropna(subset=["temp_2m_F"])  # drop trailing empty rows, if any


def add_signals(df):
    # Wind ratio: clip 10 m wind at 0.5 mph so a dead calm doesn't divide by zero
    df["wind_ratio_80_10"] = (df["wind_80m_mph"] / df["wind_10m_mph"].clip(lower=0.5)).round(1)
    # Smallest angle between the two directions, 0-180 degrees
    diff = (df["dir_80m_deg"] - df["dir_10m_deg"] + 180) % 360 - 180
    df["dir_diff_deg"] = diff.abs().round()
    # 2 m temperature change over the last hour (4 steps of 15 min)
    df["cooling_F_per_hr"] = df["temp_2m_F"].diff(4).round(1)
    return df


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    if mode == "past" and not (start and end):
        sys.exit("Usage: python science/fetch_hrrr_15min.py past YYYY-MM-DD YYYY-MM-DD")

    for name, (lat, lon) in POINTS.items():
        df = fetch_15min(lat, lon, mode, start, end)
        if df.empty:
            sys.exit(f"HRRR has no 15-min data for {start} to {end}. Try a more recent date.")
        df = add_signals(df)
        out = DATA_DIR / f"hrrr15_{name}_{mode}.csv"
        df.to_csv(out, index=False)
        print(f"{name}: {len(df)} rows (15 min) saved to data/{out.name}")


if __name__ == "__main__":
    main()
