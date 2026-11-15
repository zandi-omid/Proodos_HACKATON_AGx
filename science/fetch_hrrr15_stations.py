"""
Download 15-minute HRRR data (Open-Meteo, 3 km grid) for every station in a CSV.

Station CSV needs columns: name, latitude, longitude  (e.g. data/ks_stations_both_heights.csv)

    python science/fetch_hrrr15_stations.py                                   # Kansas, last 2 years
    python science/fetch_hrrr15_stations.py data/ks_stations_both_heights.csv 2024-11-15 2026-11-14

Variables (only the ones HRRR really outputs every 15 min - see fetch_hrrr_15min.py):
  temp_2m_F, rh_pct, wind_10m_mph, wind_80m_mph, dir_10m_deg, dir_80m_deg
  + inversion hints: wind_ratio_80_10, dir_diff_deg, cooling_F_per_hr
No 80 m temperature, so no dT here: the stations' own two-height temperatures give dT.

Output: data/hrrr15_stations_<csv name>/<station>.csv, one row per 15 min.
  time_utc is UTC (Kansas spans two time zones, and local time repeats/skips an
  hour at daylight-saving changes).
  Not committed to git (large): re-create with this script.
Already-downloaded stations are skipped, so a stopped run can just be restarted.
About 1 minute and 3 MB per station for 2 years.
"""
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fetch_hrrr import DATA_DIR, PAST_URL, get_json
from fetch_hrrr_15min import MINUTELY_15, add_signals

# ---- Inputs: station file and dates (default: the last 2 full years up to yesterday) ----
station_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "ks_stations_both_heights.csv"
yesterday = date.today() - timedelta(days=1)
start = sys.argv[2] if len(sys.argv) > 2 else str(yesterday - timedelta(days=730) + timedelta(days=1))
end = sys.argv[3] if len(sys.argv) > 3 else str(yesterday)

stations = pd.read_csv(station_csv)
out_dir = DATA_DIR / f"hrrr15_stations_{station_csv.stem}"
out_dir.mkdir(exist_ok=True)


def file_name(name):
    """'Alma 5SE' -> 'alma_5se'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


print(f"{len(stations)} stations, {start} to {end} (UTC), output: {out_dir.relative_to(DATA_DIR.parent)}/")
for i, st in stations.iterrows():
    out = out_dir / f"{file_name(st['name'])}.csv"
    if out.exists():
        print(f"[{i + 1}/{len(stations)}] {st['name']}: already done, skipping")
        continue

    params = {
        "latitude": st["latitude"], "longitude": st["longitude"],
        "minutely_15": ",".join(MINUTELY_15),
        "models": "ncep_hrrr_conus",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "GMT",
        "start_date": start, "end_date": end,
    }
    j = get_json(PAST_URL, params, timeout=300)  # 2 years takes ~1 min to serve
    df = pd.DataFrame(j["minutely_15"]).rename(columns={
        "time": "time_utc",
        "temperature_2m": "temp_2m_F", "relative_humidity_2m": "rh_pct",
        "wind_speed_10m": "wind_10m_mph", "wind_speed_80m": "wind_80m_mph",
        "wind_direction_10m": "dir_10m_deg", "wind_direction_80m": "dir_80m_deg",
    })
    df["time_utc"] = df["time_utc"].str.replace("T", " ")
    df = add_signals(df)

    tmp = out.with_suffix(".part")  # write then rename, so a crash never leaves a half file
    df.to_csv(tmp, index=False)
    tmp.rename(out)
    n_missing = int(df["temp_2m_F"].isna().sum())
    print(f"[{i + 1}/{len(stations)}] {st['name']}: {len(df)} rows, {n_missing} missing -> {out.name}"
          f"  (HRRR cell at {j['latitude']:.3f}, {j['longitude']:.3f})")

print("Done.")
