"""
Download 15-minute HRRR data (Open-Meteo, 3 km grid) for every station in a CSV.

Station CSV needs columns: name, latitude, longitude  (e.g. data/ks_stations_both_heights.csv)

    python science/fetch_hrrr15_stations.py                                   # Kansas, last year
    python science/fetch_hrrr15_stations.py data/ks_stations_both_heights.csv 2025-11-15 2026-11-14 --workers 24

Variables (real 15-min output, except rh_pct = hourly filled in; see fetch_hrrr_15min.py).
Real 15-min data starts about 2023-09-22: use dates from then on.
  temp_2m_F, rh_pct, wind_10m_mph, wind_80m_mph, dir_10m_deg, dir_80m_deg
  + inversion hints: wind_ratio_80_10, dir_diff_deg, cooling_F_per_hr
No 80 m temperature, so no dT here: the stations' own two-height temperatures give dT.

Output: data/hrrr15_stations_<csv name>_<start>_<end>/<station>.csv, one row per 15 min.
  time_utc is UTC (Kansas spans two time zones, and local time repeats/skips an
  hour at daylight-saving changes).
  Not committed to git (large): re-create with this script.
Already-downloaded stations are skipped, so a stopped run can just be restarted.
Stations are downloaded in parallel (--workers, default 8). The time is spent waiting
for Open-Meteo, not on our CPU, so threads are used; too many at once may hit
Open-Meteo's rate limit, which the script waits out.
About 30 s and 1.5 MB per station per year.
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fetch_hrrr import DATA_DIR, PAST_URL, get_json
from features import OPEN_METEO_VARS, build_features  # the ONE feature definition

# ---- Inputs: station file, dates (default: the last full year up to yesterday), workers ----
yesterday = date.today() - timedelta(days=1)
ap = argparse.ArgumentParser(description="15-min HRRR for every station in a CSV")
ap.add_argument("stations", nargs="?", default=str(DATA_DIR / "ks_stations_both_heights.csv"))
ap.add_argument("start", nargs="?", default=str(yesterday - timedelta(days=364)))
ap.add_argument("end", nargs="?", default=str(yesterday))
ap.add_argument("--workers", type=int, default=8, help="stations downloaded at the same time")
args = ap.parse_args()

station_csv = Path(args.stations)
stations = pd.read_csv(station_csv)
out_dir = DATA_DIR / f"hrrr15_stations_{station_csv.stem}_{args.start}_{args.end}"
out_dir.mkdir(exist_ok=True)


def file_name(name):
    """'Alma 5SE' -> 'alma_5se'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def download(st):
    """Download one station; return a one-line status message."""
    out = out_dir / f"{file_name(st['name'])}.csv"
    if out.exists():
        return f"{st['name']}: already done, skipping"

    params = {
        "latitude": st["latitude"], "longitude": st["longitude"],
        "minutely_15": ",".join(OPEN_METEO_VARS),
        "models": "ncep_hrrr_conus",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "GMT",
        "start_date": args.start, "end_date": args.end,
    }
    j = get_json(PAST_URL, params, timeout=300)  # a long period takes ~30-60 s to serve
    df = build_features(j["minutely_15"])

    tmp = out.with_suffix(".part")  # write then rename, so a crash never leaves a half file
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M")
    tmp.rename(out)
    n_missing = int(df["temp_2m_F"].isna().sum())
    return (f"{st['name']}: {len(df)} rows, {n_missing} missing -> {out.name}"
            f"  (HRRR cell at {j['latitude']:.3f}, {j['longitude']:.3f})")


print(f"{len(stations)} stations, {args.start} to {args.end} (UTC), {args.workers} at a time")
print(f"output: {out_dir.relative_to(DATA_DIR.parent)}/")
failed = []
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    jobs = {pool.submit(download, st): st["name"] for _, st in stations.iterrows()}
    for k, job in enumerate(as_completed(jobs), 1):
        try:
            print(f"[{k}/{len(stations)}] {job.result()}", flush=True)
        except (Exception, SystemExit) as e:  # get_json exits on hard errors
            failed.append(jobs[job])
            print(f"[{k}/{len(stations)}] {jobs[job]}: FAILED - {e}", flush=True)

if failed:
    print(f"Done, {len(failed)} failed: {', '.join(failed)}. Re-run the same command to retry them.")
else:
    print("Done, all stations downloaded.")
