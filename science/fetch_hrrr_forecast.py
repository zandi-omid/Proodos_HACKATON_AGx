"""
Download the HRRR forecast (Open-Meteo, 3 km) for many points and turn it into the
SAME feature columns as the training files, ready to pass to the trained models.

Command line (writes data/forecast/hrrr_forecast_<points>_<step>_<fetch time UTC>.csv):
    python science/fetch_hrrr_forecast.py                                  # all Kansas stations
    python science/fetch_hrrr_forecast.py --points my_points.csv           # name,latitude,longitude
    python science/fetch_hrrr_forecast.py --lat 39.12 --lon -96.64         # one point
    python science/fetch_hrrr_forecast.py --grid 37 40 -102 -94.6 0.25     # lat/lon grid (min max min max step)
    python science/fetch_hrrr_forecast.py --step hourly                    # hourly instead of 15-min

From Python (e.g. inside the app):
    from fetch_hrrr_forecast import get_forecast
    df = get_forecast(points)          # points: DataFrame with name, latitude, longitude
    X = df[FEATURES[1:]]               # -> model.predict(X)

Output: one row per point and time step:
    name, lat, lon            the requested point
    hrrr_lat, hrrr_lon        centre of the HRRR grid cell actually used
    fetched_utc               when the forecast was downloaded (UTC)
    lead_hours                hours from fetched_utc to time_utc; NEGATIVE = already past
                              (3 past hours are included so cooling_F_per_hr has its 1-hour lag)
    real_15min                False where a 15-min value is hourly data filled in: HRRR writes
                              real 15-min output only for the first ~18 forecast hours
    time_utc + feature columns, exactly as in features.FEATURES (same names, units, definitions)

Notes:
  - Default step is 15 min, like the training files. HRRR forecasts reach ~45-48 h
    (only the 00/06/12/18 UTC runs go to 48 h); hours with no data are dropped.
  - Many points go into one request (--batch). Responses are cached in data/forecast_cache/
    with their fetch time and reused for --max-age minutes (HRRR updates hourly), so
    re-running the script or the app does not re-download.
  - Forecast accuracy drops with lead time.
Requires: pip install requests pandas
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_hrrr import DATA_DIR, FORECAST_URL, get_json
from features import FEATURES, OPEN_METEO_VARS, build_features

CACHE_DIR = DATA_DIR / "forecast_cache"
OUT_DIR = DATA_DIR / "forecast"


def _fetch_batch(points, step, hours_ahead, hours_back, max_age_min):
    """One Open-Meteo request for a batch of points, using the disk cache when fresh.
    Returns (list of per-point responses, fetch time UTC)."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in points["latitude"]),
        "longitude": ",".join(f"{v:.4f}" for v in points["longitude"]),
        "models": "ncep_hrrr_conus",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "GMT",
        ("minutely_15" if step == "15min" else "hourly"): ",".join(OPEN_METEO_VARS),
    }
    # count in time steps from now (forecast_days would stop at midnight UTC tomorrow)
    if step == "15min":
        params.update({"forecast_minutely_15": hours_ahead * 4, "past_minutely_15": hours_back * 4})
    else:
        params.update({"forecast_hours": hours_ahead, "past_hours": hours_back})
    # cache file name = hash of the exact request
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{key}.json"
    now = datetime.now(timezone.utc)
    if cache.exists():
        saved = json.loads(cache.read_text())
        fetched = datetime.fromisoformat(saved["fetched_utc"])
        if (now - fetched).total_seconds() < max_age_min * 60:
            return saved["response"], fetched

    response = get_json(FORECAST_URL, params, timeout=120)
    if isinstance(response, dict):  # a single point comes back as one object, not a list
        response = [response]
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"fetched_utc": now.isoformat(), "params": params, "response": response}))
    return response, now


def get_forecast(points, step="15min", hours_ahead=48, hours_back=3, batch=50, max_age_min=60):
    """points: DataFrame with columns name, latitude, longitude.
    Returns the forecast feature table described at the top of this file."""
    block = "minutely_15" if step == "15min" else "hourly"
    frames = []
    for start in range(0, len(points), batch):
        chunk = points.iloc[start:start + batch]
        responses, fetched = _fetch_batch(chunk, step, hours_ahead, hours_back, max_age_min)
        for (_, pt), r in zip(chunk.iterrows(), responses):
            f = build_features(r[block])                    # the ONE feature definition
            f.insert(0, "lead_hours", ((f["time_utc"] - pd.Timestamp(fetched).tz_localize(None))
                                       .dt.total_seconds() / 3600).round(2))
            # HRRR writes real 15-min output only for its first 18 forecast hours; later
            # 15-min steps are hourly values filled in (:00 values are always real)
            f.insert(1, "real_15min", (step == "hourly") | (f["lead_hours"] <= 18)
                     | (f["time_utc"].dt.minute == 0))
            f.insert(0, "fetched_utc", pd.Timestamp(fetched).tz_localize(None).floor("s"))
            f.insert(0, "hrrr_lon", r["longitude"])
            f.insert(0, "hrrr_lat", r["latitude"])
            f.insert(0, "lon", pt["longitude"])
            f.insert(0, "lat", pt["latitude"])
            f.insert(0, "name", pt["name"])
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    # put time_utc right after the id columns, then the model features
    cols = (["name", "lat", "lon", "hrrr_lat", "hrrr_lon", "fetched_utc", "time_utc", "lead_hours", "real_15min"]
            + FEATURES[1:])
    return out[cols]


def grid_points(lat_min, lat_max, lon_min, lon_max, step):
    lats = np.arange(lat_min, lat_max + step / 2, step)
    lons = np.arange(lon_min, lon_max + step / 2, step)
    return pd.DataFrame([{"name": f"g_{la:.3f}_{lo:.3f}", "latitude": round(la, 4), "longitude": round(lo, 4)}
                         for la in lats for lo in lons])


def main():
    ap = argparse.ArgumentParser(description="HRRR forecast -> model features")
    ap.add_argument("--points", default=str(DATA_DIR / "ks_stations_both_heights.csv"),
                    help="CSV with name, latitude, longitude (default: Kansas stations)")
    ap.add_argument("--lat", type=float, help="single point latitude (use with --lon)")
    ap.add_argument("--lon", type=float, help="single point longitude")
    ap.add_argument("--grid", type=float, nargs=5, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "STEP"))
    ap.add_argument("--step", choices=["15min", "hourly"], default="15min")
    ap.add_argument("--hours", type=int, default=48, help="hours ahead (HRRR reaches ~48 h)")
    ap.add_argument("--batch", type=int, default=50, help="points per request")
    ap.add_argument("--max-age", type=int, default=60, help="reuse cached forecast younger than this (minutes)")
    args = ap.parse_args()

    if args.grid:
        points, tag = grid_points(*args.grid), "grid"
    elif args.lat is not None and args.lon is not None:
        points = pd.DataFrame([{"name": "point", "latitude": args.lat, "longitude": args.lon}])
        tag = f"point_{args.lat}_{args.lon}"
    else:
        points, tag = pd.read_csv(args.points), Path(args.points).stem

    df = get_forecast(points, step=args.step, hours_ahead=args.hours, batch=args.batch, max_age_min=args.max_age)
    OUT_DIR.mkdir(exist_ok=True)
    fetched = df["fetched_utc"].min()
    out = OUT_DIR / f"hrrr_forecast_{tag}_{args.step}_{fetched:%Y-%m-%d_%H%M}UTC.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M")
    future = df[df["lead_hours"] >= 0]
    print(f"{points.shape[0]} points, step {args.step}, fetched {fetched:%Y-%m-%d %H:%M} UTC")
    print(f"forecast: {future['time_utc'].min():%Y-%m-%d %H:%M} to {future['time_utc'].max():%Y-%m-%d %H:%M} UTC "
          f"(lead up to {future['lead_hours'].max():.0f} h), {len(df)} rows -> {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
