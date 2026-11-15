"""
Explore Meteostat (https://meteostat.net) weather stations in Washington State,
to see what a "single ground station" baseline can offer the Spray-or-Wait tool.

    python science/explore_meteostat.py

What it does:
  1. Lists all Meteostat stations in WA (country US, region WA) with the dates
     hourly data is available -> data/meteostat_wa_stations.csv
  2. Prints the hourly variables, their units and (standard) measurement heights
  3. Fetches hourly data for 2025-10-15 at the 5 stations closest to Wenatchee,
     converts to F and mph, and shows how much of each variable is missing
     -> data/meteostat_wenatchee_2025-10-15.csv
  4. Maps the stations, marking Wenatchee and Yakima -> data/meteostat_wa_map.png

Uses the current Meteostat API (v2: `import meteostat as ms`). If that is not
installed, falls back to the older v1 API (`from meteostat import Stations, Hourly`).
Requires: pip install meteostat pandas matplotlib  (Meteostat v2 needs pandas >= 2.3)
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # <repo>/data
WENATCHEE = (47.42, -120.31)
YAKIMA = (46.60, -120.51)
DAY = datetime(2025, 10, 15)
TZ = "America/Los_Angeles"

# ---- Which Meteostat API do we have? ----
try:
    import meteostat as ms
    ms.hourly  # only exists in v2
    NEW_API = True
except (ImportError, AttributeError):
    from meteostat import Stations, Hourly  # v1 API
    NEW_API = False
print(f"Meteostat API: {'v2 (import meteostat as ms)' if NEW_API else 'v1 (Stations, Hourly)'}")

# Hourly variables. Meteostat does NOT store measurement heights; the heights
# below are the standard for US airport / synoptic stations (ISD, METAR):
# temperature and humidity at about 2 m (~6 ft), wind at 10 m (~33 ft).
HOURLY_VARS = {  # column: (description, Meteostat unit, standard height)
    "temp": ("Air temperature", "C", "2 m"),
    "dwpt": ("Dew point", "C", "2 m"),
    "rhum": ("Relative humidity", "%", "2 m"),
    "prcp": ("Precipitation (1 h total)", "mm", "ground"),
    "snwd": ("Snow depth", "cm", "ground"),
    "wdir": ("Wind direction", "deg", "10 m"),
    "wspd": ("Wind speed (mean)", "km/h", "10 m"),
    "wpgt": ("Peak wind gust", "km/h", "10 m"),
    "pres": ("Air pressure, reduced to sea level", "hPa", "station (reduced to MSL)"),
    "tsun": ("Sunshine duration", "min", "-"),
    "cldc": ("Cloud cover", "oktas", "sky"),
    "vsby": ("Visibility", "m", "-"),
    "coco": ("Weather condition code", "code", "-"),
}


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---- 1. All WA stations + dates of hourly data ----
def wa_stations():
    if NEW_API:
        # v2 ships a station database (SQLite) we can query directly
        st = ms.stations.query("""
            SELECT s.id, n.name, s.latitude AS lat, s.longitude AS lon, s.elevation AS elevation_m,
                   MIN(i.start) AS hourly_start, MAX(i."end") AS hourly_end
            FROM stations s
            LEFT JOIN names n ON n.station = s.id AND n.language = 'en'
            LEFT JOIN inventory i ON i.station = s.id
                 AND i.provider IN ('isd_lite', 'metar', 'ghcnh')   -- hourly sources
            WHERE s.country = 'US' AND s.region = 'WA'
            GROUP BY s.id
            ORDER BY n.name
        """)
    else:
        st = Stations().region("US", "WA").fetch().reset_index()
        st = st.rename(columns={"latitude": "lat", "longitude": "lon", "elevation": "elevation_m",
                                "hourly_end": "hourly_end", "hourly_start": "hourly_start"})
        st = st[["id", "name", "lat", "lon", "elevation_m", "hourly_start", "hourly_end"]]
    for c in ("hourly_start", "hourly_end"):
        st[c] = pd.to_datetime(st[c]).dt.date
    return st


stations = wa_stations()
stations.to_csv(DATA_DIR / "meteostat_wa_stations.csv", index=False)
has_hourly = stations["hourly_start"].notna()
print(f"\n1. WA stations: {len(stations)} total, {int(has_hourly.sum())} with hourly data "
      f"-> data/meteostat_wa_stations.csv")

# ---- 2. Variables, units, heights ----
print("\n2. Hourly variables (units as Meteostat stores them; heights = station standard,")
print("   Meteostat itself has no height field):")
for col, (desc, unit, height) in HOURLY_VARS.items():
    print(f"   {col:5} {desc:36} {unit:6} {height}")
print("   -> Temperature is at ONE height (~2 m) and wind at ONE height (~10 m).")
print("      No variable gives temperature or wind at two heights, so no inversion (dT) from Meteostat alone.")

# ---- 3. Hourly data, 2025-10-15, 5 stations closest to Wenatchee ----
stations["dist_km"] = distance_km(WENATCHEE[0], WENATCHEE[1], stations["lat"], stations["lon"]).round(1)
covers_day = (pd.to_datetime(stations["hourly_start"]) <= DAY) & (pd.to_datetime(stations["hourly_end"]) >= DAY)
closest = stations[covers_day].nsmallest(5, "dist_km")
n_skipped = int((stations["dist_km"] <= closest["dist_km"].max()).sum() - len(closest))
print(f"\n3. Hourly data for {DAY.date()} at the 5 closest stations to Wenatchee with hourly data that day")
print(f"   ({n_skipped} closer stations skipped: no hourly data on that date)")
print(closest[["id", "name", "dist_km", "elevation_m"]].to_string(index=False))

frames = []
for _, s in closest.iterrows():
    day_end = DAY.replace(hour=23)  # start == end date would return only 1 hour
    if NEW_API:
        params = [ms.Parameter(c) for c in HOURLY_VARS]  # ask for all, not just the defaults
        df = ms.hourly(s["id"], DAY, day_end, timezone=TZ, parameters=params).fetch()
    else:
        df = Hourly(s["id"], DAY, day_end, timezone=TZ).fetch()
    if df is None or df.empty:
        print(f"   {s['id']} {s['name']}: no data returned")
        continue
    df = df.reset_index()
    df.columns = [getattr(c, "value", c) for c in df.columns]  # empty columns come back as enums
    df = df.drop(columns=[c for c in ("station",) if c in df.columns])
    df.insert(0, "station", s["id"])
    df.insert(1, "name", s["name"])
    frames.append(df)

hourly = pd.concat(frames, ignore_index=True)
# keep only the variables listed above, then convert to US units
hourly = hourly[["station", "name", "time"] + [c for c in HOURLY_VARS if c in hourly.columns]]
hourly["temp_F"] = (hourly["temp"] * 9 / 5 + 32).round(1)
hourly["dwpt_F"] = (hourly["dwpt"] * 9 / 5 + 32).round(1)
hourly["wspd_mph"] = (hourly["wspd"] / 1.609344).round(1)
hourly["wpgt_mph"] = (hourly["wpgt"] / 1.609344).round(1)
hourly = hourly.drop(columns=[c for c in ("temp", "dwpt", "wspd", "wpgt") if c in hourly.columns])
hourly.to_csv(DATA_DIR / "meteostat_wenatchee_2025-10-15.csv", index=False)

# % missing per variable and station (24 expected hours per station)
missing = (hourly.drop(columns=["name", "time"]).set_index("station").isna()
           .groupby(level=0).mean().mul(100).round(0).astype(int))
rows = hourly.groupby("station").size().rename("hours_returned")
# Meteostat stores temperature in whole C here, so F values come in ~1.8 F steps
step = hourly.groupby("station")["temp_F"].apply(lambda x: x.dropna().diff().abs()[lambda d: d > 0].min())
print(f"\n   Smallest temperature step seen: {step.min():.1f} F (values rounded to whole C)")
print("\n   % missing per variable (of hours returned) -> data/meteostat_wenatchee_2025-10-15.csv")
print(pd.concat([rows, missing], axis=1).to_string())

# ---- 4. Map ----
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.scatter(stations.loc[~has_hourly, "lon"], stations.loc[~has_hourly, "lat"], s=10,
           color="lightgray", label="station, no hourly data")
ax.scatter(stations.loc[has_hourly, "lon"], stations.loc[has_hourly, "lat"], s=14,
           color="tab:blue", label="station with hourly data")
ax.scatter(closest["lon"], closest["lat"], s=40, facecolors="none", edgecolors="tab:red",
           label="5 used for Wenatchee")
for name, (lat, lon) in {"Wenatchee": WENATCHEE, "Yakima": YAKIMA}.items():
    ax.plot(lon, lat, marker="*", color="black", ms=14)
    ax.annotate(name, (lon, lat), xytext=(6, 6), textcoords="offset points", weight="bold")
ax.set_aspect(1 / np.cos(np.radians(47.3)))  # so the map is not stretched
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Meteostat stations in Washington State ({len(stations)})")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(DATA_DIR / "meteostat_wa_map.png", dpi=120)
print("\n4. Map saved to data/meteostat_wa_map.png")
