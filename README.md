# Spray or Wait? – Hack Arizona AgX 2026

A tool that uses weather readings at 5 ft and 30 ft to tell orchard sprayers **Go / Caution / Stop**.

## Repository layout

- `data/` – weather data
- `science/` – analysis and decision logic
- `app/` – user-facing application

## Quick start

Needs Python 3.11+ and internet. No API key or account.
Use a virtual environment: Meteostat needs pandas >= 2.3, which can clash with
an existing Anaconda install if you `pip install` into it directly.

```bash
git clone https://github.com/zandi-omid/Proodos_HACKATON_AGx.git
cd Proodos_HACKATON_AGx
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python science/fetch_hrrr.py forecast                      # next 2 days + past 2 days
python science/fetch_hrrr.py past 2025-10-01 2025-10-07    # historical week
python science/summarize_inversions.py                     # Fresno summary + data/fresno_inversion.png
python science/plot_point.py data/hrrr_kern_almonds_past.csv  # 3-panel check plot for one point
python science/fetch_hrrr_15min.py past 2025-10-01 2025-10-07  # 15-min data (real 15-min from 2023-09-22 on)
python science/plot_15min.py kern_almonds past                 # 15-min signals vs hourly inversion
python science/explore_meteostat.py                            # Meteostat stations in WA (baseline)
```

- Scripts can be run from any folder; output always goes to `data/`.
- Each run **overwrites** the CSVs for the same point and mode.
- `forecast` = last 2 days + next 2 days. `past` works from about 2018 on; the HRRR-only
  80 m columns (`*_hrrr`) need dates from about 23 Mar 2021 on (empty before that).

### Add your own location

Edit the `POINTS` list at the top of [`science/fetch_hrrr.py`](science/fetch_hrrr.py)
(the 15-min script uses the same list). Name, then (latitude, longitude); US only (HRRR covers CONUS):

```python
POINTS = {
    "kern_almonds": (35.60, -119.33),
    "my_orchard":   (36.50, -119.80),   # <- add a line
}
```

### Data source

Weather comes from NOAA HRRR via Open-Meteo (no API key). Open-Meteo has no HRRR
temperature at 80 m, so that one column is filled from NOAA GFS (`temp_80m_source`).
As an all-HRRR check, the HRRR pressure-level profile is also interpolated to 80 m.

### Output: `data/hrrr_<point>_<mode>.csv` (one row per hour, local time)

| column | meaning |
|---|---|
| `time` | local time, `YYYY-MM-DD HH:MM` |
| `temp_2m_F` | air temperature at 2 m (HRRR) |
| `temp_80m_F` | air temperature at 80 m (GFS; see `temp_80m_source`) |
| `wind_10m_mph`, `wind_80m_mph` | wind speed at 10 m / 80 m (HRRR) |
| `dir_10m_deg`, `dir_80m_deg` | wind direction (from), degrees (HRRR) |
| `rh_pct`, `cloud_pct` | relative humidity at 2 m, cloud cover (HRRR) |
| `elev_m` | ground elevation used by Open-Meteo |
| `dT_80m_minus_2m_F`, `inversion` | T80m - T2m with GFS 80 m; inversion if > 0 |
| `temp_80m_hrrr_F` | HRRR-only 80 m temperature (profile interpolated) |
| `hrrr_lowest_level_agl_m` | height of lowest HRRR level used (~40 m good, ~200 m = rough estimate) |
| `dT_80m_hrrr_minus_2m_F`, `inversion_hrrr` | T80m - T2m with HRRR 80 m; inversion if > 0 |

### 15-minute output: `data/hrrr15_<point>_<mode>.csv`

Real 15-min HRRR output: `temp_2m_F`, `wind_10m_mph`, `wind_80m_mph`, `dir_10m_deg`, `dir_80m_deg`.
`rh_pct` is included but is **hourly values filled in** (use it as hourly).
**Real 15-min data only exists from about 2023-09-22**; earlier dates return hourly values
smoothed in between. Values at the top of the hour (:00) are jumpier than :15/:30/:45 (likely
where separate HRRR runs are joined), so smooth over ~1 hour before using extremes.
No 80 m temperature,
so no dT; instead, inversion hints:

| column | meaning |
|---|---|
| `wind_ratio_80_10` | wind 80 m / wind 10 m; > ~1.5 at night = ground layer decoupled |
| `dir_diff_deg` | wind direction difference between 10 m and 80 m (0-180) |
| `cooling_F_per_hr` | 2 m temperature change over the last hour |

Against the hourly dT (hours where GFS and HRRR agree), `wind_ratio_80_10 >= 1.5`
catches 80% of inversion hours with 16% false alarms; `>= 2` catches 64% with 4%.

## Meteostat: single-station baseline (Washington State)

`science/explore_meteostat.py` (Meteostat v2 API, falls back to v1) writes:
- `data/meteostat_wa_stations.csv`: all 51 WA stations (48 with hourly data):
  id, name, lat, lon, elevation, first/last date of hourly data
- `data/meteostat_wenatchee_2025-10-15.csv`: hourly data, 5 stations nearest Wenatchee, in F and mph
- `data/meteostat_wa_map.png`

Findings:
- Temperature/humidity at **one** height (~2 m), wind at **one** height (~10 m),
  the airport standard. Meteostat has no height field and **no variable at two heights**,
  so it cannot see an inversion; it is the "single weather station" the challenge compares against.
- Temperatures are whole degrees C (~1.8 F steps). Hourly only.
- On 2025-10-15 temperature, RH, wind speed and direction, pressure, cloud cover were complete
  at all 5 stations; dew point (v2 API), gusts, visibility, snow depth, sunshine were 100% missing.
- Stations are mostly airports (nearest to Wenatchee: Pangborn airport, 8.6 km), not orchards.

## Kansas Mesonet stations: 15-min HRRR at each station

`data/ks_stations_both_heights.csv`: 73 Kansas Mesonet (KSRE) stations that measure at two heights.

```bash
python science/fetch_hrrr15_stations.py data/ks_stations_both_heights.csv 2023-09-22 2026-11-14 --workers 4
```

Writes one CSV per station to `data/hrrr15_stations_ks_stations_both_heights_<start>_<end>/` (about 1.8 MB per
station per year; 73 stations took ~4 min; **not in git**, re-create with the command). More than ~4 workers
hits Open-Meteo's "too many concurrent requests" limit (the script waits it out). Same columns as the
15-min output above, with `time_utc` in **UTC**. Stopped runs can be restarted; finished stations are skipped.
HRRR 15-min has temperature at 2 m only, so dT must come from the stations' own two-height readings.

## Forecast: HRRR next ~48 h -> model features

`science/fetch_hrrr_forecast.py` downloads the live HRRR forecast and builds **exactly the
training-table feature columns** (same names, units, UTC, definitions), ready for the trained models.
All features come from one function, `science/features.py` (`build_features`), which the
training downloads also use (checked: it rebuilds `gove_5se.csv` byte-for-byte).

```bash
python science/fetch_hrrr_forecast.py                               # all Kansas stations, 15-min
python science/fetch_hrrr_forecast.py --lat 39.12 --lon -96.64      # one point
python science/fetch_hrrr_forecast.py --grid 37 40 -102 -94.6 0.25  # lat/lon grid
python science/fetch_hrrr_forecast.py --step hourly                 # hourly instead of 15-min
```
From Python: `from fetch_hrrr_forecast import get_forecast; df = get_forecast(points)` then
`model.predict(df[FEATURES[1:]])` (`from features import FEATURES`).

Output `data/forecast/hrrr_forecast_<points>_<step>_<fetch time>UTC.csv` (not in git):
`name, lat, lon, hrrr_lat, hrrr_lon, fetched_utc, time_utc, lead_hours, real_15min` + the 9 features.
- `lead_hours`: hours after the download; negative = the 3 past hours kept for the 1-hour cooling lag.
- HRRR reaches ~45-48 h ahead. **Real 15-min values only for the first ~18 h**; later 15-min steps are
  hourly values filled in (`real_15min = False`). Accuracy drops with lead time.
- Many points per request; responses cached in `data/forecast_cache/` and reused for 60 min.
