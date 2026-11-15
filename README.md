# Spray or Wait?

Weather intelligence to prevent pesticide drift in orchards. Built for HACKATHON AGX.

**Team:** Ioannis Lolos and Omid Zandi

The tool tells a pesticide applicator whether it is safe to spray **right now and for the next 18 hours**, and why, using conditions at two heights (about 5 ft and 30 ft): temperature inversion, wind speed, and wind shear. It works at orchards that have **no weather tower**, by learning from real two-height weather stations.

> This supports the applicator's decision. The pesticide label, on-site checks and regulations always come first.

## The idea

Two-height towers are rare, and a single ground station cannot see an inversion or wind shear. So:

1. **Learn from real towers.** 73 Kansas Mesonet stations measure temperature and wind at both 2 m and 10 m every 5 minutes. These are the ground truth (measured, not simulated).
2. **Predict the tower readings from data that exists everywhere.** A LightGBM model predicts the 10 m minus 2 m temperature difference and the wind at both heights from:
   - HRRR weather forecasts (3 km, every 15 minutes),
   - sun position and season,
   - DINOv3 features of the latest clear Sentinel-2 image around the site (what the land looks like).
3. **Apply it to orchards.** For every 3 km HRRR cell in Kansas that contains orchard (USDA Cropland Data Layer, 10 m), predict both heights every 15 minutes out to 18 hours and apply Go / Caution / Stop rules with plain-language reasons and a time-stamped log.

## Built during the hackathon

Everything in this repository was built by the team during Hack Arizona AgX 2026, on **15 November 2026**.
The git history shows it: the first commit (empty project scaffold) is at 11:35 and the full pipeline
was pushed at 16:27 (Arizona time, UTC-7). Every code file starts with a short header saying the same.

| Time (AZ) | What we built | Where |
|---|---|---|
| 11:35 | Project scaffold | `README.md`, folders |
| 11:55-12:45 | HRRR at two heights from Open-Meteo, first inversion analysis (Fresno, Kern County almonds), check of which HRRR values are real 15-minute output | `science/fetch_hrrr.py`, `science/fetch_hrrr_15min.py`, plots |
| 12:57 | Single-station baseline: Meteostat stations in Washington State (one height only, cannot see inversions) | `science/explore_meteostat.py` |
| 13:15-13:48 | 15-minute HRRR for the 73 Kansas Mesonet two-height stations (2023-09-22 to 2026-11-14), quality checks, discovery that real 15-min HRRR only starts 2023-09-22 | `science/fetch_hrrr15_stations.py`, `science/qc_station_plots.py` |
| 14:50 | One shared feature definition for training and forecasting; live HRRR forecast -> model inputs | `science/features.py`, `science/fetch_hrrr_forecast.py` |
| 14:58-15:03 | Click-a-location app (Streamlit) | `app/app.py` |
| 16:27 | Full pipeline: Kansas Mesonet ground truth, Sentinel-2 + DINOv3 land features, LightGBM training with spatial cross-validation, HRRR orchard forecast, Go / Caution / Stop | `scripts/` |

**Who did what:** Omid Zandi - HRRR data and features (`science/`), forecast download and app (`app/`).
Ioannis Lolos - station data, Sentinel-2 / DINOv3, LightGBM training and the orchard forecast pipeline (`scripts/`).
Design decisions, data checks and results were the team's.

**Tools:** we used an AI coding assistant (Claude Code) to help write and debug code; the commits
it helped with are marked `Co-Authored-By: Claude` in the git history. All data sources are public
and listed below.

## Results

Scores are on stations the model never saw (spatial cross-validation, stations held out in about 50 km blocks). This run used the 39 stations whose data had finished downloading, 3.68 million 15-minute rows, 64 predictors.

| Predicted quantity | Mean abs. error | R² | HRRR alone (MAE / R²) |
|---|---|---|---|
| Temperature difference, 10 m minus 2 m | 0.38 °C | 0.60 | not available in HRRR input |
| Wind at 2 m, east-west | 0.76 m/s | 0.73 | not available |
| Wind at 2 m, north-south | 0.82 m/s | 0.87 | not available |
| Wind at 10 m, east-west | 1.07 m/s | 0.70 | 1.13 / 0.65 |
| Wind at 10 m, north-south | 1.15 m/s | 0.83 | 1.20 / 0.81 |
| Wind speed at 2 m | 0.81 m/s | | |
| Wind speed at 10 m | 1.04 m/s | | |

**Inversion detection** (inversion defined as more than 0.5 °C warmer at 10 m; this happens 28% of the time): the model catches **85%** of real inversions; 27% of its inversion alerts are false alarms.

The strongest predictors of inversion are sun elevation (sunset and sunrise), humidity, how calm the last hour was, and the cooling rate, which matches the known physics.

## Go / Caution / Stop rules

Wind limits are the common guidance from the brief; actual limits depend on the label.

| Status | Trigger |
|---|---|
| **STOP** | Inversion (more than 0.3 °C / 0.5 °F warmer at 30 ft) with wind under 3 mph at 5 ft |
| **STOP** | Strong inversion (more than 1.0 °C / 1.8 °F) |
| **STOP** | Wind over 10 mph at either height |
| **CAUTION** | Weak inversion with wind of 3 mph or more |
| **CAUTION** | Calm at 5 ft (under 3 mph) without an inversion |
| **CAUTION** | Wind speed differs by more than 5 mph between heights |
| **CAUTION** | Wind direction differs by more than 45° between heights |
| **GO** | None of the above |

## Pipeline

All scripts are in [`scripts/`](scripts/README.md) and run from one working folder. The HRRR pull for the stations is in [`science/`](science/README.md), copied from the team repo [zandi-omid/Proodos_HACKATON_AGx](https://github.com/zandi-omid/Proodos_HACKATON_AGx).

| Step | Script | Output |
|---|---|---|
| 1. Orchard map | `cdl_orchards_zarr.py` | Orchard-only CDL 2025 raster, 10 m, CONUS |
| 2. Find two-height stations | `ks_check.py`, `make_station_list.py` | `ks_stations_both_heights.csv` (73 stations) |
| 3. Station data | `ks_pull.py` | 5-minute data, last 4 years, one Parquet per station |
| 4. Sentinel-2 at stations | `s2_pull.py` | Every image since May 2022 for the HRRR cell around each station |
| 5. DINOv3 features | `dino_feats.py` | CLS token and 20 × 20 patch grid per clear image |
| 6. HRRR at stations | [`science/fetch_hrrr15_stations.py`](science/README.md) | 15-minute HRRR per station, 2023-09-22 to 2026-11-14 |
| 7. Training table | `build_table.py`, `prep_lgbm.py` | One row per station per 15 minutes |
| 8. Train | `train_lgbm.py` | 5 models, held-out predictions, metrics |
| 9. Orchard cells | `s2_orchards.py`, then `dino_feats.py` again | Latest clear image and features for 1,845 orchard cells |
| 10. Forecast | `hrrr_fcst.py` | HRRR 15-minute forecast, 0 to 18 h, per orchard cell |
| 11. Decide | `predict_orchards.py` | Status, reasons, log, next safe window per cell |
| 12. App | `make_app.py` | `index.html`: interactive map with status, confidence, reasons, log and model accuracy |

## Data sources

| Data | Use | Terms |
|---|---|---|
| [Kansas Mesonet](https://mesonet.k-state.edu) | Ground truth: temperature and wind at 2 m and 10 m | Free for public use; cite as below |
| NOAA HRRR, 3 km, 15-minute | Training inputs (via Open-Meteo's historical forecast API) and forecast inputs (NOAA files via Herbie) | HRRR is public domain; see Open-Meteo's terms for its API |
| Sentinel-2 L2A (Earth Search on AWS) | Imagery | Copernicus open data |
| [DINOv3](https://github.com/facebookresearch/dinov3) ViT-H+/16 | Image features | DINOv3 License (gated on Hugging Face) |
| USDA NASS Cropland Data Layer 2025 (via Source Cooperative) | Orchard locations | Public domain |

Kansas Mesonet, 2026: Kansas Mesonet Historical Data. Accessed 15 November 2026, http://mesonet.k-state.edu/weather/historical

## Limitations

- **Stations are not in orchards.** They sit in open sites in Kansas farmland. The model predicts what an open-site two-height station would read at the orchard's location, not conditions inside the canopy.
- **Kansas only.** Flat plains. The model has not been tested in valleys such as California's Central Valley, where terrain drives inversions.
- **39 of 73 stations** were used for training in this run.
- **Two HRRR routes.** Training inputs come from Open-Meteo at the station's coordinates; the orchard forecast comes from NOAA's files at the grid cell. Small differences between the two (for example humidity, which is hourly in the training files) are not corrected.
- **About 3 years of overlap.** Station data covers 4 years, but 15-minute HRRR starts on 2023-09-22.
- **Short-range training.** The HRRR used for training is short lead time, so accuracy at 12 to 18 hours ahead is lower than the scores above suggest.
- **Cell level.** Predictions are per 3 km HRRR cell that contains orchard, not per individual orchard block. Many cells contain only a small orchard area.
- **No terrain input.** Elevation and cold-air drainage are not modelled directly.
- **Hyperparameters were set by hand,** not tuned.

## Setup

See [`scripts/README.md`](scripts/README.md) for how to run each step. Python packages are in `requirements.txt`. The scripts were run on a Slurm cluster; the `.sh` files contain cluster-specific paths that you need to edit.

## Other tools in this repo: HRRR downloads, forecast and app

Run instructions for the tools developed earlier in this repo (HRRR point and station downloads, live forecast features, Meteostat baseline, Streamlit app).

A tool that uses weather readings at 5 ft and 30 ft to tell orchard sprayers **Go / Caution / Stop**.

### Repository layout

- `data/` – weather data
- `science/` – analysis and decision logic
- `app/` – user-facing application

### Quick start

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

#### Add your own location

Edit the `POINTS` list at the top of [`science/fetch_hrrr.py`](science/fetch_hrrr.py)
(the 15-min script uses the same list). Name, then (latitude, longitude); US only (HRRR covers CONUS):

```python
POINTS = {
    "kern_almonds": (35.60, -119.33),
    "my_orchard":   (36.50, -119.80),   # <- add a line
}
```

#### Data source

Weather comes from NOAA HRRR via Open-Meteo (no API key). Open-Meteo has no HRRR
temperature at 80 m, so that one column is filled from NOAA GFS (`temp_80m_source`).
As an all-HRRR check, the HRRR pressure-level profile is also interpolated to 80 m.

#### Output: `data/hrrr_<point>_<mode>.csv` (one row per hour, local time)

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

#### 15-minute output: `data/hrrr15_<point>_<mode>.csv`

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

### Meteostat: single-station baseline (Washington State)

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

### Kansas Mesonet stations: 15-min HRRR at each station

`data/ks_stations_both_heights.csv`: 73 Kansas Mesonet (KSRE) stations that measure at two heights.

```bash
python science/fetch_hrrr15_stations.py data/ks_stations_both_heights.csv 2023-09-22 2026-11-14 --workers 4
```

Writes one CSV per station to `data/hrrr15_stations_ks_stations_both_heights_<start>_<end>/` (about 1.8 MB per
station per year; 73 stations took ~4 min; **not in git**, re-create with the command). More than ~4 workers
hits Open-Meteo's "too many concurrent requests" limit (the script waits it out). Same columns as the
15-min output above, with `time_utc` in **UTC**. Stopped runs can be restarted; finished stations are skipped.
HRRR 15-min has temperature at 2 m only, so dT must come from the stations' own two-height readings.

### Forecast: HRRR next ~48 h -> model features

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

### App: click a location in Kansas -> HRRR forecast

```bash
conda activate hackaton            # or the venv with requirements.txt
streamlit run app/app.py           # settings in .streamlit/config.toml (port 8501, localhost)
```
Click a location in Kansas. The app downloads NOAA's latest HRRR forecast (3 km) for that spot
(cached 1 h) and shows the next ~48 h: temperature, humidity, wind at 10 m and 80 m, wind ratio,
direction difference and temperature change (the model input features, `science/features.py`).
On the Rain server: forward port 8501 in VS Code (PORTS tab) and open http://localhost:8501.
`science/predict.py` + `models/README.md` hold the model plug-in (and a DEMO stand-in) for later.
