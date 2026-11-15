# Scripts

Run everything from **one working folder** that contains all the scripts: they read and write relative paths, and two of them import others (`s2_orchards.py` imports `s2_pull.py`; `predict_orchards.py` imports `build_table.py`).

The `.sh` files are Slurm wrappers. Before using them, change the working folder (`cd /scratch/...`), the environment name, and the Python path to your own. Steps that say "job" need more memory than a login node allows.

## 1. Orchard map

```
sbatch cdl_orchards_zarr.sh        # or: python cdl_orchards_zarr.py 8
```
Reads the 2025 Cropland Data Layer (10 m) from Source Cooperative and keeps only orchard classes (apples, pecans, peaches, cherries, almonds, walnuts and so on; grapes, Christmas trees and blueberries are excluded). Writes `cdl_orchards_2025_10m.tif` and a per-crop area table. About 16 minutes, 32 GB.

## 2. Find stations with two heights

```
python ks_check.py                 # station list + one-day test of every 5-minute station
python make_station_list.py        # -> ks_stations_both_heights.csv
```
`ks_check.py` pulls one day of 5-minute data from each Kansas Mesonet station and counts readings at 2 m and 10 m for temperature, wind speed and wind direction. `make_station_list.py` keeps the stations where all six are complete. Result on 15 Nov 2026: 73 of 112 stations (the list is in `../data/`).

## 3. Station data

```
sbatch ks_pull.sh                  # array job; each task takes every n-th station
```
Raw 5-minute data for the last 4 years, one file per station in `ks5min/`. Columns: temperature, wind speed, direction, direction spread and gust at 2 m and 10 m, plus humidity, vapour pressure deficit, pressure, solar radiation, precipitation. Re-running skips finished stations.

Be gentle with the Mesonet server: many parallel tasks make every request slower. 4 to 6 tasks is about as fast as 18.

## 4. Sentinel-2 at the stations

```
sbatch s2_pull.sh                  # one array task per station
```
For each station, finds the HRRR 3 km cell it falls in and saves every Sentinel-2 L2A pass since May 2022 (6 months before the station data, so every training row has an earlier image). Window: 320 × 320 pixels at 10 m (3.2 km = 20 × 20 DINO patches) centred on the HRRR cell. Saves red, green, blue, the scene classification layer and the clear fraction per date, in `s2/<station>.zarr`.

## 5. DINOv3 features

```
# once, on a node with internet, after requesting access to the model on Hugging Face:
export HF_HUB_CACHE=/path/to/hf_cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vith16plus-pretrain-lvd1689m')"

sbatch dino_feats.sh               # 4 GPUs
```
Runs frozen DINOv3 ViT-H+/16 on every image that is at least 80% clear. Saves the CLS token (1,280 numbers) and the 20 × 20 patch grid per image in `dino/<station>.zarr`, plus which patch contains the station.

Input preparation: raw values to reflectance `(DN - 1000) / 10000`, stretched so 0.3 reflectance is full brightness, then ImageNet normalisation. The tile is not resized.

## 6. HRRR at the stations

```
python science/fetch_hrrr15_stations.py data/ks_stations_both_heights.csv 2023-09-22 2026-11-14 --workers 4
```
Run from the repo root. 15-minute HRRR at each station from Open-Meteo, one CSV per station. Real 15-minute data starts about 2023-09-22. Put the CSVs in `hrrr/` in the working folder. Details in `../science/README.md`.

## 7. Training table

```
sbatch build_table.sh              # joins stations + HRRR; --watch picks up new stations
sbatch prep_lgbm.sh                # -> lgbm_data.parquet, lgbm_meta.json, dino_pca.npz
```
`build_table.py`:
- detects the station clock's time zone from the solar radiation curve and converts to UTC;
- averages 5-minute data to 15 minutes (wind direction through its east-west and north-south components);
- builds the targets and joins HRRR (converted to °C and m/s) and sun position;
- writes `table/<station>.parquet` and `dinofeat/<station>.parquet`.

`prep_lgbm.py`:
- reduces the DINO features with PCA (16 components each for CLS, mean patch and station patch);
- attaches to every row the latest image taken **at or before** that time, with its age in days;
- assigns spatial folds (0.5° blocks, 5 folds).

Targets: `y_dT` (10 m minus 2 m temperature), `u2`, `v2`, `u10`, `v10` (wind components).
Predictors: HRRR 2 m temperature and humidity, 10 m and 80 m wind, 1-hour temperature change, 1-hour mean wind; sun elevation, azimuth and elevation one hour later; season; DINO components; image age. No station measurements, coordinates or elevation.

## 8. Train

```
sbatch train_lgbm.sh
```
One LightGBM model per target, 5 spatial folds for scoring, then a final model on all stations. Writes `models/`, `oof.parquet` (held-out predictions) and `metrics.csv`. About 6 minutes for 3.7 million rows on 16 cores.

## 9. Orchard cells

```
python s2_orchards.py cells        # -> ks_orchard_cells.csv (HRRR cells in Kansas with orchard)
sbatch s2_orchards.sh              # latest >=80% clear image per cell -> s2_orchards/
DINO_SRC=s2_orchards DINO_OUT=dino_orchards sbatch --export=ALL dino_feats.sh
```
Kansas has 1,845 HRRR cells with any orchard (1,619 ha in total; most cells hold well under 1 ha).

## 10. Forecast

```
python hrrr_fcst.py
```
Downloads the newest HRRR run's 15-minute output (f00 to f18) with Herbie and extracts 2 m temperature and humidity and 10 m and 80 m wind for every orchard cell. Writes `hrrr_fcst_<cycle>.parquet`, 73 time steps per cell.

## 11. Decide

```
sbatch predict_orchards.sh
```
Predicts both heights for every orchard cell and time, applies the rules, and writes:
- `orchard_forecast_<cycle>.parquet`: status, reasons, readings, and what a single 5 ft station would say;
- `orchard_log_<cycle>.csv`: the time-stamped alert log;
- `orchard_summary_<cycle>.csv`: status now and the next safe window (GO for at least 1 hour) per cell.

## 12. App

```
python make_app.py                 # as a job: needs about 32 GB
```
Builds `index.html` (also saved as `spray_or_wait.html`), one self-contained page: a map of Kansas with the orchard cells coloured by status, a time slider for the next 18 hours, and for each cell the status with confidence, the reasons, both-height readings, the single-station comparison, the next safe window and a downloadable time-stamped log.

Confidence: the model's held-out errors (`oof.parquet`) are added to each forecast 100 times and the rules re-applied; the share of each outcome is its probability. GO is shown only when it is at least 60% likely, otherwise CAUTION.
