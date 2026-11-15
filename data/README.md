# Data

Only the station list is stored in the repo. Everything else is downloaded or produced by the scripts and is ignored by git (see `.gitignore`).

## `ks_stations_both_heights.csv`

The 73 Kansas Mesonet stations that reported temperature, wind speed and wind direction at both 2 m and 10 m for a full test day (12 to 13 November 2026). Columns: name, county, latitude, longitude, elevation (m), network.

The test covers one day, so it shows the sensors were working then, not how long each station has had them. Copy this file into the working folder before running the pipeline.

Source: Kansas Mesonet, http://mesonet.k-state.edu. Data are free for public use; cite the Mesonet when sharing or publishing.

## HRRR at the stations

Made by `science/fetch_hrrr15_stations.py` (see `../science/README.md`); not stored in git. The training step expects one CSV per station in `hrrr/`, named after the station in lower case with underscores (for example `alma_5se.csv`), every 15 minutes in UTC:

| Column | Meaning |
|---|---|
| `time_utc` | `YYYY-MM-DD HH:MM` |
| `temp_2m_F` | 2 m temperature, °F |
| `rh_pct` | 2 m relative humidity, % |
| `wind_10m_mph`, `wind_80m_mph` | wind speed, mph |
| `dir_10m_deg`, `dir_80m_deg` | wind direction, degrees |
| `wind_ratio_80_10`, `dir_diff_deg`, `cooling_F_per_hr` | present in the files, not used by the model |

The run described in the main README used 2023-09-22 to 2026-11-14 (110,400 rows per station). `scripts/hrrr_fcst.py` writes the same columns for the orchard forecast.

## Folders the pipeline creates

| Folder / file | Made by | Contents |
|---|---|---|
| `cdl_orchards_2025_10m.tif` | `cdl_orchards_zarr.py` | Orchard-only crop raster |
| `ks5min/` | `ks_pull.py` | 5-minute station data |
| `s2/`, `s2_orchards/` | `s2_pull.py`, `s2_orchards.py` | Sentinel-2 RGB tiles |
| `dino/`, `dino_orchards/` | `dino_feats.py` | DINOv3 features |
| `table/`, `dinofeat/` | `build_table.py` | Per-station training tables |
| `lgbm_data.parquet`, `dino_pca.npz` | `prep_lgbm.py` | Training set and PCA |
| `models/`, `oof.parquet`, `metrics.csv` | `train_lgbm.py` | Models and scores |
| `hrrr_fcst_*.parquet` | `hrrr_fcst.py` | Orchard forecast inputs |
| `orchard_*` | `predict_orchards.py` | Decisions, log, summary |
