# HRRR at the stations (training inputs)

These three files are copied unchanged from the team repo
[zandi-omid/Proodos_HACKATON_AGx](https://github.com/zandi-omid/Proodos_HACKATON_AGx) (`science/`, commit of 15 Nov 2026), written by its author. They produced the HRRR files used for training.

| File | Role |
|---|---|
| `fetch_hrrr15_stations.py` | Downloads 15-minute HRRR for every station in a CSV, one CSV per station |
| `features.py` | The single definition of the output columns |
| `fetch_hrrr.py` | Shared settings and the download helper |

## Run

```
python science/fetch_hrrr15_stations.py data/ks_stations_both_heights.csv 2023-09-22 2026-11-14 --workers 4
```

- **Source:** Open-Meteo's historical forecast API, model `ncep_hrrr_conus` (HRRR, 3 km), `minutely_15` values at each station's coordinates. No API key and no GRIB files.
- **Period:** real 15-minute HRRR starts about 2023-09-22, so the pull is about 3 years and 2 months, not the full 4 years of station data. Training uses the overlap.
- **Output:** `data/hrrr15_stations_ks_stations_both_heights_2023-09-22_2026-11-14/<station>.csv`, 110,400 rows per station, times in UTC. Copy or unzip these into `hrrr/` in the working folder for `build_table.py`.
- **Columns:** `time_utc, temp_2m_F, rh_pct, wind_10m_mph, wind_80m_mph, dir_10m_deg, dir_80m_deg, wind_ratio_80_10, dir_diff_deg, cooling_F_per_hr`. Relative humidity is hourly, filled in to 15 minutes. The last three are derived; the model does not use them.
- **Speed:** about 30 s and 1.5 MB per station per year. More than about 4 workers can hit Open-Meteo's rate limit; the script waits and can be restarted (finished stations are skipped).

Check Open-Meteo's terms of use before reusing the data.
