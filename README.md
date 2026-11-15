# Spray or Wait? – Hack Arizona AgX 2026

A tool that uses weather readings at 5 ft and 30 ft to tell orchard sprayers **Go / Caution / Stop**.

## Repository layout

- `data/` – weather data
- `science/` – analysis and decision logic
- `app/` – user-facing application

## Quick start

```bash
pip install -r requirements.txt
python science/fetch_hrrr.py forecast                      # next 2 days + past 2 days
python science/fetch_hrrr.py past 2025-10-01 2025-10-07    # historical week
python science/summarize_inversions.py                     # Fresno summary + data/fresno_inversion.png
python science/plot_point.py data/hrrr_kern_almonds_past.csv  # 3-panel check plot for one point
```

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
