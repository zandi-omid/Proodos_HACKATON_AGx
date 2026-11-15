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
```

Weather comes from NOAA HRRR via Open-Meteo (no API key). Open-Meteo has no HRRR
temperature at 80 m, so that one column is filled from NOAA GFS (`temp_80m_source`).
