"""
The ONE place where raw Open-Meteo HRRR data becomes model features.
Used by the training-data downloads (fetch_hrrr15_stations.py, fetch_hrrr_15min.py),
the forecast script (fetch_hrrr_forecast.py) and the app - so features are always
computed the same way.

    from features import build_features, FEATURES
    df = build_features(open_meteo_json["hourly"])          # or ["minutely_15"]

Input: the "hourly" (or "minutely_15") block of an Open-Meteo response requested with
  temperature_unit=fahrenheit, wind_speed_unit=mph, timezone=GMT and the variables in
  OPEN_METEO_VARS. Works for hourly and 15-min data alike.

Output columns (exactly the training-table columns, in this order):
  time_utc          UTC, "YYYY-MM-DD HH:MM" (pandas datetime)
  temp_2m_F         air temperature at 2 m (F)
  rh_pct            relative humidity at 2 m (%)
  wind_10m_mph      wind speed at 10 m (mph)
  wind_80m_mph      wind speed at 80 m (mph)
  dir_10m_deg       wind direction at 10 m (degrees, from)
  dir_80m_deg       wind direction at 80 m (degrees, from)
  wind_ratio_80_10  wind_80m / wind_10m, with the 10 m wind floored at 0.5 mph so a
                    calm 10 m wind cannot divide by ~0 (rounded to 0.1)
  dir_diff_deg      smallest angle between the 10 m and 80 m directions, 0-180 (rounded)
  cooling_F_per_hr  T(t) - T(t - 1 hour) at 2 m (F); negative = cooling. Looked up by
                    TIME, not by row, so hourly and 15-min data give the same value.
                    Empty for the first hour (no earlier value).
"""
import pandas as pd

# Variables to request from Open-Meteo (hourly= or minutely_15=)
OPEN_METEO_VARS = [
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m",
]
RENAME = {
    "time": "time_utc",
    "temperature_2m": "temp_2m_F", "relative_humidity_2m": "rh_pct",
    "wind_speed_10m": "wind_10m_mph", "wind_speed_80m": "wind_80m_mph",
    "wind_direction_10m": "dir_10m_deg", "wind_direction_80m": "dir_80m_deg",
}
FEATURES = ["time_utc", "temp_2m_F", "rh_pct", "wind_10m_mph", "wind_80m_mph",
            "dir_10m_deg", "dir_80m_deg", "wind_ratio_80_10", "dir_diff_deg", "cooling_F_per_hr"]
MIN_WIND_10M_MPH = 0.5  # floor for the ratio's denominator


def add_derived(df):
    """Add wind_ratio_80_10, dir_diff_deg and cooling_F_per_hr to a frame that already
    has time_utc (datetime), temp_2m_F, wind_10m_mph, wind_80m_mph, dir_10m_deg, dir_80m_deg."""
    df = df.copy()
    df["wind_ratio_80_10"] = (df["wind_80m_mph"] / df["wind_10m_mph"].clip(lower=MIN_WIND_10M_MPH)).round(1)
    diff = (df["dir_80m_deg"] - df["dir_10m_deg"] + 180) % 360 - 180   # signed, -180..180
    df["dir_diff_deg"] = diff.abs().round()
    # temperature exactly one hour earlier, matched by timestamp
    temp_by_time = df.set_index("time_utc")["temp_2m_F"]
    t_hour_ago = (df["time_utc"] - pd.Timedelta(hours=1)).map(temp_by_time)
    df["cooling_F_per_hr"] = (df["temp_2m_F"] - t_hour_ago).round(1)
    return df


def build_features(block):
    """Open-Meteo "hourly"/"minutely_15" block (dict of lists, or a DataFrame) -> feature table."""
    df = pd.DataFrame(block).rename(columns=RENAME)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df = df.dropna(subset=["temp_2m_F"])  # trailing empty rows at the end of a forecast
    df = add_derived(df)[FEATURES].reset_index(drop=True)
    # whole-number columns stay whole numbers (as in the training files), even if
    # Open-Meteo sent empty cells that made pandas store them as decimals
    for c in ("rh_pct", "dir_10m_deg", "dir_80m_deg", "dir_diff_deg"):
        df[c] = df[c].round().astype("Int64")
    return df
