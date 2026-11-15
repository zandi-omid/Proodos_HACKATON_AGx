"""Build the 15-min training table, incrementally, from whatever has been downloaded.

python build_table.py            build everything new, then exit
python build_table.py --watch    keep checking every 5 min until all stations are built

For every station in ks5min/ that also has hrrr/<station>.csv:
  table/<station>.parquet      targets (station) + HRRR + sun, 15-min, UTC
For every station with dino/<station>.zarr:
  dinofeat/<station>.parquet   per image date: cls, mean patch, station patch (1280 each)
Stations already built are skipped (rebuilt only if their input file is newer).
PCA of the DINO features is done at training time, so adding stations never invalidates old files.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("table"); OUT.mkdir(exist_ok=True)
DOUT = Path("dinofeat"); DOUT.mkdir(exist_ok=True)
NPC = 16                                   # PCA components per DINO feature group
MPH, F2C = 0.44704, lambda f: (f - 32) / 1.8


def key(name):
    return name.lower().replace(" ", "_")


# ---------- sun position (NOAA approximation), t in UTC ----------
def sun(t, lat, lon):
    t = pd.DatetimeIndex(t)
    doy = t.day_of_year.values; hr = t.hour.values + t.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    cz = np.clip(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha), -1, 1)
    el = 90 - np.degrees(np.arccos(cz))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la))) + 180
    return el.astype("float32"), (az % 360).astype("float32")


def detect_tz(d, lat, lon):
    """Pick the shift that best lines up solar radiation with the sun: UTC, CST, or US/Central."""
    s = d[["timestamp_raw", "SRAVG"]].dropna().iloc[::6]
    cands = {"UTC": pd.DatetimeIndex(s.timestamp_raw),
             "CST(UTC-6)": pd.DatetimeIndex(s.timestamp_raw) + pd.Timedelta(hours=6),
             "America/Chicago": pd.DatetimeIndex(s.timestamp_raw).tz_localize("America/Chicago", ambiguous="NaT",
                                                                         nonexistent="NaT").tz_convert(None)}
    score = {}
    for k, t in cands.items():
        ok = ~pd.isna(t)
        el, _ = sun(t[ok], lat, lon)
        score[k] = np.corrcoef(np.maximum(np.sin(np.radians(el)), 0), s.SRAVG.values[ok])[0, 1]
    return max(score, key=score.get), score


def to_utc(ts, tz):
    t = pd.DatetimeIndex(ts)
    if tz == "UTC": return t
    if tz == "CST(UTC-6)": return t + pd.Timedelta(hours=6)
    return t.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").tz_convert(None)


def circ_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def station_15min(d, tz):
    d = d.copy()
    d["t"] = to_utc(d.timestamp_raw, tz)
    d = d.dropna(subset=["t"]).set_index("t").sort_index()
    for h in ("2M", "10M"):                                  # wind vectors for direction averaging
        r = np.radians(d[f"WDIR{h}"])
        d[f"u{h}"] = -d[f"WSPD{h}AVG"] * np.sin(r); d[f"v{h}"] = -d[f"WSPD{h}AVG"] * np.cos(r)
    cols = ["TEMP2MAVG", "TEMP10MAVG", "WSPD2MAVG", "WSPD10MAVG", "u2M", "v2M", "u10M", "v10M",
            "RELHUM2MAVG", "SRAVG", "WSPD2MMAX", "WSPD10MMAX"]
    g = d[cols].resample("15min", label="right", closed="right")
    m = g.mean(); n = g.count()
    m.loc[n[["TEMP2MAVG", "TEMP10MAVG", "WSPD2MAVG", "WSPD10MAVG"]].min(axis=1) < 3] = np.nan
    m["WSPD2MMAX"] = g["WSPD2MMAX"].max(); m["WSPD10MMAX"] = g["WSPD10MMAX"].max()
    out = pd.DataFrame(index=m.index)
    out["y_dT"] = m.TEMP10MAVG - m.TEMP2MAVG                  # >0 = inversion
    out["y_ws2"] = m.WSPD2MAVG; out["y_ws10"] = m.WSPD10MAVG
    out["y_wd2"] = (np.degrees(np.arctan2(-m.u2M, -m.v2M)) % 360)
    out["y_wd10"] = (np.degrees(np.arctan2(-m.u10M, -m.v10M)) % 360)
    out["y_ddir"] = circ_diff(out.y_wd10, out.y_wd2)
    out["y_t2"] = m.TEMP2MAVG
    out["obs_rh"] = m.RELHUM2MAVG; out["obs_srad"] = m.SRAVG
    out["obs_gust2"] = m.WSPD2MMAX; out["obs_gust10"] = m.WSPD10MMAX
    return out.dropna(subset=["y_dT", "y_ws2", "y_ws10"]).astype("float32")


def hrrr_feats(f):
    h = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
    o = pd.DataFrame(index=h.index)
    o["h_t2"] = F2C(h.temp_2m_F); o["h_rh"] = h.rh_pct
    o["h_ws10"] = h.wind_10m_mph * MPH; o["h_ws80"] = h.wind_80m_mph * MPH
    for z in ("10", "80"):
        r = np.radians(h[f"dir_{z}m_deg"])
        o[f"h_u{z}"] = -o[f"h_ws{z}"] * np.sin(r); o[f"h_v{z}"] = -o[f"h_ws{z}"] * np.cos(r)
    o["h_ratio"] = h.wind_ratio_80_10; o["h_ddir"] = h.dir_diff_deg
    o["h_cool"] = h.cooling_F_per_hr / 1.8
    o["h_dt1h"] = o.h_t2.diff(4)                               # 1-h temperature change
    o["h_ws10_1h"] = o.h_ws10.rolling(4).mean()
    return o.astype("float32")


def dino_raw(n):
    """Per image date: cls, mean patch and station patch vectors (no PCA here)."""
    import xarray as xr
    z = xr.open_zarr(Path("dino") / f"{fn(n)}.zarr")
    r, c = z.attrs["station_prow"], z.attrs["station_pcol"]
    parts = {"cls": z.cls.values.astype("float32"),                       # small pieces, low memory
             "mean": z.patches.astype("float32").mean(("prow", "pcol")).values,
             "pt": z.patches.isel(prow=r, pcol=c).values.astype("float32")}
    cols = {f"{g}{k}": v[:, k] for g, v in parts.items() for k in range(v.shape[1])}
    o = pd.DataFrame(cols, index=pd.DatetimeIndex(z.time.values, name="img_time")).astype("float16")
    o["clear_frac"] = z.clear_frac.values
    return o.reset_index()


def fn(n):
    return n.replace(" ", "_")


def stale(out, *inputs):
    return not out.exists() or any(i.exists() and i.stat().st_mtime > out.stat().st_mtime for i in inputs)


def build_station(n, r, tz):
    s = station_15min(pd.read_parquet(Path("ks5min") / f"{fn(n)}.parquet"), tz)
    h = hrrr_feats(Path("hrrr") / f"{key(n)}.csv")
    t = s.join(h, how="inner")
    t["sun_el"], t["sun_az"] = sun(t.index, r.latitude, r.longitude)
    t["sun_el_1h"], _ = sun(t.index + pd.Timedelta(hours=1), r.latitude, r.longitude)   # rising / setting
    doy = t.index.day_of_year
    t["doy_s"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    t["doy_c"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    t["elev"] = np.float32(r.elevation)
    t.insert(0, "station", n)
    t.reset_index(names="time").to_parquet(OUT / f"{fn(n)}.parquet", index=False)
    return t


def run_once(st, tz_cache):
    names = set(st.name)
    done_any = False
    for f in sorted(Path("ks5min").glob("*.parquet")):
        n = next((x for x in names if fn(x) == f.stem), None)
        if n is None: continue
        r = st[st.name == n].iloc[0]
        hf = Path("hrrr") / f"{key(n)}.csv"
        out = OUT / f"{fn(n)}.parquet"
        if hf.exists() and stale(out, f, hf):
            if tz_cache.get("tz") is None:
                tz_cache["tz"], sc = detect_tz(pd.read_parquet(f), r.latitude, r.longitude)
                print("station time zone:", tz_cache["tz"], {k: round(v, 3) for k, v in sc.items()}, flush=True)
            t = build_station(n, r, tz_cache["tz"])
            print(f"table  {n}: {len(t):,} rows, {t.y_dT.gt(0).mean():.0%} inversion", flush=True)
            done_any = True
    for z in sorted(Path("dino").glob("*.zarr")) if Path("dino").exists() else []:
        if z.name.endswith(".tmp.zarr"): continue
        n = next((x for x in names if fn(x) == z.stem), None)
        out = DOUT / f"{z.stem}.parquet"
        if n is None or not stale(out, z): continue
        d = dino_raw(n); d.to_parquet(out, index=False)
        print(f"dino   {n}: {len(d)} images", flush=True)
        done_any = True
    return done_any


if __name__ == "__main__":
    st = pd.read_csv("ks_stations_both_heights.csv")
    watch = "--watch" in sys.argv
    tz_cache = {}
    while True:
        run_once(st, tz_cache)
        nt, nd = len(list(OUT.glob("*.parquet"))), len(list(DOUT.glob("*.parquet")))
        print(f"-- {time.strftime('%H:%M')}  tables {nt}/{len(st)}, dino {nd}/{len(st)}", flush=True)
        if not watch or (nt == len(st) and nd == len(st)): break
        time.sleep(300)
