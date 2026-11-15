# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""
Quality check of the per-station 15-min HRRR files from fetch_hrrr15_stations.py.

    python science/qc_station_plots.py                       # 6 random station/week picks
    python science/qc_station_plots.py --n 10 --days 5 --seed 7

1. Range check over ALL stations: min/max of every variable, impossible values,
   gaps in the 15-min time steps.
2. Plots of random stations and random windows (all variables), with night
   shaded from the sun's position -> data/qc_hrrr15_ks/<station>_<start>.png
Times on the plots are US Central (most of Kansas; the far west is Mountain).
The seed makes the random picks repeatable.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ap = argparse.ArgumentParser()
ap.add_argument("--folder", default=str(DATA_DIR / "hrrr15_stations_ks_stations_both_heights_2025-11-15_2026-11-14"))
ap.add_argument("--stations", default=str(DATA_DIR / "ks_stations_both_heights.csv"))
ap.add_argument("--n", type=int, default=6, help="number of random plots")
ap.add_argument("--days", type=int, default=5, help="length of each plotted window")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

folder = Path(args.folder)
out_dir = DATA_DIR / "qc_hrrr15_ks"
out_dir.mkdir(exist_ok=True)
meta = pd.read_csv(args.stations)
meta["file"] = meta["name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
meta = meta.set_index("file")

# Physically possible ranges (anything outside = a data problem)
LIMITS = {"temp_2m_F": (-40, 120), "rh_pct": (0, 100), "wind_10m_mph": (0, 100),
          "wind_80m_mph": (0, 150), "dir_10m_deg": (0, 360), "dir_80m_deg": (0, 360)}


def sun_elevation(times_utc, lat, lon):
    """Approximate solar elevation angle (degrees); below 0 = night."""
    t = pd.DatetimeIndex(times_utc)
    doy = t.day_of_year.values
    hour = t.hour.values + t.minute.values / 60
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)        # sun declination
    hour_angle = np.radians(15 * (hour + lon / 15 - 12))                   # local solar time
    lat = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(lat) * np.sin(decl)
                                + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)))


# ---- 1. Range check over all stations ----
files = sorted(folder.glob("*.csv"))
rows = []
for f in files:
    d = pd.read_csv(f, parse_dates=["time_utc"])
    bad = sum(((d[c] < lo) | (d[c] > hi)).sum() for c, (lo, hi) in LIMITS.items())
    steps = d["time_utc"].diff().dropna()
    rows.append({"station": f.stem, "rows": len(d), "missing": int(d[list(LIMITS)].isna().sum().sum()),
                 "out_of_range": int(bad), "bad_steps": int((steps != pd.Timedelta("15min")).sum()),
                 "T_min": d["temp_2m_F"].min(), "T_max": d["temp_2m_F"].max(),
                 "T_mean": round(d["temp_2m_F"].mean(), 1),
                 "wind10_max": d["wind_10m_mph"].max(), "wind80_max": d["wind_80m_mph"].max(),
                 "ratio_median": d["wind_ratio_80_10"].median()})
qc = pd.DataFrame(rows)
qc.to_csv(out_dir / "qc_summary_all_stations.csv", index=False)
print(f"1. Checked {len(qc)} stations -> data/qc_hrrr15_ks/qc_summary_all_stations.csv")
print(f"   rows per station: {qc['rows'].min()}-{qc['rows'].max()} | missing values: {qc['missing'].sum()} | "
      f"out of range: {qc['out_of_range'].sum()} | bad time steps: {qc['bad_steps'].sum()}")
print(f"   temp range over all stations: {qc['T_min'].min()} to {qc['T_max'].max()} F | "
      f"annual mean: {qc['T_mean'].min()}-{qc['T_mean'].max()} F")
print(f"   max wind: 10 m {qc['wind10_max'].max()} mph, 80 m {qc['wind80_max'].max()} mph | "
      f"median 80/10 ratio: {qc['ratio_median'].min()}-{qc['ratio_median'].max()}")
coldest, warmest = qc.loc[qc["T_mean"].idxmin(), "station"], qc.loc[qc["T_mean"].idxmax(), "station"]
print(f"   coldest annual mean: {coldest} ({meta.loc[coldest, 'elevation']:.0f} m), "
      f"warmest: {warmest} ({meta.loc[warmest, 'elevation']:.0f} m)")

# ---- 2. Random plots ----
rng = np.random.default_rng(args.seed)
picks = rng.choice(len(files), size=args.n, replace=False)
print(f"\n2. Plots (seed {args.seed}):")
for i in picks:
    f = files[i]
    st = meta.loc[f.stem]
    d = pd.read_csv(f, parse_dates=["time_utc"])
    start = d["time_utc"].iloc[rng.integers(0, len(d) - args.days * 96)].normalize()
    w = d[(d["time_utc"] >= start) & (d["time_utc"] < start + pd.Timedelta(days=args.days))].copy()
    w["local"] = w["time_utc"].dt.tz_localize("UTC").dt.tz_convert("America/Chicago").dt.tz_localize(None)
    night = sun_elevation(w["time_utc"], st["latitude"], st["longitude"]) < 0

    fig, ax = plt.subplots(7, 1, figsize=(12, 13), sharex=True)
    ax[0].plot(w["local"], w["temp_2m_F"]); ax[0].set_ylabel("Temp 2 m\n(F)")
    ax[1].plot(w["local"], w["rh_pct"], c="tab:cyan"); ax[1].set_ylabel("RH 2 m\n(%)"); ax[1].set_ylim(0, 105)
    ax[2].plot(w["local"], w["wind_10m_mph"], c="tab:green", label="10 m")
    ax[2].plot(w["local"], w["wind_80m_mph"], c="tab:green", ls="--", label="80 m")
    ax[2].set_ylabel("Wind\n(mph)"); ax[2].legend(loc="upper left", ncol=2)
    ax[3].scatter(w["local"], w["dir_10m_deg"], s=4, c="tab:orange", label="10 m")
    ax[3].scatter(w["local"], w["dir_80m_deg"], s=4, c="tab:brown", label="80 m")
    ax[3].set_ylabel("Wind dir\n(deg, from)"); ax[3].set_ylim(0, 360); ax[3].set_yticks([0, 90, 180, 270, 360])
    ax[3].legend(loc="upper left", ncol=2, markerscale=3)
    ax[4].plot(w["local"], w["wind_ratio_80_10"], c="tab:purple"); ax[4].axhline(1.5, c="k", ls=":", lw=0.8)
    ax[4].set_ylabel("Wind ratio\n80/10"); ax[4].set_ylim(0, 8)
    ax[5].plot(w["local"], w["dir_diff_deg"], c="tab:red"); ax[5].set_ylabel("Dir diff\n80 vs 10 (deg)")
    ax[5].set_ylim(0, 180)
    ax[6].plot(w["local"], w["cooling_F_per_hr"]); ax[6].axhline(0, c="k", lw=0.8)
    ax[6].set_ylabel("2 m change\n(F per hr)")
    for a in ax:
        a.fill_between(w["local"], 0, 1, where=night, transform=a.get_xaxis_transform(),
                       color="0.85", lw=0, step="mid", zorder=0)  # behind the data
        a.grid(alpha=0.3)
    fig.suptitle(f"{st['name']} ({st['county']} Co., {st['elevation']:.0f} m) - HRRR 15-min, "
                 f"{w['local'].iloc[0]:%Y-%m-%d} +{args.days} days, US Central time (gray = night)")
    fig.autofmt_xdate()
    fig.tight_layout()
    png = out_dir / f"{f.stem}_{start:%Y-%m-%d}.png"
    fig.savefig(png, dpi=90)
    plt.close(fig)
    print(f"   {st['name']:20} {start:%Y-%m-%d}  T {w['temp_2m_F'].min()}-{w['temp_2m_F'].max()} F, "
          f"wind10 max {w['wind_10m_mph'].max()} mph -> data/qc_hrrr15_ks/{png.name}")
