# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""Spray or Wait? Go / Caution / Stop for every Kansas orchard cell, every 15 min, next 18 h.

python predict_orchards.py [hrrr_fcst_YYYYMMDDHH.parquet]     (default: newest hrrr_fcst_*.parquet)
Reads  hrrr forecast, dino_orchards/*.zarr, dino_pca.npz, models/*.txt + models/features.json,
       ks_orchard_cells.csv
Writes orchard_forecast_<cycle>.parquet  every cell x 15 min: predicted 2 m / 10 m wind + inversion,
                                          status, plain-language reasons, and what a single ground
                                          station (2 m wind only) would have said
       orchard_log_<cycle>.csv          time-stamped log: one line per alert (status != GO)
       orchard_summary_<cycle>.csv      per cell: status now, next safe (GO >= 1 h) window
"""
import sys, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from build_table import sun

MS2MPH, C2F = 2.23694, 1.8
# ---- rules (thresholds from the brief; label always governs) ----
CALM, WINDY = 3.0, 10.0          # mph
INV, STRONG_INV = 0.3, 1.0       # deg C, 10 m minus 2 m (0.5 / 1.8 F)
SPEED_DIFF, DIR_DIFF = 5.0, 45.0 # mph, degrees


def rules(dT, ws2, ws10, ddir):
    """Vectorised Go/Caution/Stop + reasons. ws in mph, dT in C."""
    n = len(dT); status = np.full(n, "GO", object); reasons = [[] for _ in range(n)]

    def flag(mask, level, text):
        for k in np.where(mask)[0]:
            reasons[k].append(text(k))
            if level == "STOP" or status[k] == "GO": status[k] = level

    inv = dT > INV
    flag(inv & (ws2 < CALM), "STOP", lambda k: f"Inversion: air at 30 ft is {dT[k]*C2F:.1f}°F warmer than at 5 ft, with calm wind ({ws2[k]:.1f} mph) - spray will hang and drift")
    flag(dT > STRONG_INV, "STOP", lambda k: f"Strong inversion ({dT[k]*C2F:.1f}°F warmer at 30 ft)")
    flag(inv & (dT <= STRONG_INV) & (ws2 >= CALM), "CAUTION", lambda k: f"Weak inversion ({dT[k]*C2F:.1f}°F warmer at 30 ft)")
    flag(np.maximum(ws2, ws10) > WINDY, "STOP", lambda k: f"Too windy: {max(ws2[k], ws10[k]):.0f} mph (limit {WINDY:.0f})")
    flag(~inv & (ws2 < CALM), "CAUTION", lambda k: f"Calm at canopy height ({ws2[k]:.1f} mph < {CALM:.0f}) - inversion may be forming")
    flag(np.abs(ws10 - ws2) > SPEED_DIFF, "CAUTION", lambda k: f"Wind shear: {ws10[k]:.0f} mph at 30 ft vs {ws2[k]:.0f} mph at 5 ft")
    flag((ddir > DIR_DIFF) & (ws2 >= 2) & (ws10 >= 2), "CAUTION", lambda k: f"Wind direction differs by {ddir[k]:.0f}° between 5 ft and 30 ft")
    return status, ["; ".join(r) if r else "Conditions within guidance at both heights" for r in reasons]


def ground_only(ws2):
    """What a single 5-ft station would say: wind 3-10 mph = fine."""
    return np.where((ws2 >= CALM) & (ws2 <= WINDY), "GO", "NOT OK")


def hrrr_features(h):
    o = pd.DataFrame({"cell": h.cell, "time": pd.to_datetime(h.time_utc)})
    o["h_t2"] = (h.temp_2m_F - 32) / 1.8; o["h_rh"] = h.rh_pct
    o["h_ws10"] = h.wind_10m_mph * 0.44704; o["h_ws80"] = h.wind_80m_mph * 0.44704
    for z in ("10", "80"):
        r = np.radians(h[f"dir_{z}m_deg"])
        o[f"h_u{z}"] = -o[f"h_ws{z}"] * np.sin(r); o[f"h_v{z}"] = -o[f"h_ws{z}"] * np.cos(r)
    o["h_ratio"] = h.wind_ratio_80_10; o["h_ddir"] = h.dir_diff_deg
    o = o.sort_values(["cell", "time"])
    o["h_dt1h"] = o.groupby("cell").h_t2.diff(4)
    o["h_ws10_1h"] = o.groupby("cell").h_ws10.transform(lambda s: s.rolling(4).mean())
    return o


def dino_features(npc):
    pca = np.load("dino_pca.npz")
    rows = []
    for f in sorted(Path("dino_orchards").glob("*.zarr")):
        if f.name.endswith(".tmp.zarr"): continue
        z = xr.open_zarr(f)
        if z.sizes["time"] == 0: continue
        r, c = z.attrs["station_prow"], z.attrs["station_pcol"]
        vec = {"cls": z.cls.values[-1].astype("float32"),
               "mean": z.patches.values[-1].astype("float32").mean((0, 1)),
               "pt": z.patches.values[-1, r, c].astype("float32")}
        row = {"cell": f"{int(z.attrs['hrrr_i'])}_{int(z.attrs['hrrr_j'])}", "img_time": pd.Timestamp(z.time.values[-1])}
        for g, v in vec.items():
            Z = (v - pca[f"{g}_mu"]) @ pca[f"{g}_V"].T
            row.update({f"d_{g}{k}": Z[k] for k in range(npc)})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("hrrr_fcst_*.parquet"))[-1]
    cycle = Path(f).stem.split("_")[-1]
    meta = json.load(open("models/features.json")); feats = meta["features"]
    cells = pd.read_csv("ks_orchard_cells.csv")
    cells["cell"] = cells.i.astype(int).astype(str) + "_" + cells.j.astype(int).astype(str)
    X = hrrr_features(pd.read_parquet(f)).merge(cells[["cell", "lon", "lat", "orchard_ha", "main_crop"]], on="cell")
    X["sun_el"], X["sun_az"] = sun(X.time, X.lat.values, X.lon.values)
    X["sun_el_1h"], _ = sun(X.time + pd.Timedelta(hours=1), X.lat.values, X.lon.values)
    doy = X.time.dt.day_of_year
    X["doy_s"], X["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if any(c.startswith("d_") for c in feats):
        D = dino_features(json.load(open("lgbm_meta.json"))["npc"])
        X = X.merge(D, on="cell", how="left")
        X["img_age_d"] = (X.time - X.img_time).dt.total_seconds() / 86400
        print(f"DINO features for {D.cell.nunique()} of {X.cell.nunique()} cells", flush=True)
    for c in feats:
        if c not in X: X[c] = np.nan                               # missing features -> NaN (LightGBM copes)
    for t in meta["targets"]:
        X[f"p_{t}"] = lgb.Booster(model_file=f"models/{t}.txt").predict(X[feats].astype("float32"))
    for h in ("2", "10"):
        X[f"ws{h}_mph"] = np.hypot(X[f"p_u{h}"], X[f"p_v{h}"]) * MS2MPH
        X[f"wd{h}"] = np.degrees(np.arctan2(-X[f"p_u{h}"], -X[f"p_v{h}"])) % 360
    X["ddir"] = np.abs((X.wd10 - X.wd2 + 180) % 360 - 180)
    X["inv_F"] = X.p_y_dT * C2F
    X["status"], X["reason"] = rules(X.p_y_dT.values, X.ws2_mph.values, X.ws10_mph.values, X.ddir.values)
    X["single_station"] = ground_only(X.ws2_mph.values)
    X["missed_by_single_station"] = (X.single_station == "GO") & (X.status != "GO")
    X["time_local"] = X.time.dt.tz_localize("UTC").dt.tz_convert("America/Chicago").dt.tz_localize(None)
    keep = ["cell", "lon", "lat", "orchard_ha", "main_crop", "time", "time_local", "status", "reason",
            "inv_F", "ws2_mph", "ws10_mph", "wd2", "wd10", "ddir", "single_station", "missed_by_single_station"]
    out = X[keep].sort_values(["cell", "time"])
    out.to_parquet(f"orchard_forecast_{cycle}.parquet", index=False)
    log = out[out.status != "GO"][["time_local", "cell", "status", "reason", "inv_F", "ws2_mph", "ws10_mph", "ddir"]].round(1)
    log.to_csv(f"orchard_log_{cycle}.csv", index=False)

    now = out.time.min()
    def summary(g):
        g = g.reset_index(drop=True)
        go = (g.status == "GO").astype(int)
        run = go.groupby((go != go.shift()).cumsum()).transform("sum") * go   # length of each GO streak
        nxt = g[run >= 4]                                                   # >= 4 x 15 min = 1 h
        return pd.Series({"status_now": g.status.iloc[0], "reason_now": g.reason.iloc[0],
                          "next_safe_window": nxt.time_local.iloc[0] if len(nxt) else pd.NaT,
                          "orchard_ha": g.orchard_ha.iloc[0], "lon": g.lon.iloc[0], "lat": g.lat.iloc[0]})
    s = out.groupby("cell").apply(summary).reset_index()
    s.to_csv(f"orchard_summary_{cycle}.csv", index=False)
    print(f"cycle {cycle}: {out.cell.nunique()} cells x {out.time.nunique()} times, from {now} UTC")
    print("status share over the 18 h:", out.status.value_counts(normalize=True).round(3).to_dict())
    print("status now:", s.status_now.value_counts().to_dict())
    print(f"single 5-ft station says GO but tool says CAUTION/STOP: {out.missed_by_single_station.mean():.1%} of cell-times")
    print("\nexample alerts:\n" + log.sort_values("inv_F", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
