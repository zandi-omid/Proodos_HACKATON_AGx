# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""Latest HRRR forecast, every 15 min out to 18 h, for every Kansas orchard cell.

python hrrr_fcst.py                 (needs: pip install herbie-data)
Reads  ks_orchard_cells.csv (HRRR i, j per cell)
Writes hrrr_fcst_<cycle>.parquet : cell, i, j, time_utc + the same columns as the training HRRR CSVs
       (temp_2m_F, rh_pct, wind_10m_mph, wind_80m_mph, dir_10m_deg, dir_80m_deg,
        wind_ratio_80_10, dir_diff_deg, cooling_F_per_hr)
Uses HRRR's 15-minute ("subh") output, f00-f18 of the newest cycle that has f18 out.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie

SAVE = Path("/scratch/ilolos/herbie")
SEARCH = r":(?:TMP|DPT):2 m above ground|:(?:UGRD|VGRD):(?:10|80) m above ground"
LEVELS = {2: ("t", "d"), 10: ("u", "v"), 80: ("u", "v")}
NAMES = {"t": {"2t", "t"}, "d": {"2d", "d"}, "u": {"10u", "u"}, "v": {"10v", "v"}}
LOV, ROT = -97.5, np.sin(np.radians(38.5))          # HRRR grid -> earth wind rotation


def latest_cycle():
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    for back in range(1, 7):
        c = now - pd.Timedelta(hours=back)
        if Herbie(c, model="hrrr", product="subh", fxx=18, verbose=False).grib is not None:
            return c
    raise RuntimeError("no recent HRRR cycle with f18 found")


def read_file(cycle, f, I, J):
    H = Herbie(cycle, model="hrrr", product="subh", fxx=f, save_dir=SAVE, verbose=False)
    H.download(SEARCH, verbose=False)
    path = H.get_localFilePath(SEARCH)
    out, vt = {}, None
    for lev, wants in LEVELS.items():
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": lev}, "indexpath": ""})
        for w in wants:
            v = [k for k in ds.data_vars if ds[k].attrs.get("GRIB_shortName") in NAMES[w]][0]
            a = ds[v].values
            a = a[None] if a.ndim == 2 else a                          # (time, y, x)
            out[f"{w}{lev}"] = a[:, J, I]
        vt = np.atleast_1d(ds.valid_time.values)
    return vt, out


def main():
    cells = pd.read_csv("ks_orchard_cells.csv")
    I, J = cells.i.astype(int).values, cells.j.astype(int).values
    cycle = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else latest_cycle()
    print(f"HRRR cycle {cycle:%Y-%m-%d %H}Z, {len(cells)} cells", flush=True)
    frames = []
    for f in range(0, 19):
        t0 = time.time()
        try:
            vt, o = read_file(cycle, f, I, J)
        except Exception as ex:
            print(f"f{f:02d}: skipped ({str(ex)[:80]})", flush=True); continue
        for k, t in enumerate(vt):
            frames.append(pd.DataFrame({"cell": cells.i.astype(int).astype(str) + "_" + cells.j.astype(int).astype(str),
                                        "i": I, "j": J, "time_utc": pd.Timestamp(t),
                                        **{n: a[k] for n, a in o.items()}}))
        print(f"f{f:02d}: {len(vt)} times  ({time.time() - t0:.0f} s)", flush=True)
    d = pd.concat(frames).drop_duplicates(["cell", "time_utc"]).sort_values(["cell", "time_utc"])
    lon = pd.read_csv("ks_orchard_cells.csv").set_index(
        cells.i.astype(int).astype(str) + "_" + cells.j.astype(int).astype(str)).lon
    a = np.radians(ROT * (d.cell.map(lon).values - LOV))
    for z in ("10", "80"):                                        # grid-relative -> earth-relative
        u, v = d[f"u{z}"].values, d[f"v{z}"].values
        d[f"u{z}"], d[f"v{z}"] = np.cos(a) * u + np.sin(a) * v, -np.sin(a) * u + np.cos(a) * v
    T = d.t2.values - 273.15; Td = d.d2.values - 273.15
    o = d[["cell", "i", "j", "time_utc"]].copy()
    o["temp_2m_F"] = (T * 9 / 5 + 32).round(1)
    o["rh_pct"] = np.clip(100 * np.exp(17.625 * Td / (243.04 + Td) - 17.625 * T / (243.04 + T)), 0, 100).round()
    for z in ("10", "80"):
        u, v = d[f"u{z}"].values, d[f"v{z}"].values
        o[f"wind_{z}m_mph"] = (np.hypot(u, v) / 0.44704).round(1)
        o[f"dir_{z}m_deg"] = (np.degrees(np.arctan2(-u, -v)) % 360).round()
    o["wind_ratio_80_10"] = (o.wind_80m_mph / o.wind_10m_mph.clip(lower=0.1)).round(1)
    o["dir_diff_deg"] = np.abs((o.dir_80m_deg - o.dir_10m_deg + 180) % 360 - 180)
    o["cooling_F_per_hr"] = -o.groupby("cell").temp_2m_F.diff(4)  # not used by the model
    out = f"hrrr_fcst_{cycle:%Y%m%d%H}.parquet"
    o.to_parquet(out, index=False)
    print(f"saved {out}: {o.cell.nunique()} cells x {o.time_utc.nunique()} times "
          f"({o.time_utc.min()} -> {o.time_utc.max()} UTC)")


if __name__ == "__main__":
    main()
