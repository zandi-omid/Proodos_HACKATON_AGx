"""Latest >=80%-clear Sentinel-2 RGB for every HRRR cell in Kansas that contains orchard.

Step 1 (login node, once):   python s2_orchards.py cells
    -> ks_orchard_cells.csv  (HRRR i, j, cell centre lon/lat, orchard ha, main CDL crop code)
Step 2 (Slurm array):        sbatch s2_orchards.sh
    -> s2_orchards/cell_<i>_<j>.zarr : one date, red/green/blue/scl, same 320x320 10 m window
       (3.2 km, centred on the HRRR cell) as the station pull; attrs as s2_pull.
Needs s2_pull.py in the same folder (reuses its HRRR grid + window code).
Re-running skips cells already saved.
"""
import os, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
import s2_pull as S

KS = [[-101.906, 40.002], [-95.306, 40.002], [-95.208, 39.909], [-94.885, 39.832], [-95.109, 39.542],
      [-94.983, 39.443], [-94.824, 39.207], [-94.611, 39.158], [-94.616, 37.0], [-100.088, 37.0],
      [-102.043, 36.995], [-102.054, 40.002], [-101.906, 40.002]]
CELLS = Path("ks_orchard_cells.csv")
OUT = Path("s2_orchards"); OUT.mkdir(exist_ok=True)
MIN_CLEAR = 0.8
LOOKBACK_DAYS = [60, 180, 400]          # widen the search only if nothing clear is found
MAX_TILE_CLOUD = 60                      # skip scenes whose whole-tile cloud cover is above this


def find_cells(raster="cdl_orchards_2025_10m.tif"):
    import rasterio
    from rasterio.windows import from_bounds, Window
    from matplotlib.path import Path as MPath
    src = rasterio.open(raster)
    tr = Transformer.from_crs(4326, src.crs, always_xy=True)
    inv = Transformer.from_crs(src.crs, 4326, always_xy=True)
    xs, ys = zip(*[tr.transform(*p) for p in KS])
    poly = MPath(np.c_[xs, ys])
    w = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
    c0, r0, W, H = int(w.col_off), int(w.row_off), int(w.width), int(w.height)
    parts = []
    for r in range(r0, r0 + H, 1000):                       # 1000-row strips, low memory
        a = src.read(1, window=Window(c0, r, W, min(1000, r0 + H - r)))
        iy, ix = np.nonzero(a)
        if not len(iy): continue
        x, y = src.transform * (ix + c0 + 0.5, iy + r + 0.5)
        x, y = np.asarray(x), np.asarray(y)
        k = poly.contains_points(np.c_[x, y])
        lon, lat = inv.transform(x[k], y[k])
        hx, hy = S.LL2H.transform(lon, lat)
        parts.append(pd.DataFrame({"i": np.rint((np.asarray(hx) - S.X0) / S.DX).astype(int),
                                   "j": np.rint((np.asarray(hy) - S.Y0) / S.DX).astype(int),
                                   "crop": a[iy[k], ix[k]]}))
    d = pd.concat(parts)
    g = d.groupby(["i", "j"]).agg(orchard_px=("crop", "size"),
                                  main_crop=("crop", lambda s: int(s.mode().iloc[0]))).reset_index()
    g["orchard_ha"] = g.orchard_px * 0.01
    g["lon"], g["lat"] = S.H2LL.transform((S.X0 + g.i * S.DX).values, (S.Y0 + g.j * S.DX).values)
    g.sort_values("orchard_ha", ascending=False).to_csv(CELLS, index=False)
    print(f"{len(g)} HRRR cells with orchard in Kansas, {g.orchard_ha.sum():,.0f} ha total")
    print(g.orchard_ha.describe().round(2).to_string())


def day(it):
    return pd.Timestamp(it.datetime).tz_convert(None).normalize()


def pull_cell(row, catalog):
    import odc.stac
    from odc.geo.geobox import GeoBox
    f = OUT / f"cell_{row.i}_{row.j}.zarr"
    if f.exists():
        return "skipped"
    i, j, (clon, clat), corners_ll = S.hrrr_cell(row.lon, row.lat)
    utm, bbox, corners_utm, ll = S.window(clon, clat, corners_ll)
    poly = {"type": "Polygon", "coordinates": [ll + [ll[0]]]}
    gbox = GeoBox.from_bbox(bbox, crs=utm.to_string(), resolution=S.RES)
    end = pd.Timestamp.now().normalize()
    tried = set()
    for lb in LOOKBACK_DAYS:
        items = list(catalog.search(collections=[S.COLLECTION], intersects=poly,
                                    datetime=f"{(end - pd.Timedelta(days=lb)).date()}/{end.date()}").items())
        items = [it for it in items if not it.assets["red"].href.endswith(".jp2")
                 and float(it.properties.get("eo:cloud_cover", 0) or 0) <= MAX_TILE_CLOUD]
        for d in sorted({day(it) for it in items} - tried, reverse=True):      # newest first
            tried.add(d)
            its = [it for it in items if day(it) == d]
            scl = odc.stac.load(its, bands=["scl"], geobox=gbox, groupby="solar_day",
                                fail_on_error=False).scl.isel(time=0).values
            clear = float(np.isin(scl, S.CLEAR).mean())
            if clear < MIN_CLEAR:
                continue
            ds = odc.stac.load(its, bands=["red", "green", "blue", "scl"], geobox=gbox,
                               groupby="solar_day", pool=8, fail_on_error=False).isel(time=[0]).load()
            valid = float((ds.red > 0).mean())
            if valid < 0.999:
                continue
            ds["valid_frac"] = ("time", np.array([valid], "float32"))
            ds["clear_frac"] = ("time", np.array([float(np.isin(ds.scl.values, S.CLEAR).mean())], "float32"))
            ds.attrs.update(cell=f"{i}_{j}", hrrr_i=i, hrrr_j=j, hrrr_center_lon=clon, hrrr_center_lat=clat,
                            station_lon=clon, station_lat=clat,        # "point" = cell centre (for dino_feats)
                            orchard_ha=float(row.orchard_ha), main_crop=int(row.main_crop),
                            hrrr_corners_utm=str([tuple(round(v, 1) for v in c) for c in corners_utm]),
                            crs=utm.to_string(), collection=S.COLLECTION, size_px=S.SIZE, res_m=S.RES)
            if "spatial_ref" in ds.coords:
                ds = ds.drop_vars("spatial_ref")
            tmp = f.with_suffix(".tmp.zarr"); shutil.rmtree(tmp, ignore_errors=True)
            ds.to_zarr(tmp, mode="w"); tmp.rename(f)
            return f"{d.date()} clear {clear:.0%}"
    return "NONE clear enough"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cells":
        find_cells(); sys.exit()
    import odc.stac
    from pystac_client import Client
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    cells = pd.read_csv(CELLS)
    shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    nshard = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    cells = cells.iloc[shard::nshard]
    print(f"shard {shard}/{nshard}: {len(cells)} cells", flush=True)
    catalog = Client.open(S.STAC)
    t0 = time.time()
    for k, (_, r) in enumerate(cells.iterrows()):
        try:
            msg = pull_cell(r, catalog)
        except Exception as ex:
            msg = f"ERROR {str(ex)[:120]}"
        print(f"[{k + 1}/{len(cells)}] cell {r.i}_{r.j} ({r.orchard_ha:.1f} ha): {msg}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
