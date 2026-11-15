"""Sentinel-2 L2A RGB (+ cloud mask) for the HRRR 3 km cell around each station.

python s2_pull.py [stations_csv]      (default ks_stations_both_heights.csv)
pip install pystac-client odc-stac pyproj zarr

Per station -> s2/<station>.zarr with:
  red, green, blue (uint16 raw DN, 0 = no data), scl (uint8 scene class)
  valid_frac, clear_frac per date
  window: 320 x 320 px at 10 m (3.2 km, = 20 x 20 DINO patches) centred on the
  HRRR cell centre, in the local UTM zone; covers the 3 km cell + ~100 m buffer.
  attrs: HRRR i/j (0-based), cell centre lat/lon, cell corners in UTM.
Also s2_manifest.csv. Re-running skips stations already saved.

Reflectance: all dates here are processing baseline >= 04.00, so
reflectance = (DN - 1000) / 10000. baseline is stored per date to check.
"""
import sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

COLLECTION = "sentinel-2-l2a"
STAC = "https://earth-search.aws.element84.com/v1"
START = "2022-05-14"          # 6 months before the Mesonet pull starts
END = pd.Timestamp.now().strftime("%Y-%m-%d")
SIZE = 320                    # pixels, multiple of 16
RES = 10
CLEAR = [4, 5, 6, 11]         # SCL: vegetation, bare, water, snow
OUT = Path("s2"); OUT.mkdir(exist_ok=True)

# ---- HRRR grid (NCEP 3 km Lambert conformal, grid points = cell centres) ----
HRRR = CRS.from_proj4("+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
                      "+a=6371229 +b=6371229 +units=m +no_defs")
LL2H = Transformer.from_crs(4326, HRRR, always_xy=True)
H2LL = Transformer.from_crs(HRRR, 4326, always_xy=True)
X0, Y0 = LL2H.transform(-122.719528, 21.138123)   # grid point (1,1)
DX = 3000.0
NX, NY = 1799, 1059


def hrrr_cell(lon, lat):
    x, y = LL2H.transform(lon, lat)
    i, j = int(round((x - X0) / DX)), int(round((y - Y0) / DX))
    assert 0 <= i < NX and 0 <= j < NY, (lon, lat)
    xc, yc = X0 + i * DX, Y0 + j * DX
    corners = [(xc + sx * DX / 2, yc + sy * DX / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return i, j, H2LL.transform(xc, yc), [H2LL.transform(*c) for c in corners]


def window(clon, clat, corners_ll):
    zone = int((clon + 180) // 6) + 1
    utm = CRS.from_epsg(32600 + zone)
    t = Transformer.from_crs(4326, utm, always_xy=True)
    cx, cy = t.transform(clon, clat)
    cx, cy = round(cx / RES) * RES, round(cy / RES) * RES      # snap to S2 10 m grid
    h = SIZE * RES / 2
    bbox = (cx - h, cy - h, cx + h, cy + h)
    cu = [t.transform(*c) for c in corners_ll]
    assert all(bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3] for x, y in cu), "cell not inside window"
    inv = Transformer.from_crs(utm, 4326, always_xy=True)
    ll = [inv.transform(x, y) for x, y in ((bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3]))]
    return utm, bbox, cu, ll


def pull(row, catalog):
    import odc.stac
    from odc.geo.geobox import GeoBox
    name = row["name"]
    f = OUT / f"{name.replace(' ', '_')}.zarr"
    if f.exists():
        return None
    i, j, (clon, clat), corners_ll = hrrr_cell(row.longitude, row.latitude)
    utm, bbox, corners_utm, ll = window(clon, clat, corners_ll)
    poly = {"type": "Polygon", "coordinates": [ll + [ll[0]]]}
    items = list(catalog.search(collections=[COLLECTION], intersects=poly,
                                datetime=f"{START}/{END}").items())
    base = {}
    for it in items:
        d = pd.Timestamp(it.datetime).tz_localize(None).normalize()
        base[d] = max(base.get(d, 0.0), float(it.properties.get("s2:processing_baseline", 0) or 0))
    gbox = GeoBox.from_bbox(bbox, crs=utm.to_string(), resolution=RES)
    ds = odc.stac.load(items, bands=["red", "green", "blue", "scl"], geobox=gbox,
                       groupby="solar_day", pool=32, fail_on_error=False)
    ds = ds.sortby("time")
    valid = (ds.red > 0).mean(("y", "x")).compute()
    ds = ds.sel(time=valid.time[valid > 0])                  # drop dates outside the swath
    ds["valid_frac"] = (ds.red > 0).mean(("y", "x")).astype("float32")
    ds["clear_frac"] = ds.scl.isin(CLEAR).mean(("y", "x")).astype("float32")
    days = pd.DatetimeIndex(ds.time.values).normalize()
    ds["baseline"] = ("time", np.array([base.get(d, np.nan) for d in days], "float32"))
    ds.attrs.update(station=name, station_lon=row.longitude, station_lat=row.latitude,
                    hrrr_i=i, hrrr_j=j, hrrr_center_lon=clon, hrrr_center_lat=clat,
                    hrrr_corners_utm=str([tuple(round(v, 1) for v in c) for c in corners_utm]),
                    crs=utm.to_string(), collection=COLLECTION, size_px=SIZE, res_m=RES)
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    tmp = f.with_suffix(".tmp.zarr")
    shutil.rmtree(tmp, ignore_errors=True)
    ds.chunk({"time": 64}).to_zarr(tmp, mode="w")
    tmp.rename(f)
    return dict(station=name, hrrr_i=i, hrrr_j=j, crs=utm.to_string(), n_dates=ds.sizes["time"],
                n_clear80=int((ds.clear_frac > 0.8).sum()),
                first=str(ds.time.values[0])[:10], last=str(ds.time.values[-1])[:10])


if __name__ == "__main__":
    import odc.stac
    from pystac_client import Client
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    st = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "ks_stations_both_heights.csv")
    import os
    shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    nshard = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    st = st.iloc[shard::nshard]
    print(f"shard {shard}/{nshard}: {len(st)} stations", flush=True)
    cells = st.apply(lambda r: hrrr_cell(r.longitude, r.latitude)[:2], axis=1)
    print(f"{len(st)} stations in {cells.nunique()} HRRR cells; {START} -> {END}", flush=True)
    catalog = Client.open(STAC)
    rows, t0 = [], time.time()
    for k, (_, r) in enumerate(st.iterrows()):
        try:
            out = pull(r, catalog)
            msg = "skipped" if out is None else f"{out['n_dates']} dates, {out['n_clear80']} >80% clear"
            if out: rows.append(out)
        except Exception as ex:
            msg = f"ERROR {str(ex)[:120]}"
        print(f"[{k + 1}/{len(st)}] {r['name']}: {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if rows:
        m = pd.DataFrame(rows)
        m.to_csv(f"s2_manifest_{shard}.csv", index=False)
        print(m.to_string(index=False))
