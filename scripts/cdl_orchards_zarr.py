# Orchard-only 2025 CDL at native 10 m, read straight from the Source Cooperative copy of the USDA NASS CDL
# (Icechunk/Zarr, EPSG:5070). No zip download needed.
# Usage, from /scratch/ilolos/HACKATHON_AGX/CDL:  python cdl_orchards_zarr.py [threads]
# Output: cdl_orchards_2025_10m.tif        = CDL class where the pixel is an orchard, 0 everywhere else
#         cdl_orchards_2025_10m_counts.csv = pixels and hectares per orchard class
import csv, sys, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
import numpy as np, rasterio, icechunk, xarray as xr
from affine import Affine
from rasterio.windows import Window

YEAR, TILE = 2025, 8192                       # tile = 16 x 16 inner chunks of 512
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = f"cdl_orchards_{YEAR}_10m.tif"

ORCHARDS = {
    66: "Cherries", 67: "Peaches", 68: "Apples", 71: "Other Tree Crops", 72: "Citrus",
    74: "Pecans", 75: "Almonds", 76: "Walnuts", 77: "Pears", 204: "Pistachios",
    210: "Prunes", 211: "Olives", 212: "Oranges", 215: "Avocados", 217: "Pomegranates",
    218: "Nectarines", 220: "Plums", 223: "Apricots",
}   # tree orchards only; grapes (69), Christmas trees (70) and blueberries (242) left out
KEEP = np.zeros(256, bool); KEEP[list(ORCHARDS)] = True

storage = icechunk.s3_storage(bucket="chill", prefix="usda-cropland-data-layer/v0.1.0.icechunk",
                              endpoint_url="https://data.source.coop", region="us-east-1",
                              anonymous=True, force_path_style=True)
store = icechunk.Repository.open(storage).readonly_session("main").store
ds = xr.open_zarr(store, group="10m", chunks=None)
da = ds["crop_type"].sel(year=YEAR)
ny, nx = da.shape

# georeferencing: prefer GDAL GeoTransform on spatial_ref, else build it from pixel-centre coords
sr = ds["spatial_ref"].attrs if "spatial_ref" in ds else {}
if "GeoTransform" in sr:
    x0, dx, _, y0, _, dy = map(float, sr["GeoTransform"].split())
else:
    x, y = ds["x"].values, ds["y"].values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    x0, y0 = float(x[0]) - dx / 2, float(y[0]) - dy / 2
if dy > 0:
    sys.exit("y coordinate is ascending; this script expects north-up data")
transform = Affine(dx, 0, x0, 0, dy, y0)
crs = sr.get("crs_wkt", "EPSG:5070")
print(f"{YEAR}: grid {ny} x {nx}, pixel {dx} x {-dy} m, {THREADS} threads", flush=True)

prof = dict(driver="GTiff", width=nx, height=ny, count=1, dtype="uint8", nodata=0, crs=crs,
            transform=transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
            BIGTIFF="YES", SPARSE_OK="TRUE")      # empty tiles are never written -> small file

tiles = [(r, c) for r in range(0, ny, TILE) for c in range(0, nx, TILE)]

def read(rc):
    r, c = rc
    for attempt in range(5):                     # retry on network hiccups
        try:
            a = np.array(da.isel(y=slice(r, r + TILE), x=slice(c, c + TILE)).values, dtype="uint8")
            a[~KEEP[a]] = 0                      # in place, no extra copies
            return r, c, a
        except Exception as e:
            print(f"tile {r},{c} attempt {attempt + 1} failed: {e}", flush=True); time.sleep(10)
    raise RuntimeError(f"tile {r},{c} failed 5 times")

counts = np.zeros(256, np.int64); t0 = time.time(); i = 0
todo = iter(tiles)
with rasterio.open(OUT, "w", **prof) as dst, ThreadPoolExecutor(THREADS) as pool:
    pending = {pool.submit(read, rc) for rc in islice(todo, 2 * THREADS)}   # at most 2x threads tiles in memory
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            r, c, o = fut.result(); i += 1
            counts += np.bincount(o.ravel(), minlength=256)
            if o.any():
                dst.write(o, 1, window=Window(c, r, o.shape[1], o.shape[0]))
            del o
            nxt = next(todo, None)
            if nxt is not None:
                pending.add(pool.submit(read, nxt))
            if i % 50 == 0 or i == len(tiles):
                print(f"{i}/{len(tiles)} tiles, {(time.time() - t0) / 60:.1f} min", flush=True)

ha_per_px = dx * -dy / 10_000
with open(f"cdl_orchards_{YEAR}_10m_counts.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["code", "crop", "pixels", "hectares"])
    for code, name in sorted(ORCHARDS.items()):
        wr.writerow([code, name, int(counts[code]), round(counts[code] * ha_per_px, 1)])
print(f"{YEAR}: {OUT} done, total orchard area {counts[list(ORCHARDS)].sum() * ha_per_px:,.0f} ha", flush=True)
