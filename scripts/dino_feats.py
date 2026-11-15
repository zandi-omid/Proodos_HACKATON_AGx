# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""DINOv3 ViT-H+ (web) features for every clear Sentinel-2 date of every station.

python dino_feats.py            (needs a GPU; reads s2/*.zarr, writes dino/<station>.zarr)
Per station: dates with clear_frac >= MIN_CLEAR and full coverage.
Saved per date:
  cls      (time, 1280)              CLS token
  patches  (time, 20, 20, 1280)      patch grid, 160 m patches, row 0 = north
  clear_frac, days of acquisition (time coord)
attrs: station patch row/col (patch containing the station), model name.
Re-running skips stations already done.
"""
import os, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch
from pyproj import Transformer
from transformers import AutoModel

MODEL = os.environ.get("DINO_MODEL", "facebook/dinov3-vith16plus-pretrain-lvd1689m")
MIN_CLEAR = 0.8
BATCH = 32
STRETCH = 0.3                       # reflectance mapped to full brightness (true-colour look)
MEAN = np.array([0.485, 0.456, 0.406], "float32")   # ImageNet stats (web-pretrained model)
STD = np.array([0.229, 0.224, 0.225], "float32")
SRC = Path(os.environ.get("DINO_SRC", "s2"))          # s2_orchards for the orchard cells
OUT = Path(os.environ.get("DINO_OUT", "dino")); OUT.mkdir(exist_ok=True)


def to_input(ds, idx):
    rgb = np.stack([ds[b].isel(time=idx).values for b in ("red", "green", "blue")], -1).astype("float32")
    rgb = np.clip((rgb - 1000) / 10000 / STRETCH, 0, 1)          # baseline >= 04.00 offset
    x = (rgb - MEAN) / STD                                         # (n, H, W, 3)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def station_patch(ds, p):
    tr = Transformer.from_crs(4326, ds.attrs["crs"], always_xy=True)
    sx, sy = tr.transform(ds.attrs["station_lon"], ds.attrs["station_lat"])
    x0 = float(ds.x.min()) - 5; y0 = float(ds.y.max()) + 5       # tile edges (10 m pixel centres)
    return int((y0 - sy) // (16 * 10)), int((sx - x0) // (16 * 10))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    nreg = model.config.num_register_tokens
    files = sorted(SRC.glob("*.zarr"))
    files = [f for f in files if not f.name.endswith(".tmp.zarr")]
    shard, nshard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    files = files[shard::nshard]
    print(f"{MODEL} on {dev}, {len(files)} stations, {nreg} registers", flush=True)
    t0 = time.time()
    for k, f in enumerate(files):
        out = OUT / f.name
        if out.exists():
            print(f"[{k+1}/{len(files)}] {f.stem}: skipped", flush=True); continue
        ds = xr.open_zarr(f)
        keep = np.where((ds.clear_frac.values >= MIN_CLEAR) & (ds.valid_frac.values >= 0.999))[0]
        H, W = ds.sizes["y"], ds.sizes["x"]
        ph, pw = H // 16, W // 16
        cls = np.zeros((len(keep), model.config.hidden_size), "float16")
        pat = np.zeros((len(keep), ph, pw, model.config.hidden_size), "float16")
        for i in range(0, len(keep), BATCH):
            idx = keep[i:i + BATCH]
            x = to_input(ds, idx).to(dev, model.dtype)
            with torch.inference_mode():
                h = model(pixel_values=x).last_hidden_state              # (b, 1+nreg+ph*pw, D)
            cls[i:i + len(idx)] = h[:, 0].float().cpu().numpy()
            pat[i:i + len(idx)] = h[:, 1 + nreg:].float().cpu().numpy().reshape(len(idx), ph, pw, -1)
        r, c = station_patch(ds, (ph, pw))
        o = xr.Dataset(
            {"cls": (("time", "feat"), cls),
             "patches": (("time", "prow", "pcol", "feat"), pat),
             "clear_frac": (("time",), ds.clear_frac.values[keep])},
            coords={"time": ds.time.values[keep]},
            attrs=dict(ds.attrs, model=MODEL, station_prow=r, station_pcol=c, patch_m=160,
                       stretch=STRETCH, min_clear=MIN_CLEAR))
        tmp = out.with_suffix(".tmp.zarr"); shutil.rmtree(tmp, ignore_errors=True)
        o.chunk({"time": 16}).to_zarr(tmp, mode="w"); tmp.rename(out)
        print(f"[{k+1}/{len(files)}] {f.stem}: {len(keep)} images, station patch ({r},{c})"
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
