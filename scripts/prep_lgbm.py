"""Assemble predictors + targets for LightGBM from table/ and dinofeat/.

python prep_lgbm.py
Writes
  lgbm_data.parquet   one row per station x 15 min: time, station, fold, targets, predictors
  lgbm_meta.json      lists of predictor and target columns
  dino_pca.npz        PCA (mean + components) per DINO group, to apply to orchard cells later
Uses whatever stations are in table/ right now; re-run when more arrive.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

NPC = 16
NFOLD = 5
BLOCK_DEG = 0.5                                   # ~50 km spatial blocks for leave-area-out folds

HRRR = ["h_t2", "h_rh", "h_ws10", "h_ws80", "h_u10", "h_v10", "h_u80", "h_v80",
        "h_dt1h", "h_ws10_1h"]      # h_cool, h_ratio, h_ddir left out (derived / not reproducible)
SUN = ["sun_el", "sun_az", "sun_el_1h", "doy_s", "doy_c"]
TARGETS = ["y_dT", "u2", "v2", "u10", "v10"]
KEEP_EVAL = ["y_ws2", "y_ws10", "y_wd2", "y_wd10", "y_ddir"]       # for scoring only, not predictors


def fit_pca(X):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu.astype("float32"), Vt[:NPC].astype("float32")


def main():
    tabs = sorted(Path("table").glob("*.parquet"))
    st = pd.read_csv("ks_stations_both_heights.csv").set_index("name")
    print(f"{len(tabs)} station tables", flush=True)

    # ---- DINO: one PCA per group, fitted on all station images ----
    dfeat = {f.stem: pd.read_parquet(f) for f in sorted(Path("dinofeat").glob("*.parquet"))}
    pca, groups = {}, ("cls", "mean", "pt")
    if dfeat:
        allimg = pd.concat(dfeat.values(), ignore_index=True)
        for g in groups:
            cols = [c for c in allimg.columns if c.startswith(g) and c[len(g):].isdigit()]
            pca[g] = (cols, *fit_pca(allimg[cols].to_numpy("float32")))
        np.savez("dino_pca.npz", **{f"{g}_{k}": v for g, (cols, mu, V) in pca.items()
                                   for k, v in (("mu", mu), ("V", V))})
        print(f"DINO PCA fitted on {len(allimg)} images", flush=True)

    def dino_reduced(stem):
        d = dfeat[stem]
        o = pd.DataFrame({"img_time": pd.to_datetime(d.img_time).astype("datetime64[ns]")})
        for g, (cols, mu, V) in pca.items():
            Z = (d[cols].to_numpy("float32") - mu) @ V.T
            for k in range(NPC): o[f"d_{g}{k}"] = Z[:, k]
        return o.sort_values("img_time")

    # ---- spatial folds: 0.5-degree blocks shuffled into NFOLD folds ----
    blk = (np.floor(st.latitude / BLOCK_DEG).astype(int).astype(str) + "_" +
           np.floor(st.longitude / BLOCK_DEG).astype(int).astype(str))
    ub = np.random.default_rng(0).permutation(blk.unique())
    fold_of_block = {b: i % NFOLD for i, b in enumerate(ub)}

    parts = []
    for f in tabs:
        t = pd.read_parquet(f)
        n = t.station.iloc[0]
        for h in ("2", "10"):
            r = np.radians(t[f"y_wd{h}"]); s = t[f"y_ws{h}"]
            t[f"u{h}"] = (-s * np.sin(r)).astype("float32"); t[f"v{h}"] = (-s * np.cos(r)).astype("float32")
        t["time"] = pd.to_datetime(t.time).astype("datetime64[ns]")
        t = t.sort_values("time")
        if f.stem in dfeat:
            t = pd.merge_asof(t, dino_reduced(f.stem), left_on="time", right_on="img_time", direction="backward")
            t["img_age_d"] = ((t.time - t.img_time).dt.total_seconds() / 86400).astype("float32")
            t = t.drop(columns="img_time")
        t["fold"] = fold_of_block[blk[n]]
        parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    dino_cols = [c for c in d.columns if c.startswith("d_")] + (["img_age_d"] if "img_age_d" in d else [])
    feats = HRRR + SUN + dino_cols
    d = d[["time", "station", "fold"] + TARGETS + KEEP_EVAL + feats]
    d = d.dropna(subset=TARGETS)
    d[feats + TARGETS] = d[feats + TARGETS].astype("float32")
    d.to_parquet("lgbm_data.parquet", index=False)
    json.dump({"features": feats, "targets": TARGETS, "eval": KEEP_EVAL, "npc": NPC}, open("lgbm_meta.json", "w"), indent=1)
    print(f"{len(d):,} rows, {d.station.nunique()} stations, {len(feats)} predictors "
          f"({len(dino_cols)} DINO), targets {TARGETS}")
    print("rows per fold:", d.fold.value_counts().sort_index().to_dict())
    print("stations per fold:", d.groupby("fold").station.nunique().to_dict())
    print(f"inversion share (y_dT>0): {(d.y_dT > 0).mean():.1%}; rows without a DINO image: "
          f"{d[dino_cols[0]].isna().mean():.1%}" if dino_cols else "")


if __name__ == "__main__":
    main()
