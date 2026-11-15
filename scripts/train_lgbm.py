# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""Train LightGBM for inversion + wind at 2 m and 10 m, with spatial cross-validation.

python train_lgbm.py
Reads  lgbm_data.parquet, lgbm_meta.json
Writes models/<target>.txt         final models (trained on all stations)
       oof.parquet                 held-out predictions for every row (for scoring / demo)
       metrics.csv                 per target: MAE, R2, and HRRR-only baseline where one exists
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROUNDS = 200
SUB = 2                                            # train on every 2nd row (speed); score on all
PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              num_threads=0, verbose=-1)
DROP = {"h_ratio", "h_ddir", "h_cool"}              # derived / not reproducible: never used
BASE = {"u10": "h_u10", "v10": "h_v10"}            # HRRR's own 10 m wind as the baseline


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    m = json.load(open("lgbm_meta.json"))
    feats = [f for f in m["features"] if f not in DROP]
    targets = m["targets"]
    d = pd.read_parquet("lgbm_data.parquet")
    print(f"{len(d):,} rows, {d.station.nunique()} stations, {len(feats)} predictors", flush=True)
    Path("models").mkdir(exist_ok=True)
    json.dump({"features": feats, "targets": targets}, open("models/features.json", "w"), indent=1)
    oof = d[["time", "station", "fold"] + targets + m["eval"]].copy()
    rows = []
    for tgt in targets:
        t0 = time.time()
        p = np.full(len(d), np.nan, "float32")
        for k in sorted(d.fold.unique()):
            tr, te = (d.fold != k) & (np.arange(len(d)) % SUB == 0), d.fold == k
            b = lgb.train(PARAMS, lgb.Dataset(d.loc[tr, feats], d.loc[tr, tgt]), ROUNDS)
            p[te.values] = b.predict(d.loc[te, feats])
        oof[f"p_{tgt}"] = p
        y = d[tgt].values
        r = {"target": tgt, "MAE": np.mean(np.abs(y - p)), "R2": r2(y, p)}
        if tgt in BASE:
            r["MAE_hrrr"] = np.mean(np.abs(y - d[BASE[tgt]].values)); r["R2_hrrr"] = r2(y, d[BASE[tgt]].values)
        rows.append(r)
        sub = d.iloc[::SUB]
        final = lgb.train(PARAMS, lgb.Dataset(sub[feats], sub[tgt]), ROUNDS)
        final.save_model(f"models/{tgt}.txt")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    # derived quantities from predicted wind components
    for h in ("2", "10"):
        oof[f"p_ws{h}"] = np.hypot(oof[f"p_u{h}"], oof[f"p_v{h}"])
        oof[f"p_wd{h}"] = np.degrees(np.arctan2(-oof[f"p_u{h}"], -oof[f"p_v{h}"])) % 360
    oof["p_ddir"] = np.abs((oof.p_wd10 - oof.p_wd2 + 180) % 360 - 180)
    yi, pi = oof.y_dT > 0.5, oof.p_y_dT > 0.5                 # "real" inversion: > 0.5 C over 8 m
    rows.append({"target": "inversion>0.5C", "hit_rate": (pi & yi).sum() / yi.sum(),
                 "false_alarm": (pi & ~yi).sum() / max(pi.sum(), 1), "share_obs": yi.mean()})
    for h in ("2", "10"):
        rows.append({"target": f"ws{h}", "MAE": np.mean(np.abs(oof[f"y_ws{h}"] - oof[f"p_ws{h}"]))})
    met = pd.DataFrame(rows)
    met.to_csv("metrics.csv", index=False)
    oof.to_parquet("oof.parquet", index=False)
    print("\n" + met.round(3).to_string(index=False))
    imp = pd.Series(lgb.Booster(model_file="models/y_dT.txt").feature_importance("gain"), feats)
    print("\ntop predictors for inversion:\n" + (imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
