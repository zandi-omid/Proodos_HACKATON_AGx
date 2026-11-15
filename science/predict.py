# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""
Predictions + GO / CAUTION / STOP for the Spray-or-Wait app.

Three predictions per time step (from the features in features.FEATURES):
  dT_pred          temperature, upper minus lower sensor height (F); > 0 = inversion
  speed_diff_pred  wind speed, upper minus lower height (mph)
  dir_diff_pred    wind direction difference between the heights (deg, 0-180)

REAL MODELS: put three scikit-learn models saved with joblib in models/:
    models/dT.joblib   models/speed_diff.joblib   models/dir_diff.joblib
Each must take a DataFrame with the columns MODEL_INPUTS (in that order) in .predict().
When all three files exist they are used; otherwise a DEMO stand-in is used.

DEMO model: simple hand-made rules from the HRRR signals (NOT trained, NOT validated).
It only exists so the app can be shown end to end before the real models are ready.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILES = {"dT_pred": "dT.joblib", "speed_diff_pred": "speed_diff.joblib", "dir_diff_pred": "dir_diff.joblib"}
MODEL_INPUTS = FEATURES[1:]  # the 9 features, without time_utc

# GO / CAUTION / STOP rules (edit here). Wind limits follow common label guidance
# (spray in ~3-10 mph wind); the inversion and wind-shear limits are placeholders to tune.
RULES = {
    "dT_stop": 2.0,          # F warmer above than below -> STOP (strong inversion)
    "dT_caution": 0.5,       # F -> CAUTION (weak inversion)
    "wind_low": 3.0,         # mph at 10 m: calmer -> CAUTION (drift can hang / inversion likely)
    "wind_high": 10.0,       # mph at 10 m: windier -> CAUTION
    "wind_stop": 15.0,       # mph at 10 m: windier -> STOP
    "dir_diff_caution": 45,  # deg between heights -> CAUTION (layers moving differently)
    "speed_diff_caution": 5, # mph between heights -> CAUTION
}


def load_models(models_dir=MODELS_DIR):
    """Return {prediction column: model} if all three model files exist, else None (-> demo)."""
    paths = {col: Path(models_dir) / f for col, f in MODEL_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    import joblib
    return {col: joblib.load(p) for col, p in paths.items()}


def demo_predict(X):
    """DEMO stand-in, NOT a trained model: rough rules from HRRR wind shear and cooling."""
    stable = (X["wind_ratio_80_10"] - 1.3).clip(0, 4)          # ground layer decoupling
    calm = ((6 - X["wind_10m_mph"]) / 6).clip(0, 1)            # light wind near the ground
    cooling = (-X["cooling_F_per_hr"].fillna(0)).clip(0, 6)    # air near the ground cooling
    dT = 1.8 * stable + 0.6 * cooling + 1.5 * calm * (stable > 0) - 1.0
    return pd.DataFrame({
        "dT_pred": dT.clip(-3, 15).round(1),
        "speed_diff_pred": (0.35 * (X["wind_80m_mph"] - X["wind_10m_mph"])).round(1),
        "dir_diff_pred": (0.6 * X["dir_diff_deg"].astype(float)).round(),
    }, index=X.index)


def predict(features, models=None):
    """Add dT_pred, speed_diff_pred, dir_diff_pred and model ('trained' or 'DEMO')."""
    df = features.copy()
    X = df[MODEL_INPUTS].astype(float)
    if models:
        ok = X.notna().all(axis=1)  # e.g. the first hour has no cooling value yet
        for col, m in models.items():
            df[col] = np.nan
            df.loc[ok, col] = m.predict(X[ok])
        df["model"] = "trained"
    else:
        df = df.join(demo_predict(X))
        df["model"] = "DEMO"
    return df


def add_status(df, rules=RULES):
    """Add status (GO / CAUTION / STOP) and reasons (plain language) per row."""
    statuses, reasons = [], []
    for _, r in df.iterrows():
        stop, caution = [], []
        if r["dT_pred"] >= rules["dT_stop"]:
            stop.append(f"inversion: air {r['dT_pred']:.1f} F warmer above")
        elif r["dT_pred"] >= rules["dT_caution"]:
            caution.append(f"weak inversion ({r['dT_pred']:+.1f} F)")
        w = r["wind_10m_mph"]
        if w >= rules["wind_stop"]:
            stop.append(f"wind too strong ({w:.0f} mph)")
        elif w > rules["wind_high"]:
            caution.append(f"windy ({w:.0f} mph)")
        elif w < rules["wind_low"]:
            caution.append(f"nearly calm ({w:.1f} mph)")
        if r["dir_diff_pred"] >= rules["dir_diff_caution"]:
            caution.append(f"wind turns {r['dir_diff_pred']:.0f} deg between heights")
        if abs(r["speed_diff_pred"]) >= rules["speed_diff_caution"]:
            caution.append(f"wind speed differs {r['speed_diff_pred']:+.0f} mph between heights")
        statuses.append("STOP" if stop else "CAUTION" if caution else "GO")
        reasons.append("; ".join(stop + caution) if stop or caution else "no inversion, wind in range")
    out = df.copy()
    out["status"], out["reasons"] = statuses, reasons
    return out


def go_windows(df, min_steps=2):
    """Contiguous GO periods in the future: list of (start, end) times (end = last GO step)."""
    fut = df[df["lead_hours"] >= 0].reset_index(drop=True)
    is_go = fut["status"].eq("GO")
    group = (is_go != is_go.shift()).cumsum()
    wins = []
    for _, g in fut[is_go].groupby(group[is_go]):
        if len(g) >= min_steps:
            wins.append((g["time_utc"].iloc[0], g["time_utc"].iloc[-1]))
    return wins
