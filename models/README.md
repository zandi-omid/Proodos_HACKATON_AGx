# Trained models go here

The app (`app/app.py`, via `science/predict.py`) uses these three files when **all three** exist;
otherwise it shows a clearly labelled DEMO model.

| file | predicts | unit |
|---|---|---|
| `dT.joblib` | `dT_pred`: temperature, upper minus lower sensor height (> 0 = inversion) | °F |
| `speed_diff.joblib` | `speed_diff_pred`: wind speed, upper minus lower height | mph |
| `dir_diff.joblib` | `dir_diff_pred`: wind direction difference between the heights (0-180) | degrees |

Each is a scikit-learn model saved with `joblib.dump(model, "models/dT.joblib")`, whose `.predict()` takes a
DataFrame with these columns in this order (`predict.MODEL_INPUTS`, built by `science/features.py`):

`temp_2m_F, rh_pct, wind_10m_mph, wind_80m_mph, dir_10m_deg, dir_80m_deg, wind_ratio_80_10, dir_diff_deg, cooling_F_per_hr`

If a model needs different inputs (e.g. hour of day), change `MODEL_INPUTS` / `predict()` in `science/predict.py`.
