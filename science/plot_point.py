"""
Plot one point's CSV from fetch_hrrr.py as a 3-panel sanity check.

    python science/plot_point.py data/hrrr_kern_almonds_past.csv
    python science/plot_point.py data/hrrr_kern_almonds_forecast.csv

Panels:
  1. Temperature at 2 m (HRRR) and at 80 m (GFS, and HRRR-profile estimate)
  2. dT = T80m - T2m from both 80 m sources; above 0 = inversion (shaded)
  3. Wind at 10 m and 80 m
Saves the PNG next to the CSV (same name, .png).

Requires: pip install pandas matplotlib
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/hrrr_kern_almonds_past.csv"
png_path = csv_path.replace(".csv", ".png")
df = pd.read_csv(csv_path, parse_dates=["time"])
t = df["time"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# ---- 1. Temperatures ----
ax1.plot(t, df["temp_2m_F"], color="tab:blue", label="2 m (HRRR)")
ax1.plot(t, df["temp_80m_F"], color="tab:red", label="80 m (GFS)")
ax1.plot(t, df["temp_80m_hrrr_F"], color="tab:red", ls="--", label="80 m (HRRR profile)")
ax1.set_ylabel("Temperature (F)")
ax1.legend(loc="upper left", ncol=3)

# ---- 2. dT and inversion shading ----
ax2.plot(t, df["dT_80m_minus_2m_F"], color="tab:red", label="dT with GFS 80 m")
ax2.plot(t, df["dT_80m_hrrr_minus_2m_F"], color="tab:red", ls="--", label="dT with HRRR 80 m")
ax2.axhline(0, color="black", lw=0.8)
# shade hours where the all-HRRR estimate says inversion
ax2.fill_between(t, 0, 1, where=df["inversion_hrrr"].eq(True), transform=ax2.get_xaxis_transform(),
                 color="orange", alpha=0.25, step="post", label="Inversion (HRRR)")
ax2.set_ylabel("T80m - T2m (F)\n> 0 = inversion")
ax2.legend(loc="upper left", ncol=3)

# ---- 3. Wind ----
ax3.plot(t, df["wind_10m_mph"], color="tab:green", label="10 m (HRRR)")
ax3.plot(t, df["wind_80m_mph"], color="tab:green", ls="--", label="80 m (HRRR)")
ax3.set_ylabel("Wind (mph)")
ax3.legend(loc="upper left", ncol=2)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
fig.suptitle(f"{csv_path.split('/')[-1]}  (elevation {df['elev_m'].iloc[0]:.0f} m, local time)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(png_path, dpi=110)
print(f"Plot saved to {png_path}")
