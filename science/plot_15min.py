# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""
Plot one point's 15-minute CSV from fetch_hrrr_15min.py.
Shades the hours that the hourly file (fetch_hrrr.py) flags as inversion
(both GFS and HRRR dT > 0), to check the 15-min signals against it.

    python science/plot_15min.py kern_almonds past

Saves data/hrrr15_<point>_<mode>.png
Requires: pip install pandas matplotlib
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt

point = sys.argv[1] if len(sys.argv) > 1 else "kern_almonds"
mode = sys.argv[2] if len(sys.argv) > 2 else "past"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # <repo>/data
q = pd.read_csv(DATA_DIR / f"hrrr15_{point}_{mode}.csv", parse_dates=["time"])
h = pd.read_csv(DATA_DIR / f"hrrr_{point}_{mode}.csv", parse_dates=["time"])
inv_hours = h.loc[h["inversion"].eq(True) & h["inversion_hrrr"].eq(True), "time"]  # blank = unknown

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
ax1, ax2, ax3, ax4 = axes

ax1.plot(q["time"], q["temp_2m_F"], color="tab:blue")
ax1.set_ylabel("Temp 2 m (F)")

ax2.plot(q["time"], q["wind_10m_mph"], color="tab:green", label="10 m")
ax2.plot(q["time"], q["wind_80m_mph"], color="tab:green", ls="--", label="80 m")
ax2.set_ylabel("Wind (mph)")
ax2.legend(loc="upper left", ncol=2)

ax3.plot(q["time"], q["wind_ratio_80_10"], color="tab:purple")
ax3.axhline(1.5, color="black", lw=0.8, ls=":")
ax3.set_ylabel("Wind 80 m / 10 m")
ax3.set_ylim(0, 10)  # calm 10 m wind can make the ratio huge; cap the view

ax4.plot(q["time"], q["cooling_F_per_hr"], color="tab:blue")
ax4.axhline(0, color="black", lw=0.8)
ax4.set_ylabel("2 m change\n(F per hour)")

for ax in axes:
    for t0 in inv_hours:  # shade each inversion hour
        ax.axvspan(t0, t0 + pd.Timedelta(hours=1), color="orange", alpha=0.2, lw=0)
    ax.grid(alpha=0.3)
fig.suptitle(f"{point} {mode}: 15-min HRRR  (orange = hourly dT says inversion, GFS and HRRR agree)")
fig.autofmt_xdate()
fig.tight_layout()
out = DATA_DIR / f"hrrr15_{point}_{mode}.png"
fig.savefig(out, dpi=110)
print(f"Plot saved to {out}")
