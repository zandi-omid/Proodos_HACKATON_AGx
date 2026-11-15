"""
Summarize temperature inversions in a CSV made by fetch_hrrr.py and plot them.

    python science/summarize_inversions.py                      # Fresno, past week
    python science/summarize_inversions.py data/hrrr_modesto_past.csv data/modesto_inversion.png

Prints: inversion hours, typical start/end hour of each inversion episode,
and the strongest dT (T80m - T2m). Saves a plot of T2m and T80m with
inversion hours shaded.

Requires: pip install pandas matplotlib
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/hrrr_fresno_past.csv"
png_path = sys.argv[2] if len(sys.argv) > 2 else "data/fresno_inversion.png"

df = pd.read_csv(csv_path, parse_dates=["time"])

# ---- 1. Group consecutive inversion hours into "episodes" ----
# A new episode starts every time the inversion flag flips.
df["episode"] = (df["inversion"] != df["inversion"].shift()).cumsum()
episodes = (
    df[df["inversion"]]
    .groupby("episode")
    .agg(start=("time", "first"), end=("time", "last"),
         hours=("time", "size"), max_dT=("dT_80m_minus_2m_F", "max"))
    .reset_index(drop=True)
)

# ---- 2. Print the summary ----
print(f"File: {csv_path}")
print(f"Period: {df['time'].min()} -> {df['time'].max()}  ({len(df)} hours)")
print(f"Inversion hours: {int(df['inversion'].sum())} of {len(df)} "
      f"({df['inversion'].mean():.0%})")
print(f"Episodes: {len(episodes)}")
print(f"Typical start hour (median): {episodes['start'].dt.hour.median():.0f}:00")
print(f"Typical last inversion hour (median): {episodes['end'].dt.hour.median():.0f}:00")
peak = df.loc[df["dT_80m_minus_2m_F"].idxmax()]
print(f"Strongest inversion: dT = +{peak['dT_80m_minus_2m_F']:.1f} F at {peak['time']}")
print("\nEpisodes:")
print(episodes.to_string(index=False))

# ---- 3. Plot T2m and T80m, shade inversion hours ----
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(df["time"], df["temp_2m_F"], label="T at 2 m (HRRR)", color="tab:blue")
ax.plot(df["time"], df["temp_80m_F"], label="T at 80 m (GFS)", color="tab:red")
for _, ep in episodes.iterrows():
    # shade from the start of the first hour to the end of the last hour
    ax.axvspan(ep["start"], ep["end"] + pd.Timedelta(hours=1),
               color="orange", alpha=0.25, lw=0)
ax.set_ylabel("Temperature (F)")
ax.set_title(f"Temperature at two heights - {csv_path.split('/')[-1]}")
# add a legend entry for the shaded band
handles, _ = ax.get_legend_handles_labels()
handles.append(Patch(color="orange", alpha=0.25, label="Inversion (T80m > T2m)"))
ax.legend(handles=handles, loc="upper left")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(png_path, dpi=120)
print(f"\nPlot saved to {png_path}")
