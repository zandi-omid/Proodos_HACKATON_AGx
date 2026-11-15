# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""
Spray or Wait? - click a location in Kansas, see HRRR's forecast for it.

    streamlit run app/app.py

Click the map -> the app downloads NOAA's HRRR forecast (3 km, next ~48 h, via
Open-Meteo, cached 1 h) for that spot and plots it, using the same feature
columns as the model training data (science/features.py).
"""
import sys
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "science"))
from fetch_hrrr_forecast import get_forecast  # noqa: E402

# Kansas bounding box (the north-east corner follows the Missouri River, so a few
# points just outside Kansas there are still accepted)
KS = {"lat_min": 36.99, "lat_max": 40.00, "lon_min": -102.05, "lon_max": -94.59}
TZ = "America/Chicago"
st.set_page_config(page_title="Spray or Wait? HRRR forecast", page_icon="🌾", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def forecast_for(lat, lon):
    """HRRR forecast for one point, 15-min steps (cached 1 h, like HRRR's update cycle)."""
    pts = pd.DataFrame([{"name": "selected", "latitude": lat, "longitude": lon}])
    return get_forecast(pts, step="15min")


@st.cache_data
def kansas_stations():
    return pd.read_csv(ROOT / "data" / "ks_stations_both_heights.csv")


st.title("Spray or Wait? — HRRR forecast for Kansas")
st.caption("Click a location in Kansas. The app downloads NOAA's latest HRRR forecast (3 km grid) for "
           "that spot and shows the next ~48 hours. Times are US Central.")

if "loc" not in st.session_state:
    st.session_state.loc = None

# ---------- map ----------
fmap = folium.Map(location=[38.5, -98.3], zoom_start=7, min_zoom=6, tiles="OpenStreetMap",
                  max_bounds=True, min_lat=35.5, max_lat=41.5, min_lon=-104, max_lon=-92.5)
folium.Rectangle([[KS["lat_min"], KS["lon_min"]], [KS["lat_max"], KS["lon_max"]]],
                 color="#1f4e79", weight=2, fill=False).add_to(fmap)
for _, s in kansas_stations().iterrows():
    folium.CircleMarker([s["latitude"], s["longitude"]], radius=3, color="#555", fill=True,
                        tooltip=s["name"]).add_to(fmap)
if st.session_state.loc:
    folium.Marker(st.session_state.loc, icon=folium.Icon(color="red")).add_to(fmap)
click = st_folium(fmap, height=420, use_container_width=True, returned_objects=["last_clicked"])
st.caption("Gray dots: Kansas Mesonet stations used for training.")

if click and click.get("last_clicked"):
    lat, lon = round(click["last_clicked"]["lat"], 3), round(click["last_clicked"]["lng"], 3)
    if (lat, lon) != st.session_state.loc:
        if KS["lat_min"] <= lat <= KS["lat_max"] and KS["lon_min"] <= lon <= KS["lon_max"]:
            st.session_state.loc = (lat, lon)
            st.rerun()
        else:
            st.warning(f"{lat}, {lon} is outside Kansas — please click inside the blue box.")

if not st.session_state.loc:
    st.info("Click a location in Kansas to see its forecast.")
    st.stop()

# ---------- forecast ----------
lat, lon = st.session_state.loc
with st.spinner(f"Downloading HRRR forecast for {lat}, {lon}..."):
    try:
        fc = forecast_for(lat, lon)
    except SystemExit as e:  # the download helper stops with a message on API errors
        st.error(f"Could not get the forecast: {e}")
        st.stop()

fc["local"] = fc["time_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
fut = fc[fc["lead_hours"] >= 0].reset_index(drop=True)
now = fut.iloc[0]

st.subheader(f"Forecast for {lat}, {lon}")
st.caption(f"HRRR forecast downloaded {now['fetched_utc']:%Y-%m-%d %H:%M} UTC · grid cell centre "
           f"{now['hrrr_lat']:.3f}, {now['hrrr_lon']:.3f} · {fut['local'].iloc[0]:%a %H:%M} to "
           f"{fut['local'].iloc[-1]:%a %H:%M} (lead up to {fut['lead_hours'].max():.0f} h). "
           "Accuracy drops further ahead; after ~18 h the 15-min steps are hourly values filled in.")

c = st.columns(5)
c[0].metric("Temp 2 m", f"{now['temp_2m_F']:.0f} °F")
c[1].metric("Humidity 2 m", f"{now['rh_pct']} %")
c[2].metric("Wind 10 m", f"{now['wind_10m_mph']:.0f} mph", help=f"from {now['dir_10m_deg']}°")
c[3].metric("Wind 80 m", f"{now['wind_80m_mph']:.0f} mph", help=f"from {now['dir_80m_deg']}°")
c[4].metric("Wind ratio 80/10", f"{now['wind_ratio_80_10']:.1f}",
            help="> ~1.5 at night: the air near the ground is decoupled from the air above (inversion likely)")

x = alt.X("local:T", title="Time (US Central)")
tip = [alt.Tooltip("local:T", title="time", format="%a %H:%M"), alt.Tooltip("lead_hours", title="lead (h)")]


def chart(cols, labels, title, zero=False, ymax=None):
    """Line chart of one or more columns over time."""
    long = fut.melt(id_vars=["local", "lead_hours"], value_vars=cols, var_name="series", value_name="value")
    long["series"] = long["series"].map(dict(zip(cols, labels)))
    long["value"] = long["value"].astype(float)
    y = alt.Y("value:Q", title=title, scale=alt.Scale(domainMax=ymax) if ymax else alt.Undefined)
    layers = [alt.Chart(long).mark_line().encode(
        x=x, y=y, color=alt.Color("series:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=tip + ["series", alt.Tooltip("value:Q", format=".1f")])]
    if zero:
        layers.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q"))
    return alt.layer(*layers).properties(height=220)


left, right = st.columns(2)
left.altair_chart(chart(["temp_2m_F"], ["Temp 2 m"], "Temperature (°F)"), width="stretch")
right.altair_chart(chart(["wind_10m_mph", "wind_80m_mph"], ["10 m", "80 m"], "Wind speed (mph)"), width="stretch")
left.altair_chart(chart(["wind_ratio_80_10"], ["80 m / 10 m"], "Wind ratio 80/10", ymax=8), width="stretch")
right.altair_chart(chart(["dir_diff_deg"], ["80 m vs 10 m"], "Wind direction difference (°)"), width="stretch")
left.altair_chart(chart(["cooling_F_per_hr"], ["2 m change"], "Temp change (°F per hour)", zero=True),
                  width="stretch")
right.altair_chart(chart(["rh_pct"], ["Humidity 2 m"], "Relative humidity (%)"), width="stretch")

with st.expander("All forecast values (model input features)"):
    st.dataframe(fut, width="stretch", hide_index=True)
    st.download_button("Download CSV", fut.to_csv(index=False).encode(),
                       file_name=f"hrrr_forecast_{lat}_{lon}.csv", mime="text/csv")
