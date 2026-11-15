"""
Spray or Wait? - click a location, get the next ~48 h forecast and spray guidance.

    streamlit run app/app.py

Flow: click map -> download HRRR forecast for that point (Open-Meteo, cached 1 h)
      -> features (science/features.py) -> 3 predictions (science/predict.py)
      -> GO / CAUTION / STOP per time step, next GO windows, charts.
Until the trained models are put in models/, a clearly labelled DEMO model is used.
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
from predict import add_status, go_windows, load_models, predict, RULES  # noqa: E402

STATUS_COLORS = {"GO": "#2e9d4f", "CAUTION": "#e8a33d", "STOP": "#d64545"}
st.set_page_config(page_title="Spray or Wait?", page_icon="🌾", layout="wide")


# ---------- cached helpers ----------
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_for(lat, lon, step):
    """Live HRRR forecast for one point (cached 1 h, like HRRR's update cycle)."""
    pts = pd.DataFrame([{"name": "selected", "latitude": lat, "longitude": lon}])
    return get_forecast(pts, step=step)


@st.cache_resource
def models():
    return load_models()


@st.cache_data
def kansas_stations():
    return pd.read_csv(ROOT / "data" / "ks_stations_both_heights.csv")


# ---------- sidebar ----------
st.sidebar.title("Settings")
tz = st.sidebar.selectbox("Show times in", ["America/Chicago", "America/Denver", "America/Los_Angeles",
                                            "America/New_York", "America/Phoenix", "UTC"])
step = st.sidebar.radio("Time step", ["15min", "hourly"],
                        help="15-min matches the training data. HRRR 15-min values are real only for the "
                             "first ~18 h; after that they are hourly values filled in.")
m = models()
if m:
    st.sidebar.success("Using the trained models from models/")
else:
    st.sidebar.warning("DEMO model: no trained models found in models/. Predictions are rough "
                       "hand-made rules, not real model output.")
with st.sidebar.expander("GO / CAUTION / STOP rules"):
    st.write(RULES)

# ---------- header + map ----------
st.title("Spray or Wait?")
st.caption("Click a location on the map. The app downloads the latest HRRR forecast (3 km, NOAA) for that "
           "spot and predicts the temperature inversion and wind differences between two heights for "
           "the next ~48 hours.")

if "loc" not in st.session_state:
    st.session_state.loc = None

col_map, col_now = st.columns([3, 2])
with col_map:
    fmap = folium.Map(location=[38.5, -98.3], zoom_start=6, tiles="OpenStreetMap")
    for _, s in kansas_stations().iterrows():
        folium.CircleMarker([s["latitude"], s["longitude"]], radius=3, color="#555", fill=True,
                            tooltip=s["name"]).add_to(fmap)
    if st.session_state.loc:
        folium.Marker(st.session_state.loc, icon=folium.Icon(color="red")).add_to(fmap)
    click = st_folium(fmap, height=430, use_container_width=True, returned_objects=["last_clicked"])
    if click and click.get("last_clicked"):
        new = (round(click["last_clicked"]["lat"], 3), round(click["last_clicked"]["lng"], 3))
        if new != st.session_state.loc:
            st.session_state.loc = new
            st.rerun()
    st.caption("Gray dots: Kansas Mesonet stations used for training.")

if not st.session_state.loc:
    with col_now:
        st.info("Click anywhere in the lower 48 US states to get a forecast.")
    st.stop()

lat, lon = st.session_state.loc
with st.spinner(f"Downloading HRRR forecast for {lat}, {lon}..."):
    try:
        fc = forecast_for(lat, lon, step)
    except SystemExit as e:  # get_json stops with a message on API errors
        st.error(f"Could not get the forecast: {e}")
        st.stop()
if fc.empty or fc["temp_2m_F"].isna().all():
    st.error("No HRRR data for this location. HRRR covers the lower 48 US states only.")
    st.stop()

df = add_status(predict(fc, m))
df["local"] = df["time_utc"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
fut = df[df["lead_hours"] >= 0].reset_index(drop=True)
now = fut.iloc[0]

# ---------- right now ----------
with col_now:
    st.subheader(f"{lat}, {lon}")
    st.caption(f"Forecast downloaded {now['fetched_utc']:%Y-%m-%d %H:%M} UTC  ·  "
               f"HRRR cell {now['hrrr_lat']:.3f}, {now['hrrr_lon']:.3f}  ·  model: {now['model']}")
    color = STATUS_COLORS[now["status"]]
    st.markdown(f"<div style='background:{color};color:white;padding:14px;border-radius:8px;"
                f"font-size:1.6em;text-align:center'><b>{now['status']}</b> &nbsp; "
                f"<span style='font-size:0.6em'>{now['local']:%a %H:%M}</span></div>",
                unsafe_allow_html=True)
    st.write(f"**Why:** {now['reasons']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Inversion dT", f"{now['dT_pred']:+.1f} °F", help="upper minus lower height; > 0 = inversion")
    c2.metric("Wind speed diff", f"{now['speed_diff_pred']:+.1f} mph")
    c3.metric("Wind dir diff", f"{now['dir_diff_pred']:.0f}°")
    st.caption(f"Wind at 10 m: {now['wind_10m_mph']:.0f} mph · Temp at 2 m: {now['temp_2m_F']:.0f} °F")

    st.markdown("**Next GO windows**")
    wins = go_windows(df)
    if wins:
        for a, b in wins[:4]:
            a_l = pd.Timestamp(a).tz_localize("UTC").tz_convert(tz)
            b_l = pd.Timestamp(b).tz_localize("UTC").tz_convert(tz)
            lead = (pd.Timestamp(a) - now["fetched_utc"]).total_seconds() / 3600
            st.write(f"🟢 {a_l:%a %H:%M} – {b_l:%H:%M}  (in {max(lead, 0):.0f} h)")
    else:
        st.write("No GO window in the forecast period.")

# ---------- charts ----------
st.subheader("Hour by hour")
st.caption("Background colour = GO / CAUTION / STOP. Lead time = hours after the forecast was downloaded. "
           "**Accuracy drops further ahead**, and after ~18 h the 15-min steps are hourly values filled in "
           "(dashed line).")
step_len = pd.Timedelta(minutes=15 if step == "15min" else 60)
band = fut.assign(end=fut["local"] + step_len)
x = alt.X("local:T", title=f"Time ({tz})")
tooltip = [alt.Tooltip("local:T", title="time", format="%a %H:%M"), "lead_hours", "status", "reasons",
           "dT_pred", "speed_diff_pred", "dir_diff_pred", "wind_10m_mph"]
background = alt.Chart(band).mark_rect(opacity=0.25).encode(
    x=x, x2="end:T",
    color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=tooltip)
cut = fut.loc[fut["lead_hours"] > 18, "local"]
rule = (alt.Chart(pd.DataFrame({"t": [cut.iloc[0]]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="t:T")
        if step == "15min" and len(cut) else None)


def panel(col, title, zero=False):
    line = alt.Chart(fut).mark_line(color="#1f4e79").encode(x=x, y=alt.Y(f"{col}:Q", title=title), tooltip=tooltip)
    layers = [background, line]
    if zero:
        layers.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q"))
    if rule is not None:
        layers.append(rule)
    return alt.layer(*layers).properties(height=200)


st.altair_chart(panel("dT_pred", "Inversion dT (°F)", zero=True), width="stretch")
c1, c2 = st.columns(2)
c1.altair_chart(panel("speed_diff_pred", "Wind speed diff (mph)", zero=True), width="stretch")
c2.altair_chart(panel("dir_diff_pred", "Wind direction diff (°)"), width="stretch")

with st.expander("All forecast rows"):
    show = fut[["local", "lead_hours", "status", "reasons", "dT_pred", "speed_diff_pred", "dir_diff_pred",
                "temp_2m_F", "rh_pct", "wind_10m_mph", "wind_80m_mph", "real_15min"]]
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("Download CSV", fut.to_csv(index=False).encode(),
                       file_name=f"spray_forecast_{lat}_{lon}.csv", mime="text/csv")
