# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""Build the interactive app: spray_or_wait.html (one self-contained file; open in a browser).

python make_app.py [orchard_forecast_YYYYMMDDHH.parquet]      (default: newest)
Reads  orchard_forecast_*.parquet (from predict_orchards.py), oof.parquet, metrics.csv
Writes spray_or_wait.html

Confidence: for every orchard cell and time, the model's held-out errors (oof.parquet) are added
100 times to the predicted inversion and winds, the Go/Caution/Stop rules are re-applied, and the
share of each outcome is the probability. The app shows the most likely outcome and its probability.
GO is only shown when GO is at least 60% likely; otherwise CAUTION.
"""
import sys, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

# must match predict_orchards.py
CALM, WINDY = 3.0, 10.0
INV, STRONG_INV = 0.3, 1.0
SPEED_DIFF, DIR_DIFF = 5.0, 45.0
MS2MPH, K, GO_MIN = 2.23694, 100, 0.6
KS = [[-101.906, 40.002], [-95.306, 40.002], [-95.208, 39.909], [-94.885, 39.832], [-95.109, 39.542],
      [-94.983, 39.443], [-94.824, 39.207], [-94.611, 39.158], [-94.616, 37.0], [-100.088, 37.0],
      [-102.043, 36.995], [-102.054, 40.002], [-101.906, 40.002]]
CROPS = {66: "Cherries", 67: "Peaches", 68: "Apples", 71: "Other tree crops", 72: "Citrus", 74: "Pecans",
         75: "Almonds", 76: "Walnuts", 77: "Pears", 204: "Pistachios", 210: "Prunes", 211: "Olives",
         212: "Oranges", 215: "Avocados", 217: "Pomegranates", 218: "Nectarines", 220: "Plums", 223: "Apricots"}


def status_fast(dT, ws2, ws10, ddir):
    """0 GO, 1 CAUTION, 2 STOP. dT in C, wind in mph. Same logic as predict_orchards.rules."""
    inv = dT > INV
    stop = (inv & (ws2 < CALM)) | (dT > STRONG_INV) | (np.maximum(ws2, ws10) > WINDY)
    caut = (inv & (ws2 >= CALM)) | (~inv & (ws2 < CALM)) | (np.abs(ws10 - ws2) > SPEED_DIFF) | \
           ((ddir > DIR_DIFF) & (ws2 >= 2) & (ws10 >= 2))
    return np.where(stop, 2, np.where(caut, 1, 0)).astype("int8")


def ddir_of(wd10, wd2):
    return np.abs((wd10 - wd2 + 180) % 360 - 180)


def probabilities(fc, oof):
    res = pd.DataFrame({"dT": oof.y_dT - oof.p_y_dT, "stable": oof.p_y_dT > 0,
                        **{c: oof[c] - oof[f"p_{c}"] for c in ("u2", "v2", "u10", "v10")}}).dropna()
    p_dT = (fc.inv_F.values / 1.8).astype("float32")
    comp = {}
    for h in ("2", "10"):
        s = fc[f"ws{h}_mph"].values / MS2MPH; r = np.radians(fc[f"wd{h}"].values)
        comp[f"u{h}"], comp[f"v{h}"] = (-s * np.sin(r)).astype("float32"), (-s * np.cos(r)).astype("float32")
    P = np.zeros((len(fc), 3), "float32")
    for stable in (False, True):
        m = (p_dT > 0) == stable
        if not m.any(): continue
        pool = res[res.stable == stable]
        if len(pool) < 50: pool = res
        e = pool.sample(K, random_state=0, replace=len(pool) < K).astype("float32")
        dT = p_dT[m, None] + e.dT.values[None]
        w = {k: comp[k][m, None] + e[k].values[None] for k in comp}
        ws2, ws10 = np.hypot(w["u2"], w["v2"]) * MS2MPH, np.hypot(w["u10"], w["v10"]) * MS2MPH
        wd2 = np.degrees(np.arctan2(-w["u2"], -w["v2"])) % 360
        wd10 = np.degrees(np.arctan2(-w["u10"], -w["v10"])) % 360
        st = status_fast(dT, ws2, ws10, ddir_of(wd10, wd2))
        for k in range(3): P[m, k] = (st == k).mean(1)
    return P


def extra_metrics(oof):
    o = oof.dropna(subset=["y_dT", "p_y_dT", "y_ws2", "y_ws10", "p_ws2", "p_ws10"])
    obs = status_fast(o.y_dT.values, o.y_ws2.values * MS2MPH, o.y_ws10.values * MS2MPH, o.y_ddir.values)
    prd = status_fast(o.p_y_dT.values, o.p_ws2.values * MS2MPH, o.p_ws10.values * MS2MPH, o.p_ddir.values)
    w2 = o.y_ws2.values * MS2MPH
    single_ok = (w2 >= CALM) & (w2 <= WINDY)
    risky = obs > 0
    pct = lambda x: f"{100 * x:.0f}%"
    return [
        ["Stations the model was tested on (never seen in training)", str(o.station.nunique())],
        ["Go / Caution / Stop matches what the two-height sensors showed", pct((obs == prd).mean())],
        ["Risky times (Caution or Stop) the model also flagged", pct((prd[risky] > 0).mean())],
        ["Stop times the model flagged as Caution or Stop", pct((prd[obs == 2] > 0).mean()) if (obs == 2).any() else "n/a"],
        ["Times a single 5-ft station looked fine (wind 3-10 mph) but two heights showed a risk",
         pct((risky & single_ok).sum() / max(single_ok.sum(), 1))],
        ["... and of those hidden risks, share the model caught", pct((prd[risky & single_ok] > 0).mean()) if (risky & single_ok).any() else "n/a"],
    ]


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("orchard_forecast_*.parquet"))[-1]
    cycle = Path(f).stem.split("_")[-1]
    fc = pd.read_parquet(f).sort_values(["cell", "time"]).reset_index(drop=True)
    oof = pd.read_parquet("oof.parquet")
    P = probabilities(fc, oof)
    a = P.argmax(1); conf = P.max(1)
    low = (a == 0) & (conf < GO_MIN)                     # not sure enough to say GO -> CAUTION
    a[low] = 1; conf[low] = 1 - P[low, 0]
    fc["s"], fc["c"] = a, np.rint(100 * conf).astype(int)
    times = np.sort(fc.time.unique())
    loc = pd.DatetimeIndex(times).tz_localize("UTC").tz_convert("America/Chicago")
    cells = []
    for cid, g in fc.groupby("cell", sort=False):
        g = g.set_index("time").reindex(times)
        if g.s.isna().any(): continue
        r = g.iloc[0]
        iv = lambda x, k=1: [int(v) for v in np.rint(np.nan_to_num(x.values) * k)]
        cells.append({"id": cid, "lat": round(float(r.lat), 4), "lon": round(float(r.lon), 4),
                      "ha": round(float(r.orchard_ha), 2), "crop": CROPS.get(int(r.main_crop), "Orchard"),
                      "s": "".join(str(int(v)) for v in g.s), "c": iv(g.c),
                      "inv": iv(g.inv_F, 10), "w2": iv(g.ws2_mph, 10), "w10": iv(g.ws10_mph, 10),
                      "d2": iv(g.wd2), "d10": iv(g.wd10)})
    met = pd.read_csv("metrics.csv") if Path("metrics.csv").exists() else pd.DataFrame()
    data = {"cycle": f"{cycle[:4]}-{cycle[4:6]}-{cycle[6:8]} {cycle[8:]}:00 UTC",
            "times": [t.strftime("%a %-I:%M %p") for t in loc],
            "iso": [t.strftime("%Y-%m-%d %H:%M") for t in loc],
            "thr": {"CALM": CALM, "WINDY": WINDY, "INV_F": round(INV * 1.8, 2), "STRONG_F": round(STRONG_INV * 1.8, 2),
                    "SPEED_DIFF": SPEED_DIFF, "DIR_DIFF": DIR_DIFF, "GO_MIN": int(GO_MIN * 100)},
            "cells": cells, "ks": [[y, x] for x, y in KS],
            "metrics": json.loads(met.round(3).to_json(orient="records")),
            "extra": extra_metrics(oof)}
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path("spray_or_wait.html").write_text(html, encoding="utf-8")
    Path("index.html").write_text(html, encoding="utf-8")          # same file, named for GitHub Pages
    sh = pd.Series(a).map({0: "GO", 1: "CAUTION", 2: "STOP"}).value_counts(normalize=True).round(3).to_dict()
    print(f"spray_or_wait.html: {len(cells)} orchard cells x {len(times)} times, {len(html) / 1e6:.1f} MB")
    print("status share:", sh, " mean confidence:", round(float(conf.mean()), 2))
    for k, v in data["extra"]: print(f"  {k}: {v}")


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Spray or Wait? Kansas orchards</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1c2430;background:#f4f5f2}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:10px 16px;background:#16301f;color:#fff}
header h1{font-size:19px;margin:0;font-weight:650}
header .sub{font-size:12.5px;opacity:.8}
.grow{flex:1}
.logo{display:flex;align-items:center;gap:8px;font-weight:800;letter-spacing:2.5px;font-size:15px;padding-right:14px;border-right:1px solid #ffffff40}
details summary{cursor:pointer;font-weight:600;color:#33404d;margin:12px 0 6px}
button{font:inherit;border:1px solid #c9cec6;background:#fff;color:#1c2430;border-radius:7px;padding:6px 11px;cursor:pointer}
button:hover{background:#eef1ea}
header button{border-color:#ffffff55;background:#ffffff14;color:#fff}
header button:hover{background:#ffffff2b}
#bar{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;padding:8px 16px;background:#fff;border-bottom:1px solid #dfe3da}
#bar input[type=range]{flex:1;min-width:220px;accent-color:#16301f}
#tlabel{font-weight:650;min-width:120px}
.cnt{font-size:13px;white-space:nowrap}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:4px;vertical-align:-1px}
#wrap{display:flex;height:calc(100% - 104px);min-height:420px}
#map{flex:1;background:#e9ece6}
#panel{width:440px;max-width:48vw;overflow:auto;background:#fff;border-left:1px solid #dfe3da;padding:14px 16px}
#panel h2{font-size:17px;margin:0 0 2px}
.muted{color:#5d6773;font-size:13px}
.badge{display:flex;align-items:center;gap:12px;margin:12px 0 8px;padding:12px 14px;border-radius:10px;color:#fff}
.badge .ic{font-size:26px;font-weight:700;width:38px;height:38px;border-radius:50%;background:#ffffff33;display:flex;align-items:center;justify-content:center}
.badge .wd{font-size:25px;font-weight:750;letter-spacing:.5px;line-height:1}
.badge .cf{font-size:13.5px;opacity:.95;margin-top:3px}
ul.why{margin:6px 0 10px;padding-left:19px}
ul.why li{margin:3px 0}
.cmp{border:1px solid #dfe3da;border-radius:9px;padding:9px 11px;margin:8px 0;font-size:14px;background:#fafbf8}
.cmp b.bad{color:#b3261e}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid #eceee8;vertical-align:top}
th{font-weight:600;color:#4a535e}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
h3{font-size:14px;margin:14px 0 5px;color:#33404d}
#strip{display:flex;height:22px;border-radius:5px;overflow:hidden;cursor:pointer;border:1px solid #dfe3da}
#strip div{flex:1}
#strip div.sel{outline:2px solid #111;outline-offset:-2px}
.axis{display:flex;justify-content:space-between;font-size:11.5px;color:#5d6773;margin-top:2px}
#logbox{max-height:230px;overflow:auto;border:1px solid #e4e7df;border-radius:7px}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;color:#fff;font-size:12px;font-weight:650}
#modal{position:fixed;inset:0;background:#0008;display:none;align-items:center;justify-content:center;z-index:2000}
#modal .box{background:#fff;border-radius:12px;max-width:760px;width:92vw;max-height:86vh;overflow:auto;padding:18px 22px}
.note{font-size:12.5px;color:#5d6773;margin-top:10px}
.empty{color:#5d6773;margin-top:30px;text-align:center}
</style></head><body>
<header>
  <div class="logo" title="Proodos"><svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="#fff"/><path d="M16 26V14M16 16c0-5 3-8 8-8 0 5-3 8-8 8zM16 20c0-4-2.5-6.5-7-6.5 0 4 2.5 6.5 7 6.5z" fill="#2b8a4e" stroke="#16301f" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg><span>PROODOS</span></div>
  <div><h1>Spray or Wait?</h1><div class="sub">Advancing precision agriculture across large scales</div></div>
  <div class="grow"></div>
  <button id="bMiss">Risk a ground station misses</button>
  <button id="bMet">Model accuracy</button>
</header>
<div id="bar">
  <span id="tlabel"></span>
  <input type="range" id="slider" min="0" value="0" step="1" aria-label="Forecast time">
  <span class="cnt" id="counts"></span>
  <button id="bHome">Whole state</button>
  <span class="muted" id="run"></span>
</div>
<div id="wrap"><div id="map"></div>
  <div id="panel"><div class="empty">Click an orchard on the map.</div></div>
</div>
<div id="modal"><div class="box" id="mbox"></div></div>
<script>
const D = __DATA__;
const T = D.thr, N = D.times.length;
const COL = ["#2b8a4e", "#c98a00", "#c9342c"], NAME = ["GO", "CAUTION", "STOP"], ICON = ["✔", "!", "✖"];
const ADV = ["OK to spray under common guidance", "Wait if you can, or take extra care", "Do not spray now"];
let ti = 0, sel = null, rect = null;
const $ = id => document.getElementById(id);
const S = (c, i) => +c.s[i];
const comp = d => ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"][Math.round(d / 22.5) % 16];
const dd = (c, i) => Math.abs(((c.d10[i] - c.d2[i] + 540) % 360) - 180);

function triggers(c, i) {
  const inv = c.inv[i] / 10, w2 = c.w2[i] / 10, w10 = c.w10[i] / 10, d = dd(c, i), R = [];
  const isInv = inv > T.INV_F, strong = inv > T.STRONG_F;
  if (isInv && w2 < T.CALM) R.push([2, `Inversion with calm air: it is ${inv.toFixed(1)}°F warmer at 30 ft than at 5 ft, and wind at 5 ft is only ${w2.toFixed(1)} mph. Spray would hang in the air and drift sideways.`]);
  if (strong) R.push([2, `Strong inversion: ${inv.toFixed(1)}°F warmer at 30 ft than at 5 ft.`]);
  if (isInv && !strong && w2 >= T.CALM) R.push([1, `Weak inversion: ${inv.toFixed(1)}°F warmer at 30 ft than at 5 ft.`]);
  if (Math.max(w2, w10) > T.WINDY) R.push([2, `Too windy: ${Math.max(w2, w10).toFixed(0)} mph (guidance limit ${T.WINDY} mph).`]);
  if (!isInv && w2 < T.CALM) R.push([1, `Calm at 5 ft: ${w2.toFixed(1)} mph, below ${T.CALM} mph. An inversion may be forming.`]);
  if (Math.abs(w10 - w2) > T.SPEED_DIFF) R.push([1, `Wind shear: ${w10.toFixed(0)} mph at 30 ft but ${w2.toFixed(0)} mph at 5 ft.`]);
  if (d > T.DIR_DIFF && w2 >= 2 && w10 >= 2) R.push([1, `Wind direction differs by ${d.toFixed(0)}° between 5 ft (${comp(c.d2[i])}) and 30 ft (${comp(c.d10[i])}).`]);
  return R;
}
function why(c, i) {
  const R = triggers(c, i), s = S(c, i), det = R.reduce((m, r) => Math.max(m, r[0]), 0), out = R.map(r => r[1]);
  if (!R.length && s === 0) out.push("No inversion, wind between 3 and 10 mph, and similar wind at both heights.");
  if (s > det) out.push(`The forecast is close to a limit. Allowing for the model's known error, ${NAME[s]} is the most likely outcome.`);
  if (s < det) out.push(`A value is just past a limit, but allowing for the model's known error, ${NAME[s]} is the most likely outcome.`);
  return out;
}
const singleOK = (c, i) => c.w2[i] / 10 >= T.CALM && c.w2[i] / 10 <= T.WINDY;
function nextSafe(c, from) {
  for (let i = from; i <= N - 4; i++) if (c.s.substr(i, 4) === "0000") { let j = i; while (j < N && c.s[j] === "0") j++; return [i, j - 1]; }
  return null;
}

// ---------- map ----------
const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25 });
const ksB = L.latLngBounds(D.ks);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", { attribution: "Tiles: Esri", maxZoom: 16 }).addTo(map);
const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { attribution: "Imagery © Esri", maxZoom: 18 });
map.on("zoomend", () => { if (map.getZoom() >= 11) { if (!map.hasLayer(sat)) sat.addTo(map); } else if (map.hasLayer(sat)) map.removeLayer(sat); });
L.polygon([[[85, -180], [85, 180], [-85, 180], [-85, -180]], D.ks], { stroke: false, fillColor: "#0b1a10", fillOpacity: 0.6, interactive: false }).addTo(map);
L.polygon(D.ks, { color: "#16301f", weight: 2, fill: false, interactive: false }).addTo(map);
map.fitBounds(ksB.pad(0.03));
const rad = ha => Math.max(4, Math.min(11, 4 + 2.2 * Math.sqrt(ha)));
D.cells.forEach(c => {
  c.m = L.circleMarker([c.lat, c.lon], { radius: rad(c.ha), color: "#fff", weight: 1, fillOpacity: 0.9 }).addTo(map);
  c.m.on("click", () => select(c));
  c.m.bindTooltip(`${c.crop}, ${c.ha} ha`);
});
function paint() {
  const n = [0, 0, 0];
  D.cells.forEach(c => { const s = S(c, ti); n[s]++; c.m.setStyle({ fillColor: COL[s] }); });
  $("tlabel").textContent = D.times[ti] + (ti === 0 ? " (now)" : "");
  $("counts").innerHTML = NAME.map((nm, k) => `<span class="dot" style="background:${COL[k]}"></span>${nm} ${n[k]}`).join(" &nbsp; ");
}
function setTime(i) { ti = Math.max(0, Math.min(N - 1, i)); $("slider").value = ti; paint(); if (sel) panel(); }
$("slider").max = N - 1; $("slider").addEventListener("input", e => setTime(+e.target.value));
$("run").textContent = "Central time";
$("bHome").onclick = () => map.flyToBounds(ksB.pad(0.03), { duration: 0.6 });

function select(c) {
  sel = c;
  const dl = 1.5 / 111, dn = 1.5 / (111 * Math.cos(c.lat * Math.PI / 180));
  if (rect) rect.remove();
  rect = L.rectangle([[c.lat - dl, c.lon - dn], [c.lat + dl, c.lon + dn]], { color: "#111", weight: 2, fill: false, dashArray: "5 4", interactive: false }).addTo(map);
  map.flyTo([c.lat, c.lon], 13.5, { duration: 0.8 });
  panel();
}

// ---------- panel ----------
function chart(c) {
  const W = 400, H = 120, L0 = 30, R0 = 34, Tp = 8, B = 16, mx = Math.max(14, ...c.w2.map(v => v / 10), ...c.w10.map(v => v / 10));
  const x = i => L0 + (W - L0 - R0) * i / (N - 1), y = v => Tp + (H - Tp - B) * (1 - v / mx);
  const line = a => a.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v / 10).toFixed(1)}`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Forecast wind at 5 ft and 30 ft">
  <rect x="${L0}" y="${y(T.WINDY)}" width="${W - L0 - R0}" height="${y(T.CALM) - y(T.WINDY)}" fill="#2b8a4e" opacity=".10"/>
  ${[0, T.CALM, T.WINDY].map(v => `<line x1="${L0}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}" stroke="#cfd4cb" stroke-width="1"/><text x="${L0 - 4}" y="${y(v) + 4}" font-size="10.5" fill="#5d6773" text-anchor="end">${v}</text>`).join("")}
  <path d="${line(c.w10)}" fill="none" stroke="#6b3fa0" stroke-width="2"/><path d="${line(c.w2)}" fill="none" stroke="#1668c7" stroke-width="2"/>
  <text x="${W - R0 + 3}" y="${y(c.w10[N - 1] / 10) + 4}" font-size="10.5" fill="#6b3fa0">30 ft</text>
  <text x="${W - R0 + 3}" y="${y(c.w2[N - 1] / 10) + (Math.abs(c.w10[N - 1] - c.w2[N - 1]) < 12 ? 14 : 4)}" font-size="10.5" fill="#1668c7">5 ft</text>
  <line x1="${x(ti)}" x2="${x(ti)}" y1="${Tp}" y2="${H - B}" stroke="#111" stroke-width="1.2"/>
  <text x="${L0}" y="${H - 3}" font-size="10.5" fill="#5d6773">Wind (mph). Shaded band = 3 to 10 mph guidance.</text></svg>`;
}
function panel() {
  const c = sel, i = ti, s = S(c, i), inv = c.inv[i] / 10, w2 = c.w2[i] / 10, w10 = c.w10[i] / 10;
  const ns = nextSafe(c, i), sOK = singleOK(c, i);
  const cmp = sOK && s > 0
    ? `<b class="bad">A single ground station would miss this.</b> At 5 ft the wind is ${w2.toFixed(1)} mph, inside the 3 to 10 mph guidance, so it looks fine. The second height shows the risk.`
    : sOK ? `A single 5-ft station also looks fine here: wind ${w2.toFixed(1)} mph.` : `A single 5-ft station would also flag this: wind ${w2.toFixed(1)} mph is outside 3 to 10 mph.`;
  const strip = [...c.s].map((v, k) => `<div data-k="${k}" class="${k === i ? "sel" : ""}" style="background:${COL[+v]}" title="${D.times[k]}: ${NAME[+v]}"></div>`).join("");
  const rows = [...c.s].map((v, k) => `<tr><td class="n">${D.times[k]}</td><td><span class="pill" style="background:${COL[+v]}">${NAME[+v]}</span></td><td class="n">${c.c[k]}%</td><td class="n">${(c.inv[k] / 10).toFixed(1)}</td><td class="n">${(c.w2[k] / 10).toFixed(1)}</td><td class="n">${(c.w10[k] / 10).toFixed(1)}</td><td>${triggers(c, k).map(r => r[1].split(":")[0]).join("; ") || "none"}</td></tr>`).join("");
  $("panel").innerHTML = `
  <h2>${c.crop} orchard area</h2>
  <div class="muted">${c.ha} ha of orchard in this 3 km cell &middot; ${D.times[i]}${i === 0 ? " (now)" : ""}</div>
  <div class="badge" style="background:${COL[s]}"><div class="ic">${ICON[s]}</div><div><div class="wd">${NAME[s]}</div><div class="cf">${ADV[s]} &middot; ${c.c[i]}% confident</div></div></div>
  <h3>Why</h3><ul class="why">${why(c, i).map(t => `<li>${t}</li>`).join("")}</ul>
  <div class="cmp">${cmp}</div>
  <table><tr><th></th><th>5 ft</th><th>30 ft</th></tr>
   <tr><td>Wind speed</td><td class="n">${w2.toFixed(1)} mph</td><td class="n">${w10.toFixed(1)} mph</td></tr>
   <tr><td>Wind from</td><td class="n">${comp(c.d2[i])} (${c.d2[i]}°)</td><td class="n">${comp(c.d10[i])} (${c.d10[i]}°)</td></tr>
   <tr><td>Temperature</td><td colspan="2" class="n">30 ft is ${Math.abs(inv).toFixed(1)}°F ${inv > 0 ? "warmer" : "cooler"} than 5 ft ${inv > T.INV_F ? "(inversion)" : "(no inversion)"}</td></tr></table>
  <h3>Next safe window</h3><div>${ns ? (ns[0] === i ? `Now until ${D.times[ns[1]]}` : `${D.times[ns[0]]} to ${D.times[ns[1]]}`) : "No window of at least 1 hour in this forecast."}</div>
  <h3>Next 18 hours (click to jump)</h3><div id="strip">${strip}</div><div class="axis"><span>${D.times[0]}</span><span>${D.times[N - 1]}</span></div>
  <details><summary>Time-stamped log</summary>
  <div style="text-align:right;margin-bottom:5px"><button id="bCsv" style="padding:2px 8px;font-size:12.5px">Download CSV</button></div>
  <div id="logbox"><table><tr><th>Time</th><th>Status</th><th>Conf.</th><th>ΔT °F</th><th>5 ft</th><th>30 ft</th><th>Triggered by</th></tr>${rows}</table></div></details>
  <div class="note">Model estimates. The pesticide label, on-site checks and regulations always come first.</div>`;
  $("strip").onclick = e => { const k = e.target.dataset.k; if (k !== undefined) setTime(+k); };
  $("bCsv").onclick = () => csv(c);
}
function csv(c) {
  const q = t => `"${String(t).replace(/"/g, '""')}"`;
  const L1 = ["time_central,cell,lat,lon,status,confidence_pct,temp_30ft_minus_5ft_F,wind_5ft_mph,wind_30ft_mph,dir_5ft_deg,dir_30ft_deg,single_station_says,reasons"];
  for (let k = 0; k < N; k++) L1.push([D.iso[k], c.id, c.lat, c.lon, NAME[S(c, k)], c.c[k], c.inv[k] / 10, c.w2[k] / 10, c.w10[k] / 10, c.d2[k], c.d10[k], singleOK(c, k) ? "OK" : "NOT OK", q(why(c, k).join(" | "))].join(","));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([L1.join("\n")], { type: "text/csv" }));
  a.download = `spray_log_${c.id}.csv`; a.click();
}

// ---------- demo helper + metrics ----------
$("bMiss").onclick = () => {
  let best = null;
  for (const c of D.cells) for (let k = 0; k < N; k++) {
    const s = S(c, k); if (s === 0 || !singleOK(c, k)) continue;
    const score = s * 1000 + c.c[k] + Math.min(c.ha, 50);
    if (!best || score > best[0]) best = [score, c, k];
  }
  if (!best) { alert("No such case in this forecast: whenever the tool flags a risk, the 5 ft wind is also outside 3 to 10 mph."); return; }
  ti = best[2]; $("slider").value = ti; paint(); select(best[1]);
};
const LBL = { y_dT: "Temperature difference, 30 ft minus 5 ft (°C)", u2: "Wind at 5 ft, east-west (m/s)", v2: "Wind at 5 ft, north-south (m/s)",
  u10: "Wind at 30 ft, east-west (m/s)", v10: "Wind at 30 ft, north-south (m/s)", ws2: "Wind speed at 5 ft (m/s)", ws10: "Wind speed at 30 ft (m/s)" };
$("bMet").onclick = () => {
  const f = v => v === null || v === undefined ? "" : (+v).toFixed(2);
  const reg = D.metrics.filter(m => LBL[m.target]);
  const inv = D.metrics.find(m => String(m.target).startsWith("inversion"));
  $("mbox").innerHTML = `<h2 style="margin:0 0 4px">How accurate is the model?</h2>
  <div class="muted">All numbers are for weather stations the model never saw during training (held out by area).</div>
  <h3>What it means for spraying</h3><table>${D.extra.map(r => `<tr><td>${r[0]}</td><td class="n"><b>${r[1]}</b></td></tr>`).join("")}
  ${inv ? `<tr><td>Real inversions (more than 0.9°F) the model detected</td><td class="n"><b>${Math.round(100 * inv.hit_rate)}%</b></td></tr><tr><td>Inversion alerts that were false alarms</td><td class="n"><b>${Math.round(100 * inv.false_alarm)}%</b></td></tr>` : ""}</table>
  <h3>Raw model errors</h3><table><tr><th>Predicted quantity</th><th>Average error</th><th>R²</th><th>HRRR alone: error</th><th>HRRR alone: R²</th></tr>
  ${reg.map(m => `<tr><td>${LBL[m.target]}</td><td class="n">${f(m.MAE)}</td><td class="n">${f(m.R2)}</td><td class="n">${f(m.MAE_hrrr)}</td><td class="n">${f(m.R2_hrrr)}</td></tr>`).join("")}</table>
  <div class="note">Confidence shown in the app: the model's held-out errors are added to each forecast 100 times and the rules are re-applied; the share of each outcome is its probability. GO is shown only when it is at least ${T.GO_MIN}% likely. Rules: inversion above ${T.INV_F}°F, strong above ${T.STRONG_F}°F, wind outside ${T.CALM} to ${T.WINDY} mph, speed difference above ${T.SPEED_DIFF} mph, direction difference above ${T.DIR_DIFF}°. The model was trained on Kansas Mesonet stations at 2 m and 10 m (about 5 ft and 30 ft), in open sites, not inside orchard canopy.</div>
  <div style="text-align:right;margin-top:10px"><button id="bClose">Close</button></div>`;
  $("modal").style.display = "flex"; $("bClose").onclick = () => $("modal").style.display = "none";
};
$("modal").onclick = e => { if (e.target.id === "modal") $("modal").style.display = "none"; };
paint();
</script></body></html>
"""

if __name__ == "__main__":
    main()
