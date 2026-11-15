"""Kansas Mesonet: raw 5-min data, last 4 years, for the 73 two-height stations.

python ks_pull.py [workers]          (default 4 stations in parallel)
Needs ks_stations_both_heights.csv in the same folder.
Output: ks5min/<station>.parquet (one per station) + ks5min_manifest.csv
Re-running skips stations already saved.
As a Slurm array each task takes every n-th station; afterwards run
`python ks_pull.py` once on the login node to write the manifest (it skips all saved stations).
"""
import io, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import ksmesopy as ms

VARS = [
    "TEMP2MAVG", "TEMP10MAVG",
    "WSPD2MAVG", "WSPD10MAVG",
    "WDIR2M", "WDIR10M",
    "WDIR2MSTD", "WDIR10MSTD",
    "WSPD2MMAX", "WSPD10MMAX",
    "RELHUM2MAVG", "VPDEFAVG", "PRESSUREAVG", "SRAVG", "PRECIP", "PRECIP2",
]
CORE = ["TEMP2MAVG", "TEMP10MAVG", "WSPD2MAVG", "WSPD10MAVG", "WDIR2M", "WDIR10M"]

URL = "http://mesonet.k-state.edu/rest/stationdata/"
END = pd.Timestamp.now().normalize() - pd.Timedelta(minutes=5)   # through yesterday 23:55
START = (END - pd.DateOffset(years=4)).normalize()
CHUNK = pd.Timedelta(minutes=5) * 2999                          # API limit 3000 records
OUT = Path("ks5min"); OUT.mkdir(exist_ok=True)
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def fetch(stn, s, e):
    q = urllib.parse.urlencode({"stn": stn, "int": "5min",
                                "t_start": s.strftime("%Y%m%d%H%M%S"),
                                "t_end": e.strftime("%Y%m%d%H%M%S"),
                                "vars": ",".join(VARS)}, safe=",", quote_via=urllib.parse.quote)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
                txt = r.read().decode()
            if "TIMESTAMP" not in txt[:200]:
                return None                       # "No data available": not an error
            d = pd.read_csv(io.StringIO(txt), na_values="M", parse_dates=["TIMESTAMP"])
            return d.drop(columns=["STATION"], errors="ignore")
        except Exception as ex:
            last = ex
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(last)


def pull_station(stn, first):
    f = OUT / f"{stn.replace(' ', '_')}.parquet"
    if f.exists():
        return stn, "skipped", 0
    s = max(START, first)
    parts, failed = [], 0
    while s <= END:
        e = min(s + CHUNK, END)
        try:
            d = fetch(stn, s, e)
            if d is not None and len(d):
                parts.append(d)
        except Exception:
            failed += 1
        s = e + pd.Timedelta(minutes=5)
        time.sleep(0.3)
    if not parts:
        return stn, "no data", failed
    d = pd.concat(parts, ignore_index=True).drop_duplicates("TIMESTAMP").set_index("TIMESTAMP")
    idx = pd.date_range(d.index.min(), d.index.max(), freq="5min")
    d = d.reindex(idx)                            # gaps become NaN rows
    for v in VARS:
        if v not in d.columns:
            d[v] = np.nan
    d = d[VARS].astype("float32")
    d.index.name = "timestamp_raw"
    d = d.reset_index()
    d.insert(1, "station", stn)
    d.to_parquet(f, index=False)
    return stn, "ok", failed


def summarize(f):
    d = pd.read_parquet(f)
    r = {"station": d.station.iloc[0], "rows": len(d),
         "first": d.timestamp_raw.min(), "last": d.timestamp_raw.max()}
    for v in CORE:
        r[f"{v}_pct"] = round(100 * d[v].notna().mean(), 1)
    return r


if __name__ == "__main__":
    import os
    st = pd.read_csv("ks_stations_both_heights.csv")["name"].tolist()
    shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    nshard = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    st = st[shard::nshard]
    act = ms.get_stations_active()
    act = act[act.OBS_INTERVAL == 300].drop_duplicates("STATION").set_index("STATION")
    first = {s: pd.Timestamp(act.START.get(s, START)).normalize() for s in st}
    print(f"{len(st)} stations, {START.date()} -> {END.date()}, {WORKERS} workers", flush=True)
    t0, done, fails = time.time(), 0, {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(pull_station, s, first[s]) for s in st]
        for fu in as_completed(futs):
            stn, status, nfail = fu.result()
            done += 1
            fails[stn] = nfail
            print(f"[{done}/{len(st)}] {stn}: {status}"
                  + (f", {nfail} failed chunks" if nfail else "")
                  + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if nshard > 1:
        sys.exit(0)                    # manifest: run `python ks_pull.py` once after all tasks finish
    m = pd.DataFrame([summarize(f) for f in sorted(OUT.glob("*.parquet"))])
    m["failed_chunks"] = m.station.map(fails).fillna(0).astype(int)
    m.to_csv("ks5min_manifest.csv", index=False)
    print(f"\nsaved {len(m)} stations, {m.rows.sum():,} rows, "
          f"{sum(f.stat().st_size for f in OUT.glob('*.parquet')) / 1e6:.0f} MB")
    print(m.to_string(index=False))
