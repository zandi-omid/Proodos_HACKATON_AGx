# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""Kansas Mesonet: how many stations, how far back, which have both heights.
pip install ksmesopy pandas
python ks_check.py
"""
import pandas as pd
import ksmesopy as ms

VARS = ["TEMP2MAVG", "TEMP10MAVG", "WSPD2MAVG", "WSPD10MAVG", "WDIR2M", "WDIR10M"]

allst = ms.get_stations()
print("all stations in metadata:", len(allst))
print("metadata columns:", list(allst.columns))
allst.to_csv("ks_stations_all.csv", index=False)

act = ms.get_stations_active().copy()
act = act[act["OBS_INTERVAL"] == 300].drop_duplicates("STATION").copy()  # 5-min rows only
act["START"] = pd.to_datetime(act["START"], errors="coerce")
act["END"] = pd.to_datetime(act["END"], errors="coerce")
print("\nactive stations:", len(act))
print("earliest START:", act["START"].min())
print("latest START:  ", act["START"].max())
print("obs interval (s) counts:\n", act["OBS_INTERVAL"].value_counts().to_string())
print("stations by start year:\n", act["START"].dt.year.value_counts().sort_index().to_string())

# one-day 5-min sample per station: do both heights report?
end = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
start = end - pd.Timedelta(days=1)
rows = []
for s in act["STATION"]:
    try:
        d = ms.request_data(s, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                            "5min", VARS, verbose=False)
        r = {"STATION": s, "rows": len(d)}
        for v in d.columns:
            if v.lower() in {x.lower() for x in VARS} or v in VARS:
                r[v] = int(d[v].notna().sum())
    except Exception as e:
        r = {"STATION": s, "error": str(e)[:80]}
    rows.append(r)
chk = pd.DataFrame(rows)
chk.to_csv("ks_height_check.csv", index=False)
print("\nsample", start.date(), "->", end.date())
print(chk.to_string(index=False))
