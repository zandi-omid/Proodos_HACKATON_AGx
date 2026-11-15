# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
"""Keep the stations that reported both heights (2 m and 10 m) for a full test day.

python make_station_list.py
Reads  ks_stations_all.csv, ks_height_check.csv (both written by ks_check.py)
Writes ks_stations_both_heights.csv  (name, county, latitude, longitude, elevation, network)
"""
import pandas as pd

CORE = ["TEMP2MAVG", "TEMP10MAVG", "WSPD2MAVG", "WSPD10MAVG", "WDIR2M", "WDIR10M"]
MIN_READINGS = 250                      # of 289 five-minute readings in the test day

meta = pd.read_csv("ks_stations_all.csv")
chk = pd.read_csv("ks_height_check.csv")
ok = chk[chk["error"].isna()] if "error" in chk else chk
ok = ok[(ok[CORE] >= MIN_READINGS).all(axis=1)]
out = ok[["STATION"]].merge(meta, left_on="STATION", right_on="NAME", how="left")
out = out[["NAME", "COUNTY", "LATITUDE", "LONGITUDE", "ELEVATION", "NETWORK"]].rename(columns=str.lower)
out.to_csv("ks_stations_both_heights.csv", index=False)
print(len(out), "stations with both heights")
