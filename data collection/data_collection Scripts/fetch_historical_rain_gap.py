"""
fetch_historical_rain_gap.py
----------------------------
Fetches rain data for the 2014-07-01 to 2014-12-31 gap to allow 
filling of SPI-180 and Lag-1 features in the master matrix.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ds_locations import DS_LOCATIONS
from utils import fetch_data, rate_limit, log, ProgressTracker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CSV_PATH  = DATA_DIR / "rain_sum_2014.csv"
PROG_PATH = DATA_DIR / "progress_historical.json"

FIELDNAMES = ["date", "division", "rain_sum"]
TASK = "rain_historical"

# ---------------------------------------------------------------------------
def init_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        log.info(f"Created {CSV_PATH}")

def append_rows(rows: list[dict]):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(rows)

# ---------------------------------------------------------------------------
def fetch_historical_rain(ds: dict) -> list[dict] | None:
    data = fetch_data(
        lat   = ds["lat"],
        lon   = ds["lon"],
        daily = ["rain_sum"],
        start_date = "2014-07-01",
        end_date   = "2014-12-31"
    )
    if data is None:
        return None

    daily_block = data.get("daily", {})
    dates  = daily_block.get("time", [])
    rains  = daily_block.get("rain_sum", [])

    if not dates:
        return None

    rows = []
    for date, rain in zip(dates, rains):
        rows.append({
            "date":      date,
            "division":  ds["id"],
            "rain_sum":  rain,
        })
    return rows

# ---------------------------------------------------------------------------
def main():
    init_csv()
    tracker = ProgressTracker(PROG_PATH, task=TASK)
    total   = len(DS_LOCATIONS)
    
    log.info("=== Historical Rain Gap Collection Started ===")
    
    for idx, ds in enumerate(DS_LOCATIONS, start=1):
        ds_id = ds["id"]
        if tracker.is_done(ds_id):
            continue

        log.info(f"[{idx:>3}/{total}] Fetching 2014 data for: {ds_id}")
        rows = fetch_historical_rain(ds)

        if rows:
            append_rows(rows)
            tracker.mark_done(ds_id)
        
        rate_limit()

    log.info("=== Historical Rain Gap Collection Complete ===")

if __name__ == "__main__":
    main()
