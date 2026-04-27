"""
fetch_soil_moisture.py
-----------------------
Collect soil moisture data (3 depth layers) for every Sri Lanka
Divisional Secretariat (DS) from Open-Meteo (ERA5-Land reanalysis).

The Open-Meteo API returns soil moisture at hourly resolution.
This script fetches the full hourly time series and aggregates it to
DAILY MEAN values — one row per DS per day — to keep the output at the
same scale as the temperature and rain CSVs (~3 M rows total).

Date range : 2015-01-01 → 2026-01-01
Output CSV : ../data/soil_moisture_daily.csv

CSV schema
----------
id                        : DS name  (primary identifier)
district                  : District name
province                  : Province name
latitude                  : Actual latitude snapped by Open-Meteo grid
longitude                 : Actual longitude snapped by Open-Meteo grid
date                      : YYYY-MM-DD
soil_moisture_7_to_28cm   : Daily mean volumetric soil moisture (m³/m³)
soil_moisture_28_to_100cm : Daily mean volumetric soil moisture (m³/m³)
soil_moisture_100_to_255cm: Daily mean volumetric soil moisture (m³/m³)

Run
---
    python fetch_soil_moisture.py

Resumes automatically if interrupted (progress saved per DS).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ds_locations import DS_LOCATIONS
from utils import fetch_data, rate_limit, log, ProgressTracker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CSV_PATH  = DATA_DIR / "soil_moisture_daily.csv"
PROG_PATH = DATA_DIR / "progress.json"

FIELDNAMES = [
    "date", "division",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]

HOURLY_VARS = [
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]

TASK = "soil_moisture"


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
def aggregate_hourly_to_daily(
    times: list[str],
    sm_7_28:   list[float | None],
    sm_28_100: list[float | None],
    sm_100_255:list[float | None],
) -> dict[str, dict]:
    """
    Aggregate hourly soil moisture readings to daily means.

    Parameters
    ----------
    times       : ISO datetime strings  "YYYY-MM-DDTHH:MM"
    sm_*        : Corresponding hourly values (may contain None / null)

    Returns
    -------
    dict keyed by date string "YYYY-MM-DD", each value is a dict with
    keys soil_moisture_7_to_28cm, soil_moisture_28_to_100cm,
    soil_moisture_100_to_255cm (daily means, rounded to 4 dp).
    """
    # Accumulate per-day sums and counts
    day_data: dict[str, dict] = defaultdict(
        lambda: {"s7": [], "s28": [], "s100": []}
    )

    for ts, v7, v28, v100 in zip(times, sm_7_28, sm_28_100, sm_100_255):
        date = ts[:10]           # "YYYY-MM-DDTHH:MM" → "YYYY-MM-DD"
        if v7   is not None: day_data[date]["s7"].append(v7)
        if v28  is not None: day_data[date]["s28"].append(v28)
        if v100 is not None: day_data[date]["s100"].append(v100)

    # Compute means
    daily: dict[str, dict] = {}
    for date in sorted(day_data.keys()):
        d = day_data[date]
        daily[date] = {
            "soil_moisture_7_to_28cm":    round(sum(d["s7"])   / len(d["s7"]),   4) if d["s7"]   else None,
            "soil_moisture_28_to_100cm":  round(sum(d["s28"])  / len(d["s28"]),  4) if d["s28"]  else None,
            "soil_moisture_100_to_255cm": round(sum(d["s100"]) / len(d["s100"]), 4) if d["s100"] else None,
        }
    return daily


# ---------------------------------------------------------------------------
def fetch_soil_moisture_for_ds(ds: dict) -> list[dict] | None:
    """
    Fetch hourly soil moisture for one DS and return daily-aggregated rows.

    Returns list of row dicts or None on failure.
    """
    data = fetch_data(
        lat    = ds["lat"],
        lon    = ds["lon"],
        hourly = HOURLY_VARS,
    )
    if data is None:
        return None

    hourly_block = data.get("hourly", {})
    times   = hourly_block.get("time", [])
    sm_7    = hourly_block.get("soil_moisture_7_to_28cm",    [])
    sm_28   = hourly_block.get("soil_moisture_28_to_100cm",  [])
    sm_100  = hourly_block.get("soil_moisture_100_to_255cm", [])

    if not times:
        log.warning(f"  No hourly data returned for {ds['id']}")
        return None

    log.info(f"  Aggregating {len(times)} hourly points → daily means …")
    daily = aggregate_hourly_to_daily(times, sm_7, sm_28, sm_100)

    rows = []
    for date, vals in daily.items():
        rows.append({
            "date":                       date,
            "division":                   ds["id"],
            "soil_moisture_7_to_28cm":    vals["soil_moisture_7_to_28cm"],
            "soil_moisture_28_to_100cm":  vals["soil_moisture_28_to_100cm"],
            "soil_moisture_100_to_255cm": vals["soil_moisture_100_to_255cm"],
        })
    return rows


# ---------------------------------------------------------------------------
def main():
    init_csv()
    tracker = ProgressTracker(PROG_PATH, task=TASK)
    total   = len(DS_LOCATIONS)
    failed  = []

    log.info("=== Soil Moisture Collection Started ===")
    log.info(f"Total DS divisions : {total}")
    log.info(f"Already completed  : {tracker.completed_count()}")
    log.info("(Hourly data will be aggregated to daily means per DS)")

    for idx, ds in enumerate(DS_LOCATIONS, start=1):
        ds_id = ds["id"]

        if tracker.is_done(ds_id):
            log.info(f"[{idx:>3}/{total}] SKIP (already done) : {ds_id}")
            continue

        log.info(f"[{idx:>3}/{total}] Fetching : {ds_id} ({ds['district']})")
        rows = fetch_soil_moisture_for_ds(ds)

        if rows is None:
            log.error(f"  -> FAILED : {ds_id}")
            failed.append(ds_id)
        else:
            append_rows(rows)
            tracker.mark_done(ds_id)
            log.info(f"  -> OK : {len(rows)} daily rows saved")

        rate_limit()

    log.info("\n=== Soil Moisture Collection Complete ===")
    log.info(f"Successful : {total - len(failed)}/{total}")
    if failed:
        log.warning(f"Failed DS  : {', '.join(failed)}")
        log.warning("Re-run the script to retry failed divisions.")
    else:
        log.info("All DS divisions collected successfully!")


if __name__ == "__main__":
    main()
