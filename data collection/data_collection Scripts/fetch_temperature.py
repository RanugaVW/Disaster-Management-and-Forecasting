"""
fetch_temperature.py
--------------------
Collect daily mean 2 m air temperature for every Sri Lanka
Divisional Secretariat (DS) from Open-Meteo (ERA5 reanalysis).

Date range : 2015-01-01 → 2026-01-01 (configurable in utils.py)
Output CSV : ../data/temperature_mean_2m.csv

CSV schema
----------
id                  : DS name  (primary identifier)
district            : District name
province            : Province name
latitude            : Actual latitude snapped by Open-Meteo grid
longitude           : Actual longitude snapped by Open-Meteo grid
date                : YYYY-MM-DD
temperature_2m_mean : Mean 2 m air temperature (°C)

Run
---
    python fetch_temperature.py

The script resumes automatically if interrupted (progress is saved to
../data/progress.json after each successful DS).
"""

import csv
import sys
from pathlib import Path

# Allow imports from the parent directory
sys.path.insert(0, str(Path(__file__).parent))

from ds_locations import DS_LOCATIONS
from utils import fetch_data, rate_limit, log, ProgressTracker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CSV_PATH  = DATA_DIR / "temperature_mean_2m.csv"
PROG_PATH = DATA_DIR / "progress.json"

FIELDNAMES = ["date", "division", "temperature_2m_mean"]

TASK = "temperature"


# ---------------------------------------------------------------------------
def init_csv():
    """Create the CSV with headers if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        log.info(f"Created {CSV_PATH}")


def append_rows(rows: list[dict]):
    """Append a batch of row dicts to the CSV."""
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
def fetch_temperature_for_ds(ds: dict) -> list[dict] | None:
    """
    Fetch daily temperature_2m_mean for one DS division.

    Returns a list of row dicts ready for CSV, or None on failure.
    """
    data = fetch_data(
        lat   = ds["lat"],
        lon   = ds["lon"],
        daily = ["temperature_2m_mean"],
    )
    if data is None:
        return None

    daily_block = data.get("daily", {})
    dates  = daily_block.get("time", [])
    temps  = daily_block.get("temperature_2m_mean", [])

    if not dates:
        log.warning(f"  No data returned for {ds['id']}")
        return None

    rows = []
    for date, temp in zip(dates, temps):
        rows.append({
            "date":                 date,
            "division":             ds["id"],
            "temperature_2m_mean":  temp,
        })
    return rows


# ---------------------------------------------------------------------------
def main():
    init_csv()
    tracker = ProgressTracker(PROG_PATH, task=TASK)
    total   = len(DS_LOCATIONS)
    failed  = []

    log.info(f"=== Temperature Collection Started ===")
    log.info(f"Total DS divisions : {total}")
    log.info(f"Already completed  : {tracker.completed_count()}")

    for idx, ds in enumerate(DS_LOCATIONS, start=1):
        ds_id = ds["id"]

        if tracker.is_done(ds_id):
            log.info(f"[{idx:>3}/{total}] SKIP (already done) : {ds_id}")
            continue

        log.info(f"[{idx:>3}/{total}] Fetching : {ds_id} ({ds['district']})")
        rows = fetch_temperature_for_ds(ds)

        if rows is None:
            log.error(f"  -> FAILED : {ds_id}")
            failed.append(ds_id)
        else:
            append_rows(rows)
            tracker.mark_done(ds_id)
            log.info(f"  -> OK : {len(rows)} rows saved")

        rate_limit()

    # Summary
    log.info("\n=== Temperature Collection Complete ===")
    log.info(f"Successful : {total - len(failed)}/{total}")
    if failed:
        log.warning(f"Failed DS  : {', '.join(failed)}")
        log.warning("Re-run the script to retry failed divisions.")
    else:
        log.info("All DS divisions collected successfully!")


if __name__ == "__main__":
    main()
