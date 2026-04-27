"""
fetch_all.py
------------
Master runner — executes all three data collection scripts in sequence:

  1. fetch_temperature.py  →  data/temperature_mean_2m.csv
  2. fetch_rain.py         →  data/rain_sum.csv
  3. fetch_soil_moisture.py→  data/soil_moisture_daily.csv

After all three finish, it calls generate_report.py to produce the
final Markdown summary in reports/data_collection_report.md.

Run
---
    python fetch_all.py

    # Run only one step (useful for debugging or partial re-runs):
    python fetch_all.py --only temperature
    python fetch_all.py --only rain
    python fetch_all.py --only soil_moisture
    python fetch_all.py --only report
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import log

# Steps: (label, module_name)
STEPS = [
    ("Temperature",    "fetch_temperature"),
    ("Rain",           "fetch_rain"),
    ("Soil Moisture",  "fetch_soil_moisture"),
    ("Report",         "generate_report"),
]

STEP_FLAGS = {
    "temperature":   "Temperature",
    "rain":          "Rain",
    "soil_moisture": "Soil Moisture",
    "report":        "Report",
}


def run_step(label: str, module_name: str) -> bool:
    """Dynamically import and call main() from a sibling module."""
    log.info("")
    log.info("=" * 60)
    log.info(f"  STEP : {label}")
    log.info("=" * 60)
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(module_name)
        # Reload so the module re-runs even if called a second time
        importlib.reload(mod)
        mod.main()
        elapsed = time.perf_counter() - t0
        log.info(f"  Finished in {elapsed:.1f}s")
        return True
    except Exception as exc:
        log.error(f"  STEP FAILED — {label}: {exc}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Open-Meteo data collection master runner"
    )
    parser.add_argument(
        "--only",
        choices=list(STEP_FLAGS.keys()),
        default=None,
        help="Run only one specific step.",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()
    results   = {}

    if args.only:
        target_label = STEP_FLAGS[args.only]
        for label, module in STEPS:
            if label == target_label:
                results[label] = run_step(label, module)
    else:
        for label, module in STEPS:
            results[label] = run_step(label, module)

    # ── Summary ────────────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - wall_start
    log.info("")
    log.info("=" * 60)
    log.info("  ALL STEPS COMPLETE")
    log.info(f"  Total wall time : {wall_elapsed / 60:.1f} min")
    log.info("=" * 60)
    for label, ok in results.items():
        status = "✓ OK " if ok else "✗ FAILED"
        log.info(f"  {status}  {label}")

    failed = [l for l, ok in results.items() if not ok]
    if failed:
        log.warning(f"\nFailed steps: {', '.join(failed)}. Re-run fetch_all.py to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
