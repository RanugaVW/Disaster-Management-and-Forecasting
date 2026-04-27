"""
generate_report.py
------------------
Reads the three output CSVs and writes a detailed Markdown report to
  reports/data_collection_report.md

Metrics reported
----------------
- Row count and date coverage per CSV
- DS count per district / province
- Null / missing-value rates per variable column
- Min / max / mean statistics per variable
- List of any DS IDs missing from each CSV (quality gate)
- File sizes

Run standalone after collection:
    python generate_report.py
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ds_locations import DS_LOCATIONS
from utils import log, START_DATE, END_DATE

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "data_collection_report.md"

CSV_Temperature   = DATA_DIR / "temperature_mean_2m.csv"
CSV_Rain          = DATA_DIR / "rain_sum.csv"
CSV_SoilMoisture  = DATA_DIR / "soil_moisture_daily.csv"

EXPECTED_DS = {ds["id"] for ds in DS_LOCATIONS}

# Rough expected number of days in the range
from datetime import date as _date
_d0, _d1 = _date.fromisoformat(START_DATE), _date.fromisoformat(END_DATE)
EXPECTED_DAYS = (_d1 - _d0).days          # 2015-01-01 to 2026-01-01 = 4018 days


# ---------------------------------------------------------------------------
def file_size_str(path: Path) -> str:
    if not path.exists():
        return "NOT FOUND"
    size_mb = path.stat().st_size / (1024 ** 2)
    if size_mb >= 1000:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.2f} MB"


def safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyse_csv(path: Path, value_cols: list[str]) -> dict:
    """
    Read a DS-keyed CSV and return analysis stats.

    Returns
    -------
    dict with keys:
      exists, rows, ds_found, missing_ds, date_min, date_max,
      stats: {col: {min, max, mean, null_count, null_pct}}
    """
    if not path.exists():
        return {"exists": False}

    rows          = 0
    ds_found      = set()
    date_min      = "9999-99-99"
    date_max      = "0000-00-00"
    col_vals      = {c: [] for c in value_cols}
    col_nulls     = {c: 0  for c in value_cols}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            ds_found.add(row.get("division", ""))
            d = row.get("date", row.get("datetime", ""))[:10]
            if d < date_min: date_min = d
            if d > date_max: date_max = d
            for col in value_cols:
                v = safe_float(row.get(col))
                if v is None:
                    col_nulls[col] += 1
                else:
                    col_vals[col].append(v)

    stats = {}
    for col in value_cols:
        vals = col_vals[col]
        nulls = col_nulls[col]
        pct   = 100 * nulls / rows if rows else 0
        stats[col] = {
            "min":        round(min(vals), 4)         if vals else None,
            "max":        round(max(vals), 4)         if vals else None,
            "mean":       round(sum(vals) / len(vals), 4) if vals else None,
            "null_count": nulls,
            "null_pct":   round(pct, 2),
        }

    return {
        "exists":      True,
        "rows":        rows,
        "ds_found":    ds_found,
        "missing_ds":  sorted(EXPECTED_DS - ds_found),
        "date_min":    date_min,
        "date_max":    date_max,
        "stats":       stats,
    }


def ds_by_district() -> dict[str, list[str]]:
    d = defaultdict(list)
    for ds in DS_LOCATIONS:
        d[ds["district"]].append(ds["id"])
    return dict(sorted(d.items()))


def ds_by_province() -> dict[str, list[str]]:
    d = defaultdict(list)
    for ds in DS_LOCATIONS:
        d[ds["province"]].append(ds["id"])
    return dict(sorted(d.items()))


# ---------------------------------------------------------------------------
def build_report() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    temp_analysis = analyse_csv(CSV_Temperature,  ["temperature_2m_mean"])
    rain_analysis = analyse_csv(CSV_Rain,         ["rain_sum"])
    sm_analysis   = analyse_csv(CSV_SoilMoisture, [
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm",
    ])

    dist_map = ds_by_district()
    prov_map = ds_by_province()
    total_ds = len(DS_LOCATIONS)

    # ── Helper functions ───────────────────────────────────────────────────
    def section_header(a: dict, name: str, path: Path) -> str:
        if not a.get("exists"):
            return f"\n> ⚠️ **{name}** — file not found: `{path}`\n"
        return ""

    def stats_table(stats: dict) -> str:
        lines = []
        lines.append("| Variable | Min | Max | Mean | Nulls | Null % |")
        lines.append("|---|---|---|---|---|---|")
        for col, s in stats.items():
            lines.append(
                f"| `{col}` | {s['min']} | {s['max']} | {s['mean']} "
                f"| {s['null_count']:,} | {s['null_pct']}% |"
            )
        return "\n".join(lines)

    def coverage_bar(ds_found: set) -> str:
        pct = 100 * len(ds_found) / total_ds if total_ds else 0
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        return f"`{bar}` {len(ds_found)}/{total_ds} DS ({pct:.1f}%)"

    def missing_list(missing: list) -> str:
        if not missing:
            return "_None — all DS divisions present_ ✅"
        return ", ".join(f"`{m}`" for m in missing)

    # ── District coverage table ────────────────────────────────────────────
    district_rows = []
    for dist, ids in dist_map.items():
        prov = next(ds["province"] for ds in DS_LOCATIONS if ds["district"] == dist)
        ds_names = ", ".join(f"`{d}`" for d in sorted(ids))
        district_rows.append(f"| {prov} | {dist} | {len(ids)} | {ds_names} |")

    # ── Province summary ──────────────────────────────────────────────────
    prov_rows = []
    for prov, ids in prov_map.items():
        prov_rows.append(f"| {prov} | {len(ids)} |")

    # ──────────────────────────────────────────────────────────────────────
    # Build Markdown
    # ──────────────────────────────────────────────────────────────────────
    md = f"""# Open-Meteo Historical Data Collection Report
**Group J2 — Disaster Response System : Data & Intelligence**

---

*Generated on: {now}*

---

## 1. Overview

| Item | Value |
|---|---|
| Data Source | Open-Meteo Historical Archive API (ERA5 / ERA5-Land reanalysis) |
| API Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Date Range | **{START_DATE}** → **{END_DATE}** |
| Expected Days | {EXPECTED_DAYS:,} calendar days |
| DS Divisions Registered | {total_ds} |
| Provinces Covered | {len(prov_map)} |
| Districts Covered | {len(dist_map)} |
| Report Generated | {now} |

---

## 2. Output Files

| File | Description | Size |
|---|---|---|
| `data/temperature_mean_2m.csv` | Daily mean 2 m air temperature (°C) | {file_size_str(CSV_Temperature)} |
| `data/rain_sum.csv` | Daily total rainfall (mm) | {file_size_str(CSV_Rain)} |
| `data/soil_moisture_daily.csv` | Daily mean soil moisture — 3 depth layers (m³/m³) | {file_size_str(CSV_SoilMoisture)} |
| `data/progress.json` | Collection progress tracker (resume state) | {file_size_str(DATA_DIR / 'progress.json')} |

---

## 3. Collection Summary by CSV

### 3.1 Temperature — `temperature_mean_2m.csv`

{'> ⚠️ File not found — run `fetch_temperature.py` first.' if not temp_analysis.get('exists') else f"""
| Metric | Value |
|---|---|
| Total rows | {temp_analysis['rows']:,} |
| DS Coverage | {coverage_bar(temp_analysis['ds_found'])} |
| Date Range in data | {temp_analysis['date_min']} → {temp_analysis['date_max']} |

**Variable Statistics:**

{stats_table(temp_analysis['stats'])}

**Missing DS divisions:**

{missing_list(temp_analysis['missing_ds'])}
"""}

---

### 3.2 Rain Sum — `rain_sum.csv`

{'> ⚠️ File not found — run `fetch_rain.py` first.' if not rain_analysis.get('exists') else f"""
| Metric | Value |
|---|---|
| Total rows | {rain_analysis['rows']:,} |
| DS Coverage | {coverage_bar(rain_analysis['ds_found'])} |
| Date Range in data | {rain_analysis['date_min']} → {rain_analysis['date_max']} |

**Variable Statistics:**

{stats_table(rain_analysis['stats'])}

**Missing DS divisions:**

{missing_list(rain_analysis['missing_ds'])}
"""}

---

### 3.3 Soil Moisture — `soil_moisture_daily.csv`

{'> ⚠️ File not found — run `fetch_soil_moisture.py` first.' if not sm_analysis.get('exists') else f"""
| Metric | Value |
|---|---|
| Total rows | {sm_analysis['rows']:,} |
| DS Coverage | {coverage_bar(sm_analysis['ds_found'])} |
| Date Range in data | {sm_analysis['date_min']} → {sm_analysis['date_max']} |
| Aggregation | Hourly ERA5-Land values averaged → daily mean |

**Variable Statistics:**

{stats_table(sm_analysis['stats'])}

**Missing DS divisions:**

{missing_list(sm_analysis['missing_ds'])}
"""}

---

## 4. DS Division Registry by District

| Province | District | DS Count | Collected Divisions |
|---|---|---|---|
{chr(10).join(district_rows)}
| | **TOTAL** | **{total_ds}** | |

---

## 5. DS Division Registry by Province

| Province | DS Count |
|---|---|
{chr(10).join(prov_rows)}
| **TOTAL** | **{total_ds}** |

---

## 6. Variable Descriptions

| Variable | Source Layer | Unit | Temporal Resolution | ERA5 Grid |
|---|---|---|---|---|
| `temperature_2m_mean` | ERA5 atmosphere | °C | Daily | ~9 km |
| `rain_sum` | ERA5 atmosphere | mm | Daily | ~9 km |
| `soil_moisture_7_to_28cm` | ERA5-Land | m³/m³ | Daily (mean of hourly) | ~9 km |
| `soil_moisture_28_to_100cm` | ERA5-Land | m³/m³ | Daily (mean of hourly) | ~9 km |
| `soil_moisture_100_to_255cm` | ERA5-Land | m³/m³ | Daily (mean of hourly) | ~9 km |

---

## 7. Data Quality Notes

- **ERA5 grid snapping**: Open-Meteo snaps each DS coordinate to the nearest ERA5 reanalysis
  grid cell (~9 km resolution). Multiple DS offices within the same grid cell will share
  identical values — this is expected and acceptable for the chosen modelling approach.
- **Soil moisture aggregation**: Hourly readings were averaged per day. Days with fewer
  than 24 hourly readings (e.g., at the range boundary) still have means computed from
  available readings.
- **Null values**: Any `null` entries in the API response (rare in ERA5) are preserved as
  empty cells in the CSV. The null rate per variable is reported in §3 above.
- **Resume capability**: The `data/progress.json` file tracks completed DS divisions.
  If any collection script is interrupted, simply re-run it and it will skip completed
  DS entries automatically.

---

## 8. Intended Downstream Use  (Group J2 Pipeline)

```
Open-Meteo CSVs
│
├── temperature_mean_2m.csv   ─┐
├── rain_sum.csv               ├──► Feature Engineering
└── soil_moisture_daily.csv   ─┘       │
                                        ▼
                               Flood / Drought Risk Model
                                        │
                              + IoT Sensor Feeds (real-time)
                                        │
                                        ▼
                              Real-Time Decision Pipeline
                                        │
                                        ▼
                              Resource Allocation Module
```

Variables feed into:
- **Flood risk model** — rain_sum (current + lag), soil_moisture_7_to_28cm
- **Drought risk model** — temperature_2m_mean, rain_sum deviation from mean,
  soil_moisture_28_to_100cm, soil_moisture_100_to_255cm
- **Spatial risk maps** — lat/lon of each DS used for geographic risk aggregation

---

## 9. Reproduction Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full collection (~40–60 min total)
python data_collection/fetch_all.py

# 3. Re-generate this report at any time
python data_collection/generate_report.py

# 4. Retry only failed steps
python data_collection/fetch_all.py --only temperature
python data_collection/fetch_all.py --only rain
python data_collection/fetch_all.py --only soil_moisture
```

---

*Report auto-generated by `generate_report.py` · Group J2 Disaster Response System*
"""
    return md


# ---------------------------------------------------------------------------
def main():
    log.info("Generating data collection report …")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = build_report()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"Report written to : {REPORT_PATH}")


if __name__ == "__main__":
    main()
