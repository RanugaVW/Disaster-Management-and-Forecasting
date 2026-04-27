# Open-Meteo Historical Data Collection Report
**Group J2 — Disaster Response System : Data & Intelligence**

---

*Generated on: 2026-04-22 23:11*

---

## 1. Overview

| Item | Value |
|---|---|
| Data Source | Open-Meteo Historical Archive API (ERA5 / ERA5-Land reanalysis) |
| API Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Date Range | **2015-01-01** → **2026-01-01** |
| Expected Days | 4,018 calendar days |
| DS Divisions Registered | 121 |
| Provinces Covered | 9 |
| Districts Covered | 21 |
| Report Generated | 2026-04-22 23:11 |

---

## 2. Output Files

| File | Description | Size |
|---|---|---|
| `data/temperature_mean_2m.csv` | Daily mean 2 m air temperature (°C) | 13.28 MB |
| `data/rain_sum.csv` | Daily total rainfall (mm) | 12.89 MB |
| `data/soil_moisture_daily.csv` | Daily mean soil moisture — 3 depth layers (m³/m³) | 20.17 MB |
| `data/progress.json` | Collection progress tracker (resume state) | 0.01 MB |

---

## 3. Collection Summary by CSV

### 3.1 Temperature — `temperature_mean_2m.csv`


| Metric | Value |
|---|---|
| Total rows | 486,299 |
| DS Coverage | `████████████████████` 121/121 DS (100.0%) |
| Date Range in data | 2015-01-01 → 2026-01-01 |

**Variable Statistics:**

| Variable | Min | Max | Mean | Nulls | Null % |
|---|---|---|---|---|---|
| `temperature_2m_mean` | 11.3 | 32.9 | 25.3399 | 0 | 0.0% |

**Missing DS divisions:**

_None — all DS divisions present_ ✅


---

### 3.2 Rain Sum — `rain_sum.csv`


| Metric | Value |
|---|---|
| Total rows | 486,299 |
| DS Coverage | `████████████████████` 121/121 DS (100.0%) |
| Date Range in data | 2015-01-01 → 2026-01-01 |

**Variable Statistics:**

| Variable | Min | Max | Mean | Nulls | Null % |
|---|---|---|---|---|---|
| `rain_sum` | 0.0 | 532.2 | 5.3633 | 0 | 0.0% |

**Missing DS divisions:**

_None — all DS divisions present_ ✅


---

### 3.3 Soil Moisture — `soil_moisture_daily.csv`


| Metric | Value |
|---|---|
| Total rows | 486,299 |
| DS Coverage | `████████████████████` 121/121 DS (100.0%) |
| Date Range in data | 2015-01-01 → 2026-01-01 |
| Aggregation | Hourly ERA5-Land values averaged → daily mean |

**Variable Statistics:**

| Variable | Min | Max | Mean | Nulls | Null % |
|---|---|---|---|---|---|
| `soil_moisture_7_to_28cm` | -0.0002 | 0.5185 | 0.3195 | 0 | 0.0% |
| `soil_moisture_28_to_100cm` | -0.0002 | 0.5195 | 0.3078 | 0 | 0.0% |
| `soil_moisture_100_to_255cm` | 0.125 | 0.5197 | 0.3607 | 0 | 0.0% |

**Missing DS divisions:**

_None — all DS divisions present_ ✅


---

## 4. DS Division Registry by District

| Province | District | DS Count | Collected Divisions |
|---|---|---|---|
| Eastern | Ampara | 2 | `Dehiattakandiya`, `Sammanthurai` |
| North Central | Anuradhapura | 15 | `Galenbidunuwewa`, `Horowpathana`, `Kahatagasdigiliya`, `Kebithigollewa`, `Kekirawa`, `Mahawilachchiya`, `Medawachchiya`, `Mihinthale`, `Nochchiyagama`, `Nuwaragam Palatha Central`, `Padaviya`, `Palagala`, `Rambewa`, `Thalawa`, `Thirappane` |
| Uva | Badulla | 9 | `Badulla`, `Kandeketiya`, `Lunugala`, `Mahiyanganaya`, `Meegahakiula`, `Passara`, `Rideemaliyadda`, `Soranathota`, `Welimada` |
| Eastern | Batticaloa | 3 | `Eravur Pattu`, `Koralai Pattu North`, `Koralai Pattu South` |
| Western | Gampaha | 1 | `Dompe` |
| Southern | Hambantota | 1 | `Ambalantota` |
| Northern | Jaffna | 2 | `Thenmaradchi (Chavakachcheri)`, `Vadamaradchchi East` |
| Central | Kandy | 17 | `Akurana`, `Deltota`, `Doluwa`, `Ganga Ihala Korale`, `Harispattuwa`, `Hatharaliyadda`, `Medadumbara`, `Minipe`, `Panvila`, `Pasbagekorale`, `Pathadumbara`, `Pathahewaheta`, `Poojapitiya`, `Udapalatha`, `Ududumbara`, `Udunuwara`, `Yatinuwara` |
| Sabaragamuwa | Kegalle | 6 | `Aranayake`, `Bulathkohipitiya`, `Mawanella`, `Rambukkana`, `Warakapola`, `Yatiyantota` |
| Northern | Kilinochchi | 4 | `Kandavalai`, `Karachchi`, `Pachchilaipalli`, `Poonakary` |
| North Western | Kurunegala | 10 | `Alawwa`, `Ehetuwewa`, `Galgamuwa`, `Giribawa`, `Ibbagamuwa`, `Mallawapitiya`, `Mawathagama`, `Polgahawela`, `Polpitigama`, `Rideegama` |
| Northern | Mannar | 4 | `Mannar Town`, `Manthai West`, `Musali`, `Nanaddan` |
| Central | Matale | 10 | `Ambanganga`, `Dambulla`, `Laggala`, `Matale`, `Naula`, `Pallepola`, `Rattota`, `Ukuwela`, `Wilgamuwa`, `Yatawatta` |
| Uva | Monaragala | 2 | `Bibile`, `Thanamalwila` |
| Northern | Mullaitivu | 3 | `Maritimepattu`, `Oddusuddan`, `Welioya` |
| Central | Nuwara Eliya | 9 | `Hanguranketa`, `Kothmale East`, `Kothmale West`, `Mathurata`, `Nildandahinna`, `Norwood`, `Nuwara Eliya`, `Thalawakele`, `Walapane` |
| North Central | Polonnaruwa | 6 | `Dimbulagala`, `Higurakgoda`, `Lankapura`, `Medirigiriya`, `Thamankaduwa`, `Welikanda` |
| North Western | Puttalam | 2 | `Karuwalagaswewa`, `Mundel` |
| Sabaragamuwa | Ratnapura | 3 | `Kolonna`, `Kuruvita`, `Ratnapura` |
| Eastern | Trincomalee | 8 | `Gomarankadawala`, `Kantale`, `Kinniya`, `Kuchchaweli`, `Morawewa`, `Muthur`, `Seruvila`, `Verugal` |
| Northern | Vavuniya | 4 | `Vavuniya`, `Vavuniya North`, `Vavuniya South`, `Vengalacheddikulam` |
| | **TOTAL** | **121** | |

---

## 5. DS Division Registry by Province

| Province | DS Count |
|---|---|
| Central | 36 |
| Eastern | 13 |
| North Central | 21 |
| North Western | 12 |
| Northern | 17 |
| Sabaragamuwa | 9 |
| Southern | 1 |
| Uva | 11 |
| Western | 1 |
| **TOTAL** | **121** |

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
