# Disaster Severity Prediction Models — Implementation Plan

## Background

The `master_feature_matrix.csv` contains **486,299 rows** across **121 Sri Lankan Divisional Secretariats** from 2015-01-01 to 2026-01-01, with 15 columns:

| Feature | Description |
|---|---|
| `rain_sum` | Daily rainfall (mm) |
| `temperature_2m_mean` | Mean daily temperature (°C) |
| `soil_moisture_7_to_28cm` | Shallow soil moisture |
| `soil_moisture_28_to_100cm` | Mid soil moisture |
| `soil_moisture_100_to_255cm` | Deep soil moisture |
| `rain_lag_1` | Yesterday's rainfall |
| `rain_rolling_3d` | 3-day rolling rainfall |
| `rain_rolling_7d` | 7-day rolling rainfall |
| `month_sin`, `month_cos` | Cyclical seasonal encoding |
| `spi_30d`, `spi_90d`, `spi_180d` | Standardized Precipitation Index (short/mid/long) |

---

## Phase 1 — Rule-Based Severity Labeling

Three new label columns will be created: `flood_label`, `drought_label`, `landslide_label`.
All labels use a 4-class ordinal scale: `0=NORMAL`, `1=MODERATE`, `2=SEVERE`, `3=EXTREME`.

### 1.1 Flood Labeling

Flood risk is driven by **acute rainfall accumulation** and **high soil moisture** (saturation leaves no absorption capacity).

| Label | Condition |
|---|---|
| EXTREME | `rain_rolling_7d > 300` OR (`rain_rolling_7d > 200` AND `soil_moisture_7_to_28cm > 0.45`) |
| SEVERE  | `rain_rolling_7d > 150` OR (`rain_rolling_7d > 100` AND `soil_moisture_7_to_28cm > 0.40`) |
| MODERATE | `rain_rolling_7d > 70` OR `rain_sum > 50` |
| NORMAL | Everything else |

> **Justification**: Sri Lanka's flood thresholds are aligned with IMD/DMC guidelines; 7-day accumulated rainfall is the primary operational trigger. Soil saturation compounds risk significantly.

### 1.2 Drought Labeling

Drought is driven by **SPI** (multi-timescale) — negative SPI = drier than normal.

| Label | Condition |
|---|---|
| EXTREME | `spi_30d < -2.0` AND `spi_90d < -1.5` |
| SEVERE  | `spi_30d < -1.5` OR `spi_90d < -1.5` |
| MODERATE | `spi_30d < -1.0` OR `spi_90d < -1.0` |
| NORMAL | Everything else |

> **Justification**: WMO's official SPI drought classification: Moderate (-1.0 to -1.49), Severe (-1.5 to -1.99), Extreme (< -2.0). Multi-scale agreement reduces false positives.

### 1.3 Landslide Labeling

Landslide risk is triggered by **rainfall intensity + soil moisture saturation in deeper layers** (pre-wetted slopes).

| Label | Condition |
|---|---|
| EXTREME | `rain_sum > 100` AND `soil_moisture_28_to_100cm > 0.45` |
| SEVERE  | `rain_sum > 60` AND `soil_moisture_28_to_100cm > 0.38` OR `rain_rolling_3d > 150` |
| MODERATE | `rain_sum > 30` OR `rain_rolling_3d > 80` |
| NORMAL | Everything else |

> **Justification**: NBRO (National Building Research Organisation) Sri Lanka triggers landslide warnings at >75mm/day rainfall. Deeper soil moisture (`28-100cm`) is crucial as it indicates slope pre-saturation.

---

## Phase 2 — Feature Engineering for 3-Day Ahead Prediction

To predict severity **3 days into the future**, we will use **future-shifted labels** as targets:

```
target_t+1 = label shifted back by 1 day per division
target_t+2 = label shifted back by 2 days per division
target_t+3 = label shifted back by 3 days per division
```

Or equivalently: the model learns **"given today's features → what will severity be in d days?"**

We will train **3 separate models per hazard**, one per horizon (day+1, day+2, day+3), giving **9 total models**, OR use a **multi-output classifier** predicting all 3 days simultaneously (preferred — fewer models, shared representations).

**Preferred approach: Multi-Output Classifier** (one model per hazard, 3 outputs).

### 2.1 Feature Set Per Model

All models share the same 13 input features (all numeric columns). Division will be **label-encoded** as a feature to capture regional patterns.

---

## Phase 3 — Model Architecture

### Algorithm: Random Forest + XGBoost (Ensemble)

- **Primary**: `XGBClassifier` with `multi:softmax` objective
- **Multi-output wrapper**: `sklearn.multioutput.MultiOutputClassifier`
- **Class imbalance**: `scale_pos_weight` / `class_weight='balanced'`
- **Train/Test split**: Temporal — train on 2015–2023, validate on 2024, test on 2025+

### Per-Hazard Model Files

| Model | Output File |
|---|---|
| Flood | `models/flood_severity_model.pkl` |
| Drought | `models/drought_severity_model.pkl` |
| Landslide | `models/landslide_severity_model.pkl` |

---

## Phase 4 — Scripts to Create

```
models/
  01_label_data.py          ← Rule-based labeling → saves labeled CSV
  02_train_flood.py         ← Train flood model
  02_train_drought.py       ← Train drought model
  02_train_landslide.py     ← Train landslide model
  03_predict.py             ← Load all 3 models, predict next 3 days
  flood_severity_model.pkl
  drought_severity_model.pkl
  landslide_severity_model.pkl
data/
  labeled_master.csv        ← master + 3 label columns
```

---

## Open Questions

> [!IMPORTANT]
> **Do you have any historical disaster occurrence records** (actual flood/drought/landslide events with dates and locations)? If yes, we can validate/calibrate the labeling thresholds against real events to make them more accurate for Sri Lanka.

> [!IMPORTANT]
> **Model preference**: Should we use:
> - **Option A**: XGBoost multi-output (fast, production-ready) ✅ *Recommended*
> - **Option B**: Random Forest multi-output (more interpretable)
> - **Option C**: Both in an ensemble

> [!NOTE]
> **3-day prediction strategy**: We will train separate outputs for Day+1, Day+2, Day+3 within a single model. This means each model predicts 3 severity values at once. Is this the desired behavior?

---

## Verification Plan

1. Run `01_label_data.py` → inspect class distribution (expect heavy `NORMAL` class imbalance)
2. Train each model → check validation accuracy, F1 per class
3. Run `03_predict.py` on a sample division → verify 3-day outputs
4. Cross-check a known historical flood event (e.g., Kalutara 2017 floods) against labels
