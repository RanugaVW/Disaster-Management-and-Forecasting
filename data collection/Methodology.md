# Methodology: Feature Engineering & Modeling Strategy

**Group J2 — Disaster Response System : Data & Intelligence**

---

This document outlines the core methodological decisions and steps taken regarding feature engineering, data integration, and model architecture for predicting Floods, Droughts, and Landslides in Sri Lanka.

## 1. Data Integration

### Merging on Date + Division
**Decision:** All meteorological data (`temperature_mean_2m.csv`, `rain_sum.csv`, `soil_moisture_daily.csv`) are merged using `Date` and `Division` as the primary keys, producing a single unified master feature matrix.

**Rationale:** This creates a proper spatio-temporal panel dataset. Each row uniquely identifies the environmental and weather state at a specific geographical location (DS Division) on a specific day. This structured representation natively supports time-series models while segmenting geographic locations properly.

**Current Implementation:**
- Source: **Open-Meteo Historical Climate API**
- Coverage: **121 Sri Lankan Divisional Secretariats**, spanning **2015-01-01 to 2025-12-31**
- Training split: **2015–2023** (337,760 rows per hazard)
- Test split: **2024–2025** (88,209 rows — never seen during training)
- Severity labels: **4 ordinal classes** — `Normal`, `Moderate`, `Severe`, `Extreme`

---

## 2. Feature Engineering

### Standardized Precipitation Index (SPI)

**Decision:** We compute the SPI on a **daily rolling window** (1-day granularity per division) using `scipy.stats.gamma`, fitted per division on the training window.

**Mathematical Formulation:** The SPI computation normalizes rolling cumulative rainfall into standard deviations. Because raw precipitation data is highly skewed and frequently contains zero values, it cannot be modeled using a standard normal curve inherently.
1. We compute the probability of zero rainfall, denoted as $q$, over the given window.
2. We isolate the non-zero rainfall values and fit a **Gamma Distribution** to them, determining the shape ($\alpha$) and scale ($\beta$) parameters. The cumulative distribution function for this gamma fit is $G(x)$.
3. We calculate the mixed cumulative probability $H(x)$ for all values, accounting for the zero-value probability:
   $$H(x) = q + (1 - q) \cdot G(x)$$
4. Finally, the SPI is derived by transforming $H(x)$ through the inverse standard normal cumulative distribution function (the probit function, $\Phi^{-1}$):
   $$SPI = \Phi^{-1}(H(x))$$

**Rationale:** The World Meteorological Organization (WMO) officially requires gamma distributions on rolling rainfall arrays to define drought indices. By running data through $\Phi^{-1}$, the final SPI values are expressed entirely in **Standard Deviations**, allowing the system to cross-reference standard WMO alert tables without unit mismatches:
- $SPI \le -2.0$ maps exactly to Extreme Drought (2 standard deviations below average).
- $SPI \ge 2.0$ maps exactly to Critical Landslide/Flood Risk (2 standard deviations above average saturation).

**Time-Series Edge Effects (Missing Initial Values):** Because SPI relies on cumulative rolling windows, the algorithm inherently requires a warm-up period. The first rows of each division will output `NaN` until sufficient history is accumulated. These rows are dropped before training via `dropna()`.

---

### Lags & Rolling Sums

**Decision:** The dataset includes lag features and rolling cumulative sums computed **per division** (within-group shifts), preventing cross-division leakage.

| Feature | Computation |
|---|---|
| `rain_lag_1` | `rain_sum` shifted back 1 day within each division |
| `rain_rolling_3d` | Sum of `rain_sum` over the past 3 days per division |
| `rain_rolling_7d` | Sum of `rain_sum` over the past 7 days per division |

**Rationale:** These features are essential predictors for floods and landslides because those events occur precisely when antecedent soil moisture crosses a critical saturation frontier. Cumulative sums map explicitly to that risk frontier. The "Sponge Effect" — where floods occur when the ground is already hyper-saturated — is captured through these rolling windows rather than only looking at single-day rainfall spikes.

---

### Seasonal Encoding

**Decision:** Cyclical seasonal features are created using Sine and Cosine transforms of the month number.

$$month\_sin = \sin\!\left(\frac{2\pi \times month}{12}\right), \quad month\_cos = \cos\!\left(\frac{2\pi \times month}{12}\right)$$

**Rationale:** Sri Lanka has very explicit, bimodal monsoon patterns (Yala and Maha monsoons). This mathematical transform allows December (month 12) and January (month 1) to be recognized by the model as adjacent seasons. Standard integer encoding (1–12) fails because the numerical distance between 12 and 1 is maximal, confusing the algorithm about December-January seasonality.

---

### Division Encoding

**Decision:** Division names are encoded as integers using a `LabelEncoder` saved as `master_division_encoder.pkl`. The same encoder is used at inference time to ensure consistency.

**Rationale:** Tree-based models (XGBoost, LightGBM) cannot consume string inputs. The division integer provides the model with spatial identity to learn location-specific disaster patterns (e.g., coastal vs. hill-country divisions have different flood thresholds).

---

### Final Feature Array (12 features, exact order for model input)

```python
FEATURES = [
    'rain_sum', 'temperature_2m_mean',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi', 'division_encoded'
]
```

---

## 3. Modeling Architecture

### 3 Separate Classifiers (Multi-Hazard Independence)

**Decision:** We train three **separate, specialized multi-class classifiers** — one each for Flood, Drought, and Landslide — instead of a single unified multi-label architecture.

**Rationale:** The three disasters have fundamentally different trigger logic. A single model would face conflicting gradients — e.g., high rainfall dramatically increases *flood risk* but entirely resets *drought risk*. Independent models allow each to be parameterized and optimized for its specific disaster dynamics:

| | **Flood** | **Drought** | **Landslide** |
|---|---|---|---|
| **Time scale** | Hours–days | Weeks–months | Hours (after saturation) |
| **Key signal** | Rain spike | Rain deficit | Rain + soil wetness combo |
| **SPI role** | Flood saturation indicator | Drought deficit indicator | Soil proxy |
| **Temperature role** | Minimal | Critical | Minimal |
| **Label imbalance** | Moderate | Severe | Moderate |

---

### Multi-Class Severity Labeling

**Decision:** Each hazard is labeled as one of 4 ordinal severity classes:

| Class | Label | Meaning |
|---|---|---|
| 0 | `Normal` | No significant disaster risk |
| 1 | `Moderate` | Minor risk — localised impact possible |
| 2 | `Severe` | High risk — widespread impact expected |
| 3 | `Extreme` | Emergency level — immediate response needed |

**Rationale:** A 4-class ordinal system allows the model to distinguish between alert levels for emergency response staging, rather than a binary yes/no trigger. Ordinal structure is preserved via the **Quadratic Weighted Kappa (QWK)** metric, which penalizes predictions in proportion to how far they are from the true class.

---

### Forecasting Horizon: 3-Day Forward Shift

**Decision:** We finalized the **Lagging Indicator Methodology**. The system uses 100% factual, observed ground-truth weather from *today* (features at time $t$) to predict disaster severity labels for *tomorrow*, *2 days ahead*, and *3 days ahead* (labels at $t+1$, $t+2$, $t+3$).

This is implemented by forward-shifting the target column per division:

```python
df = df.sort_values(['division', 'date'])
df['target_d1'] = df.groupby('division')['severity_col'].shift(-1)  # tomorrow
df['target_d2'] = df.groupby('division')['severity_col'].shift(-2)  # 2 days ahead
df['target_d3'] = df.groupby('division')['severity_col'].shift(-3)  # 3 days ahead
```

The last 3 rows of each division are dropped (no future labels available). Each model therefore simultaneously predicts three values per inference call.

**Rationale:**
1. **No Compounding Errors:** Weather forecasts are inherently susceptible to inaccuracies. Feeding a flawed rainfall forecast into the model forces it to produce a flawed disaster alert. Relying strictly on recorded actuals from today eliminates forecast unreliability entirely.
2. **Droughts are Slow:** The rolling SPI up until today is the definitive measuring stick for drought risk tomorrow. A single weather forecast for tomorrow has practically zero impact on drought tracking.
3. **The "Sponge Effect":** Floods and landslides occur when the ground is already hyper-saturated. Lag features and rolling sums capture this antecedent saturation from today.
4. **Architectural Simplicity:** The production system only needs to ping the Open-Meteo API once at midnight for daily actuals, execute the model, and alert users — without complex short-term forecast API integrations.

---

## 4. Ensemble Architecture: Soft-Voting

### Individual Algorithms

**XGBoost:**
- Level-wise gradient boosted trees; industry standard for tabular data.
- Regularisation via `reg_alpha`, `reg_lambda`, `gamma` prevents overfitting on rare event classes.
- `hist` tree method for fast training on 300k+ row datasets.

**LightGBM:**
- Leaf-wise growth finds complex non-linear patterns ~3× faster than XGBoost.
- Built-in `class_weight='balanced'` for additional imbalance protection.
- Excellent at capturing granular weather-disaster correlations.

### Soft-Voting Ensemble (Production Model)

**Decision:** The production model is a `SoftVotingEnsemble` that averages the probability arrays of both XGBoost and LightGBM sub-models. For each day horizon $d \in \{1, 2, 3\}$:

$$P_{ensemble}(c \mid \mathbf{x}, d) = \frac{P_{XGB}(c \mid \mathbf{x}, d) + P_{LGB}(c \mid \mathbf{x}, d)}{2}$$

The final predicted class is then:

$$\hat{y}_d = \underset{c \in \{0,1,2,3\}}{\arg\max} \; P_{ensemble}(c \mid \mathbf{x}, d)$$

**Rationale — Why soft voting over hard voting:**

| Method | How it works | Advantage |
|---|---|---|
| Hard voting | Each model votes for a class; majority wins | Simple |
| **Soft voting** | Average the probability arrays; take argmax | Preserves uncertainty; reduces overconfidence |

By averaging probabilities instead of class labels, the ensemble benefits from both models' confidence estimates, damping individual model overconfidence and reducing prediction variance.

---

## 5. Class Imbalance Strategy

Disaster severity is heavily skewed towards `Normal` (~75–85% of rows). We applied a multi-layer defence:

| Technique | Effect |
|---|---|
| `compute_sample_weight('balanced')` on XGBoost | Penalises misclassifying rare Extreme/Severe events |
| `class_weight='balanced'` in LightGBM | Additional internal balancing |
| Temporal validation split (last 15% of training) | No future leakage; realistic evaluation |
| Optuna objective on Day+1 Accuracy | Directly optimises the primary forecast horizon |

> **Why NOT SMOTE?** SMOTE generates synthetic rows by interpolating between samples. For time-series climate data this breaks temporal continuity and causes data leakage across the temporal fold boundary, producing wildly over-optimistic metrics.

---

## 6. Hyperparameter Optimization (Optuna TPE)

**Decision:** Hyperparameters are optimized using **Optuna with Tree-structured Parzen Estimators (TPE)** — far more efficient than grid search or random search for high-dimensional parameter spaces.

| Parameter | Search Range (XGBoost) | Best Found |
|---|---|---|
| `n_estimators` | 100–500 | 250 |
| `max_depth` | 3–10 | 10 |
| `learning_rate` | 0.01–0.30 | 0.1206 |
| `subsample` | 0.50–1.00 | 0.7993 |
| `colsample_bytree` | 0.50–1.00 | 0.578 |
| `reg_alpha` | 0–2 | 1.7324 |
| `reg_lambda` | 0.5–5 | 3.205 |

| Parameter | Search Range (LightGBM) | Best Found |
|---|---|---|
| `n_estimators` | 100–500 | 126 |
| `num_leaves` | 20–150 | 144 |
| `learning_rate` | 0.01–0.30 | 0.2669 |
| `min_child_samples` | 5–100 | 14 |
| `reg_alpha` | 0–2 | 1.3685 |
| `reg_lambda` | 0–5 | 2.2008 |

---

## 7. Evaluation Metrics

| Metric | Why Use It | Limitation |
|---|---|---|
| **Accuracy** | Simple % of correct predictions | Misleading with skewed data (80% by always predicting Normal) |
| **F1-Macro** | Equal weight to all classes including rare Extreme | Primary metric for disaster models |
| **F1-Weighted** | Weighted by class frequency; realistic overall score | Favours dominant Normal class |
| **QWK (Quadratic Weighted Kappa)** | Penalises predictions by ordinal distance from truth | Best for ranked severity classes |
| **Exact Match** | All 3 future days must be exactly correct simultaneously | Strictest end-to-end metric |

> **Recommended primary metric:** F1-Macro on Day+1 (most actionable horizon).
> **Recommended secondary:** QWK on Day+1 (ordinal error awareness).

---

## 8. Population-Weighted Risk Score

The final output to the disaster management dashboard combines model probability with the at-risk population:

$$\text{Risk Score} = P(\text{Severe or Extreme} \mid Day+1) \times \text{Division Population}$$

Where:

$$P(\text{Severe or Extreme}) = P_{ensemble}(c=2) + P_{ensemble}(c=3)$$

and Division Population is loaded from `division_population_map.csv`. Divisions are ranked by Risk Score to prioritize emergency response resource allocation.

---

## 9. Current State & Validation Results

The models have been fully trained, validated, and evaluated on held-out test data:

| Hazard | Ensemble D+1 Accuracy (Val) | Ensemble D+1 Accuracy (Test) |
|---|---|---|
| Flood | 89.46% | 88.11% |
| Landslide | 89.00% | 87.91% |
| Drought | 93.87% | 94.57% |

All 9 models (XGBoost × 3, LightGBM × 3, Ensemble × 3) are serialized as `.pkl` files in `Dataset Separation/models/` and are ready for integration into the production inference pipeline.

> **Deployment Recommendation:** Use the `*_ensemble.pkl` models for production. Soft-voting between XGBoost and LightGBM reduces prediction variance and prevents a single model's overconfidence from triggering false alarms.
