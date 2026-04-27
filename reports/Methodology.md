# Methodology: Feature Engineering & Modeling Strategy

**Group J2 — Disaster Response System : Data & Intelligence**

---

This document outlines the core methodological decisions and steps taken regarding feature engineering, data integration, and model architecture for predicting Floods, Droughts, and Landslides in Sri Lanka.

## 1. Data Integration

### Merging on Date + Division
**Decision:** All meteorological data (`temperature_mean_2m.csv`, `rain_sum.csv`, `soil_moisture_daily.csv`) will be merged using `Date` and `Division` as the primary keys.
**Rationale:** This creates a proper spatio-temporal panel dataset. Each row will uniquely identify the environmental and weather state at a specific geographical location (DS Division) on a specific day. This structured representation natively supports time-series models while segmenting geographic locations properly.

## 2. Feature Engineering

### Standardized Precipitation Index (SPI)
**Decision:** We will compute the SPI for 30, 90, and 180-day rolling windows using `scipy.stats.gamma`.

**Mathematical Formulation:** The SPI computation normalizes rolling cumulative rainfall into standard deviations. Because raw precipitation data is highly skewed and frequently contains zero values, it cannot be modeled using a standard normal curve inherently. 
1. We compute the probability of zero rainfall, denoted as $q$, over the given window.
2. We isolate the non-zero rainfall values and fit a **Gamma Distribution** to them, determining the shape ($\alpha$) and scale ($\beta$) parameters. The cumulative distribution function for this gamma fit is $G(x)$.
3. We calculate the mixed cumulative probability $H(x)$ for all values, accounting for the zero-value probability:
   $$H(x) = q + (1 - q) \cdot G(x)$$
4. Finally, the SPI is derived by transforming $H(x)$ through the inverse standard normal cumulative distribution function (the probit function, $\Phi^{-1}$):
   $$SPI = \Phi^{-1}(H(x))$$

**Rationale:** The World Meteorological Organization (WMO) officially strictly requires gamma distributions on rolling rainfall arrays to define drought indices. Computing 30/90/180-day SPI covers short-term agricultural drought up to long-term hydrological drought.
Crucially, by running the data through $\Phi^{-1}$, the final SPI values are expressed entirely in **Standard Deviations**. This allows our machine learning system to perfectly cross-reference standard WMO alert tables without unit mismatches:
- $SPI \le -2.0$ maps exactly to Extreme Drought (2 standard deviations below average).
- $SPI \ge 2.0$ maps exactly to Critical Landslide/Flood Risk (2 standard deviations above average saturation).
Implementing this custom logic via `scipy.stats.gamma` is mathematically robust and perfectly aligns Open-Meteo's raw millimeter readings to global disaster warning metrics.
**Time-Series Edge Effects (Missing Initial Values):** Because SPI relies on cumulative rolling windows, the algorithm inherently requires a warm-up period. For instance, computing a 30-day SPI mandates 30 days of prior rainfall history. Consequently, the first 29 days of the dataset (e.g., January 2015) will intrinsically output empty cells (`NaN`). This structural requirement scales proportionally—the `spi_180d` column will exclusively contain empty cells until exactly 180 consecutive days of boundary context have passed (late June 2015). This behavior occurs by design; it is a fundamental property of rolling time-series calculations rather than a data processing error.

### Lags & Rolling Sums
**Decision:** The dataset will include lag features such as yesterday's rainfall, as well as rolling cumulative sums (e.g., 3-day and 7-day rolling rain sums).
**Rationale:** These features are essential predictors for floods and landslides because those events occur precisely when antecedent soil moisture crosses a critical saturation frontier. Cumulative sums map explicitly to that risk frontier.

### Seasonal Encoding
**Decision:** Cyclical seasonal features will be created using Sine and Cosine transforms of the month number (`sin(month)`, `cos(month)`).
**Rationale:** Sri Lanka has very explicit, bimodal monsoon patterns (Yala and Maha monsoons). Mathematical transforms using Sin/Cos allow December (month 12) and January (month 1) to be recognized by the model as adjacent seasons sequentially. Standard integer scaling (1 to 12) fails because the numerical distance between 12 and 1 is maximal, confusing the algorithm about December-January seasonality.

## 3. Modeling Architecture

### 3 Separate Classifiers vs 1 Unified Multi-Class Model
**Decision:** We will train three separate, specialized binary classifiers (Flood, Drought, Landslide) instead of a single multi-class/multi-label architecture.

**Rationale:** The three disasters have fundamentally different "trigger logic." If you put all three into one model, it has to learn contradictory patterns simultaneously — e.g., high rainfall dramatically increases *flood risk* but entirely resets *drought risk*. That creates conflicting gradients during backpropagation and the model will severely underperform on all targets.

| | **Flood** | **Drought** | **Landslide** |
|---|---|---|---|
| **Time scale** | Hours-days | Weeks-months | Hours (after saturation) |
| **Key signal** | Rain spike | Rain deficit | Rain + soil wetness combo |
| **SPI window** | Not needed | SPI-1, 3, 6 | SPI as soil proxy |
| **Temperature role** | Minimal | Critical | Minimal |
| **Label imbalance** | Moderate | Severe | Moderate |

By separating them into independent models, each algorithm is parameterized explicitly to find its specific trigger dynamic without contradictory noise.

### Class Imbalance Handling
**Decision:** We will use algorithmic class weighting methods (e.g., `scale_pos_weight` in XGBoost/LightGBM or `param class_weight='balanced'`) instead of SMOTE.
**Rationale:** The vast majority of our historical data points will represent a "NONE" disaster state (often >99%), causing severe class imbalance. SMOTE must be treated with extreme caution on time-series—if interpolated synthetic points bleed information across folds, it causes catastrophic data leakage and wildly over-optimistic precision metrics. Standard class weights handle zero-states efficiently by heavily penalizing false negatives natively within the loss function without altering the temporal continuity of the data.

## 4. Forecasting Strategy: The Lagging Indicator Method
**Decision:** We finalized the Lagging Indicator Methodology (Approach A). The system utilizes 100% factual, observed ground-truth weather from *today* (features at time $t$) to predict disaster risk labels for *tomorrow* (labels at time $t+1$), completely bypassing the reliance on weather forecasts.

**Rationale:**
1. **No Compounding Errors:** Weather forecasts are inherently susceptible to inaccuracies. Feeding a flawed rainfall forecast into a machine learning model forces it to produce a flawed disaster alert. By relying strictly on recorded actuals from today to forecast tomorrow, we eliminate forecast unreliability entirely.
2. **Droughts are Slow:** Droughts take weeks or months to develop. The 180-day SPI up until today is the definitive measuring stick for drought risk tomorrow. A single weather forecast for tomorrow has practically zero impact on drought tracking mathematically.
3. **The "Sponge Effect":** A common misconception is that heavy rain on a single day directly causes a flood. In reality, floods and landslides typically occur when the ground is already hyper-saturated like a wet sponge. The lag features and rolling sums explicitly track this antecedent "sponge effect" from today, forming the most reliable baseline indicator for tomorrow's structural risk.
4. **Massive Architectural Simplicity:** The data pipeline remains drastically simpler. The production system only needs to ping the Open-Meteo API once at midnight for the daily actuals, execute the model, and alert users for the subsequent 24 hours without juggling complex short-term Forecast API integrations.

*Implementation Note:* To execute this securely, during the data labeling and training phase, the historical Target Label timeline will be explicitly shifted backwards by one day.

## 5. Current State & Next Steps
- **Data Labeling:** The ground truth targets (`HIGH`, `MODERATE`, `LOW`, `NONE`) are not yet labeled or merged into the initial feature matrix. Strict labeling constraints are pending and will be executed in a distinct phase.
