# Model Decisions and Architecture

This document outlines the reasoning, architectural decisions, and specific configurations used in creating the predictive models for the Disaster Management System J2.

## 1. Goal
Predict the severity (`Normal`, `Moderate`, `Severe`, `Extreme`) for three distinct hazards (**Flood**, **Landslide**, **Drought**) up to **3 days into the future** (Day+1, Day+2, Day+3), using meteorological and soil moisture data.

## 2. Selected Algorithm: XGBoost (eXtreme Gradient Boosting)
We selected **XGBoost** over Random Forest or other algorithms for the following reasons:
- **Performance and Speed:** XGBoost is highly optimized, handles missing values naturally, and is faster to train on tabular datasets than deep learning methods.
- **Production-Ready:** It provides native support for serialization (JSON/C++ compatible) and integrates well with almost all deployment frameworks.
- **Non-Linear Relationships:** Weather interactions (e.g., how soil moisture interacts with rolling rainfall to cause landslides) are highly non-linear. Tree-based ensemble models excel at capturing these thresholds automatically without requiring extensive feature scaling.

## 3. Modeling Strategy: Multi-Output Classification
Instead of training 9 separate models (3 hazards $\times$ 3 time horizons), we opted for **3 models** (one for each hazard), wrapped in a `MultiOutputClassifier`.
- **Why?** A Multi-Output model predicts the Day+1, Day+2, and Day+3 severities simultaneously. This simplifies the deployment architecture, reduces the memory footprint, and ensures that the predictions for the three horizons are generated from the exact same feature state.

## 4. Feature Selection
The raw dataset contains several features. We selected the following as inputs to our model:
- `rain_sum`: Immediate daily rainfall (critical for flash floods and immediate landslide risk).
- `temperature_2m_mean`: Influences evaporation and drought conditions.
- `soil_moisture_7_to_28cm`, `soil_moisture_28_to_100cm`, `soil_moisture_100_to_255cm`: Crucial for determining ground saturation. High saturation + high rain = Flood/Landslide. Low saturation + lack of rain = Drought.
- `rain_lag_1`, `rain_rolling_3d`, `rain_rolling_7d`: Accumulated rainfall captures the *history* of the weather, which is the primary driver for delayed disasters (like riverine floods).
- `month_sin`, `month_cos`: Cyclical encoding of the month to give the model a sense of seasonality (monsoon vs. dry season).
- `spi`: The Standardized Precipitation Index provides a normalized view of rainfall anomaly, serving as a unified proxy for drought (negative SPI) and flood risk (positive SPI).
- `division`: Label-encoded to allow the model to learn geographic-specific vulnerabilities (e.g., hill country vs. coastal plains).

## 5. Handling Target Horizons (Data Shifting)
To predict the future, we grouped the data by `division` and shifted the severity labels *backward* in time using Pandas `shift(-1)`, `shift(-2)`, and `shift(-3)`. 
- **Decision:** This means the features for *today* are aligned with the labels of *tomorrow*, *day after tomorrow*, and *three days from now*.
- **Caveat:** The last 3 days of data for each division must be dropped from training since we cannot observe their future labels.

## 6. Addressing Class Imbalance
Disasters are inherently rare events. The distribution of severities in our dataset is heavily skewed towards `Normal`. 
- **Decision:** We address this using the `sample_weight` parameter during XGBoost training. We calculate class weights inversely proportional to class frequencies using `sklearn.utils.class_weight.compute_sample_weight`. This penalizes the model heavily for misclassifying rare `Extreme` or `Severe` events, forcing it to pay attention to disaster conditions rather than lazily predicting `Normal` all the time.

## 7. Hyperparameter Tuning (Optuna)
Given the non-linear complexity of meteorological disaster prediction, we implemented **Optuna** for automated hyperparameter tuning.
- **Why Optuna?** Optuna uses Tree-structured Parzen Estimators (TPE) to intelligently search the hyperparameter space (learning rate, max depth, subsampling), which is vastly more efficient than exhaustive Grid Search or random Random Search.
- **Objective Function:** We tune the hyperparameters to maximize the exact match accuracy on a temporal validation set (the subset immediately preceding the final test set).
- **Practical Constraint:** The script is configured to run a small number of trials (`n_trials=5`) by default to ensure training completes in a reasonable time. In a full production run, this can be scaled up to 50-100 trials.

## 8. Model Evaluation & Saving
- The dataset is split temporally (the last $N$ rows act as the test set).
- The models are saved as standard `joblib` pickle files (`.pkl`) in the `models/` directory for seamless loading during the deployment phase.
