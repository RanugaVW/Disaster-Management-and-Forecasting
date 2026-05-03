# Probability Calculation Report

This report outlines how the continuous disaster probability (0.00 to 1.00) is calculated and appended to the respective training datasets for Floods, Landslides, and Droughts.

## The Approach
We are leveraging our pre-trained, heavily-tuned XGBoost Multi-Output Classifiers located in the `tuned_models/` directory. Even though these models were trained to output a discrete class (`Normal`, `Moderate`, `Severe`, `Extreme`), XGBoost intrinsically calculates the continuous probability for every single class before making its final decision.

### How the Probability is Calculated:
1. **Feature Generation**: The raw daily data (`rain_sum`, `soil_moisture`s, `temperature_2m_mean`, etc.) is loaded. The script groups the data by `division` and calculates the historical context: `rain_lag_1`, `rain_rolling_3d`, and `rain_rolling_7d` on the fly. It also encodes the date into `month_sin` and `month_cos`.
2. **Model Inference**: The data is passed to the respective model's `predict_proba()` function instead of the standard `predict()` function.
3. **Probability Extraction**: For each row, the model outputs 4 continuous probabilities corresponding to the 4 severity classes. 
4. **Final Risk Summation**: To get the continuous probability of a disaster happening (ignoring "Normal" conditions), we sum the probabilities of the danger classes:
   `Probability of Disaster = Probability(Moderate) + Probability(Severe) + Probability(Extreme)`
5. **Formatting**: This sum is a continuous float bounded between `0.0` and `1.0`. We round this value to exactly 2 decimal places (e.g., `0.74` for 74% likelihood) and store it in a new column named `{hazard}_probability` (e.g., `flood_probability`).

> [!NOTE]
> The multi-output models predict for Day+1, Day+2, and Day+3. To ensure the probability accurately reflects the immediate risk given the day's features, the scripts extract the probability specifically for the **Day+1** horizon (i.e., the probability of the disaster occurring tomorrow given today's data).

## Execution
Three independent scripts have been created in the `Method 2/` directory:
- `add_flood_prob.py`
- `add_landslide_prob.py`
- `add_drought_prob.py`

When run, they load their respective datasets from `Method 2/Data/`, compute the probabilities, append the column, and save the updated file back to disk.
