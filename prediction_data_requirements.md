# Prediction Data Requirements

To predict the severity level for Floods, Landslides, and Droughts over a 3-day horizon, the XGBoost models require specific meteorological and geographical data. 

While the machine learning model itself requires **12 specific features**, your live application (e.g., connected to the Open-Meteo API) only needs to fetch **7 raw data points**. The remaining features are automatically calculated by the system.

---

## 1. Raw Data Required (From Open-Meteo API)
For any given day and location you wish to predict, you must provide the following raw values:

| Data Point | Type | Description |
| :--- | :--- | :--- |
| **`date`** | String/Date | The current date (Format: `YYYY-MM-DD`). |
| **`division`** | String | The Sri Lankan Divisional Secretariat (e.g., "Colombo", "Akurana"). |
| **`rain_sum`** | Float | Total daily rainfall in millimeters (mm). |
| **`temperature_2m_mean`** | Float | Average daily temperature at 2 meters above ground (°C). |
| **`soil_moisture_7_to_28cm`** | Float | Volumetric soil moisture at shallow depth ($m^3/m^3$). |
| **`soil_moisture_28_to_100cm`** | Float | Volumetric soil moisture at mid depth ($m^3/m^3$). |
| **`soil_moisture_100_to_255cm`** | Float | Volumetric soil moisture at deep depth ($m^3/m^3$). |

> [!IMPORTANT]
> **Historical Rainfall Context Needed**
> To accurately predict disasters (especially floods and landslides), the model relies heavily on ground saturation from previous days. Therefore, your application must keep track of the `rain_sum` for the **previous 7 days** for that specific `division`. 

---

## 2. Computed Features (Calculated on the Fly)
Using the raw data above, your inference script will automatically compute the following variables before passing the data to the models:

1. **`rain_lag_1`**: Yesterday's total rainfall.
2. **`rain_rolling_3d`**: The sum of rainfall over the last 3 days.
3. **`rain_rolling_7d`**: The sum of rainfall over the last 7 days.
4. **`spi`**: The 1-Day Standardized Precipitation Index. (Calculated by fitting a Gamma distribution to the daily rain using historical baseline data).
5. **`month_sin` / `month_cos`**: The `date` is converted into its integer month (1-12) and passed through sine and cosine functions. This tells the model what season it is (e.g., Monsoon season vs. Dry season) without hardcoding months.
6. **`division_encoded`**: The text name of the `division` is converted into a unique integer ID using the saved `_division_encoder.pkl` file.

---

## 3. Final Model Input Array
When all computations are finished, the exact array passed into the `model.predict()` function will look exactly like this, in this specific order:

```python
[
    'rain_sum', 
    'temperature_2m_mean', 
    'soil_moisture_7_to_28cm', 
    'soil_moisture_28_to_100cm', 
    'soil_moisture_100_to_255cm',
    'rain_lag_1', 
    'rain_rolling_3d', 
    'rain_rolling_7d',
    'month_sin', 
    'month_cos', 
    'spi',
    'division_encoded'
]
```

By providing these 12 data points, all three disaster models (Flood, Landslide, Drought) will successfully output the severity predictions for Day+1, Day+2, and Day+3.
