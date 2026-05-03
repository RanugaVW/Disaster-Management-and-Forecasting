import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def main():
    model_name = 'Flood'
    dataset_path = r'Method 2\Data\training_data_Flood.csv'
    
    print(f"Loading {model_name} model and encoder...")
    # Go up one directory to access tuned_models/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'tuned_models', f'{model_name}_xgboost.pkl')
    encoder_path = os.path.join(base_dir, 'tuned_models', f'{model_name}_division_encoder.pkl')
    
    model = joblib.load(model_path)
    le_div = joblib.load(encoder_path)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(os.path.join(base_dir, dataset_path))
    
    # Feature Engineering
    print("Computing features on the fly...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['division', 'date']).reset_index(drop=True)
    df['rain_lag_1'] = df.groupby('division')['rain_sum'].shift(1)
    df['rain_rolling_3d'] = df.groupby('division')['rain_sum'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
    df['rain_rolling_7d'] = df.groupby('division')['rain_sum'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Encode division safely
    known_divs = set(le_div.classes_)
    valid_mask = df['division'].isin(known_divs)
    df.loc[valid_mask, 'division_encoded'] = le_div.transform(df.loc[valid_mask, 'division'])
    
    FEATURES = [
        'rain_sum', 'temperature_2m_mean', 
        'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
        'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
        'month_sin', 'month_cos', 'spi', 'division_encoded'
    ]
    
    # We can only predict on rows that have valid lag data and division encoding
    predict_mask = df[FEATURES].notna().all(axis=1)
    X = df.loc[predict_mask, FEATURES]
    
    print("Predicting continuous probabilities...")
    # predict_proba returns [Day1_probs, Day2_probs, Day3_probs]
    # We extract Day1_probs: shape (n_samples, 4)
    probs_all_days = model.predict_proba(X)
    day1_probs = probs_all_days[0]
    
    # Probability of any flood (Moderate + Severe + Extreme) = classes 1, 2, 3
    # day1_probs[:, 1:] sums the columns for Moderate(1), Severe(2), Extreme(3)
    flood_prob = np.sum(day1_probs[:, 1:], axis=1)
    
    # Assign rounded probability back to dataframe
    df['flood_probability'] = np.nan
    df.loc[predict_mask, 'flood_probability'] = np.round(flood_prob, 2)
    
    print("Saving updated dataset...")
    df.to_csv(os.path.join(base_dir, dataset_path), index=False)
    print("Done! Appended 'flood_probability' (0.00 to 1.00)")

if __name__ == "__main__":
    main()
