import pandas as pd
import numpy as np
import joblib

def main():
    # 1. Load the pre-trained Flood model and encoder
    print("Loading existing Flood model...")
    model = joblib.load('tuned_models/Flood_xgboost.pkl')
    le_div = joblib.load('tuned_models/Flood_division_encoder.pkl')
    
    # 2. Get a single test row (e.g., the very first row from test_data)
    df_test = pd.read_csv('data/test_data.csv')
    
    # Create the rolling features on the fly for demonstration
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test = df_test.sort_values(by=['division', 'date']).reset_index(drop=True)
    df_test['rain_lag_1'] = df_test.groupby('division')['rain_sum'].shift(1)
    df_test['rain_rolling_3d'] = df_test.groupby('division')['rain_sum'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
    df_test['rain_rolling_7d'] = df_test.groupby('division')['rain_sum'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
    df_test['month'] = df_test['date'].dt.month
    df_test['month_sin'] = np.sin(2 * np.pi * df_test['month'] / 12)
    df_test['month_cos'] = np.cos(2 * np.pi * df_test['month'] / 12)
    df_test = df_test.dropna(subset=['rain_lag_1'])
    
    # Safely encode division
    known_divs = set(le_div.classes_)
    df_test = df_test[df_test['division'].isin(known_divs)]
    df_test['division_encoded'] = le_div.transform(df_test['division'])
    
    FEATURES = [
        'rain_sum', 'temperature_2m_mean', 
        'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
        'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
        'month_sin', 'month_cos', 'spi', 'division_encoded'
    ]
    
    # Grab just 5 arbitrary rows to demonstrate
    sample_data = df_test.head(5)
    X_sample = sample_data[FEATURES]
    
    # 3. USE predict_proba INSTEAD OF predict!
    # predict_proba returns a list of 3 arrays (one for Day 1, Day 2, Day 3)
    # Each array has shape (n_samples, 4 classes)
    probabilities = model.predict_proba(X_sample)
    
    # Let's look at Day 1 predictions for these 5 rows
    day1_probs = probabilities[0] 
    
    print("\n--- CONTINUOUS PROBABILITY PREDICTIONS (DAY+1) ---")
    for i in range(len(X_sample)):
        div_name = sample_data.iloc[i]['division']
        date_val = sample_data.iloc[i]['date'].strftime('%Y-%m-%d')
        
        # Classes are: 0=Normal, 1=Moderate, 2=Severe, 3=Extreme
        prob_normal = day1_probs[i][0]
        prob_moderate = day1_probs[i][1]
        prob_severe = day1_probs[i][2]
        prob_extreme = day1_probs[i][3]
        
        # Continuous Probability of ANY Flood = Sum of Moderate, Severe, Extreme
        prob_any_flood = prob_moderate + prob_severe + prob_extreme
        
        # Continuous Probability of DANGEROUS Flood = Sum of Severe, Extreme
        prob_danger_flood = prob_severe + prob_extreme
        
        print(f"\nLocation: {div_name} on {date_val}")
        print(f"  Probability of Normal:         {prob_normal:.2%}")
        print(f"  Probability of ANY Flood:      {prob_any_flood:.2%}  <-- Continuous (0 to 1)")
        print(f"  Probability of DANGER Flood:   {prob_danger_flood:.2%}  <-- Continuous (0 to 1)")
        
if __name__ == "__main__":
    main()
