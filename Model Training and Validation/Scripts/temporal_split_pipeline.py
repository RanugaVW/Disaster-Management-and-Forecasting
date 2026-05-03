import pandas as pd
import numpy as np
import scipy.stats as stats
import os
from tqdm import tqdm

def compute_spi(series):
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return np.full(len(series), np.nan)
        
    zeros = valid_data[valid_data == 0]
    q = len(zeros) / len(valid_data)
    non_zeros = valid_data[valid_data > 0]
    
    out = np.full(len(series), np.nan)
    
    if len(non_zeros) > 0:
        try:
            a, loc, scale = stats.gamma.fit(non_zeros, floc=0)
            gamma_cdf = stats.gamma.cdf(valid_data, a, loc=loc, scale=scale)
            h_x = q + (1 - q) * gamma_cdf
            h_x = np.clip(h_x, 0.0001, 0.9999)
            spi = stats.norm.ppf(h_x)
            out[series.notna()] = spi
        except Exception:
            pass
            
    return out

def label_severities(df):
    df = df.copy()
    
    # 1. Flood
    flood_conds = [
        (df['spi'] < 0.8),
        (df['spi'] >= 0.8) & (df['spi'] < 1.3),
        (df['spi'] >= 1.3) & (df['spi'] < 1.6),
        (df['spi'] >= 1.6)
    ]
    df['flood_severity'] = np.select(flood_conds, ['Normal', 'Moderate', 'Severe', 'Extreme'], default='Normal')
    
    # 2. Drought
    drought_conds = [
        (df['spi'] <= -2.00),
        (df['spi'] > -2.00) & (df['spi'] <= -1.50),
        (df['spi'] > -1.50) & (df['spi'] <= -1.00),
        (df['spi'] > -1.00)
    ]
    df['drought_severity'] = np.select(drought_conds, ['Extreme', 'Severe', 'Moderate', 'Normal'], default='Normal')
    
    # 3. Landslide
    landslide_conds = [
        (df['rain_sum'] >= 150) | (df['spi'] >= 2.00),
        (df['rain_sum'] >= 100) | (df['spi'] >= 1.50),
        (df['rain_sum'] >= 75) | (df['spi'] >= 1.00)
    ]
    df['landslide_severity'] = np.select(landslide_conds, ['Extreme', 'Severe', 'Moderate'], default='Normal')
    
    return df

def process_dataset(df, name):
    print(f"\nProcessing {name} Data...")
    df['spi'] = np.nan
    divisions = df['division'].unique()
    
    for div in tqdm(divisions, desc=f"Computing SPI for {name}"):
        div_mask = df['division'] == div
        daily_rain = df.loc[div_mask, 'rain_sum']
        df.loc[div_mask, 'spi'] = compute_spi(daily_rain)
        
    # Round SPI to 3 decimal places as requested
    df['spi'] = df['spi'].round(3)
    
    print(f"Applying severity labels to {name} Data...")
    df = label_severities(df)
    return df

def main():
    print("Loading master_feature_matrix.csv...")
    df_master = pd.read_csv('data/master_feature_matrix.csv')
    df_master['date'] = pd.to_datetime(df_master['date'])
    df_master['year'] = df_master['date'].dt.year
    
    # Temporal Split
    df_train = df_master[df_master['year'] <= 2023].copy()
    df_test = df_master[df_master['year'] >= 2024].copy()
    
    print(f"Split complete. Train rows: {len(df_train)}, Test rows: {len(df_test)}")
    
    # Process both
    df_train = process_dataset(df_train, "Training")
    df_test = process_dataset(df_test, "Testing")
    
    # Drop the temporary year column
    df_train.drop(columns=['year'], inplace=True)
    df_test.drop(columns=['year'], inplace=True)
    
    # Create output directory
    out_dir = 'Dataset Separation'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save main datasets
    train_path = os.path.join(out_dir, 'training_data.csv')
    test_path = os.path.join(out_dir, 'test_data.csv')
    print(f"\nSaving {train_path}...")
    df_train.to_csv(train_path, index=False)
    print(f"Saving {test_path}...")
    df_test.to_csv(test_path, index=False)
    
    # Save isolated hazard training sets
    print("Generating isolated training datasets...")
    
    # Flood
    flood_train = df_train.drop(columns=['drought_severity', 'landslide_severity'])
    flood_train.to_csv(os.path.join(out_dir, 'training_data_Flood.csv'), index=False)
    
    # Landslide
    landslide_train = df_train.drop(columns=['flood_severity', 'drought_severity'])
    landslide_train.to_csv(os.path.join(out_dir, 'training_data_Landslide.csv'), index=False)
    
    # Drought
    drought_train = df_train.drop(columns=['flood_severity', 'landslide_severity'])
    drought_train.to_csv(os.path.join(out_dir, 'training_data_Drought.csv'), index=False)
    
    print("\nProcess fully complete! All data stored in 'Dataset Separation' folder.")

if __name__ == "__main__":
    main()
