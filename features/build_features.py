import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

def compute_spi(series):
    """
    Computes the Standardized Precipitation Index (SPI) for a given pandas Series of precipitation.
    Fits a Gamma distribution, accounting for zeros.
    """
    # Exclude NaNs
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return np.full(len(series), np.nan)
        
    # Calculate probability of zero precipitation
    zeros = valid_data[valid_data == 0]
    q = len(zeros) / len(valid_data)
    
    # Non-zero data for gamma fit
    non_zeros = valid_data[valid_data > 0]
    
    out = np.full(len(series), np.nan)
    
    if len(non_zeros) > 0:
        # Fit gamma distribution (a=shape, loc, scale=beta)
        try:
            a, loc, scale = stats.gamma.fit(non_zeros, floc=0)
            
            # Compute CDF for all valid data
            # H(x) = q + (1 - q) * G(x) where G(x) is the gamma CDF
            gamma_cdf = stats.gamma.cdf(valid_data, a, loc=loc, scale=scale)
            h_x = q + (1 - q) * gamma_cdf
            
            # The CDF can sometimes perfectly output 0.0 or 1.0 which causes Inf in norm.ppf.
            # Clip between small probabilities to avoid infinite SPI values
            h_x = np.clip(h_x, 0.0001, 0.9999)
            
            # Inverse normal to get SPI
            spi = stats.norm.ppf(h_x)
            
            # Place back into the original shape
            out[series.notna()] = spi
            
        except Exception:
            # Fallback if fit fails (extremely rare but possible with weird data)
            pass
            
    return out

def main():
    print("Loading datasets...")
    df_temp = pd.read_csv('data/temperature_mean_2m.csv')
    df_rain = pd.read_csv('data/rain_sum.csv')
    df_soil = pd.read_csv('data/soil_moisture_daily.csv')
    
    print("Merging on date and division...")
    # Setup Date columns
    for df in [df_temp, df_rain, df_soil]:
        df['date'] = pd.to_datetime(df['date'])
        
    df_merged = pd.merge(df_rain, df_temp, on=['date', 'division'], how='outer')
    df_merged = pd.merge(df_merged, df_soil, on=['date', 'division'], how='outer')
    
    # Sort for time-series operations
    df_merged = df_merged.sort_values(['division', 'date']).reset_index(drop=True)
    
    print("Computing Lags and Rolling Sums...")
    # Shift operations grouped by division
    # 1-day lag
    df_merged['rain_lag_1'] = df_merged.groupby('division')['rain_sum'].shift(1)
    
    # Rolling sums (3-day and 7-day)
    # Using min_periods=1 ensures that we get partial sums at the edges instead of NaNs, but we could also use strict 3/7
    df_merged['rain_rolling_3d'] = df_merged.groupby('division')['rain_sum'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
    df_merged['rain_rolling_7d'] = df_merged.groupby('division')['rain_sum'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
    
    print("Computing Seasonal Encodings...")
    # sin and cos of month
    df_merged['month'] = df_merged['date'].dt.month
    df_merged['month_sin'] = np.sin(2 * np.pi * df_merged['month'] / 12)
    df_merged['month_cos'] = np.cos(2 * np.pi * df_merged['month'] / 12)
    df_merged.drop(columns=['month'], inplace=True)
    
    print("Computing SPI (30, 90, 180 days)... This may take a minute.")
    divisions = df_merged['division'].unique()
    
    # Create empty columns
    for d in [30, 90, 180]:
        df_merged[f'spi_{d}d'] = np.nan
        
    # We first calculate the rolling sum for each window size, then calculate SPI
    for d in [30, 90, 180]:
        df_merged[f'rain_rolling_{d}d'] = df_merged.groupby('division')['rain_sum'].rolling(window=d, min_periods=d).sum().reset_index(0, drop=True)
        
    # Compute SPI per division and window
    # We loop via tqdm to show progress since the gamma fit per division can take a few seconds
    for div in tqdm(divisions, desc="Processing Divisions for SPI"):
        div_mask = df_merged['division'] == div
        
        for d in [30, 90, 180]:
            rolling_series = df_merged.loc[div_mask, f'rain_rolling_{d}d']
            spi_res = compute_spi(rolling_series)
            df_merged.loc[div_mask, f'spi_{d}d'] = spi_res
            
    # Optional cleanup: drop the heavy long-term rolling rain sums if you only want SPI
    df_merged.drop(columns=['rain_rolling_30d', 'rain_rolling_90d', 'rain_rolling_180d'], inplace=True)
            
    print("Saving master feature matrix...")
    df_merged = df_merged.round(3)
    df_merged.to_csv('data/master_feature_matrix.csv', index=False)
    
    print(f"Done! Created master_feature_matrix.csv with {len(df_merged)} rows and {len(df_merged.columns)} columns.")
    print("Columns:", list(df_merged.columns))

if __name__ == "__main__":
    main()
