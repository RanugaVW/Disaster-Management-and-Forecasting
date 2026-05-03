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
            gamma_cdf = stats.gamma.cdf(valid_data, a, loc=loc, scale=scale)
            h_x = q + (1 - q) * gamma_cdf
            
            # Clip between small probabilities to avoid infinite SPI values
            h_x = np.clip(h_x, 0.0001, 0.9999)
            
            # Inverse normal to get SPI
            spi = stats.norm.ppf(h_x)
            
            # Place back into the original shape
            out[series.notna()] = spi
            
        except Exception:
            pass
            
    return out

def main():
    file_path = 'data/test_data.csv'
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Computing 1-day SPI...")
    df['spi'] = np.nan
    divisions = df['division'].unique()
    
    for div in tqdm(divisions, desc="Processing Divisions for SPI"):
        div_mask = df['division'] == div
        daily_rain = df.loc[div_mask, 'rain_sum']
        spi_res = compute_spi(daily_rain)
        df.loc[div_mask, 'spi'] = spi_res
        
    # Round to 3 decimal places to keep it clean
    df['spi'] = df['spi'].round(3)
        
    print(f"Saving updated data back to {file_path}...")
    df.to_csv(file_path, index=False)
    print("Done! Added 'spi' column to test_data.csv")

if __name__ == "__main__":
    main()
