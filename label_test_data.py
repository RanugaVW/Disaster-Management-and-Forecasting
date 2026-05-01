import pandas as pd
import numpy as np

def main():
    file_path = 'data/test_data.csv'
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    if 'spi' not in df.columns:
        print("Error: 'spi' column not found. Please run add_spi_to_test.py first.")
        return
        
    print("Labeling Flood Severity...")
    flood_conds = [
        (df['spi'] < 0.8),
        (df['spi'] >= 0.8) & (df['spi'] < 1.3),
        (df['spi'] >= 1.3) & (df['spi'] < 1.6),
        (df['spi'] >= 1.6)
    ]
    choices = ['Normal', 'Moderate', 'Severe', 'Extreme']
    df['flood_severity'] = np.select(flood_conds, choices, default='Normal')
    
    print("Labeling Drought Severity...")
    drought_conds = [
        (df['spi'] <= -2.00),
        (df['spi'] > -2.00) & (df['spi'] <= -1.50),
        (df['spi'] > -1.50) & (df['spi'] <= -1.00),
        (df['spi'] > -1.00)
    ]
    drought_choices = ['Extreme', 'Severe', 'Moderate', 'Normal']
    df['drought_severity'] = np.select(drought_conds, drought_choices, default='Normal')
    
    print("Labeling Landslide Severity...")
    landslide_conds = [
        (df['rain_sum'] >= 150) | (df['spi'] >= 2.00),
        (df['rain_sum'] >= 100) | (df['spi'] >= 1.50),
        (df['rain_sum'] >= 75) | (df['spi'] >= 1.00)
    ]
    landslide_choices = ['Extreme', 'Severe', 'Moderate']
    df['landslide_severity'] = np.select(landslide_conds, landslide_choices, default='Normal')
    
    print("Saving updated test dataset...")
    df.to_csv(file_path, index=False)
    
    print("\n--- Class Distributions in Test Data ---")
    print("\nFlood:")
    print(df['flood_severity'].value_counts())
    print("\nDrought:")
    print(df['drought_severity'].value_counts())
    print("\nLandslide:")
    print(df['landslide_severity'].value_counts())
    print("\nDone! Added all 3 severity labels to test_data.csv")

if __name__ == "__main__":
    main()
