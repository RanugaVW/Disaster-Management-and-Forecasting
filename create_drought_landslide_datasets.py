import pandas as pd
import numpy as np
import os

def main():
    input_file = 'data/training_data_Flood.csv'
    drought_output = 'data/training_data_Drought.csv'
    landslide_output = 'data/training_data_Landslide.csv'
    
    print(f"Loading base dataset from {input_file}...")
    df_base = pd.read_csv(input_file)
    
    # Remove flood_severity for the other datasets
    if 'flood_severity' in df_base.columns:
        df_base = df_base.drop(columns=['flood_severity'])
        
    # ---------------------------------------------------------
    # 1. Create Drought Dataset
    # ---------------------------------------------------------
    print("\nProcessing Drought Dataset...")
    df_drought = df_base.copy()
    
    drought_conds = [
        (df_drought['spi'] <= -2.00),
        (df_drought['spi'] > -2.00) & (df_drought['spi'] <= -1.50),
        (df_drought['spi'] > -1.50) & (df_drought['spi'] <= -1.00)
    ]
    drought_choices = ['Extreme', 'Severe', 'Moderate']
    
    df_drought['drought_severity'] = np.select(drought_conds, drought_choices, default='Normal')
    
    print("Class distribution for drought_severity:")
    print(df_drought['drought_severity'].value_counts())
    
    print(f"Saving to {drought_output}...")
    df_drought.to_csv(drought_output, index=False)
    
    # ---------------------------------------------------------
    # 2. Create Landslide Dataset
    # ---------------------------------------------------------
    print("\nProcessing Landslide Dataset...")
    df_landslide = df_base.copy()
    
    landslide_conds = [
        (df_landslide['rain_sum'] >= 150) | (df_landslide['spi'] >= 2.00),
        (df_landslide['rain_sum'] >= 100) | (df_landslide['spi'] >= 1.50),
        (df_landslide['rain_sum'] >= 75) | (df_landslide['spi'] >= 1.00)
    ]
    landslide_choices = ['Extreme', 'Severe', 'Moderate']
    
    df_landslide['landslide_severity'] = np.select(landslide_conds, landslide_choices, default='Normal')
    
    print("Class distribution for landslide_severity:")
    print(df_landslide['landslide_severity'].value_counts())
    
    print(f"Saving to {landslide_output}...")
    df_landslide.to_csv(landslide_output, index=False)
    
    print("\nDone! Created drought and landslide datasets successfully.")

if __name__ == "__main__":
    main()
