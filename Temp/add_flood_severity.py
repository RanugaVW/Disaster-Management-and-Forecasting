import pandas as pd
import numpy as np

def main():
    file_path = 'data/training_data.csv'
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    if 'spi' not in df.columns:
        print("Error: 'spi' column not found in dataset. Please compute it first.")
        return
        
    print("Labeling flood_severity based on SPI thresholds...")
    
    # Define the conditions based on the provided image
    conditions = [
        (df['spi'] < 0.8),
        (df['spi'] >= 0.8) & (df['spi'] < 1.3),
        (df['spi'] >= 1.3) & (df['spi'] < 1.6),
        (df['spi'] >= 1.6)
    ]
    
    # Define the corresponding exact labels requested by user
    choices = ['Normal', 'Moderate', 'Severe', 'Extreme']
    
    # Apply conditions to create the label
    df['flood_severity'] = np.select(conditions, choices, default='Normal')
    
    print("Class distribution for flood_severity:")
    print(df['flood_severity'].value_counts())
    
    print(f"Saving updated data back to {file_path}...")
    df.to_csv(file_path, index=False)
    print("Done! Added 'flood_severity' column to training_data.csv")

if __name__ == "__main__":
    main()
