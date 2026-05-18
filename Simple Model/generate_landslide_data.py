import pandas as pd
import numpy as np
import random
import os
import math

os.makedirs('Simple Model/Datasets', exist_ok=True)

def generate_landslide_data(num_samples=50000):
    samples_per_class = num_samples // 4
    
    data = []
    
    # Normal
    for _ in range(samples_per_class):
        moist = random.randint(0, 1229)
        hum = round(random.uniform(40, 80), 2)
        temp = round(random.uniform(25, 33), 2)
        # tilt < 1 deg
        ax, ay = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
        az = random.uniform(9.7, 9.9)
        gx, gy, gz = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
        data.append(["LANDSLIDE", temp, hum, moist, ax, ay, az, gx, gy, gz, "Normal"])
        
    # Moderate
    for _ in range(samples_per_class):
        moist = random.randint(1230, 2048)
        hum = round(random.uniform(70, 90), 2)
        temp = round(random.uniform(15, 30), 2)
        # tilt 1 to 3 deg. 1 deg config
        ax = random.uniform(0.15, 0.5) * random.choice([1, -1])
        ay = random.uniform(0.15, 0.5) * random.choice([1, -1])
        az = random.uniform(9.5, 9.8)
        gx, gy, gz = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
        data.append(["LANDSLIDE", temp, hum, moist, ax, ay, az, gx, gy, gz, "Moderate"])

    # Severe
    for _ in range(samples_per_class):
        moist = random.randint(2049, 2457)
        hum = round(random.uniform(80, 95), 2)
        temp = round(random.uniform(15, 30), 2)
        # tilt 3 to 5 deg
        ax = random.uniform(0.5, 1.0) * random.choice([1, -1])
        ay = random.uniform(0.5, 1.0) * random.choice([1, -1])
        az = random.uniform(9.0, 9.6)
        gx, gy, gz = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
        data.append(["LANDSLIDE", temp, hum, moist, ax, ay, az, gx, gy, gz, "Severe"])

    # Extreme
    for _ in range(num_samples - 3 * samples_per_class):
        moist = random.randint(2458, 4095)
        hum = round(random.uniform(85, 100), 2)
        temp = round(random.uniform(20, 30), 2) # sudden change
        # tilt > 5 deg
        ax = random.uniform(1.0, 5.0) * random.choice([1, -1])
        ay = random.uniform(1.0, 5.0) * random.choice([1, -1])
        az = random.uniform(4.0, 9.0)
        gx, gy = random.uniform(1.1, 5.0) * random.choice([1, -1]), random.uniform(1.1, 5.0) * random.choice([1, -1]) 
        gz = random.uniform(-2.0, 2.0)
        data.append(["LANDSLIDE", temp, hum, moist, ax, ay, az, gx, gy, gz, "Extreme"])

    df = pd.DataFrame(data, columns=["type", "temp", "hum", "moist", "ax", "ay", "az", "gx", "gy", "gz", "label"])
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Train-val test split
    split_idx = int(num_samples * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_df.to_csv('Simple Model/Datasets/landslide_train.csv', index=False)
    val_df.to_csv('Simple Model/Datasets/landslide_val.csv', index=False)

if __name__ == "__main__":
    generate_landslide_data(50000)
    print("Landslide data generated.")
