import pandas as pd
import numpy as np
import random
import os

os.makedirs('Simple Model/Datasets', exist_ok=True)

def generate_flood_data(num_samples=50000):
    samples_per_class = num_samples // 4
    
    data = []
    
    # Normal
    for _ in range(samples_per_class):
        depth = round(random.uniform(0, 3.0), 2)
        depth_prev = round(random.uniform(max(0, depth - 0.5), depth + 0.1), 2)
        hum = round(random.uniform(40, 70), 2)
        temp = round(random.uniform(25, 33), 2)
        data.append(["FLOOD", temp, hum, depth_prev, depth, "Normal"])
        
    # Moderate
    for _ in range(samples_per_class):
        depth = round(random.uniform(3.0, 8.0), 2)
        depth_prev = round(random.uniform(0, depth - 0.5), 2) # Rising water
        hum = round(random.uniform(70, 80), 2)
        temp = round(random.uniform(15, 30), 2)
        data.append(["FLOOD", temp, hum, depth_prev, depth, "Moderate"])

    # Severe
    for _ in range(samples_per_class):
        depth = round(random.uniform(8.0, 15.0), 2)
        depth_prev = round(random.uniform(3.0, depth - 1.0), 2)
        hum = round(random.uniform(80, 95), 2)
        temp = round(random.uniform(18, 28), 2)
        data.append(["FLOOD", temp, hum, depth_prev, depth, "Severe"])

    # Extreme
    for _ in range(num_samples - 3 * samples_per_class):
        depth = round(random.uniform(15.0, 50.0), 2)
        depth_prev = round(random.uniform(5.0, depth - 2.0), 2)
        hum = round(random.uniform(85, 100), 2)
        temp = round(random.uniform(20, 30), 2)
        data.append(["FLOOD", temp, hum, depth_prev, depth, "Extreme"])

    df = pd.DataFrame(data, columns=["type", "temp", "hum", "depth_prev", "depth", "label"])
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Train-val test split
    # Since we need a test set, let's just make a single file or generate train/val directly
    split_idx = int(num_samples * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_df.to_csv('Simple Model/Datasets/flood_train.csv', index=False)
    val_df.to_csv('Simple Model/Datasets/flood_val.csv', index=False)

if __name__ == "__main__":
    generate_flood_data(50000)
    print("Flood data generated.")
