import pandas as pd
import os

def main():
    print("Loading master feature matrix...")
    df = pd.read_csv('data/master_feature_matrix.csv')
    
    total_rows = len(df)
    print(f"Total rows loaded: {total_rows}")
    
    test_size = 86300
    
    print("Splitting data...")
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    print(f"Split completed. Training rows: {len(train_df)}, Test rows: {len(test_df)}")
    
    # Save the split datasets
    train_path = 'data/training_data.csv'
    test_path = 'data/test_data.csv'
    
    print("Saving training data...")
    train_df.to_csv(train_path, index=False)
    
    print("Saving test data...")
    test_df.to_csv(test_path, index=False)
    
    print("Data splitting and saving complete!")

if __name__ == "__main__":
    main()
