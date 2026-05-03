import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def main():
    base_dir = 'Dataset Separation'
    
    print("Loading datasets...")
    train_file = os.path.join(base_dir, 'training_data.csv')
    test_file = os.path.join(base_dir, 'test_data.csv')
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    print("Fitting LabelEncoder on all 121 divisions...")
    le = LabelEncoder()
    # Fit strictly on the training data divisions
    df_train['division_encoded'] = le.fit_transform(df_train['division'])
    
    # Transform test data using the EXACT same encoder
    df_test['division_encoded'] = le.transform(df_test['division'])
    
    print("Saving master division encoder...")
    joblib.dump(le, os.path.join(base_dir, 'master_division_encoder.pkl'))
    
    print("Saving updated base datasets...")
    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
    
    # Now process the specific hazard training sets
    hazards = ['Flood', 'Landslide', 'Drought']
    for hazard in hazards:
        hazard_file = os.path.join(base_dir, f'training_data_{hazard}.csv')
        print(f"Processing {hazard_file}...")
        df_hazard = pd.read_csv(hazard_file)
        df_hazard['division_encoded'] = le.transform(df_hazard['division'])
        df_hazard.to_csv(hazard_file, index=False)
        
    print("\nDone! 'division_encoded' column successfully added to all datasets.")

if __name__ == "__main__":
    main()
