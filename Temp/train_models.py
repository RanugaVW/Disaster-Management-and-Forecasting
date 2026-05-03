import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Define features used for training
FEATURES = [
    'rain_sum', 'temperature_2m_mean', 
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi'
]

# We also need 'division' as a feature, so we will label encode it
ALL_FEATURES = FEATURES + ['division_encoded']

def prepare_data_for_horizon(df, target_col):
    """
    Given a dataframe and a target severity column, create shifted targets for Day 1, Day 2, and Day 3.
    """
    df = df.copy()
    
    # Sort by division and date to ensure proper shifting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['division', 'date']).reset_index(drop=True)
    
    # Compute missing features on the fly
    df['rain_lag_1'] = df.groupby('division')['rain_sum'].shift(1)
    df['rain_rolling_3d'] = df.groupby('division')['rain_sum'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
    df['rain_rolling_7d'] = df.groupby('division')['rain_sum'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
    
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop NaNs that appear due to the 1-day lag
    df = df.dropna(subset=['rain_lag_1'])
    
    # Label encode division
    le_div = LabelEncoder()
    df['division_encoded'] = le_div.fit_transform(df['division'])
    
    # Label encode the target severity
    # Mapping: Normal=0, Moderate=1, Severe=2, Extreme=3
    severity_mapping = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}
    df[target_col] = df[target_col].map(severity_mapping)
    
    # Create shifted targets
    df['target_day1'] = df.groupby('division')[target_col].shift(-1)
    df['target_day2'] = df.groupby('division')[target_col].shift(-2)
    df['target_day3'] = df.groupby('division')[target_col].shift(-3)
    
    # Drop rows with NaN targets (the last 3 days of each division)
    df = df.dropna(subset=['target_day1', 'target_day2', 'target_day3', target_col])
    
    # Ensure targets are integers
    for col in ['target_day1', 'target_day2', 'target_day3']:
        df[col] = df[col].astype(int)
        
    X = df[ALL_FEATURES]
    y = df[['target_day1', 'target_day2', 'target_day3']]
    
    return X, y, le_div

def train_and_save_model(data_path, target_col, model_name):
    print(f"\n{'='*50}\nTraining {model_name} Model...\n{'='*50}")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preparing features and shifting 3-day horizons...")
    X, y, le_div = prepare_data_for_horizon(df, target_col)
    
    print(f"Dataset shape after preparation: Features {X.shape}, Targets {y.shape}")
    
    # Temporal Split: Use the last 20% of the dataset as the test set
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Calculate sample weights to handle severe class imbalance
    print("Computing class weights to handle imbalance...")
    sample_weights = compute_sample_weight('balanced', y_train['target_day1'])
    
    # Define Optuna objective
    import optuna
    
    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'eval_metric': 'mlogloss',
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': -1,
            'random_state': 42
        }
        
        # We split the training data to get a validation set for Optuna
        val_idx = int(len(X_train) * 0.8)
        X_t, X_v = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
        y_t, y_v = y_train.iloc[:val_idx], y_train.iloc[val_idx:]
        w_t = sample_weights[:val_idx]
        
        base_xgb = xgb.XGBClassifier(**params)
        model = MultiOutputClassifier(base_xgb, n_jobs=-1)
        model.fit(X_t, y_t, **{'sample_weight': w_t})
        
        # Evaluate on validation set
        preds = model.predict(X_v)
        # Using exact match accuracy
        acc = (y_v.values == preds).all(axis=1).mean()
        return acc

    print("Running Optuna hyperparameter tuning (5 trials)...")
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=5)
    
    print("Best trial:")
    print(f"  Value (Accuracy): {study.best_trial.value:.4f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    print("Training final model with best parameters on full training set...")
    best_params = study.best_trial.params
    best_params.update({
        'objective': 'multi:softmax',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'n_jobs': -1,
        'random_state': 42
    })
    
    final_base_xgb = xgb.XGBClassifier(**best_params)
    multi_target_model = MultiOutputClassifier(final_base_xgb, n_jobs=-1)
    multi_target_model.fit(X_train, y_train, **{'sample_weight': sample_weights})
    
    # Simple evaluation
    train_score = multi_target_model.score(X_train, y_train)
    test_score = multi_target_model.score(X_test, y_test)
    print(f"Exact Match Accuracy - Train: {train_score:.4f} | Test: {test_score:.4f}")
    print("(Note: Exact Match means the model got ALL 3 days correct simultaneously)")
    
    # Save the model and label encoder
    os.makedirs('tuned_models', exist_ok=True)
    model_path = f'tuned_models/{model_name}_xgboost.pkl'
    encoder_path = f'tuned_models/{model_name}_division_encoder.pkl'
    
    joblib.dump(multi_target_model, model_path)
    joblib.dump(le_div, encoder_path)
    
    print(f"Saved model to {model_path}")
    print(f"Saved encoder to {encoder_path}")

def main():
    # 1. Train Flood Model
    train_and_save_model(
        data_path='data/training_data_Flood.csv',
        target_col='flood_severity',
        model_name='Flood'
    )
    
    # 2. Train Landslide Model
    train_and_save_model(
        data_path='data/training_data_Landslide.csv',
        target_col='landslide_severity',
        model_name='Landslide'
    )
    
    # 3. Train Drought Model
    train_and_save_model(
        data_path='data/training_data_Drought.csv',
        target_col='drought_severity',
        model_name='Drought'
    )
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main()
