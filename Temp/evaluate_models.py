import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

FEATURES = [
    'rain_sum', 'temperature_2m_mean', 
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi'
]
ALL_FEATURES = FEATURES + ['division_encoded']
SEVERITY_MAPPING = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}

def prepare_data_for_horizon(df, target_col, le_div=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['division', 'date']).reset_index(drop=True)
    
    df['rain_lag_1'] = df.groupby('division')['rain_sum'].shift(1)
    df['rain_rolling_3d'] = df.groupby('division')['rain_sum'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
    df['rain_rolling_7d'] = df.groupby('division')['rain_sum'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
    
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df = df.dropna(subset=['rain_lag_1'])
    
    if le_div is not None:
        # Handle unseen divisions safely
        known_divs = set(le_div.classes_)
        df = df[df['division'].isin(known_divs)]
        df['division_encoded'] = le_div.transform(df['division'])
    else:
        from sklearn.preprocessing import LabelEncoder
        le_div = LabelEncoder()
        df['division_encoded'] = le_div.fit_transform(df['division'])
        
    df[target_col] = df[target_col].map(SEVERITY_MAPPING)
    
    df['target_day1'] = df.groupby('division')[target_col].shift(-1)
    df['target_day2'] = df.groupby('division')[target_col].shift(-2)
    df['target_day3'] = df.groupby('division')[target_col].shift(-3)
    
    df = df.dropna(subset=['target_day1', 'target_day2', 'target_day3', target_col])
    
    for col in ['target_day1', 'target_day2', 'target_day3']:
        df[col] = df[col].astype(int)
        
    X = df[ALL_FEATURES]
    y = df[['target_day1', 'target_day2', 'target_day3']]
    
    return X, y, le_div

def evaluate_model_on_test(model_name, target_col):
    print(f"\n{'='*60}\nEvaluating {model_name} Model\n{'='*60}")
    
    # Load assets
    try:
        model = joblib.load(f'tuned_models/{model_name}_xgboost.pkl')
        le_div = joblib.load(f'tuned_models/{model_name}_division_encoder.pkl')
    except Exception as e:
        print(f"Failed to load {model_name} models: {e}")
        return
        
    # Prepare Test Data
    print(f"Loading and preparing test_data.csv for {model_name}...")
    df_test = pd.read_csv('data/test_data.csv')
    X_test, y_test, _ = prepare_data_for_horizon(df_test, target_col, le_div)
    
    print("Making predictions...")
    preds = model.predict(X_test)
    
    # Overall Exact Match Accuracy
    exact_match = (y_test.values == preds).all(axis=1).mean()
    print(f"\n[OVERALL] Exact Match Accuracy (All 3 Days Correct): {exact_match:.4f}")
    
    # Per-Day Metrics
    for i, day in enumerate(['target_day1', 'target_day2', 'target_day3']):
        y_true = y_test[day].values
        y_pred = preds[:, i]
        
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro')
        f1_wei = f1_score(y_true, y_pred, average='weighted')
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        print(f"\n--- {day.upper()} METRICS ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Macro : {f1_mac:.4f}  |  F1-Weighted: {f1_wei:.4f}")
        print(f"QWK Score: {qwk:.4f}")
        
    # -------------------------------------------------------------------------
    # Stratified CV on Training Data (stratified by Day 1)
    # -------------------------------------------------------------------------
    print("\n--- Running 3-Fold Stratified Cross-Validation on Training Data ---")
    train_path = f'data/training_data_{model_name}.csv'
    df_train = pd.read_csv(train_path)
    X_train, y_train, _ = prepare_data_for_horizon(df_train, target_col, le_div)
    
    # Get base estimator from the MultiOutputClassifier
    base_estimator = model.estimator
    
    # To use standard StratifiedKFold with cross_validate, we need a single target.
    # The user approved stratifying by Day 1.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Cross-validating base estimator strictly for Day+1 target...")
    y_day1 = y_train['target_day1']
    
    cv_results = cross_validate(
        base_estimator, X_train, y_day1, 
        cv=skf, 
        scoring=['accuracy', 'f1_macro', 'f1_weighted'],
        n_jobs=-1
    )
    
    print(f"Stratified CV Mean Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Stratified CV Mean F1-Macro: {np.mean(cv_results['test_f1_macro']):.4f}")
    print(f"Stratified CV Mean F1-Weight: {np.mean(cv_results['test_f1_weighted']):.4f}")

def main():
    evaluate_model_on_test('Flood', 'flood_severity')
    evaluate_model_on_test('Landslide', 'landslide_severity')
    evaluate_model_on_test('Drought', 'drought_severity')
    print("\nAll evaluations complete!")

if __name__ == "__main__":
    main()
