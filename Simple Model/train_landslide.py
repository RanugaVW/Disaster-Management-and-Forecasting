import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

os.makedirs('Models', exist_ok=True)

print("--- Training Landslide Model with Optuna Hyperparameter Optimization ---")
train_df = pd.read_csv('Datasets/landslide_train.csv')
val_df = pd.read_csv('Datasets/landslide_val.csv')

X_train = train_df[['temp', 'hum', 'moist', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']]
y_train = train_df['label']

X_val = val_df[['temp', 'hum', 'moist', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']]
y_val = val_df['label']

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

def objective(trial):
    """Objective function for Optuna optimization."""
    # Suggest hyperparameters for XGBoost
    xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
    xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
    xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
    xgb_subsample = trial.suggest_float('xgb_subsample', 0.5, 1.0)
    xgb_colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
    
    # Suggest hyperparameters for LightGBM
    lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 50, 300)
    lgb_learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3)
    lgb_max_depth = trial.suggest_int('lgb_max_depth', 3, 10)
    lgb_num_leaves = trial.suggest_int('lgb_num_leaves', 20, 100)
    lgb_subsample = trial.suggest_float('lgb_subsample', 0.5, 1.0)
    
    # Create models with suggested hyperparameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_n_estimators,
        learning_rate=xgb_learning_rate,
        max_depth=xgb_max_depth,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        random_state=42,
        verbosity=0
    )
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=lgb_n_estimators,
        learning_rate=lgb_learning_rate,
        max_depth=lgb_max_depth,
        num_leaves=lgb_num_leaves,
        subsample=lgb_subsample,
        random_state=42,
        verbose=-1
    )
    
    # Create ensemble with suggested hyperparameters
    ensemble_model = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft'
    )
    
    # Train and evaluate
    ensemble_model.fit(X_train, y_train_enc)
    y_pred_enc = ensemble_model.predict(X_val)
    accuracy = accuracy_score(y_val_enc, y_pred_enc)
    
    return accuracy

# Create a study and optimize
print("Optimizing hyperparameters with Optuna...")
sampler = TPESampler(seed=42)
pruner = MedianPruner()
study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

# Get best hyperparameters
best_params = study.best_params
print(f"\nBest parameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"Best accuracy: {study.best_value:.4f}\n")

# Train final model with best hyperparameters
xgb_model = xgb.XGBClassifier(
    n_estimators=best_params['xgb_n_estimators'],
    learning_rate=best_params['xgb_learning_rate'],
    max_depth=best_params['xgb_max_depth'],
    subsample=best_params['xgb_subsample'],
    colsample_bytree=best_params['xgb_colsample_bytree'],
    random_state=42,
    verbosity=0
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=best_params['lgb_n_estimators'],
    learning_rate=best_params['lgb_learning_rate'],
    max_depth=best_params['lgb_max_depth'],
    num_leaves=best_params['lgb_num_leaves'],
    subsample=best_params['lgb_subsample'],
    random_state=42,
    verbose=-1
)

ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
    voting='soft'
)

ensemble_model.fit(X_train, y_train_enc)

y_pred_enc = ensemble_model.predict(X_val)
y_pred = le.inverse_transform(y_pred_enc)

print("--- Final Model Performance ---")
print(classification_report(y_val, y_pred))
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")

joblib.dump(ensemble_model, 'Models/landslide_ensemble_model.pkl')
joblib.dump(le, 'Models/landslide_label_encoder.pkl')

print("Landslide model trained and saved.")
