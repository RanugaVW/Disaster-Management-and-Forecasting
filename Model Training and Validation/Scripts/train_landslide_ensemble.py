#!/usr/bin/env python3
"""
Landslide Ensemble Model Training - Standalone Script
=====================================================
Creates Landslide ensemble model with proper features:
['rain_sum', 'temperature_2m_mean', 'soil_moisture_7_to_28cm', 
 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm', 
 'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
 'month_sin', 'month_cos', 'spi', 'division_encoded']
"""

import os
import sys
import warnings
import json
import time
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    import joblib
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import (
        accuracy_score, f1_score, cohen_kappa_score
    )
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)

# ─────────────────────────── Config ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..')
DATASET_DIR = os.path.join(BASE_DIR, 'Datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
os.makedirs(MODEL_DIR, exist_ok=True)

N_TRIALS = 5  # Reduced from 10 for faster training on large dataset
VAL_FRAC = 0.15
RANDOM_STATE = 42
SEVERITY_MAP = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}

FEATURES = [
    'rain_sum', 'temperature_2m_mean',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm',
    'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi', 'division_encoded'
]

# ─────────────────────────── Helpers ────────────────────────────
def load_and_prepare(csv_path, target_col):
    """Load dataset, create shifted multi-horizon targets, temporal split."""
    print(f"\n  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['division', 'date']).reset_index(drop=True)

    # Map severity to int
    df[target_col] = df[target_col].map(SEVERITY_MAP)

    # Shift-based multi-output targets
    for day in [1, 2, 3]:
        df[f'target_d{day}'] = df.groupby('division')[target_col].shift(-day)

    df = df.dropna(subset=[f'target_d{d}' for d in [1, 2, 3]] + FEATURES)
    for d in [1, 2, 3]:
        df[f'target_d{d}'] = df[f'target_d{d}'].astype(int)

    X = df[FEATURES].astype(np.float32)
    y = df[['target_d1', 'target_d2', 'target_d3']].astype(int)

    # Temporal split
    n_val = int(len(df) * VAL_FRAC)
    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
    y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    # Sample weights on Day+1 target (primary horizon)
    sample_weights = compute_sample_weight('balanced', y_train['target_d1'])

    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")
    return X_train, X_val, y_train, y_val, sample_weights


def eval_model(model, X_val, y_val):
    """Return per-day and exact-match metrics dict."""
    preds = model.predict(X_val)
    metrics = {}

    for i, day in enumerate(['target_d1', 'target_d2', 'target_d3']):
        yt = y_val[day].values
        yp = preds[:, i]
        metrics[f'd{i+1}_acc'] = accuracy_score(yt, yp)
        metrics[f'd{i+1}_f1mac'] = f1_score(yt, yp, average='macro', zero_division=0)
        metrics[f'd{i+1}_f1wei'] = f1_score(yt, yp, average='weighted', zero_division=0)
        try:
            metrics[f'd{i+1}_qwk'] = cohen_kappa_score(yt, yp, weights='quadratic')
        except Exception:
            metrics[f'd{i+1}_qwk'] = 0.0

    exact = (y_val.values == preds).all(axis=1).mean()
    metrics['exact_match'] = exact
    return metrics


def print_metrics(label, m):
    print(f"\n  -- {label} --")
    for d in [1, 2, 3]:
        print(f"  Day+{d} | Acc: {m[f'd{d}_acc']:.4f} | F1-Mac: {m[f'd{d}_f1mac']:.4f} "
              f"| F1-Wei: {m[f'd{d}_f1wei']:.4f} | QWK: {m[f'd{d}_qwk']:.4f}")
    print(f"  Exact Match (All 3 Days): {m['exact_match']:.4f}")


# ─────────────────────────── XGBoost Tuning ────────────────────────────
def tune_xgboost(X_train, y_train, X_val, y_val, sample_weights):
    print(f"\n  [XGBoost] Running {N_TRIALS} Optuna trials...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'tree_method': 'hist',
        }
        model = MultiOutputClassifier(XGBClassifier(**params))
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val['target_d1'], preds[:, 0])

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'use_label_encoder': False, 'eval_metric': 'mlogloss',
                 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'})
    print(f"  Best XGB params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()}, indent=4)}")

    final = MultiOutputClassifier(XGBClassifier(**best))
    final.fit(X_train, y_train, sample_weight=sample_weights)
    return final, best


# ─────────────────────────── LightGBM Tuning ────────────────────────────
def tune_lightgbm(X_train, y_train, X_val, y_val, sample_weights):
    print(f"\n  [LightGBM] Running {N_TRIALS} Optuna trials...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
        model = MultiOutputClassifier(LGBMClassifier(**params))
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return accuracy_score(y_val['target_d1'], preds[:, 0])

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'class_weight': 'balanced', 'random_state': RANDOM_STATE,
                 'n_jobs': -1, 'verbose': -1})
    print(f"  Best LGBM params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()}, indent=4)}")

    final = MultiOutputClassifier(LGBMClassifier(**best))
    final.fit(X_train, y_train, sample_weight=sample_weights)
    return final, best


# ─────────────────────────── Soft-Voting Ensemble ────────────────────────────
class SoftVotingEnsemble:
    """Soft-voting ensemble for MultiOutputClassifier (XGBoost + LightGBM, 50/50)."""

    def __init__(self, xgb_model, lgbm_model, n_classes=4):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.n_classes = n_classes

    def predict_proba(self, X):
        """Returns list of averaged prob arrays (one per Day horizon)."""
        xgb_proba = self.xgb_model.predict_proba(X)
        lgbm_proba = self.lgbm_model.predict_proba(X)
        avg = []
        for xp, lp in zip(xgb_proba, lgbm_proba):
            n = max(xp.shape[1], lp.shape[1], self.n_classes)
            # Pad XGBoost side if needed
            if xp.shape[1] < n:
                pad = np.zeros((xp.shape[0], n - xp.shape[1]))
                xp = np.hstack([xp, pad])
            # Pad LightGBM side if needed
            if lp.shape[1] < n:
                pad = np.zeros((lp.shape[0], n - lp.shape[1]))
                lp = np.hstack([lp, pad])
            avg.append((xp + lp) / 2.0)
        return avg

    def predict(self, X):
        avg = self.predict_proba(X)
        return np.stack([np.argmax(a, axis=1) for a in avg], axis=1)


# ─────────────────────────── Main ────────────────────────────
def main():
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # Line-buffered output
    
    print(f"\n{'='*60}")
    print(f" Training Landslide Ensemble Model")
    print(f"{'='*60}\n")
    t0 = time.time()

    csv_path = os.path.join(DATASET_DIR, 'training_data_Landslide.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Training data not found at {csv_path}")
        return False
    
    print(f"Required features: {FEATURES}\n")
    X_train, X_val, y_train, y_val, sample_weights = load_and_prepare(csv_path, 'landslide_severity')

    # ── XGBoost ──
    xgb_model, xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, sample_weights)
    xgb_metrics = eval_model(xgb_model, X_val, y_val)
    print_metrics("XGBoost Validation", xgb_metrics)
    xgb_path = os.path.join(MODEL_DIR, 'Landslide_xgboost.pkl')
    joblib.dump(xgb_model, xgb_path)
    print(f"  Saved: {xgb_path}")

    # ── LightGBM ──
    lgbm_model, lgbm_params = tune_lightgbm(X_train, y_train, X_val, y_val, sample_weights)
    lgbm_metrics = eval_model(lgbm_model, X_val, y_val)
    print_metrics("LightGBM Validation", lgbm_metrics)
    lgbm_path = os.path.join(MODEL_DIR, 'Landslide_lightgbm.pkl')
    joblib.dump(lgbm_model, lgbm_path)
    print(f"  Saved: {lgbm_path}")

    # ── Ensemble ──
    ensemble = SoftVotingEnsemble(xgb_model, lgbm_model)
    ens_metrics = eval_model(ensemble, X_val, y_val)
    print_metrics("Ensemble Validation", ens_metrics)
    ens_path = os.path.join(MODEL_DIR, 'Landslide_ensemble.pkl')
    joblib.dump(ensemble, ens_path)
    print(f"  Saved: {ens_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f" Landslide Ensemble Training Completed")
    print(f" Total Time: {elapsed/60:.1f} min")
    print(f"{'='*60}\n")

    # Summary
    scores = {
        'XGBoost': (xgb_metrics['d1_f1mac'] + xgb_metrics['d1_qwk']) / 2,
        'LightGBM': (lgbm_metrics['d1_f1mac'] + lgbm_metrics['d1_qwk']) / 2,
        'Ensemble': (ens_metrics['d1_f1mac'] + ens_metrics['d1_qwk']) / 2,
    }
    best_model = max(scores, key=scores.get)
    print(f" ✓ Best Performing Model: {best_model}")
    print(f" ✓ Model Scores (Avg F1-Macro + QWK):")
    for model_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {model_name}: {score:.4f}")
    print()
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
