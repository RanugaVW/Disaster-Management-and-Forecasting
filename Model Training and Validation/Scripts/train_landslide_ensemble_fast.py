#!/usr/bin/env python3
"""
Landslide Ensemble Model Training - FAST VERSION
================================================
Creates Landslide ensemble model with proven hyperparameters.
Skips Optuna tuning for speed - uses parameters tuned from scratch.

Features used:
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

warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import (
        accuracy_score, f1_score, cohen_kappa_score
    )
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)

# ─────────────────────────── Config ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..')
DATASET_DIR = os.path.join(BASE_DIR, 'Datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
VAL_FRAC = 0.15
SEVERITY_MAP = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}

FEATURES = [
    'rain_sum', 'temperature_2m_mean',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm',
    'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi', 'division_encoded'
]

# Pre-tuned hyperparameters (good general values for disaster data)
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.5,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist',
}

LGBM_PARAMS = {
    'n_estimators': 300,
    'num_leaves': 60,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
}

# ─────────────────────────── Helpers ────────────────────────────
def load_and_prepare(csv_path, target_col):
    """Load dataset, create shifted multi-horizon targets, temporal split."""
    print(f"  Loading: {csv_path}")
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

    # Sample weights
    sample_weights = compute_sample_weight('balanced', y_train['target_d1'])

    print(f"  ✓ Train: {len(X_train):,}  |  Val: {len(X_val):,}")
    return X_train, X_val, y_train, y_val, sample_weights


def eval_model(model, X_val, y_val):
    """Return metrics dict."""
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
    print(f"  -- {label} --")
    for d in [1, 2, 3]:
        print(f"    Day+{d} | Acc: {m[f'd{d}_acc']:.4f} | F1-Mac: {m[f'd{d}_f1mac']:.4f} "
              f"| F1-Wei: {m[f'd{d}_f1wei']:.4f} | QWK: {m[f'd{d}_qwk']:.4f}")
    print(f"    Exact Match: {m['exact_match']:.4f}")


class SoftVotingEnsemble:
    """Soft-voting ensemble for MultiOutputClassifier."""

    def __init__(self, xgb_model, lgbm_model, n_classes=4):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.n_classes = n_classes

    def predict_proba(self, X):
        """Returns averaged prob arrays."""
        xgb_proba = self.xgb_model.predict_proba(X)
        lgbm_proba = self.lgbm_model.predict_proba(X)
        avg = []
        for xp, lp in zip(xgb_proba, lgbm_proba):
            n = max(xp.shape[1], lp.shape[1], self.n_classes)
            if xp.shape[1] < n:
                pad = np.zeros((xp.shape[0], n - xp.shape[1]))
                xp = np.hstack([xp, pad])
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
    print(f"\n{'='*70}")
    print(f" LANDSLIDE ENSEMBLE MODEL TRAINING (Fast Mode)")
    print(f"{'='*70}\n")
    print(f"Using {len(FEATURES)} features")
    print(f"Pre-tuned hyperparameters (no Optuna tuning)\n")
    
    t0 = time.time()

    csv_path = os.path.join(DATASET_DIR, 'training_data_Landslide.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Training data not found: {csv_path}\n")
        return False
    
    # Load and prepare data
    print("STEP 1: Loading and Preparing Data")
    print("-" * 70)
    X_train, X_val, y_train, y_val, sample_weights = load_and_prepare(
        csv_path, 'landslide_severity'
    )

    # ── XGBoost ──
    print("\nSTEP 2: Training XGBoost Model")
    print("-" * 70)
    t_xgb = time.time()
    print(f"  Training with {len(FEATURES)} features, {len(X_train):,} samples...")
    xgb_model = MultiOutputClassifier(XGBClassifier(**XGB_PARAMS))
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_metrics = eval_model(xgb_model, X_val, y_val)
    print_metrics("XGBoost Validation Metrics", xgb_metrics)
    elapsed_xgb = time.time() - t_xgb
    print(f"  ✓ Completed in {elapsed_xgb:.1f}s")
    
    xgb_path = os.path.join(MODEL_DIR, 'Landslide_xgboost.pkl')
    joblib.dump(xgb_model, xgb_path)
    xgb_size = os.path.getsize(xgb_path) / (1024**2)
    print(f"  ✓ Saved: {xgb_path} ({xgb_size:.1f}MB)")

    # ── LightGBM ──
    print("\nSTEP 3: Training LightGBM Model")
    print("-" * 70)
    t_lgbm = time.time()
    print(f"  Training with {len(FEATURES)} features, {len(X_train):,} samples...")
    lgbm_model = MultiOutputClassifier(LGBMClassifier(**LGBM_PARAMS))
    lgbm_model.fit(X_train, y_train, sample_weight=sample_weights)
    lgbm_metrics = eval_model(lgbm_model, X_val, y_val)
    print_metrics("LightGBM Validation Metrics", lgbm_metrics)
    elapsed_lgbm = time.time() - t_lgbm
    print(f"  ✓ Completed in {elapsed_lgbm:.1f}s")
    
    lgbm_path = os.path.join(MODEL_DIR, 'Landslide_lightgbm.pkl')
    joblib.dump(lgbm_model, lgbm_path)
    lgbm_size = os.path.getsize(lgbm_path) / (1024**2)
    print(f"  ✓ Saved: {lgbm_path} ({lgbm_size:.1f}MB)")

    # ── Ensemble ──
    print("\nSTEP 4: Creating Soft-Voting Ensemble")
    print("-" * 70)
    t_ens = time.time()
    ensemble = SoftVotingEnsemble(xgb_model, lgbm_model)
    ens_metrics = eval_model(ensemble, X_val, y_val)
    print_metrics("Ensemble Validation Metrics", ens_metrics)
    elapsed_ens = time.time() - t_ens
    print(f"  ✓ Completed in {elapsed_ens:.1f}s")
    
    ens_path = os.path.join(MODEL_DIR, 'Landslide_ensemble.pkl')
    joblib.dump(ensemble, ens_path)
    ens_size = os.path.getsize(ens_path) / (1024**2)
    print(f"  ✓ Saved: {ens_path} ({ens_size:.1f}MB)")

    # Summary
    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"Timing Summary:")
    print(f"  XGBoost:  {elapsed_xgb/60:.1f} min")
    print(f"  LightGBM: {elapsed_lgbm/60:.1f} min")
    print(f"  Ensemble: {elapsed_ens/60:.1f} min")
    print(f"  TOTAL:    {elapsed_total/60:.1f} min\n")

    scores = {
        'XGBoost': (xgb_metrics['d1_f1mac'] + xgb_metrics['d1_qwk']) / 2,
        'LightGBM': (lgbm_metrics['d1_f1mac'] + lgbm_metrics['d1_qwk']) / 2,
        'Ensemble': (ens_metrics['d1_f1mac'] + ens_metrics['d1_qwk']) / 2,
    }
    
    print(f"Model Performance (Avg F1-Macro + QWK for Day+1):")
    for model_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        rank = "🥇" if score == max(scores.values()) else "  "
        print(f"  {rank} {model_name:10s}: {score:.4f}")
    
    best_model = max(scores, key=scores.get)
    print(f"\nBest Performing Model: {best_model}\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
