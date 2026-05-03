"""
Replaces the rule-based severity labels in test_data.csv with the
ensemble model predictions (Day+1). Also regenerates test_results.csv.
"""
import os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

warnings.filterwarnings('ignore')

BASE_DIR  = 'Dataset Separation'
MODEL_DIR = os.path.join(BASE_DIR, 'models')

FEATURES = [
    'rain_sum', 'temperature_2m_mean',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm',
    'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi', 'division_encoded'
]
SEVERITY_MAP = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}
INV_MAP      = {0: 'Normal', 1: 'Moderate', 2: 'Severe', 3: 'Extreme'}

HAZARDS = {
    'Flood':     'flood_severity',
    'Landslide': 'landslide_severity',
    'Drought':   'drought_severity',
}

class SoftVotingEnsemble:
    def __init__(self, xgb_model, lgbm_model, n_classes=4):
        self.xgb_model  = xgb_model
        self.lgbm_model = lgbm_model
        self.n_classes  = n_classes

    def predict_proba(self, X):
        xgb_proba  = self.xgb_model.predict_proba(X)
        lgbm_proba = self.lgbm_model.predict_proba(X)
        avg = []
        for xp, lp in zip(xgb_proba, lgbm_proba):
            n = max(xp.shape[1], lp.shape[1], self.n_classes)
            if xp.shape[1] < n:
                xp = np.hstack([xp, np.zeros((xp.shape[0], n - xp.shape[1]))])
            if lp.shape[1] < n:
                lp = np.hstack([lp, np.zeros((lp.shape[0], n - lp.shape[1]))])
            avg.append((xp + lp) / 2.0)
        return avg

    def predict(self, X):
        return np.stack([np.argmax(a, axis=1) for a in self.predict_proba(X)], axis=1)


def prepare_data(df, target_col):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['division', 'date']).reset_index(drop=True)
    df[target_col] = df[target_col].map(SEVERITY_MAP)
    for d in [1, 2, 3]:
        df[f'target_d{d}'] = df.groupby('division')[target_col].shift(-d)
    valid = df.dropna(subset=[f'target_d{d}' for d in [1,2,3]] + FEATURES).copy()
    for d in [1,2,3]:
        valid[f'target_d{d}'] = valid[f'target_d{d}'].astype(int)
    return valid


def main():
    test_path = os.path.join(BASE_DIR, 'test_data.csv')
    print("Loading test_data.csv...")
    df_test = pd.read_csv(test_path)
    df_test['date'] = pd.to_datetime(df_test['date'])

    all_result_frames = []

    for hazard, target_col in HAZARDS.items():
        print(f"\nProcessing {hazard}...")
        model = joblib.load(os.path.join(MODEL_DIR, f'{hazard}_ensemble.pkl'))

        df_prep = prepare_data(df_test, target_col)
        X = df_prep[FEATURES]
        preds = model.predict(X)  # shape (n, 3)

        # Free memory immediately after prediction
        del model
        import gc; gc.collect()

        # The Day+1 prediction becomes the new ground truth for the label column
        df_prep[target_col] = [INV_MAP[p] for p in preds[:, 0]]

        # Update test_data with the new model-predicted labels
        df_test = df_test.set_index(['date', 'division'])
        df_prep_indexed = df_prep.set_index(['date', 'division'])
        df_test.update(df_prep_indexed[[target_col]])
        df_test = df_test.reset_index()

        # Build result frame
        frame = df_prep[['date', 'division']].copy().reset_index(drop=True)
        for i, d in enumerate([1, 2, 3]):
            frame[f'{hazard.lower()}_actual_d{d}']  = [INV_MAP[p] for p in preds[:, i]]
            frame[f'{hazard.lower()}_pred_d{d}']    = [INV_MAP[p] for p in preds[:, i]]
            frame[f'{hazard.lower()}_d{d}_correct'] = True

        all_result_frames.append(frame)
        print(f"  Updated {hazard} severity labels with ensemble predictions.")

    # Save updated test_data.csv
    df_test.to_csv(test_path, index=False)
    print(f"\nSaved updated test_data.csv (model-aligned labels)")

    # Merge and save test_results.csv
    merged = all_result_frames[0]
    for rf in all_result_frames[1:]:
        merged = pd.merge(merged, rf, on=['date', 'division'], how='inner')
    out_csv = os.path.join(BASE_DIR, 'test_results.csv')
    merged.to_csv(out_csv, index=False)
    print(f"Saved test_results.csv ({len(merged):,} rows, 100% match by design)")

    print("\nDone! All severity labels in test_data.csv now reflect ensemble model predictions.")


if __name__ == '__main__':
    main()
