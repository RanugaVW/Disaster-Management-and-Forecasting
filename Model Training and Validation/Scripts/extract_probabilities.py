"""
Extracts Day+1 probabilistic predictions from all 3 ensemble models
(Flood, Landslide, Drought) on the 2024-2025 test dataset.

Output CSV columns per hazard:
  date | division | flood_severity | flood_probability
  date | division | landslide_severity | landslide_probability
  date | division | drought_severity | drought_probability
  date | division | flood_risk_score | landslide_risk_score | drought_risk_score

Also saves a separate CSV per hazard and a merged all-hazards CSV.

probability = P(predicted class) from the ensemble's averaged softmax output
risk_probability = P(Severe) + P(Extreme)  (non-Normal crisis probability)
"""
import os, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

BASE_DIR  = r'C:\Users\Ranuga\Disaster Managemet System J2\Main Repo'
MODEL_DIR = r'C:\Users\Ranuga\Disaster Managemet System J2\Dataset Preprocessing\models'
OUT_DIR   = os.path.join(BASE_DIR, 'Datasets')
TEST_CSV  = os.path.join(OUT_DIR, 'test_data.csv')

FEATURES = [
    'rain_sum', 'temperature_2m_mean',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm',
    'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi', 'division_encoded'
]
SEVERITY_MAP = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}
INV_MAP      = {0: 'Normal', 1: 'Moderate', 2: 'Severe', 3: 'Extreme'}
CLASS_LABELS = ['Normal', 'Moderate', 'Severe', 'Extreme']

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
            if xp.shape[1] < n: xp = np.hstack([xp, np.zeros((xp.shape[0], n-xp.shape[1]))])
            if lp.shape[1] < n: lp = np.hstack([lp, np.zeros((lp.shape[0], n-lp.shape[1]))])
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
    df = df.dropna(subset=[f'target_d{d}' for d in [1,2,3]] + FEATURES).copy()
    for d in [1, 2, 3]:
        df[f'target_d{d}'] = df[f'target_d{d}'].astype(int)
    return df[FEATURES], df[['target_d1', 'target_d2', 'target_d3']], df[['date', 'division', 'target_d1']]


os.makedirs(OUT_DIR, exist_ok=True)

print("Loading test_data.csv ...")
df_test = pd.read_csv(TEST_CSV)

all_hazard_frames = []

for hazard, target_col in HAZARDS.items():
    print(f"\n{'='*55}")
    print(f" Processing: {hazard}")
    print(f"{'='*55}")

    X, y, meta = prepare_data(df_test, target_col)

    model = joblib.load(os.path.join(MODEL_DIR, f'{hazard}_ensemble.pkl'))

    # ── Get full probability arrays (Day+1 is index 0) ────────────────────────
    proba_list = model.predict_proba(X)   # list of 3 arrays, each (n, 4)
    proba_d1   = proba_list[0]            # shape (n, 4) for Day+1

    # ── Predicted class (argmax) ───────────────────────────────────────────────
    pred_class   = np.argmax(proba_d1, axis=1)           # 0/1/2/3
    pred_label   = [INV_MAP[c] for c in pred_class]      # text label

    # ── Probability of the predicted class (confidence) ───────────────────────
    pred_prob = proba_d1[np.arange(len(proba_d1)), pred_class]   # scalar per row

    # ── Crisis probability: P(Severe) + P(Extreme) ────────────────────────────
    crisis_prob = proba_d1[:, 2] + proba_d1[:, 3]

    # ── Individual class probabilities ────────────────────────────────────────
    h = hazard.lower()
    frame = meta.copy().reset_index(drop=True)
    frame['date'] = pd.to_datetime(frame['date']).dt.date

    frame[f'{h}_actual_severity']   = frame['target_d1'].map(INV_MAP)
    frame[f'{h}_predicted_severity'] = pred_label
    frame[f'{h}_predicted_probability'] = np.round(pred_prob, 6)    # P(predicted class)
    frame[f'{h}_crisis_probability']    = np.round(crisis_prob, 6)  # P(Severe or Extreme)
    frame[f'{h}_p_normal']    = np.round(proba_d1[:, 0], 6)
    frame[f'{h}_p_moderate']  = np.round(proba_d1[:, 1], 6)
    frame[f'{h}_p_severe']    = np.round(proba_d1[:, 2], 6)
    frame[f'{h}_p_extreme']   = np.round(proba_d1[:, 3], 6)

    frame = frame.drop(columns=['target_d1'])

    # ── Save per-hazard CSV ────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f'{hazard.lower()}_probabilistic_predictions.csv')
    frame.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(frame):,} rows)")
    print(f"  Sample P(Severe+Extreme) range: "
          f"{crisis_prob.min():.4f} – {crisis_prob.max():.4f}")

    all_hazard_frames.append(frame)

# ── Merged all-hazards CSV ─────────────────────────────────────────────────────
print("\nMerging all hazards ...")
merged = all_hazard_frames[0]
for f in all_hazard_frames[1:]:
    merged = pd.merge(merged, f, on=['date', 'division'], how='inner')

merged_path = os.path.join(OUT_DIR, 'all_hazards_probabilistic_predictions.csv')
merged.to_csv(merged_path, index=False)
print(f"Saved merged: {merged_path}  ({len(merged):,} rows, {len(merged.columns)} columns)")

# ── Print column summary ───────────────────────────────────────────────────────
print("\n\nColumns in merged CSV:")
for col in merged.columns:
    print(f"  {col}")

print(f"\nDone! All files saved to: {OUT_DIR}")
