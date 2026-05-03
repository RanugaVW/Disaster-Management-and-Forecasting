"""
Evaluates all 9 models on the 2024-2026 test_data.csv and:
  1. Produces test_results.csv with actual vs predicted for all 3 hazards x 3 days
  2. Appends the full results section into Model_Training_Report.md
"""
import os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

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

MODELS = ['xgboost', 'lightgbm', 'ensemble']

# ─── Ensemble class needed for unpickling ───
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


def prepare_test_data(df, target_col):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['division', 'date']).reset_index(drop=True)
    df[target_col] = df[target_col].map(SEVERITY_MAP)
    for d in [1, 2, 3]:
        df[f'target_d{d}'] = df.groupby('division')[target_col].shift(-d)
    df = df.dropna(subset=[f'target_d{d}' for d in [1,2,3]] + FEATURES)
    for d in [1,2,3]:
        df[f'target_d{d}'] = df[f'target_d{d}'].astype(int)
    X = df[FEATURES]
    y = df[['target_d1','target_d2','target_d3']]
    return X, y, df


def compute_metrics(y_true, y_pred):
    m = {}
    for i, day in enumerate(['d1','d2','d3']):
        yt = y_true.iloc[:, i].values
        yp = y_pred[:, i]
        m[f'{day}_acc']   = accuracy_score(yt, yp)
        m[f'{day}_f1mac'] = f1_score(yt, yp, average='macro',    zero_division=0)
        m[f'{day}_f1wei'] = f1_score(yt, yp, average='weighted', zero_division=0)
        try:    m[f'{day}_qwk'] = cohen_kappa_score(yt, yp, weights='quadratic')
        except: m[f'{day}_qwk'] = float('nan')
    m['exact'] = (y_true.values == y_pred).all(axis=1).mean()
    return m


def main():
    print("Loading test data (2024-2026)...")
    df_test = pd.read_csv(os.path.join(BASE_DIR, 'test_data.csv'))

    all_metrics = {}
    result_frames = []

    for hazard, target_col in HAZARDS.items():
        print(f"\n{'='*50}")
        print(f" Testing: {hazard.upper()}")
        print(f"{'='*50}")

        X_test, y_test, df_meta = prepare_test_data(df_test, target_col)

        hazard_metrics = {}
        hazard_preds   = {}

        for model_name in MODELS:
            pkl_path = os.path.join(MODEL_DIR, f'{hazard}_{model_name}.pkl')
            print(f"  Loading {os.path.basename(pkl_path)}...")
            model = joblib.load(pkl_path)

            preds = model.predict(X_test)
            hazard_preds[model_name] = preds
            m = compute_metrics(y_test, preds)
            hazard_metrics[model_name] = m

            print(f"  [{model_name.upper()}] D+1 Acc:{m['d1_acc']:.4f} | F1-Mac:{m['d1_f1mac']:.4f} | QWK:{m['d1_qwk']:.4f} | Exact:{m['exact']:.4f}")

        all_metrics[hazard] = hazard_metrics

        ens_preds = hazard_preds['ensemble']
        xgb_preds = hazard_preds['xgboost']
        lgb_preds = hazard_preds['lightgbm']
        frame = df_meta[['date','division', target_col]].copy().reset_index(drop=True)

        # Actual labels (decoded)
        frame[f'{hazard.lower()}_actual_d1'] = y_test['target_d1'].values
        frame[f'{hazard.lower()}_actual_d2'] = y_test['target_d2'].values
        frame[f'{hazard.lower()}_actual_d3'] = y_test['target_d3'].values

        # Ensemble predictions (decoded to text)
        frame[f'{hazard.lower()}_ensemble_pred_d1'] = [INV_MAP[p] for p in ens_preds[:, 0]]
        frame[f'{hazard.lower()}_ensemble_pred_d2'] = [INV_MAP[p] for p in ens_preds[:, 1]]
        frame[f'{hazard.lower()}_ensemble_pred_d3'] = [INV_MAP[p] for p in ens_preds[:, 2]]

        # XGBoost predictions
        frame[f'{hazard.lower()}_xgboost_pred_d1'] = [INV_MAP[p] for p in xgb_preds[:, 0]]
        frame[f'{hazard.lower()}_xgboost_pred_d2'] = [INV_MAP[p] for p in xgb_preds[:, 1]]
        frame[f'{hazard.lower()}_xgboost_pred_d3'] = [INV_MAP[p] for p in xgb_preds[:, 2]]

        # LightGBM predictions
        frame[f'{hazard.lower()}_lightgbm_pred_d1'] = [INV_MAP[p] for p in lgb_preds[:, 0]]
        frame[f'{hazard.lower()}_lightgbm_pred_d2'] = [INV_MAP[p] for p in lgb_preds[:, 1]]
        frame[f'{hazard.lower()}_lightgbm_pred_d3'] = [INV_MAP[p] for p in lgb_preds[:, 2]]

        # Decode actual to text
        frame[f'{hazard.lower()}_actual_d1'] = frame[f'{hazard.lower()}_actual_d1'].map(INV_MAP)
        frame[f'{hazard.lower()}_actual_d2'] = frame[f'{hazard.lower()}_actual_d2'].map(INV_MAP)
        frame[f'{hazard.lower()}_actual_d3'] = frame[f'{hazard.lower()}_actual_d3'].map(INV_MAP)

        # Match flags for Ensemble
        frame[f'{hazard.lower()}_d1_correct'] = frame[f'{hazard.lower()}_actual_d1'] == frame[f'{hazard.lower()}_ensemble_pred_d1']
        frame[f'{hazard.lower()}_d2_correct'] = frame[f'{hazard.lower()}_actual_d2'] == frame[f'{hazard.lower()}_ensemble_pred_d2']
        frame[f'{hazard.lower()}_d3_correct'] = frame[f'{hazard.lower()}_actual_d3'] == frame[f'{hazard.lower()}_ensemble_pred_d3']



        frame = frame.drop(columns=[target_col])
        result_frames.append(frame)

    # Merge all hazard results on date + division
    print("\nMerging results into single CSV...")
    merged = result_frames[0]
    for rf in result_frames[1:]:
        merged = pd.merge(merged, rf, on=['date', 'division'], how='inner')

    out_csv = os.path.join(BASE_DIR, 'test_results.csv')
    merged.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(merged):,} rows)")

    # ─── Append Results to Report ───
    update_report(all_metrics)


def update_report(all_metrics):
    report_path = os.path.join(BASE_DIR, 'Model_Training_Report.md')

    section = [
        "",
        "---",
        "",
        "## Test Set Results (2024-2026 Unseen Data)",
        "",
        "These scores reflect how each model performs on data it has **never seen** during training.",
        "The test set covers all 121 Sri Lankan divisions from 2024 to 2026.",
        "",
    ]

    for hazard, models in all_metrics.items():
        xm = models['xgboost']
        lm = models['lightgbm']
        em = models['ensemble']

        section += [
            f"### {hazard} — Test Set Scores",
            "",
            "| Model | D+1 Acc | D+1 F1-Mac | D+1 QWK | D+1 F1-Wei | Exact Match |",
            "|---|---|---|---|---|---|",
            f"| XGBoost  | {xm['d1_acc']:.4f} | {xm['d1_f1mac']:.4f} | {xm['d1_qwk']:.4f} | {xm['d1_f1wei']:.4f} | {xm['exact']:.4f} |",
            f"| LightGBM | {lm['d1_acc']:.4f} | {lm['d1_f1mac']:.4f} | {lm['d1_qwk']:.4f} | {lm['d1_f1wei']:.4f} | {lm['exact']:.4f} |",
            f"| **Ensemble** | **{em['d1_acc']:.4f}** | **{em['d1_f1mac']:.4f}** | **{em['d1_qwk']:.4f}** | **{em['d1_f1wei']:.4f}** | **{em['exact']:.4f}** |",
            "",
            "> Predictions from the **Ensemble model** have been saved row-by-row into `test_results.csv`",
            "",
        ]

    section += [
        "---",
        "",
        "## Data Attributes Required for Model Inference",
        "",
        "When feeding live data from Open-Meteo into the model, you must provide the following attributes.",
        "The first 5 come **directly** from the Open-Meteo API response.",
        "The remaining are **computed on-the-fly** before passing to the model.",
        "",
        "### 1. Raw Inputs (Fetch from Open-Meteo API)",
        "",
        "| Attribute | Type | Unit | Description |",
        "|---|---|---|---|",
        "| `date` | String | YYYY-MM-DD | The date of the forecast |",
        "| `division` | String | — | Sri Lankan Divisional Secretariat name (e.g. 'Colombo') |",
        "| `rain_sum` | Float | mm | Total daily rainfall |",
        "| `temperature_2m_mean` | Float | °C | Mean daily temperature at 2m height |",
        "| `soil_moisture_7_to_28cm` | Float | m³/m³ | Volumetric soil water, shallow layer |",
        "| `soil_moisture_28_to_100cm` | Float | m³/m³ | Volumetric soil water, mid layer |",
        "| `soil_moisture_100_to_255cm` | Float | m³/m³ | Volumetric soil water, deep layer |",
        "",
        "### 2. Computed Features (Calculate Before Inference)",
        "",
        "| Feature | How to Compute |",
        "|---|---|",
        "| `rain_lag_1` | `rain_sum` from yesterday for the same division |",
        "| `rain_rolling_3d` | Sum of `rain_sum` over the last 3 days for the same division |",
        "| `rain_rolling_7d` | Sum of `rain_sum` over the last 7 days for the same division |",
        "| `month_sin` | `sin(2 * pi * month / 12)` where month is the integer month (1-12) |",
        "| `month_cos` | `cos(2 * pi * month / 12)` |",
        "| `spi` | 1-Day Standardised Precipitation Index via Gamma distribution fit |",
        "| `division_encoded` | Integer encoding via `master_division_encoder.pkl` |",
        "",
        "### 3. Final Feature Array Passed to Model (in exact order)",
        "",
        "```python",
        "features = [",
        "    'rain_sum', 'temperature_2m_mean',",
        "    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',",
        "    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',",
        "    'month_sin', 'month_cos', 'spi', 'division_encoded'",
        "]",
        "```",
        "",
        "### 4. Model Output",
        "",
        "Each ensemble model returns predictions for **3 future days simultaneously**:",
        "",
        "| Output | Meaning |",
        "|---|---|",
        "| `Day+1 Severity` | Severity forecast for tomorrow |",
        "| `Day+2 Severity` | Severity forecast for the day after tomorrow |",
        "| `Day+3 Severity` | Severity forecast for 3 days from now |",
        "",
        "Each severity is one of: `Normal`, `Moderate`, `Severe`, `Extreme`",
        "",
        "For a **continuous probability** (0.000 to 1.000), use `model.predict_proba()` instead of `model.predict()`:",
        "```python",
        "proba_list = model.predict_proba(X)  # List of 3 arrays [Day1, Day2, Day3]",
        "flood_prob_day1 = sum(proba_list[0][0][1:])  # P(Moderate) + P(Severe) + P(Extreme)",
        "```",
        "",
        "---",
        "*Full test predictions saved in: `Dataset Separation/test_results.csv`*",
    ]

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '\n'.join(section))

    print(f"Report updated: {report_path}")


if __name__ == '__main__':
    main()
