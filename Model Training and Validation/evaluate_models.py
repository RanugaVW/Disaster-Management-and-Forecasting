import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

base_dir = "/home/ranuga-weerasekara/Desktop/Disaster-Management-and-Forecasting/Model Training and Validation"
models_dir = os.path.join(base_dir, "Models")
data_path = os.path.join(base_dir, "Datasets/test_data.csv")

SEVERITY_MAP = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}
INV_SEVERITY = {v: k for k, v in SEVERITY_MAP.items()}

FEATURES = [
    'rain_sum', 'temperature_2m_mean',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm',
    'soil_moisture_100_to_255cm',
    'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d',
    'month_sin', 'month_cos', 'spi', 'division_encoded'
]

class SoftVotingEnsemble:
    """Soft-voting ensemble for MultiOutputClassifier (XGBoost + LightGBM, 50/50)."""
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

def load_and_prepare(csv_path, target_col):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['division', 'date']).reset_index(drop=True)
    df[target_col] = df[target_col].map(SEVERITY_MAP)
    for day in [1, 2, 3]:
        df[f'target_d{day}'] = df.groupby('division')[target_col].shift(-day)
    df = df.dropna(subset=[f'target_d{d}' for d in [1, 2, 3]] + FEATURES)
    for d in [1, 2, 3]:
        df[f'target_d{d}'] = df[f'target_d{d}'].astype(int)
    X = df[FEATURES]
    y = df[['target_d1', 'target_d2', 'target_d3']]
    return X, y

models = {
    "Drought": ("Drought_ensemble.pkl", "drought_severity"),
    "Flood": ("Flood_ensemble.pkl", "flood_severity"),
    "Landslide": ("Landslide_ensemble.pkl", "landslide_severity")
}

output_lines = []

for name, (model_file, target_col) in models.items():
    output_lines.append(f"# {name} Model Evaluation\n")
    model_path = os.path.join(models_dir, model_file)
    model = joblib.load(model_path)
    
    X_test, y_test = load_and_prepare(data_path, target_col)
    preds = model.predict(X_test)
    
    class_names = [INV_SEVERITY[i] for i in range(4)]
    
    for i, day in enumerate(['target_d1', 'target_d2', 'target_d3']):
        output_lines.append(f"## Day+{i+1} Horizon\n")
        
        yt = y_test[day].values
        yp = preds[:, i]
        
        # Determine which classes are present in the predictions and true labels
        present_classes = sorted(list(set(yt).union(set(yp))))
        present_class_names = [INV_SEVERITY[c] for c in present_classes]
        
        rep = classification_report(yt, yp, labels=present_classes, target_names=present_class_names, zero_division=0)
        output_lines.append("### Classification Report\n")
        output_lines.append("```\n" + rep + "\n```\n")
        
        cm = confusion_matrix(yt, yp, labels=present_classes)
        output_lines.append("### Confusion Matrix\n")
        output_lines.append("```\n")
        
        # Create a nice formatted confusion matrix
        header = f"{'True \ Pred':<15}" + "".join([f"{cn:<10}" for cn in present_class_names])
        output_lines.append(header + "\n")
        for idx, row in enumerate(cm):
            row_str = f"{present_class_names[idx]:<15}" + "".join([f"{val:<10}" for val in row])
            output_lines.append(row_str + "\n")
            
        output_lines.append("```\n\n")

with open(os.path.join(base_dir, 'model_eval_report.md'), 'w') as f:
    f.writelines(output_lines)
    
print("Evaluation complete. Report saved to model_eval_report.md")
