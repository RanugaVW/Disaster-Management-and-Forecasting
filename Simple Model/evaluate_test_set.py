import pandas as pd
import joblib
from sklearn.metrics import classification_report

print("=== Flood Model Test Report ===")
val_df_f = pd.read_csv('Datasets/flood_val.csv')
model_f = joblib.load('Models/flood_ensemble_model.pkl')
le_f = joblib.load('Models/flood_label_encoder.pkl')
X_f = val_df_f[['temp', 'hum', 'depth_prev', 'depth']]
y_f = val_df_f['label']
preds_f = le_f.inverse_transform(model_f.predict(X_f))
print(classification_report(y_f, preds_f))

print("\n=== Landslide Model Test Report ===")
val_df_l = pd.read_csv('Datasets/landslide_val.csv')
model_l = joblib.load('Models/landslide_ensemble_model.pkl')
le_l = joblib.load('Models/landslide_label_encoder.pkl')
X_l = val_df_l[['temp', 'hum', 'moist', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']]
y_l = val_df_l['label']
preds_l = le_l.inverse_transform(model_l.predict(X_l))
print(classification_report(y_l, preds_l))
