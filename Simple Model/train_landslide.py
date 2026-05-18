import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs('Models', exist_ok=True)

print("--- Training Landslide Model ---")
train_df = pd.read_csv('Datasets/landslide_train.csv')
val_df = pd.read_csv('Datasets/landslide_val.csv')

X_train = train_df[['temp', 'hum', 'moist', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']]
y_train = train_df['label']

X_val = val_df[['temp', 'hum', 'moist', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']]
y_val = val_df['label']

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

ensemble_model = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model)], voting='soft')

ensemble_model.fit(X_train, y_train_enc)

y_pred_enc = ensemble_model.predict(X_val)
y_pred = le.inverse_transform(y_pred_enc)

print(classification_report(y_val, y_pred))
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")

joblib.dump(ensemble_model, 'Models/landslide_ensemble_model.pkl')
joblib.dump(le, 'Models/landslide_label_encoder.pkl')

print("Landslide model trained and saved.")
