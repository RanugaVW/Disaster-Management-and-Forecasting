import pandas as pd, numpy as np, joblib

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

df = pd.read_csv('Dataset Separation/test_data.csv')
model = joblib.load('Dataset Separation/models/Flood_ensemble.pkl')
FEATURES = ['rain_sum', 'temperature_2m_mean', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm', 'rain_lag_1', 'rain_rolling_3d', 'rain_rolling_7d', 'month_sin', 'month_cos', 'spi', 'division_encoded']
valid_mask = df[FEATURES].notna().all(axis=1)

xgb_preds = model.xgb_model.predict(df[valid_mask][FEATURES])[:, 0]
lgb_preds = model.lgbm_model.predict(df[valid_mask][FEATURES])[:, 0]
ens_preds = model.predict(df[valid_mask][FEATURES])[:, 0]

agree = (xgb_preds == ens_preds) & (lgb_preds == ens_preds)
print('All three agree on:', np.mean(agree))
print('XGB agrees with Ens on:', np.mean(xgb_preds == ens_preds))
print('LGB agrees with Ens on:', np.mean(lgb_preds == ens_preds))
