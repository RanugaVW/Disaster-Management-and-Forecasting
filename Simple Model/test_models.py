"""
Script to test models explicitly with formatted JSON outputs as requested:
Flood Data Format
{
  "id": "J1_TX_01",
  "type": "FLOOD",
  "temp": 28.5,
  "hum": 65.0,
  "depth": 1.24
}
(note: training data used depth_prev to allow depth difference logic, so we'll simulate depth_prev as well, but for inference we will assume we maintain state or provide it).

Landslide Data Format
{
  "id": "J1_TX_02",
  "type": "LANDSLIDE",
  "temp": 22.3,
  "hum": 55.0,
  "moist": 512,
  "ax": 0.15,
  "ay": -0.08,
  "az": 9.81,
  "gx": 0.02,
  "gy": 0.01,
  "gz": -0.03
}
"""
import joblib
import json
import random

flood_model = joblib.load('Models/flood_ensemble_model.pkl')
flood_le = joblib.load('Models/flood_label_encoder.pkl')

landslide_model = joblib.load('Models/landslide_ensemble_model.pkl')
landslide_le = joblib.load('Models/landslide_label_encoder.pkl')

def test_flood_sample():
    # simulate some data like JSON
    req = {
      "id": "J1_TX_01",
      "type": "FLOOD",
      "temp": 28.5,
      "hum": 85.0,
      "depth": 10.5
    }
    # To compute difference, assuming previous depth is maintained in cache, here we mock it
    prev_depth = 5.0 
    
    # Feature order: temp, hum, depth_prev, depth
    features = [[req["temp"], req["hum"], prev_depth, req["depth"]]]
    pred_enc = flood_model.predict(features)
    pred_label = flood_le.inverse_transform(pred_enc)[0]
    
    out = req.copy()
    out["predicted_status"] = pred_label
    print("\n--- Flood Test Result ---")
    print(json.dumps(out, indent=2))
    
def test_landslide_sample():
    req = {
      "id": "J1_TX_02",
      "type": "LANDSLIDE",
      "temp": 22.3,
      "hum": 90.0,
      "moist": 2500,
      "ax": 1.5,
      "ay": -0.8,
      "az": 9.81,
      "gx": 2.02,
      "gy": 0.01,
      "gz": -0.03
    }
    
    # Feature order: temp, hum, moist, ax, ay, az, gx, gy, gz
    features = [[req["temp"], req["hum"], req["moist"], req["ax"], req["ay"], req["az"], req["gx"], req["gy"], req["gz"]]]
    
    pred_enc = landslide_model.predict(features)
    pred_label = landslide_le.inverse_transform(pred_enc)[0]
    
    out = req.copy()
    out["predicted_status"] = pred_label
    print("\n--- Landslide Test Result ---")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    test_flood_sample()
    test_landslide_sample()
    print("\nTesting Complete.")
