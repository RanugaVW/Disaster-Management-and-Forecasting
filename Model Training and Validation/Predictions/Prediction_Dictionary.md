# Prediction Column Dictionary: Probabilistic Output Guide

This document explains the structure and meaning of the columns in the generated disaster prediction datasets (e.g., `all_hazards_probabilistic_predictions.csv`).

## 1. Why are there so many columns?
Unlike simple "Yes/No" models, this system is a **Multi-Class Ordinal Classifier**. It doesn't just predict *if* a disaster will happen; it predicts the **Severity Level**.

For every single day and division, the model calculates **four separate probabilities** that always sum to 100%:
1. Probability of being **Normal**
2. Probability of being **Moderate**
3. Probability of being **Severe**
4. Probability of being **Extreme**

The multiple columns give you full transparency into the model's "internal thoughts" and confidence levels.

---

## 2. Column Definitions (Per Hazard)

Taking **Flood** as an example, here is what each column actually means:

| Column Name | Type | Description |
|---|---|---|
| `flood_actual_severity` | Text | The "Ground Truth" — what actually happened on that day (or the shifted label we are testing against). |
| `flood_predicted_severity` | Text | **The Final Answer.** This is the class (Normal, Moderate, Severe, or Extreme) that the model thinks is most likely. |
| `flood_predicted_probability` | 0.0 – 1.0 | **Model Confidence.** How certain is the model about its `predicted_severity`? (e.g., 0.85 means 85% certain). |
| **`flood_crisis_probability`** | 0.0 – 1.0 | **The Danger Gauge.** This is the sum of `p_severe` + `p_extreme`. **This is the most important value for risk mapping.** It represents the total probability of a high-impact event. |
| `flood_p_normal` | 0.0 – 1.0 | Probability that conditions will remain Normal. |
| `flood_p_moderate` | 0.0 – 1.0 | Probability of reaching Moderate levels (low risk). |
| `flood_p_severe` | 0.0 – 1.0 | Probability of reaching Severe levels (high risk). |
| `flood_p_extreme` | 0.0 – 1.0 | Probability of reaching Extreme levels (critical emergency). |

---

## 3. On which values should we look?

Depending on your objective, you should focus on different columns:

### A. For Dashboard Alerts (The Map)
**Look at:** `flood_predicted_severity`
- Use this to color-code divisions on a map (Green = Normal, Yellow = Moderate, Orange = Severe, Red = Extreme).
- This is the "Hard Prediction."

### B. For Risk Scoring & Ranking (The Priority List)
**Look at:** `flood_crisis_probability`
- Use this if you want a **continuous value** to sort by. 
- A division with `crisis_probability` of 0.95 is much more dangerous than one with 0.55, even if they are both labeled "Extreme."
- **Risk Score Formula:** `crisis_probability` × `Population`.

### C. For Model Debugging / Uncertainty
**Look at:** `flood_predicted_probability`
- If this value is low (e.g., 0.35), the model is "confused" or the situation is highly uncertain.
- If it is high (e.g., 0.98), the model is extremely confident in its alert.

---

## 4. Example Interpretation

**Case Study: Akurana on 2024-01-01**
- `actual_severity`: **Moderate** (What actually happened)
- `predicted_severity`: **Moderate** (The Model's Choice)
- `predicted_probability`: **0.396833**
- `crisis_probability`: **0.482496**
- `p_normal`: 0.120671
- `p_moderate`: 0.396833
- `p_severe`: 0.193918
- `p_extreme`: 0.288578

**What this actually means:**
In this case, the model **correctly predicted the severity level** (Moderate), matching the actual data. Even though it chose **Moderate** (because it's the single highest value at ~40%), the **Crisis Probability was ~48%**. This indicates the model saw a very high risk of the situation escalating further.

**This is why looking at the Probabilities is more powerful than just the Labels!**
