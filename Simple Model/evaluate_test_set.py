import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score
)

def evaluate_model(model, le, X, y, model_name):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Metrics used and why:
    - Accuracy: Overall correctness (but can be misleading for imbalanced data)
    - Precision: Of predicted positives, how many are correct? (minimize false alarms)
    - Recall (Sensitivity): Of actual positives, how many did we catch? (CRITICAL - we can't miss disasters)
    - Specificity: Of actual negatives, how many did we correctly identify? (true negative rate)
    - F1-Score: Harmonic mean of precision & recall (balanced metric)
    - MCC (Matthews Correlation Coefficient): Correlation between predicted and actual (handles imbalanced data)
    - Confusion Matrix: Breakdown of TP, FP, TN, FN for detailed analysis
    - ROC-AUC: Model's discrimination ability across different thresholds
    
    For disaster prediction, RECALL is most important because missing a real disaster
    (false negative) has severe consequences (lives at risk), while false positives
    (unnecessary alerts) have lower cost.
    """
    
    # Get predictions
    preds_enc = model.predict(X)
    preds = le.inverse_transform(preds_enc)
    
    # Get probabilities for ROC-AUC (soft voting gets probabilities)
    try:
        probs = model.predict_proba(X)
        # For multi-class, use One-vs-Rest approach
        if probs.shape[1] > 2:
            roc_auc = roc_auc_score(le.transform(y), probs, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(y, probs[:, 1])
    except:
        roc_auc = None
    
    # Calculate metrics
    accuracy = accuracy_score(y, preds)
    precision_macro = precision_score(y, preds, average='macro', zero_division=0)
    recall_macro = recall_score(y, preds, average='macro', zero_division=0)
    f1_macro = f1_score(y, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y, preds)
    
    # Confusion matrix
    cm = confusion_matrix(y, preds)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"{'='*70}")
    
    print("\n--- AGGREGATE METRICS ---")
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Precision (macro avg):   {precision_macro:.4f}")
    print(f"Recall (macro avg):      {recall_macro:.4f}  *** MOST CRITICAL FOR DISASTER PREDICTION ***")
    print(f"F1-Score (macro avg):    {f1_macro:.4f}")
    print(f"Matthews Correlation:    {mcc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:                 {roc_auc:.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    print("(Rows: Actual, Columns: Predicted)")
    print(cm)
    
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print("(Shows precision, recall, f1-score per class)")
    print(classification_report(y, preds, zero_division=0))
    
    # Calculate specificity and sensitivity per class
    print("\n--- PER-CLASS ANALYSIS ---")
    for i, class_label in enumerate(le.classes_):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - cm[i, :].sum() - fp
        fn = cm[i, :].sum() - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nClass '{class_label}':")
        print(f"  True Positives:  {tp}   (correctly predicted as {class_label})")
        print(f"  False Positives: {fp}   (incorrectly predicted as {class_label})")
        print(f"  True Negatives:  {tn}   (correctly predicted as NOT {class_label})")
        print(f"  False Negatives: {fn}   (incorrectly predicted as NOT {class_label})")
        print(f"  Sensitivity (Recall):    {sensitivity:.4f}  (of actual {class_label}, caught {sensitivity*100:.2f}%)")
        print(f"  Specificity:             {specificity:.4f}  (of actual non-{class_label}, correct {specificity*100:.2f}%)")

# Flood Model Evaluation
print("\n" + "="*70)
print("FLOOD MODEL EVALUATION")
print("="*70)
val_df_f = pd.read_csv('Datasets/flood_val.csv')
model_f = joblib.load('Models/flood_ensemble_model.pkl')
le_f = joblib.load('Models/flood_label_encoder.pkl')
X_f = val_df_f[['temp', 'hum', 'depth_prev', 'depth']]
y_f = val_df_f['label']
evaluate_model(model_f, le_f, X_f, y_f, "=== FLOOD MODEL TEST REPORT ===")

# Landslide Model Evaluation
print("\n" + "="*70)
print("LANDSLIDE MODEL EVALUATION")
print("="*70)
val_df_l = pd.read_csv('Datasets/landslide_val.csv')
model_l = joblib.load('Models/landslide_ensemble_model.pkl')
le_l = joblib.load('Models/landslide_label_encoder.pkl')
X_l = val_df_l[['temp', 'hum', 'moist', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']]
y_l = val_df_l['label']
evaluate_model(model_l, le_l, X_l, y_l, "=== LANDSLIDE MODEL TEST REPORT ===")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
