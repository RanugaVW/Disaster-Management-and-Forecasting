# Drought Model Evaluation
## Day+1 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       1.00      0.92      0.96     80509
    Moderate       0.76      0.92      0.84      7409
      Severe       0.05      0.93      0.10       141
     Extreme       0.06      0.94      0.11       150

    accuracy                           0.92     88209
   macro avg       0.47      0.93      0.50     88209
weighted avg       0.97      0.92      0.94     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         74107     2119      2139      2144      
Moderate       194       6838      199       178       
Severe         6         2         131       2         
Extreme        4         2         3         141       
```

## Day+2 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       1.00      0.94      0.97     80515
    Moderate       0.83      0.93      0.88      7403
      Severe       0.07      0.93      0.13       141
     Extreme       0.08      0.93      0.14       150

    accuracy                           0.94     88209
   macro avg       0.49      0.93      0.53     88209
weighted avg       0.98      0.94      0.96     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         75997     1460      1520      1538      
Moderate       171       6912      163       157       
Severe         4         3         131       3         
Extreme        5         3         2         140       
```

## Day+3 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       1.00      0.93      0.96     80529
    Moderate       0.79      0.92      0.85      7390
      Severe       0.06      0.94      0.12       140
     Extreme       0.07      0.95      0.13       150

    accuracy                           0.93     88209
   macro avg       0.48      0.93      0.51     88209
weighted avg       0.98      0.93      0.95     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         75165     1838      1742      1784      
Moderate       220       6805      191       174       
Severe         4         3         131       2         
Extreme        1         4         3         142       
```

# Flood Model Evaluation
## Day+1 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.97      0.92      0.95     42779
    Moderate       0.92      0.93      0.93     21624
      Severe       0.85      0.93      0.89     11243
     Extreme       0.87      0.93      0.90     12563

    accuracy                           0.93     88209
   macro avg       0.90      0.93      0.91     88209
weighted avg       0.93      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         39400     1172      1098      1109      
Moderate       498       20216     456       454       
Severe         276       277       10447     243       
Extreme        297       290       307       11669     
```

## Day+2 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.97      0.93      0.95     42801
    Moderate       0.93      0.92      0.93     21634
      Severe       0.85      0.94      0.89     11235
     Extreme       0.87      0.93      0.90     12539

    accuracy                           0.93     88209
   macro avg       0.91      0.93      0.92     88209
weighted avg       0.93      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         39895     986       960       960       
Moderate       516       20005     554       559       
Severe         242       242       10520     231       
Extreme        310       302       307       11620     
```

## Day+3 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.98      0.93      0.95     42780
    Moderate       0.93      0.93      0.93     21639
      Severe       0.85      0.95      0.90     11251
     Extreme       0.87      0.93      0.90     12539

    accuracy                           0.93     88209
   macro avg       0.91      0.93      0.92     88209
weighted avg       0.93      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         39615     1003      1066      1096      
Moderate       462       20185     503       489       
Severe         182       201       10667     201       
Extreme        293       322       292       11632     
```

# Landslide Model Evaluation
## Day+1 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.98      0.92      0.95     46921
    Moderate       0.92      0.94      0.93     20628
      Severe       0.86      0.92      0.89     12092
     Extreme       0.80      0.93      0.86      8568

    accuracy                           0.93     88209
   macro avg       0.89      0.93      0.91     88209
weighted avg       0.93      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         43180     1242      1256      1243      
Moderate       448       19296     417       467       
Severe         330       290       11177     295       
Extreme        201       192       194       7981      
```

## Day+2 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.98      0.92      0.95     46951
    Moderate       0.92      0.93      0.93     20645
      Severe       0.86      0.94      0.90     12066
     Extreme       0.81      0.94      0.87      8547

    accuracy                           0.93     88209
   macro avg       0.89      0.93      0.91     88209
weighted avg       0.93      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         43246     1280      1207      1218      
Moderate       433       19293     474       445       
Severe         254       268       11311     233       
Extreme        179       164       174       8030      
```

## Day+3 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.98      0.93      0.95     46947
    Moderate       0.93      0.93      0.93     20650
      Severe       0.87      0.93      0.90     12057
     Extreme       0.82      0.95      0.88      8555

    accuracy                           0.93     88209
   macro avg       0.90      0.93      0.91     88209
weighted avg       0.94      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         43731     1044      1126      1046      
Moderate       514       19164     480       492       
Severe         279       302       11190     286       
Extreme        151       150       130       8124      
```

