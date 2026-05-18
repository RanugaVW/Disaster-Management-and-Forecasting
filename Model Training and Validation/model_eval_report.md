# Drought Model Evaluation
## Day+1 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.97      0.97      0.97     80659
    Moderate       0.68      0.68      0.68      7409
      Severe       0.40      0.39      0.39       141

    accuracy                           0.95     88209
   macro avg       0.68      0.68      0.68     88209
weighted avg       0.95      0.95      0.95     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    
Normal         78318     2297      44        
Moderate       2323      5046      40        
Severe         48        38        55        
```

## Day+2 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.95      0.98      0.97     80665
    Moderate       0.67      0.47      0.55      7403
      Severe       0.32      0.14      0.20       141

    accuracy                           0.93     88209
   macro avg       0.65      0.53      0.57     88209
weighted avg       0.93      0.93      0.93     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    
Normal         78907     1738      20        
Moderate       3867      3514      22        
Severe         93        28        20        
```

## Day+3 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.94      0.98      0.96     80679
    Moderate       0.60      0.38      0.46      7390
      Severe       0.37      0.12      0.18       140

    accuracy                           0.93     88209
   macro avg       0.64      0.49      0.54     88209
weighted avg       0.91      0.93      0.92     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    
Normal         78853     1811      15        
Moderate       4589      2787      14        
Severe         95        28        17        
```

# Flood Model Evaluation
## Day+1 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.88      0.96      0.92     42779
    Moderate       0.88      0.88      0.88     21624
      Severe       0.88      0.72      0.79     11243
     Extreme       0.88      0.76      0.81     12563

    accuracy                           0.88     88209
   macro avg       0.88      0.83      0.85     88209
weighted avg       0.88      0.88      0.88     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         41109     859       381       430       
Moderate       1849      19004     359       412       
Severe         1855      818       8111      459       
Extreme        1819      856       390       9498      
```

## Day+2 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.55      0.95      0.70     42801
    Moderate       0.39      0.10      0.16     21634
      Severe       0.32      0.03      0.06     11235
     Extreme       0.43      0.27      0.33     12539

    accuracy                           0.53     88209
   macro avg       0.42      0.34      0.31     88209
weighted avg       0.47      0.53      0.43     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         40848     895       111       947       
Moderate       17235     2101      285       2013      
Severe         8094      1179      349       1613      
Extreme        7614      1206      332       3387      
```

## Day+3 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.53      0.97      0.68     42780
    Moderate       0.36      0.07      0.12     21639
      Severe       0.29      0.02      0.03     11251
     Extreme       0.42      0.15      0.22     12539

    accuracy                           0.51     88209
   macro avg       0.40      0.30      0.26     88209
weighted avg       0.44      0.51      0.39     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         41409     662       91        618       
Moderate       18956     1538      159       986       
Severe         9233      859       193       966       
Extreme        9240      1231      229       1839      
```

# Landslide Model Evaluation
## Day+1 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.86      0.78      0.82     46921
    Moderate       0.56      0.52      0.54     20628
      Severe       0.42      0.50      0.46     12092
     Extreme       0.37      0.51      0.43      8568

    accuracy                           0.66     88209
   macro avg       0.55      0.58      0.56     88209
weighted avg       0.68      0.66      0.67     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         36772     5051      2438      2660      
Moderate       2636      10724     4577      2691      
Severe         1771      2191      6035      2095      
Extreme        1773      1081      1348      4366      
```

## Day+2 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.64      0.92      0.76     46951
    Moderate       0.47      0.20      0.28     20645
      Severe       0.38      0.17      0.24     12066
     Extreme       0.34      0.24      0.28      8547

    accuracy                           0.59     88209
   macro avg       0.46      0.38      0.39     88209
weighted avg       0.53      0.59      0.53     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         43368     1865      780       938       
Moderate       13083     4205      1724      1633      
Severe         6694      1879      2061      1432      
Extreme        4558      1094      858       2037      
```

## Day+3 Horizon
### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.60      0.94      0.73     46947
    Moderate       0.41      0.13      0.20     20650
      Severe       0.34      0.12      0.18     12057
     Extreme       0.32      0.12      0.17      8555

    accuracy                           0.56     88209
   macro avg       0.42      0.33      0.32     88209
weighted avg       0.49      0.56      0.48     88209

```
### Confusion Matrix
```
True \ Pred    Normal    Moderate  Severe    Extreme   
Normal         44241     1439      687       580       
Moderate       15817     2725      1281      827       
Severe         8247      1556      1466      788       
Extreme        5765      933       847       1010      
```

