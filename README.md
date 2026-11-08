# Dry Bean Classification – ML Assignment 2

## a. Problem statement
Classify dry-bean grains into one of 7 varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) from
geometric measurements taken from images. Five classifiers are trained on the same data, compared on six
metrics, and served through a Streamlit app.

## b. Dataset description  [1 mark]
- **Source:** UCI Machine Learning Repository – *Dry Bean Dataset* (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Instances:** 13,611 · **Features:** 16 numeric (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
  Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1–4) · **Target:** `Class` (7 classes, multi-class)
- **Split:** 80/20 stratified, `random_state=42`. Models are evaluated on the 20% hold-out (2,723 rows).
- Scaling (`StandardScaler`) is part of the saved pipeline for Logistic Regression, KNN and SVM.

## c. GitHub repository link  [1 mark]
https://github.com/2026aj05556/ml-classification-streamlit

Live app: Pending deployment

## d. Models used  [5 marks]
Metrics on the 20% hold-out split. Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8979 | 0.9570 | 0.9131 | 0.9120 | 0.9125 | 0.8766 |
| KNN | 0.9137 | 0.9858 | 0.9290 | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

### Observations  [3 marks]
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong for a linear model (92.1% acc, AUC 0.995): after scaling, the classes are largely linearly separable. Nearly matches SVM and trains fastest. |
| Decision Tree | Lower than the other non-NB models (89.8%). Axis-aligned splits on highly correlated size features overfit slightly (AUC 0.957 vs ~0.99 for LR/SVM); easy to interpret. |
| KNN | Good (91.4%, AUC 0.986) once features are standardised; slower at prediction time on 13k rows and sensitive to k and scaling. |
| Naive Bayes | Weakest (76.4%, MCC 0.715). Features (Area, Perimeter, ConvexArea, EquivDiameter…) are strongly correlated, violating the independence assumption. Decent AUC shows ranking is fine but hard decisions are poor. |
| Support Vector Machine | Best on every metric (92.2% acc, AUC 0.9949, MCC 0.906); the RBF kernel captures non-linear boundaries between similar classes. |
| **Overall winner** | **SVM (RBF)** – marginally ahead of Logistic Regression on all six metrics; Logistic Regression is the simpler near-equal alternative. |

## Repository structure
```
app.py            Streamlit app (upload CSV, pick model, metrics, confusion matrix, report)
train.py          Downloads dataset, trains 5 models, writes model/
metrics.py        Shared metric function
requirements.txt  Pinned dependencies
model/            Saved models (.joblib) + metrics_table.csv
data/             Full dataset CSV
```

## Run locally
```
pip install -r requirements.txt
python train.py        # optional – models are already saved
streamlit run app.py
```
The app opens on `data/dry_bean_full.csv`; you can also load it from GitHub or upload any CSV with the same columns. The app loads the saved models and does **not** retrain.

