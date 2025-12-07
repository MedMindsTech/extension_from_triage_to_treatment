import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load CSV
df = pd.read_csv("outputs/cv_predictions.csv")

y_true = df["y_true"].values
y_proba = df["y_proba"].values
y_pred = (y_proba >= 0.5).astype(int)

# Metrics
print("AUROC:", roc_auc_score(y_true, y_proba))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))
