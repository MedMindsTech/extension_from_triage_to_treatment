# src/train_baseline_structured.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def main():
    cohort_path = os.path.join("outputs", "cohort.parquet")
    cohort = pd.read_parquet(cohort_path)

    print("Cohort shape:", cohort.shape)
    print(cohort.head())

    # Target
    y = cohort["icu_within_24h"].values

    # Sadece basit structured features:
    feature_cols = ["gender_encoded", "los_days"]
    X = cohort[feature_cols].values

    print("X shape:", X.shape, "y shape:", y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale + logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("AUROC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
