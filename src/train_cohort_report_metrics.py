import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def main():
    # Load cohort
    cohort_path = os.path.join("outputs", "cohort.parquet")
    cohort = pd.read_parquet(cohort_path)
    print("cohort:", cohort.shape)

    # Load report embeddings
    rep_path = os.path.join("outputs", "report_embeddings_from_mimiciv.parquet")
    rep_emb = pd.read_parquet(rep_path)
    print("report embeddings:", rep_emb.shape)

    # Identify embedding columns
    rep_id_cols = {"subject_id", "hadm_id"}
    rep_emb_cols = [
        c for c in rep_emb.columns
        if (c not in rep_id_cols)
        and (rep_emb[c].dtype in [np.float32, np.float64, np.int32, np.int64])
    ]
    print(f"Detected {len(rep_emb_cols)} embedding columns.")

    # Aggregate reports
    rep_small = (
        rep_emb[["subject_id", "hadm_id"] + rep_emb_cols]
        .groupby(["subject_id", "hadm_id"])
        .mean()
        .reset_index()
    )

    print("After aggregating reports:", rep_small.shape)

    # Merge cohort + reports
    df = cohort.merge(
        rep_small,
        on=["subject_id", "hadm_id"],
        how="inner"
    )
    print("Merged dataset:", df.shape)

    # Features & labels
    y = df["icu_within_24h"].values
    feature_cols = ["gender_encoded", "los_days"] + rep_emb_cols
    X = df[feature_cols].values

    print("X shape:", X.shape, "y shape:", y.shape)
    print("Label distribution:")
    print(pd.Series(y).value_counts())

    # Cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # For final metrics
    fold_metrics = []

    print("\n===== Structured + Reports (ClinicalBERT) - Metrics =====")

    fold_idx = 1

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n--- Fold {fold_idx} ---")
        print(f"AUROC:     {auc:.3f}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

        fold_metrics.append([auc, acc, precision, recall, f1])
        fold_idx += 1

    fold_metrics = np.array(fold_metrics)

    print("\n===== FINAL METRICS (Mean ± Std) =====")
    print(f"AUROC:     {fold_metrics[:,0].mean():.3f} ± {fold_metrics[:,0].std():.3f}")
    print(f"Accuracy:  {fold_metrics[:,1].mean():.3f} ± {fold_metrics[:,1].std():.3f}")
    print(f"Precision: {fold_metrics[:,2].mean():.3f} ± {fold_metrics[:,2].std():.3f}")
    print(f"Recall:    {fold_metrics[:,3].mean():.3f} ± {fold_metrics[:,3].std():.3f}")
    print(f"F1-score:  {fold_metrics[:,4].mean():.3f} ± {fold_metrics[:,4].std():.3f}")


if __name__ == "__main__":
    main()
