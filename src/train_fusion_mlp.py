import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_csv_and_embeddings():
    # Load Cohort
    cohort_path = os.path.join("outputs", "cohort.parquet")
    cohort = pd.read_parquet(cohort_path)
    print("cohort:", cohort.shape)

    # Load radiology report embeddings (ClinicalBERT)
    rep_path = os.path.join("outputs", "report_embeddings_from_mimiciv.parquet")
    rep_emb = pd.read_parquet(rep_path)
    print("report embeddings:", rep_emb.shape)

    # Load CXR embeddings
    cxr_path = os.path.join("cxr_embedding_metadata.csv")
    cxr = pd.read_csv(cxr_path)
    print("cxr embeddings:", cxr.shape)

    return cohort, rep_emb, cxr

def main(cohort, rep_emb, cxr):
    # CXR embedding columns
    cxr_emb_cols = [c for c in cxr.columns if c.startswith("e")]
    print(f"Detected {len(cxr_emb_cols)} CXR embedding columns.")

    cxr_small = (
        cxr[["subject_id"] + cxr_emb_cols]
        .groupby("subject_id")
        .mean()
        .reset_index()
    )

    print("After aggregating CXR per subject_id:", cxr_small.shape)

    if len(cxr_emb_cols) == 0:
        raise ValueError("No CXR embedding columns starting with 'e' found.")

    # Report embedding columns
    rep_id_cols = {"subject_id", "hadm_id"}
    rep_emb_cols = [
        c for c in rep_emb.columns
        if (c not in rep_id_cols) and (rep_emb[c].dtype in [np.float32, np.float64, np.int32, np.int64])
    ]
    print(f"Detected {len(rep_emb_cols)} report embedding columns.")

    rep_small = (
        rep_emb[["subject_id", "hadm_id"] + rep_emb_cols]
        .groupby(["subject_id", "hadm_id"])
        .mean()
        .reset_index()
    )

    print("After aggregating reports per (subject_id, hadm_id):", rep_small.shape)


    if len(rep_emb_cols) == 0:
        raise ValueError("No numeric report embedding columns found.")

    # Merge: cohort + reports + CXR
    # step 1 -> cohort + reports (subject_id + hadm_id)
    df = cohort.merge(
        rep_small,
        on=["subject_id", "hadm_id"],
        how="inner"
    )

    print("After merging with reports:", df.shape)

    # step 2 -> add CXR embeddings by subject_id
    df = df.merge(
        cxr_small,
        on="subject_id",
        how="inner"
    )

    print("After adding CXR embeddings:", df.shape)
    print(df.head())

    # If too few rows, warn
    n_samples = df.shape[0]
    if n_samples < 10:
        print(f"WARNING: Only {n_samples} multimodal samples. Results will be very unstable.")

    # build feature matrix X and label vector y
    y = df["icu_within_24h"].values

    struct_cols = ["gender_encoded", "los_days"]
    feature_cols = struct_cols + rep_emb_cols + cxr_emb_cols

    X = df[feature_cols].values

    print("X shape:", X.shape, "y shape:", y.shape)
    label_counts = pd.Series(y).value_counts()
    print("Label distribution in multimodal DF:")
    print(label_counts)

    # Setup Stratified K-Fold CV
    min_class_count = label_counts.min()
    n_splits = min(5, int(min_class_count)) if min_class_count >= 2 else 2
    if n_splits < 2:
        raise ValueError("Not enough samples per class for CV.")

    print(f"\nUsing StratifiedKFold with n_splits={n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Logistic Regression CV
    log_aucs = []
    fold_idx = 1

    print("\n===== Logistic Regression (Structured + Reports + CXR) - CV =====")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logreg = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",  # small, imbalanced data
            solver="lbfgs"
        )

        logreg.fit(X_train_scaled, y_train)
        y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            print(f"Fold {fold_idx}: AUROC undefined (only one class in y_test). Skipping.")
            fold_idx += 1
            continue

        log_aucs.append(auc)
        print(f"Fold {fold_idx} LOGREG AUROC: {auc:.3f}")
        fold_idx += 1

    if log_aucs:
        print(f"\nLOGREG AUROC mean: {np.mean(log_aucs):.3f}  std: {np.std(log_aucs):.3f}")
    else:
        print("\nLOGREG: No valid AUROC values (too few samples per fold).")


if __name__ == "__main__":
    cohort, rep_emb, cxr = load_csv_and_embeddings()
    main(cohort, rep_emb, cxr)