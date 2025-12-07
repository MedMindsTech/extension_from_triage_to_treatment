import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    # Load cohort
    cohort_path = os.path.join("outputs", "cohort.parquet")
    cohort = pd.read_parquet(cohort_path)
    print("cohort:", cohort.shape)

    # Load report embeddings
    rep_path = os.path.join("outputs", "report_embeddings_from_mimiciv.parquet")
    rep_emb = pd.read_parquet(rep_path)
    print("report embeddings:", rep_emb.shape)

    # Identify embedding columns (numeric only)
    rep_id_cols = {"subject_id", "hadm_id"}
    rep_emb_cols = [
        c for c in rep_emb.columns
        if (c not in rep_id_cols)
        and (rep_emb[c].dtype in [np.float32, np.float64, np.int32, np.int64])
    ]
    print(f"Detected {len(rep_emb_cols)} report embedding columns.")

    # Aggregate multiple reports per admission → 1 embedding per (subject_id, hadm_id)
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
    print("After merging with cohort:", df.shape)

    # Build features
    y = df["icu_within_24h"].values

    struct_cols = ["gender_encoded", "los_days"]
    feature_cols = struct_cols + rep_emb_cols

    X = df[feature_cols].values

    print("X shape:", X.shape, "y shape:", y.shape)
    print("Label distribution:")
    print(pd.Series(y).value_counts())

    # Stratified K-Fold
    n_samples = len(y)
    min_class = pd.Series(y).value_counts().min()

    # Ensure enough positive/negative per fold
    n_splits = min(5, min_class) if min_class >= 2 else 2
    print(f"\nUsing StratifiedKFold with n_splits={n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Logistic Regression
    log_aucs = []
    fold_idx = 1

    print("\n===== Logistic Regression (Structured + Reports) - CV =====")

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
            class_weight="balanced",
            solver="lbfgs"
        )

        logreg.fit(X_train_scaled, y_train)
        y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

        # Save predictions to CSV
        save_df = pd.DataFrame({
            "fold": fold_idx,
            "y_true": y_test,
            "y_proba": y_proba
        })
        csv_path = os.path.join("outputs", "cv_predictions.csv")

        if not os.path.exists(csv_path):
            save_df.to_csv(csv_path, index=False)
            #print(f"Saved predictions to {csv_path}")
        else:
            save_df.to_csv(csv_path, mode="a", header=False, index=False)

        try:
            auc = roc_auc_score(y_test, y_proba)
            log_aucs.append(auc)
            print(f"Fold {fold_idx} AUROC: {auc:.3f}")
        except:
            print(f"Fold {fold_idx} AUROC undefined.")
        fold_idx += 1

    # Final CV score
    if log_aucs:
        print(f"\nFinal AUROC mean: {np.mean(log_aucs):.3f}  std: {np.std(log_aucs):.3f}")
    else:
        print("\nNo valid AUROC scores.")


if __name__ == "__main__":
    main()
