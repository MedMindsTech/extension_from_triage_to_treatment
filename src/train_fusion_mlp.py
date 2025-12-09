
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report
)
import xgboost as xgb


def load_csv_and_embeddings():
    cohort = pd.read_parquet("outputs/cohort.parquet")
    rep_emb = pd.read_parquet("outputs/report_embeddings_from_mimiciv.parquet")
    cxr = pd.read_csv("cxr_embedding_metadata.csv")

    print("cohort:", cohort.shape)
    print("report embeddings:", rep_emb.shape)
    print("cxr embeddings:", cxr.shape)
    return cohort, rep_emb, cxr


def main(cohort, rep_emb, cxr):

    # -------------------------------
    # CXR subject-level
    # -------------------------------
    cxr_emb_cols = [c for c in cxr.columns if c.startswith("e")]
    cxr_small = cxr.groupby("subject_id")[cxr_emb_cols].mean().reset_index()

    # -------------------------------
    # Report subject-level
    # -------------------------------
    rep_id_cols = {"subject_id", "hadm_id"}
    rep_emb_cols = [
        c for c in rep_emb.columns
        if c not in rep_id_cols and rep_emb[c].dtype in (np.float32, np.float64)
    ]

    rep_small = rep_emb.groupby("subject_id")[rep_emb_cols].mean().reset_index()

    # -------------------------------
    # Cohort subject-level
    # -------------------------------
    cohort_small = (
        cohort.groupby("subject_id")
        .agg(
            icu_within_24h=("icu_within_24h", "max"),
            gender_encoded=("gender_encoded", "first"),
            los_days=("los_days", "mean")
        )
        .reset_index()
    )

    # -------------------------------
    # Merge modalities
    # -------------------------------
    merged = (
        cohort_small
        .merge(rep_small, on="subject_id", how="inner")
        .merge(cxr_small, on="subject_id", how="inner")
    )

    print("\nMultimodal subject count:", merged.shape[0])
    print(merged[["subject_id", "icu_within_24h"]])

    if merged.shape[0] < 3:
        print("Not enough multimodal subjects.")
        return

    # -------------------------------
    # Prepare features
    # -------------------------------
    y = merged["icu_within_24h"].values
    X_struct = merged[["gender_encoded", "los_days"]].values
    X_rep = merged[rep_emb_cols].values
    X_cxr = merged[cxr_emb_cols].values

    # PCA aggressive reduction
    pca_rep = PCA(n_components=1)
    X_rep_pca = pca_rep.fit_transform(X_rep)

    pca_cxr = PCA(n_components=1)
    X_cxr_pca = pca_cxr.fit_transform(X_cxr)

    X = np.concatenate([X_struct, X_rep_pca, X_cxr_pca], axis=1)
    print("Final X shape:", X.shape)

    # -------------------------------
    # Leave-One-Out CV
    # -------------------------------
    loo = LeaveOneOut()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # XGBoost model
        model = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.05,
            n_estimators=100,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) else 1)
        )

        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)

        y_true_all.append(int(y_test[0]))
        y_prob_all.append(float(y_prob[0]))
        y_pred_all.append(int(y_pred[0]))

    # -------------------------------
    # Metrics
    # -------------------------------
    auroc = roc_auc_score(y_true_all, y_prob_all)
    auprc = average_precision_score(y_true_all, y_prob_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    print("\n========== FINAL MULTIMODAL RESULTS ==========")
    print("AUROC:", round(auroc, 3))
    print("AUPRC:", round(auprc, 3))
    print("Balanced Accuracy:", round(bal_acc, 3))

    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all))
    
    # ---------------------------------------------------

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fusion_pr_curve.png")
    plt.show()

    # ---------------------------------------------------

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fusion_roc_curve.png")
    plt.show()

        
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)   # normalize rows

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("fusion_confusion_matrix.png")
    plt.show()



if __name__ == "__main__":
    cohort, rep_emb, cxr = load_csv_and_embeddings()
    main(cohort, rep_emb, cxr)


# import os
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score


# def load_csv_and_embeddings():
#     # Load Cohort
#     cohort_path = os.path.join("outputs", "cohort.parquet")
#     cohort = pd.read_parquet(cohort_path)
#     print("cohort:", cohort.shape)

#     # Load radiology report embeddings (ClinicalBERT)
#     rep_path = os.path.join("outputs", "report_embeddings_from_mimiciv.parquet")
#     rep_emb = pd.read_parquet(rep_path)
#     print("report embeddings:", rep_emb.shape)

#     # Load CXR embeddings
#     cxr_path = os.path.join("cxr_embedding_metadata.csv")
#     cxr = pd.read_csv(cxr_path)
#     print("cxr embeddings:", cxr.shape)

#     return cohort, rep_emb, cxr

# def main(cohort, rep_emb, cxr):
#     # CXR embedding columns
#     cxr_emb_cols = [c for c in cxr.columns if c.startswith("e")]
#     print(f"Detected {len(cxr_emb_cols)} CXR embedding columns.")

#     cxr_small = (
#         cxr[["subject_id"] + cxr_emb_cols]
#         .groupby("subject_id")
#         .mean()
#         .reset_index()
#     )

#     print("After aggregating CXR per subject_id:", cxr_small.shape)

#     if len(cxr_emb_cols) == 0:
#         raise ValueError("No CXR embedding columns starting with 'e' found.")

#     # Report embedding columns
#     rep_id_cols = {"subject_id", "hadm_id"}
#     rep_emb_cols = [
#         c for c in rep_emb.columns
#         if (c not in rep_id_cols) and (rep_emb[c].dtype in [np.float32, np.float64, np.int32, np.int64])
#     ]
#     print(f"Detected {len(rep_emb_cols)} report embedding columns.")

#     rep_small = (
#         rep_emb[["subject_id", "hadm_id"] + rep_emb_cols]
#         .groupby(["subject_id", "hadm_id"])
#         .mean()
#         .reset_index()
#     )

#     print("After aggregating reports per (subject_id, hadm_id):", rep_small.shape)


#     if len(rep_emb_cols) == 0:
#         raise ValueError("No numeric report embedding columns found.")

#     # Merge: cohort + reports + CXR
#     # step 1 -> cohort + reports (subject_id + hadm_id)
#     df = cohort.merge(
#         rep_small,
#         on=["subject_id", "hadm_id"],
#         how="inner"
#     )

#     print("After merging with reports:", df.shape)

#     # step 2 -> add CXR embeddings by subject_id
#     df = df.merge(
#         cxr_small,
#         on="subject_id",
#         how="inner"
#     )

#     print("After adding CXR embeddings:", df.shape)
#     print(df.head())

#     # If too few rows, warn
#     n_samples = df.shape[0]
#     if n_samples < 10:
#         print(f"WARNING: Only {n_samples} multimodal samples. Results will be very unstable.")

#     # build feature matrix X and label vector y
#     y = df["icu_within_24h"].values

#     struct_cols = ["gender_encoded", "los_days"]
#     feature_cols = struct_cols + rep_emb_cols + cxr_emb_cols

#     X = df[feature_cols].values

#     print("X shape:", X.shape, "y shape:", y.shape)
#     label_counts = pd.Series(y).value_counts()
#     print("Label distribution in multimodal DF:")
#     print(label_counts)

#     # Setup Stratified K-Fold CV
#     min_class_count = label_counts.min()
#     n_splits = min(5, int(min_class_count)) if min_class_count >= 2 else 2
#     if n_splits < 2:
#         raise ValueError("Not enough samples per class for CV.")

#     print(f"\nUsing StratifiedKFold with n_splits={n_splits}")

#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

#     # Logistic Regression CV
#     log_aucs = []
#     fold_idx = 1

#     print("\n===== Logistic Regression (Structured + Reports + CXR) - CV =====")

#     for train_idx, test_idx in skf.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]

#         # Scale features
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         logreg = LogisticRegression(
#             penalty="l2",
#             C=0.1,
#             max_iter=1000,
#             class_weight="balanced",  # small, imbalanced data
#             solver="lbfgs"
#         )

#         logreg.fit(X_train_scaled, y_train)
#         y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

#         try:
#             auc = roc_auc_score(y_test, y_proba)
#         except ValueError:
#             print(f"Fold {fold_idx}: AUROC undefined (only one class in y_test). Skipping.")
#             fold_idx += 1
#             continue

#         log_aucs.append(auc)
#         print(f"Fold {fold_idx} LOGREG AUROC: {auc:.3f}")
#         fold_idx += 1

#     if log_aucs:
#         print(f"\nLOGREG AUROC mean: {np.mean(log_aucs):.3f}  std: {np.std(log_aucs):.3f}")
#     else:
#         print("\nLOGREG: No valid AUROC values (too few samples per fold).")


# if __name__ == "__main__":
#     cohort, rep_emb, cxr = load_csv_and_embeddings()
#     main(cohort, rep_emb, cxr)

