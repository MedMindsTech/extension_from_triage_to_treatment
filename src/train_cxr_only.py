import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_cxr_and_labels():
    """
    Load cohort labels and CXR embeddings,
    aggregate to subject-level, and return X, y, merged_df.
    """
    # 1) Load cohort (labels)
    cohort_path = os.path.join("outputs", "cohort.parquet")
    cohort = pd.read_parquet(cohort_path)
    print("Cohort shape:", cohort.shape)

    # 2) Load CXR embeddings
    cxr_path = "cxr_embedding_metadata.csv"
    cxr = pd.read_csv(cxr_path)
    print("CXR embedding table shape:", cxr.shape)

    # 3) Identify embedding columns (e0, e1, ..., e1023)
    cxr_emb_cols = [c for c in cxr.columns if c.startswith("e")]
    if not cxr_emb_cols:
        raise ValueError("No CXR embedding columns starting with 'e' found.")

    print(f"Detected {len(cxr_emb_cols)} CXR embedding columns.")

    # 4) Aggregate CXR embeddings to subject-level (mean over images)
    cxr_subj = (
        cxr[["subject_id"] + cxr_emb_cols]
        .groupby("subject_id")
        .mean()
        .reset_index()
    )
    print("CXR subject-level shape:", cxr_subj.shape)

    # 5) Build subject-level labels from cohort
    cohort_subj = (
        cohort.groupby("subject_id")
        .agg(
            icu_within_24h=("icu_within_24h", "max")
        )
        .reset_index()
    )
    print("Cohort subject-level shape:", cohort_subj.shape)

    # 6) Merge on subject_id: only subjects with both label and CXR
    merged = cohort_subj.merge(cxr_subj, on="subject_id", how="inner")
    print("\nMerged subject-level shape (CXR-only):", merged.shape)

    # Show label distribution
    label_counts = merged["icu_within_24h"].value_counts()
    print("Label distribution (CXR-only subjects):")
    print(label_counts)

    if merged.shape[0] < 3:
        print("\nWARNING: Very few multimodal subjects. "
              "Results will be extremely unstable.")
    if label_counts.min() == 0:
        print("\nWARNING: One class has zero samples after merging. "
              "Cannot train a classifier in this case.")

    # 7) Build X and y
    y = merged["icu_within_24h"].values
    X = merged[cxr_emb_cols].values

    return X, y, merged


def main():
    X, y, merged = load_cxr_and_labels()

    n_samples, n_features = X.shape
    print(f"\nRaw X shape: {X.shape}")

    if len(np.unique(y)) < 2:
        print("\nERROR: Only one class present in y. "
              "Cannot compute metrics like AUROC/AUPRC.")
        return

    # ------------------------------
    # Optional dimensionality reduction with PCA
    # ------------------------------
    # We limit components to keep the model simple and avoid overfitting.
    n_components = min(10, n_samples - 1, n_features)
    use_pca = n_components >= 2

    if use_pca:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print(f"Using PCA with n_components={n_components}")
        print("PCA-transformed X shape:", X_pca.shape)
    else:
        X_pca = X
        print("Skipping PCA (too few samples/features).")

    # ------------------------------
    # Leave-One-Out Cross-Validation
    # ------------------------------
    loo = LeaveOneOut()

    y_true = []
    y_prob = []
    y_pred = []

    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)

        y_true.append(int(y_test[0]))
        y_prob.append(float(prob[0]))
        y_pred.append(int(pred[0]))

    # ------------------------------
    # Metrics
    # ------------------------------
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print("\n========== CXR-ONLY MODEL RESULTS ==========")
    print("Number of subjects:", len(y_true))
    print("AUROC:", round(auroc, 3))
    print("AUPRC:", round(auprc, 3))
    print("Accuracy:", round(acc, 3))
    print("Balanced Accuracy:", round(bal_acc, 3))

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    # ------------------------------
    # Visualizations
    # ------------------------------
    os.makedirs("plots", exist_ok=True)

    # ROC Curve
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve — CXR-only Logistic Regression")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "roc_curve_cxr_only.png"))
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Precision–Recall Curve — CXR-only Logistic Regression")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "pr_curve_cxr_only.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))

    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    else:
        plt.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.title("Confusion Matrix — CXR-only Logistic Regression")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "cm_cxr_only.png"))
    plt.close()
    
    from sklearn.metrics import matthews_corrcoef, confusion_matrix

    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    print("MCC:", round(mcc, 3))
    print("Specificity:", round(specificity, 3))


    print("\nSaved plots to 'plots/' folder:")
    print(" - roc_curve_cxr_only.png")
    print(" - pr_curve_cxr_only.png")
    print(" - cm_cxr_only.png")


if __name__ == "__main__":
    main()
