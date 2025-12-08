import pandas as pd

# 1. Load cohort
cohort = pd.read_parquet("outputs/cohort.parquet")
cohort_subs = set(cohort["subject_id"].unique())
print("Cohort subjects:", len(cohort_subs))

# 2. Load radiology report embeddings (ClinicalBERT)
rep = pd.read_parquet("outputs/report_embeddings_from_mimiciv.parquet")
rep_subs = set(rep["subject_id"].unique())
print("Subjects with radiology reports:", len(rep_subs))

# 3. Load existing CXR metadata (DenseNet embeddings)
try:
    cxr = pd.read_csv("cxr_embedding_metadata.csv")
    cxr_subs = set(cxr["subject_id"].unique())
    print("Subjects with existing CXR embeddings:", len(cxr_subs))
except FileNotFoundError:
    print("No CXR metadata file found. Assuming 0 downloaded CXRs.")
    cxr_subs = set()

# 4. Subjects we need for multimodal fusion
needed = cohort_subs & rep_subs
print("Subjects in cohort + reports:", len(needed))

# 5. Subjects still missing CXR data
missing_cxr = sorted(list(needed - cxr_subs))
print("Subjects missing CXR files:", len(missing_cxr))

# 6. Save to txt file
pd.Series(missing_cxr).to_csv("subjects_to_download.txt", index=False, header=False)
print("Saved subjects_to_download.txt")
