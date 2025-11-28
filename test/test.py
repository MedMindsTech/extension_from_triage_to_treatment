import pandas as pd

adm = pd.read_csv("data/mimiciv_demo/hosp/admissions.csv")
pat = pd.read_csv("data/mimiciv_demo/hosp/patients.csv")

print("admissions unique subject_id:", adm["subject_id"].nunique())
print("patients unique subject_id:", pat["subject_id"].nunique())

common = set(adm["subject_id"]).intersection(set(pat["subject_id"]))
print("Common subject_id count:", len(common))
print("Common IDs:", list(common)[:10])
