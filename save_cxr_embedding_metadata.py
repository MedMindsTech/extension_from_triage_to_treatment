import os
import numpy as np
import pandas as pd

embeddings = np.load("cxr_embeddings.npy") # Load embeddings


files = sorted([f for f in os.listdir("processed_images") if f.endswith(".png")]) # Filenames in same order as extract_cxr_embeddings.py

meta = pd.read_csv("cxr-record-list.csv.gz")  # Load MIMIC-CXR metadata 

rows = []

for file, emb in zip(files, embeddings):
  
    uid = file.replace(".png", "")   # Remove .png to get the original DICOM UUID

    
    row = meta[meta["dicom_id"] == uid] # Lookup subject_id and study_id in metadata
 
    if row.empty:
        print("Metadata not found for:", file)
        continue

    subject_id = int(row["subject_id"].values[0])
    study_id   = int(row["study_id"].values[0])

    rows.append([file, subject_id, study_id] + emb.tolist())
    
columns = ["filename", "subject_id", "study_id"] + [f"e{i}" for i in range(1024)]
df = pd.DataFrame(rows, columns=columns)

df.to_csv("cxr_embedding_metadata.csv", index=False)

print("Saved metadata CSV: cxr_embedding_metadata.csv")

