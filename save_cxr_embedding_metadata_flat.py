import os
import numpy as np
import pandas as pd

# Load embeddings
print("Loading cxr_embeddings.npy...")
emb = np.load("cxr_embeddings.npy")    # shape: (N_images, 1024)

# Get PNG files in sorted order so they match embedding order
png_files = sorted([
    f for f in os.listdir("processed_images")
    if f.lower().endswith(".png")
])

print(f"Found {len(png_files)} PNG images.")
print(f"Embedding array shape: {emb.shape}")

if len(png_files) != emb.shape[0]:
    print("\nWARNING: Number of PNG files does not match number of embeddings!")
    print("PNG files:", len(png_files))
    print("Embeddings:", emb.shape[0])
    print("This mismatch can cause incorrect metadata assignment.\n")
    # Continue anyway, but warn user

# Load the mapping file that links dicom_id → subject_id + study_id
print("Loading cxr-record-list.csv.gz...")
df_map = pd.read_csv("cxr-record-list.csv.gz")

# Extract dicom_id from filename (remove .png extension)
df_png = pd.DataFrame({
    "png_file": png_files,
    "dicom_id": [f.replace(".png", "") for f in png_files]
})

# Merge metadata
print("Merging PNG filenames with subject & study information...")
merged = df_png.merge(df_map, on="dicom_id", how="left")

# Adding embedding vectors
emb_df = pd.DataFrame(
    emb, 
    columns=[f"e{i}" for i in range(emb.shape[1])]
)

final = pd.concat([merged, emb_df], axis=1)

# Save final metadata CSV
output_path = "cxr_embedding_metadata.csv"
final.to_csv(output_path, index=False)

print(f"\nSaved {output_path} successfully!")
print(f"Rows: {final.shape[0]}, Columns: {final.shape[1]}")

import pandas as pd
df = pd.read_csv("cxr_embedding_metadata.csv")
print([c for c in df.columns if c.startswith("e")][:10])
