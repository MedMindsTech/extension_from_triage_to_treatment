import pandas as pd

df = pd.read_parquet("outputs/report_embeddings_from_mimiciv.parquet")

print(df.shape)  # row, colummns
print(df.head())
print(df.columns)
