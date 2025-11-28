import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mean_pool(last_hidden_state, attention_mask):
    # attention mask ile mean pooling
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

def main():
    # Load cohort and radiology.csv.gz
    cohort = pd.read_parquet("outputs/cohort.parquet")
    print("Cohort:", cohort.shape)

    rad_path = os.path.join("data", "mimiciv_demo", "note", "radiology.csv.gz")
    radiology = pd.read_csv(rad_path)
    print("Radiology:", radiology.shape)
    print("Radiology columns:", list(radiology.columns)[:20])

    # I need to find name of the report text column in radiology.csv.gz
    # According to MIMIC-IV docs, it could be 'text' or 'report' or similar
    TEXT_COL = "text"  # If it doesn't exist, check the printed column names and change them

    if TEXT_COL not in radiology.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in radiology. Check the printed columns and update TEXT_COL.")

    # Features extraction from radiology reports
    rad_small = radiology[["subject_id", "hadm_id", TEXT_COL]].dropna(subset=[TEXT_COL])
    rad_small.rename(columns={TEXT_COL: "report_text"}, inplace=True)

    print("Radiology (reduced):", rad_small.shape)

    # Cohort + Radiology join (subject_id + hadm_id)
    merged = cohort.merge(
        rad_small,
        on=["subject_id", "hadm_id"],
        how="inner"
    ).dropna(subset=["report_text"])

    print("Cohort with radiology reports:", merged.shape)

    if merged.empty:
        raise ValueError("Merged dataframe is empty. Check subject_id/hadm_id consistency.")

    # Load ClinicalBERT model
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    texts = merged["report_text"].astype(str).tolist()
    batch_size = 8
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding radiology reports"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeddings.append(pooled.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    print("Embedding shape:", embeddings.shape)

    emb_df = pd.DataFrame(embeddings.numpy())
    emb_df.insert(0, "subject_id", merged["subject_id"].values)
    emb_df.insert(1, "hadm_id", merged["hadm_id"].values)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "report_embeddings_from_mimiciv.parquet")
    emb_df.to_parquet(out_path, index=False)
    print(f"Report embeddings saved to {out_path}")


if __name__ == "__main__":
    main()
