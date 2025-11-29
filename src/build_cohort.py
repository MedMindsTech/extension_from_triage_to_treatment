import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "mimiciv_demo")

def main():
    # Read data from CSV files
    adm = pd.read_csv(
        os.path.join(DATA_DIR, "hosp", "admissions.csv"),
        parse_dates=["admittime", "dischtime", "deathtime"]
    )
    pat = pd.read_csv(
        os.path.join(DATA_DIR, "hosp", "patients.csv")
    )
    icu = pd.read_csv(
        os.path.join(DATA_DIR, "icu", "icustays.csv"),
        parse_dates=["intime", "outtime"]
    )

    print("Admissions:", adm.shape)
    print("Patients:", pat.shape)
    print("ICU stays:", icu.shape)

    # admissions + patients merge
    data = adm.merge(pat, on="subject_id", how="inner")
    print("After joining adm+pat:", data.shape)

    # === Simple structured features ===

    # gender_encoded: M = 1, others(F) = 0
    if "gender" in data.columns:
        data["gender_encoded"] = (data["gender"] == "M").astype(int)
    else:
        data["gender_encoded"] = 0

    # length of stay
    data["los_days"] = (data["dischtime"] - data["admittime"]).dt.total_seconds() / (3600 * 24)


    # ICU merge and label: ICU within 24h
    adm_icu = data.merge(
        icu[["subject_id", "hadm_id", "intime", "outtime"]],
        on=["subject_id", "hadm_id"],
        how="left"
    )

    has_icu = ~adm_icu["intime"].isna()
    time_diff_hours = (adm_icu["intime"] - adm_icu["admittime"]).dt.total_seconds() / 3600

    adm_icu["icu_within_24h"] = np.where(
        has_icu & (time_diff_hours >= 0) & (time_diff_hours <= 24),
        1,
        0
    )

    print("Label distribution (icu_within_24h):")
    print(adm_icu["icu_within_24h"].value_counts())

    # Cohort columns
    keep_cols = [
        "subject_id",
        "hadm_id",
        "icu_within_24h",
        "gender_encoded",
        "los_days",
    ]

    cohort = adm_icu[keep_cols].copy()
    print("Final cohort shape:", cohort.shape)

    # Save cohort to parquet
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "cohort.parquet")
    cohort.to_parquet(out_path, index=False)
    print(f"Cohort saved to {out_path}")


if __name__ == "__main__":
    main()
