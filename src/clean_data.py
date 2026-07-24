"""
Data cleaning and feature engineering pipeline for the Diabetes 130-US Hospitals
readmission dataset (Strack et al., 2014; UCI ML Repository, DOI 10.24432/C5230J).

Produces a model-ready dataframe with:
  - Deduplicated encounters (first encounter per patient, to avoid leakage)
  - Cleaned categorical fields (missing markers standardized)
  - Engineered utilization features
  - Binary target: readmitted_30d (1 if readmitted <30 days, else 0)
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/diabetic_data.csv"
IDS_PATH = "data/IDS_mapping.csv"
OUT_PATH = "data/diabetic_data_clean.csv"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    return df


def drop_high_missing_and_ids(df):
    # weight is ~97% missing in this dataset — not usable
    # payer_code is largely administrative, not clinically predictive; high missingness
    drop_cols = ["weight", "payer_code"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def dedupe_patients(df):
    """
    Keep only the FIRST encounter per patient (by encounter_id, which is
    monotonically increasing / roughly chronological in this dataset).
    This avoids leaking information about a patient's future encounters
    into the training data, and prevents the same patient appearing in
    both train and test splits.
    """
    df = df.sort_values("encounter_id")
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    return df


def filter_valid_discharge(df):
    """
    Discharge disposition codes 11, 13, 14, 19, 20, 21 correspond to
    death or hospice — these patients cannot be readmitted, and including
    them would bias the target definition. Standard practice in prior
    work on this dataset (Strack et al. 2014) is to exclude them.
    """
    death_hospice_codes = [11, 13, 14, 19, 20, 21]
    return df[~df["discharge_disposition_id"].isin(death_hospice_codes)]


def engineer_target(df):
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    return df


def engineer_features(df):
    # Total prior healthcare utilization in the year before this encounter
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    # Any prior inpatient stay (strong readmission risk factor in the literature)
    df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    # Number of diabetes medications actually changed at this encounter
    med_change_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "insulin",
    ]
    med_change_cols = [c for c in med_change_cols if c in df.columns]
    df["n_meds_changed"] = (df[med_change_cols] == "Up").sum(axis=1) + \
                            (df[med_change_cols] == "Down").sum(axis=1)
    return df


def clean_categoricals(df):
    # medical_specialty: collapse rare categories, fill missing as "Missing"
    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
    top_specialties = df["medical_specialty"].value_counts().nlargest(15).index
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top_specialties), "Other"
    )

    df["race"] = df["race"].fillna("Missing")

    # diag_1/2/3 are ICD-9 codes with huge cardinality; for a first model,
    # bucket by leading digit / category is a common simplification.
    # Left as raw strings here — grouping happens in the modeling notebook.
    return df


def run_pipeline():
    df = load_raw()
    n0 = len(df)
    df = drop_high_missing_and_ids(df)
    df = dedupe_patients(df)
    n1 = len(df)
    df = filter_valid_discharge(df)
    n2 = len(df)
    df = engineer_target(df)
    df = engineer_features(df)
    df = clean_categoricals(df)

    print(f"Raw encounters:              {n0:,}")
    print(f"After dedup (1st encounter): {n1:,}  ({n0-n1:,} removed)")
    print(f"After removing death/hospice:{n2:,}  ({n1-n2:,} removed)")
    print(f"Final target distribution:")
    print(df["readmitted_30d"].value_counts(normalize=True).rename("proportion"))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned data to {OUT_PATH}")
    return df


if __name__ == "__main__":
    run_pipeline()
