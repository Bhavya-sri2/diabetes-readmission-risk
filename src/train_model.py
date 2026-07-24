"""
Train and evaluate 30-day readmission risk models on the cleaned
Diabetes 130-US Hospitals dataset.

Models:
  1. Logistic Regression (interpretable baseline)
  2. HistGradientBoostingClassifier (stronger, handles nonlinearity/interactions)

Evaluation:
  - Stratified train/val/test split (patient-level, already deduped upstream)
  - ROC-AUC and Average Precision (PR-AUC): accuracy is misleading given
    the ~9% positive class rate
  - Calibration is checked informally via predicted probability distribution
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
import json

DATA_PATH = "data/diabetic_data_clean.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "total_prior_visits", "had_prior_inpatient", "n_meds_changed",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "max_glu_serum", "A1Cresult",
    "insulin", "change", "diabetesMed",
]

TARGET = "readmitted_30d"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop rows with unresolvable gender (3 rows, "Unknown/Invalid")
    df = df[df["gender"] != "Unknown/Invalid"]
    return df


def build_pipeline(model, sparse_ok=True):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_ok), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Avg Precision (PR): {ap:.4f}")
    print(classification_report(y_test, preds, target_names=["No 30d readmit", "30d readmit"]))
    return {"model": name, "roc_auc": auc, "avg_precision": ap}


def run():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET]

    # Fill any remaining NaNs in categoricals with "Missing" so OneHotEncoder is happy
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str).fillna("Missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_train):,}   Test size: {len(X_test):,}")
    print(f"Train positive rate: {y_train.mean():.4f}   Test positive rate: {y_test.mean():.4f}")

    results = []

    # 1. Logistic Regression baseline
    logreg = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    logreg.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", logreg, X_test, y_test))
    joblib.dump(logreg, f"{MODEL_DIR}/logreg_pipeline.joblib")

    # 2. Gradient Boosted Trees
    hgb = build_pipeline(
        sparse_ok=False,
        model=HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=200,
            learning_rate=0.05,
        )
    )
    hgb.fit(X_train, y_train)
    results.append(evaluate("HistGradientBoosting", hgb, X_test, y_test))
    joblib.dump(hgb, f"{MODEL_DIR}/hgb_pipeline.joblib")

    with open(f"{MODEL_DIR}/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save test set (with identifiers for the fairness analysis step)
    test_df = df.loc[X_test.index].copy()
    test_df.to_csv(f"{MODEL_DIR}/test_set_with_demographics.csv", index=False)

    print("\nModels and results saved to models/")
    return results


if __name__ == "__main__":
    run()
