"""
Fairness / equity analysis of the readmission risk model across
demographic subgroups (race, gender, age).

Rather than only checking raw accuracy, this looks at:
  - AUC per subgroup (does the model discriminate equally well for everyone?)
  - False Negative Rate per subgroup (does the model MISS high-risk patients
    more often in some groups? This is the clinically important error —
    a missed readmission risk means a patient doesn't get preventive care)
  - Predicted positive rate per subgroup (does the model flag some groups
    as high-risk more/less often than their true base rate would suggest?)

This mirrors the kind of subgroup analysis expected in health equity /
algorithmic fairness research (e.g., Obermeyer et al. 2019, Science).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix
import json

MODEL_PATH = "models/hgb_pipeline.joblib"
TEST_SET_PATH = "models/test_set_with_demographics.csv"
OUT_PATH = "reports/fairness_results.json"

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


def subgroup_metrics(y_true, y_pred, y_proba):
    if len(np.unique(y_true)) < 2:
        return {"n": len(y_true), "note": "only one class present, AUC undefined"}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # missed high-risk patients
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "true_positive_rate_base": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
    }


def run():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_SET_PATH)
    df = df[df["gender"] != "Unknown/Invalid"]

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str).fillna("Missing")
    y = df[TARGET].values

    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    results = {"overall": subgroup_metrics(y, preds, proba)}

    for group_col in ["race", "gender", "age"]:
        results[group_col] = {}
        for grp in df[group_col].dropna().unique():
            mask = (df[group_col] == grp).values
            if mask.sum() < 30:  # skip tiny subgroups, not statistically meaningful
                continue
            results[group_col][str(grp)] = subgroup_metrics(
                y[mask], preds[mask], proba[mask]
            )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print a readable summary
    print("=== OVERALL ===")
    print(json.dumps(results["overall"], indent=2))

    for group_col in ["race", "gender", "age"]:
        print(f"\n=== BY {group_col.upper()} ===")
        rows = []
        for grp, m in results[group_col].items():
            if "auc" in m:
                rows.append((grp, m["n"], m["auc"], m["false_negative_rate"], m["predicted_positive_rate"]))
        rows.sort(key=lambda r: -r[1])
        print(f"{'Group':<20}{'N':>8}{'AUC':>8}{'FNR':>8}{'PredPosRate':>14}")
        for grp, n, auc, fnr, ppr in rows:
            print(f"{grp:<20}{n:>8}{auc:>8.3f}{fnr:>8.3f}{ppr:>14.3f}")

    print(f"\nSaved full results to {OUT_PATH}")


if __name__ == "__main__":
    run()
