# Predicting 30-Day Hospital Readmission in Diabetes Patients: A Model Development and Equity Audit

**Bhavya Sri Dasari**

## Abstract

Hospital readmissions within 30 days are a key quality and cost metric in U.S. healthcare, and diabetes patients are readmitted at disproportionately high rates. This project develops a machine learning pipeline to predict 30-day readmission risk using the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset (Strack et al., 2014), covering 101,766 encounters across 130 hospitals. After deduplicating to one encounter per patient and excluding death/hospice discharges, 69,973 encounters remained, with a 30-day readmission rate of 9.0%. A HistGradientBoostingClassifier achieved a ROC-AUC of 0.641 and average precision of 0.179, modestly outperforming a logistic regression baseline (ROC-AUC 0.644, average precision 0.171) on precision while trailing slightly on AUC. Beyond aggregate performance, this project audits the model for demographic subgroup disparities, following the framework used in algorithmic fairness research such as Obermeyer et al. (2019). The audit finds that model performance is not uniform across race: Asian patients (n=90 in the test set) had the lowest AUC (0.556) and highest false negative rate (0.667) of any racial group, compared to an overall false negative rate of 0.458. These findings are presented alongside an interactive dashboard for risk scoring and subgroup performance exploration.

## 1. Introduction

Unplanned hospital readmissions are costly and, in many cases, preventable. Diabetes is one of the conditions most strongly associated with readmission risk due to its chronic, multi-system nature and sensitivity to medication management, follow-up care, and social determinants of health. The Centers for Medicare & Medicaid Services (CMS) penalizes hospitals for excess readmissions under the Hospital Readmissions Reduction Program, making accurate, well-calibrated readmission risk prediction operationally valuable.

However, predictive models deployed in clinical or operational settings can encode or amplify existing disparities if subgroup performance is not explicitly evaluated. This project has two goals: (1) build a reasonably performant 30-day readmission risk model from structured EHR-derived data, and (2) rigorously test whether that model's errors are evenly distributed across race, gender, and age groups, rather than assuming aggregate metrics tell the whole story.

## 2. Data

The dataset is the UCI Diabetes 130-US Hospitals dataset (Strack, DeShazo, Gennings, Olmo, Ventura, Cios, & Clore, 2014, *BioMed Research International*; UCI ML Repository, DOI: 10.24432/C5230J), comprising 101,766 inpatient encounters for diabetic patients across 130 U.S. hospitals between 1999 and 2008. Each row represents a hospital encounter and includes demographics (race, gender, age band), admission/discharge/referral source codes, utilization history (prior outpatient, emergency, and inpatient visits), clinical indicators (number of lab procedures, procedures, medications, diagnoses), diabetes-specific lab results (A1C, glucose serum), 23 medication-change fields, and the outcome variable `readmitted` (values: `NO`, `>30`, `<30`).

### 2.1 Preprocessing

Two columns were dropped outright: `weight` (~97% missing) and `payer_code` (high missingness, low clinical relevance). The dataset was then deduplicated to the first encounter per patient (sorted by `encounter_id`), reducing 101,766 rows to 71,518, to avoid data leakage from a single patient contributing multiple correlated encounters across train/test splits. Encounters ending in death or discharge to hospice (discharge disposition codes 11, 13, 14, 19, 20, 21) were excluded, since these patients cannot be readmitted by definition. This removed a further 1,545 rows, leaving 69,973 encounters. Three rows with `gender = "Unknown/Invalid"` were dropped before modeling.

The binary target `readmitted_30d` was defined as 1 if `readmitted == "<30"`, else 0. The resulting class distribution is 91.0% negative and 9.0% positive, a substantial imbalance that shaped both the modeling and evaluation strategy.

### 2.2 Feature Engineering

Three derived features were added: `total_prior_visits` (sum of prior outpatient, emergency, and inpatient visit counts), `had_prior_inpatient` (binary indicator of any prior inpatient stay, a well-documented risk factor in the readmission literature), and `n_meds_changed` (count of diabetes medications with a dosage change at the current encounter, derived from 16 medication columns). `medical_specialty` was collapsed to the 15 most frequent categories plus an "Other" bucket to control cardinality; missing race and specialty values were coded as an explicit "Missing" category rather than imputed, since missingness may itself be informative (e.g., correlated with hospital data practices).

## 3. Modeling

Eleven numeric and twelve categorical features were used as predictors (demographics, admission/discharge/source codes, utilization counts, lab results, and medication indicators). Numeric features were standardized; categorical features were one-hot encoded with unknown categories handled gracefully at inference time. Data was split 80/20 (train/test) with stratification on the target, yielding 55,976 training and 13,994 test encounters.

Two models were trained, both with `class_weight="balanced"` to address the 9:1 class imbalance:

1. **Logistic Regression**, an interpretable linear baseline.
2. **HistGradientBoostingClassifier**, a gradient-boosted tree ensemble capable of capturing nonlinear interactions.

Given the class imbalance, ROC-AUC and average precision (area under the precision-recall curve) were used as primary metrics rather than raw accuracy, which is uninformative when the majority class is 91% of the data.

### 3.1 Results

| Model | ROC-AUC | Average Precision |
|---|---|---|
| Logistic Regression | 0.644 | 0.171 |
| HistGradientBoosting | 0.641 | 0.179 |

The two models perform comparably; HistGradientBoosting was selected for the fairness audit and dashboard due to its slightly better precision-recall tradeoff and capacity to model feature interactions. Both models' AUCs (~0.64) are consistent with the broader readmission-prediction literature, which generally finds that administrative and utilization data alone, without clinical narrative, social determinants, or post-discharge information, caps achievable discrimination in this range. This is a known limitation of the dataset, not a modeling artifact, and is discussed further in Section 5.

## 4. Fairness / Equity Audit

Aggregate performance metrics can mask meaningfully different error rates across subgroups, a concern with direct clinical consequences, since a missed high-risk prediction (a false negative) means a patient does not receive the preventive follow-up that a positive flag would trigger. Following the approach used in algorithmic fairness audits of clinical risk scores (e.g., Obermeyer et al., 2019, *Science*), subgroup performance was evaluated across race, gender, and age band on the held-out test set. Subgroups with fewer than 30 test cases were excluded from analysis as statistically unreliable.

### 4.1 Overall Test-Set Performance

- N = 13,994; AUC = 0.641; False Negative Rate = 0.458; Predicted Positive Rate = 0.354 (vs. true base rate 0.090)

The predicted positive rate substantially exceeds the true base rate because `class_weight="balanced"` shifts the decision threshold to reduce false negatives at the cost of more false positives. This is a deliberate choice favoring sensitivity, appropriate for a screening-style clinical use case where missing a readmission is more costly than an unnecessary follow-up call.

### 4.2 Performance by Race

| Race | N | AUC | False Negative Rate |
|---|---|---|---|
| Caucasian | 10,455 | 0.633 | 0.452 |
| African American | 2,544 | 0.654 | 0.482 |
| Missing | 397 | 0.741 | 0.350 |
| Hispanic | 291 | 0.672 | 0.464 |
| Other | 217 | 0.675 | 0.538 |
| **Asian** | **90** | **0.556** | **0.667** |

Asian patients had both the lowest AUC and the highest false negative rate of any racial group in the test set. The model missed two-thirds of true 30-day readmissions in this subgroup, compared to 45.8% overall. The sample size (n=90) is small enough that some of this gap likely reflects sampling noise rather than a stable, generalizable disparity, and it should not be over-interpreted as a definitive finding. However, it is large enough to warrant explicit disclosure and caution. A disparity of this magnitude, in the clinically important direction (missed high-risk patients), is exactly the kind of gap that should be flagged before a model is used to guide real follow-up care decisions, rather than discovered after deployment.

### 4.3 Performance by Gender

| Gender | N | AUC | False Negative Rate |
|---|---|---|---|
| Female | 7,413 | 0.644 | 0.429 |
| Male | 6,581 | 0.637 | 0.491 |

Male patients had a modestly higher false negative rate than female patients (49.1% vs. 42.9%), a smaller but still notable gap given the much larger sample sizes involved.

### 4.4 Performance by Age

Age-group disparities were the largest observed of any dimension, with false negative rates ranging from 0.265 ([80-90) age band) to 0.667 ([50-60) age band), and 0.625 in the [30-40) band. This suggests the model is systematically better at catching readmission risk in the oldest patients and worse in middle-aged patients, plausibly because utilization history (a strong predictor in this feature set) is more information-rich for older patients with longer care histories, while younger and middle-aged patients present with sparser records.

### 4.5 Interpretation

These results illustrate a general principle in clinical ML: a model with acceptable aggregate discrimination can still under-serve specific subgroups, particularly smaller ones. The Asian-patient finding in particular should be treated as a hypothesis for further investigation with more data, not a conclusive result, given n=90. Before any real-world use, this model would require subgroup-stratified threshold calibration, a larger and more representative sample of underrepresented groups, and ideally, external validation on a more recent, more diverse hospital cohort, since this dataset spans 1999-2008 and clinical practice has changed substantially since then.

## 5. Limitations

The dataset is administrative/coded EHR data from 1999-2008; it predates many changes in diabetes care and discharge practice, and results should not be assumed to generalize to current clinical settings. The AUC ceiling (~0.64) reflects the absence of clinical narrative, social determinants of health, and post-discharge information, all known to meaningfully improve readmission prediction. The small sample size for some subgroups (e.g., Asian patients, n=90 in test) limits the statistical confidence of subgroup findings. Discharge disposition and admission source are represented as numeric IDs mapped via an external `IDS_mapping.csv` file rather than as clinically grouped categories, which is a simplification. Finally, this project uses a single train/test split rather than cross-validation; results should be interpreted as a single point estimate rather than a fully stability-tested benchmark.

## 6. Conclusion

This project builds an end-to-end pipeline (data cleaning, feature engineering, model training, and an interactive dashboard) for 30-day diabetes readmission risk prediction, and pairs it with a subgroup fairness audit that surfaces a meaningful, if statistically uncertain, performance gap for Asian patients and smaller but consistent gaps by gender and age. The core takeaway is methodological: aggregate model metrics are not sufficient evidence that a clinical risk model is safe or fair to deploy, and subgroup auditing should be a standard, not optional, step in healthcare ML development.

## References

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.
