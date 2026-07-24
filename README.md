# Diabetes Hospital Readmission Risk & Equity Audit

Predicting 30-day hospital readmission risk for diabetes patients, with an explicit fairness/equity audit of the model's performance across demographic subgroups. Built as an end-to-end ML project (data pipeline, model, equity audit, interactive dashboard).

**[Read the full methods report](reports/methods_report.md)**

## Why this project

Diabetes patients are readmitted to the hospital at disproportionately high rates, and hospitals face financial penalties (CMS Hospital Readmissions Reduction Program) for excess readmissions. Predictive models can help target follow-up care, but a model with good *aggregate* accuracy can still fail specific patient groups. This project builds a working risk model and then explicitly audits it for exactly that kind of hidden disparity, following the approach used in algorithmic fairness research (e.g. Obermeyer et al., 2019, *Science*).

## Dataset

[UCI Diabetes 130-US Hospitals for Years 1999-2008](https://doi.org/10.24432/C5230J) (Strack et al., 2014): 101,766 inpatient encounters across 130 U.S. hospitals. After deduplicating to one encounter per patient and excluding death/hospice discharges, **69,973 encounters** remain, with a 9.0% 30-day readmission rate.

The raw/cleaned CSVs aren't committed to this repo (they're patient-level health records, even if de-identified and public). Download `diabetic_data.csv` and `IDS_mapping.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place them in `data/` to reproduce.

## Pipeline

```
data/diabetic_data.csv
      │
      ▼
src/clean_data.py          → dedup, drop death/hospice, feature engineering
      │
      ▼
src/train_model.py         → Logistic Regression + HistGradientBoosting, 80/20 split
      │
      ▼
src/fairness_analysis.py   → subgroup AUC / false-negative-rate audit (race, gender, age)
      │
      ▼
dashboard/app.py           → Streamlit app: risk predictor + equity visualizations
```

## Results

| Model | ROC-AUC | Average Precision |
|---|---|---|
| Logistic Regression | 0.644 | 0.171 |
| HistGradientBoosting | 0.641 | 0.179 |

**Equity finding:** Asian patients (n=90 in test set) had the lowest AUC (0.556) and highest false negative rate (0.667) of any racial group, versus 0.458 overall. That means the model missed two-thirds of true readmissions in this subgroup. Age also showed wide disparities (false negative rate ranged from 0.265 to 0.667 across age bands). Full breakdown and discussion in the [methods report](reports/methods_report.md).

## Dashboard

Interactive Streamlit app with three views: a risk predictor (enter patient features, get a predicted readmission risk score), a model equity explorer (AUC / false negative rate by race, gender, age), and a data overview.

```bash
pip install -r requirements.txt
python src/clean_data.py       # produces data/diabetic_data_clean.csv
python src/train_model.py      # trains and saves models/
python src/fairness_analysis.py
streamlit run dashboard/app.py
```

## Repo structure

```
src/                  data cleaning, model training, fairness analysis
dashboard/app.py      Streamlit dashboard
reports/              fairness_results.json, methods_report.md
models/               trained pipelines (git-ignored if large)
data/                 IDS_mapping.csv (raw patient data not committed)
requirements.txt
```

## Limitations

This is administrative EHR data from 1999-2008 and predates many changes in diabetes care; the model should not be read as clinically validated. The ~0.64 AUC ceiling reflects the absence of clinical narrative and social-determinant features. See the [methods report](reports/methods_report.md) for full discussion.

## References

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*. UCI ML Repository, DOI: 10.24432/C5230J.

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.
