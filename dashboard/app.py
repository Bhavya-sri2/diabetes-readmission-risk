"""
Streamlit dashboard for the Diabetes 130-US Hospitals 30-day readmission
risk project.

Two tabs:
  1. Risk Predictor — enter patient/encounter features, get a predicted
     30-day readmission risk score from the trained model.
  2. Model Equity — visualize how model performance (AUC, false negative
     rate) varies across demographic subgroups, based on the fairness
     analysis computed in src/fairness_analysis.py.

Run locally with:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Diabetes Readmission Risk & Equity Dashboard",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hgb_pipeline.joblib")
FAIRNESS_PATH = os.path.join(BASE_DIR, "reports", "fairness_results.json")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "diabetic_data_clean.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_fairness_results():
    with open(FAIRNESS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_clean_data():
    return pd.read_csv(CLEAN_DATA_PATH)


st.title("Diabetes Hospital Readmission: Risk Prediction & Model Equity")
st.caption(
    "Built on the UCI Diabetes 130-US Hospitals dataset (Strack et al., 2014; "
    "101,766 encounters, 130 hospitals, 1999-2008). Predicts 30-day readmission "
    "risk and evaluates whether the model performs equally well across demographic groups."
)

tab1, tab2, tab3 = st.tabs(["🩺 Risk Predictor", "⚖️ Model Equity", "📊 Data Overview"])

# ---------------------------------------------------------------------------
# TAB 1: Risk Predictor
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Predict 30-day readmission risk for a patient encounter")
    st.write("Adjust the inputs below to see how predicted risk changes.")

    col1, col2, col3 = st.columns(3)

    with col1:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox(
            "Age group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
             "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            index=6,
        )
        admission_type_id = st.selectbox("Admission type ID", [1, 2, 3, 4, 5, 6, 7, 8], index=0)
        discharge_disposition_id = st.number_input("Discharge disposition ID", min_value=1, max_value=29, value=1)
        admission_source_id = st.number_input("Admission source ID", min_value=1, max_value=25, value=7)

    with col2:
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of lab procedures", 0, 130, 43)
        num_procedures = st.slider("Number of procedures", 0, 6, 1)
        num_medications = st.slider("Number of medications", 1, 80, 16)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
        medical_specialty = st.selectbox(
            "Admitting medical specialty",
            ["Missing", "InternalMedicine", "Family/GeneralPractice", "Emergency/Trauma",
             "Cardiology", "Surgery-General", "Other"],
        )

    with col3:
        number_outpatient = st.slider("Prior outpatient visits (past yr)", 0, 20, 0)
        number_emergency = st.slider("Prior emergency visits (past yr)", 0, 20, 0)
        number_inpatient = st.slider("Prior inpatient visits (past yr)", 0, 10, 0)
        max_glu_serum = st.selectbox("Max glucose serum test result", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C test result", ["None", "Norm", ">7", ">8"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        change = st.selectbox("Medication changed at this visit?", ["No", "Ch"])
        diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])

    total_prior_visits = number_outpatient + number_emergency + number_inpatient
    had_prior_inpatient = int(number_inpatient > 0)
    n_meds_changed = 1 if change == "Ch" else 0

    input_row = pd.DataFrame([{
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "total_prior_visits": total_prior_visits,
        "had_prior_inpatient": had_prior_inpatient,
        "n_meds_changed": n_meds_changed,
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "medical_specialty": medical_specialty,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed,
    }])

    if st.button("Predict readmission risk", type="primary"):
        model = load_model()
        proba = model.predict_proba(input_row)[0, 1]
        st.metric("Predicted 30-day readmission risk", f"{proba*100:.1f}%")

        if proba > 0.35:
            st.warning("Above the model's overall predicted-positive threshold — flagged as elevated risk.")
        else:
            st.success("Below the model's overall predicted-positive threshold.")

        st.caption(
            "Note: this model has an overall ROC-AUC of ~0.64 — it is a research/demo "
            "model, not a validated clinical decision tool. See the Model Equity tab "
            "for known performance gaps across demographic subgroups."
        )

# ---------------------------------------------------------------------------
# TAB 2: Model Equity
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Does the model perform equally well across demographic groups?")
    st.write(
        "A model can have good overall accuracy while still performing worse for "
        "some subgroups. Here we compare AUC (ability to discriminate risk) and "
        "False Negative Rate (share of truly-readmitted patients the model misses) "
        "across race, gender, and age groups."
    )

    results = load_fairness_results()

    metric_choice = st.radio("Metric", ["AUC", "False Negative Rate", "Predicted Positive Rate"], horizontal=True)
    metric_key = {"AUC": "auc", "False Negative Rate": "false_negative_rate", "Predicted Positive Rate": "predicted_positive_rate"}[metric_choice]

    group_choice = st.selectbox("Group by", ["race", "gender", "age"])

    rows = []
    for grp, m in results[group_choice].items():
        if metric_key in m:
            rows.append({"Group": grp, "N": m["n"], metric_choice: m[metric_key]})
    plot_df = pd.DataFrame(rows).sort_values(metric_choice, ascending=False)

    fig = px.bar(
        plot_df, x="Group", y=metric_choice, text="N",
        title=f"{metric_choice} by {group_choice}",
        color=metric_choice, color_continuous_scale="RdYlGn_r" if metric_choice != "AUC" else "RdYlGn",
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Overall model:** AUC = {results['overall']['auc']:.3f}, "
        f"False Negative Rate = {results['overall']['false_negative_rate']:.3f}"
    )

    st.info(
        "⚠️ Small subgroups (e.g. Asian patients, n≈90) show notably worse AUC and "
        "higher false negative rates than larger groups. With this sample size the "
        "gap could partly reflect noise, but it is large enough to warrant caution "
        "before deploying this model without subgroup-specific validation or "
        "recalibration. This is exactly the kind of gap that health equity-focused "
        "algorithmic auditing is meant to catch before a model reaches clinical use."
    )

# ---------------------------------------------------------------------------
# TAB 3: Data Overview
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Dataset overview")
    df = load_clean_data()
    st.write(f"**{len(df):,}** unique patient encounters (deduplicated to first encounter per patient)")

    c1, c2 = st.columns(2)
    with c1:
        readmit_counts = df["readmitted_30d"].value_counts().rename({0: "No 30-day readmit", 1: "30-day readmit"})
        fig1 = px.pie(values=readmit_counts.values, names=readmit_counts.index, title="30-day readmission rate")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        age_readmit = df.groupby("age")["readmitted_30d"].mean().reset_index()
        fig2 = px.bar(age_readmit, x="age", y="readmitted_30d", title="Readmission rate by age group")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "**Source:** Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., "
        "Cios, K.J., Clore, J.N. (2014). *Impact of HbA1c Measurement on Hospital "
        "Readmission Rates.* BioMed Research International. "
        "[UCI ML Repository, DOI: 10.24432/C5230J](https://doi.org/10.24432/C5230J)"
    )
