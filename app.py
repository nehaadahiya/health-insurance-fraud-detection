import streamlit as st
st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide")

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Load model ===
@st.cache_resource
def load_stacked_model():
    model = joblib.load("models/stacked_model.pkl")
    top_features = joblib.load("models/top_25_features.pkl")
    return model, top_features

model, top_features = load_stacked_model()

# === Streamlit Tabs ===
tab1, tab2 = st.tabs(["🧠 Supervised Ensemble Prediction", "🕵️ Unsupervised Anomaly Scoring"])

# ========================================
# 🔍 TAB 1: Supervised Prediction
# ========================================
with tab1:
    st.title("🧠 Supervised Ensemble Prediction (Stacked Model)")

    uploaded_file = st.file_uploader("Upload `model_ready.csv`-style data", type=["csv"])
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📋 Preview")
        st.dataframe(df_raw.head())

        # Preprocessing
        df = df_raw.drop(columns=['BeneID', 'ClaimID', 'Provider'], errors='ignore')
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]
        df.fillna(0, inplace=True)
        X_all = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
        X_sel = X_all[top_features]

        threshold = st.sidebar.slider(
        label="Threshold for Fraud Classification",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01,
            format="%.2f"  
        )

        # Predict
        probs = model.predict_proba(X_sel)[:, 1]
        preds = (probs >= 0.4).astype(int)

        df_result = df_raw.copy()
        df_result["Fraud_Probability"] = probs
        df_result["Prediction (1=Fraud)"] = preds
        st.subheader("📈 Results")
        st.dataframe(df_result.head(10))
        # Bar plot: Fraud vs Non-Fraud Predictions
        st.subheader("🔢 Prediction Summary")
        fraud_count = df_result["Prediction (1=Fraud)"].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        fraud_count.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
        ax1.set_title("Fraud vs Non-Fraud Predictions")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Histogram: Probability Distribution
        st.subheader("🎯 Probability Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df_result["Fraud_Probability"], bins=30, kde=True, ax=ax2, color='purple')
        ax2.set_title("Distribution of Fraud Probabilities")
        ax2.set_xlabel("Fraud Probability")
        st.pyplot(fig2)

        st.download_button("⬇ Download Predictions", df_result.to_csv(index=False), "predictions.csv")

# ========================================
# 🕵️ TAB 2: Isolation Forest
# ========================================
with tab2:
    st.title("🕵️ Isolation Forest - Anomaly Score Analysis")
    iso_file = st.file_uploader("Upload the same or raw claim file here", type=["csv"], key="iso")

    if iso_file:
        df_iso = pd.read_csv(iso_file)

        st.subheader("📋 Preview")
        st.dataframe(df_iso.head())

        # Preprocess
        df = df_iso.drop(columns=['BeneID', 'ClaimID'], errors='ignore')
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]
        df.fillna(0, inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        st.sidebar.markdown("🛠️ **Anomaly Detection Settings**")
        contamination = st.sidebar.slider(
            label="Contamination (expected fraud %)",
            min_value=0.01,
            max_value=0.20, 
            value=0.05,
            step=0.01,
            format="%.2f"
        )

        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)

        iso.fit(X_scaled)
        scores = iso.decision_function(X_scaled)
        anomaly_flags = iso.predict(X_scaled)

        df_iso['AnomalyScore'] = scores
        df_iso['AnomalyFlag'] = anomaly_flags
        anomaly_summary = df_iso['AnomalyFlag'].value_counts().rename({1: "Normal", -1: "Anomaly"})
        st.subheader("📌 Anomaly Prediction Summary")
        st.write(anomaly_summary)

        fig3, ax3 = plt.subplots()
        anomaly_summary.plot(kind="bar", color=["green", "red"], ax=ax3)
        ax3.set_ylabel("Number of Records")
        ax3.set_title(f"Anomaly Count at Contamination = {contamination:.2f}")
        st.pyplot(fig3)

        df_iso['RiskLevel'] = pd.cut(scores, bins=[-np.inf, -0.2, 0.1, np.inf], labels=["High", "Medium", "Low"])

        st.subheader("📉 Anomaly Scores")
        st.dataframe(df_iso[['AnomalyScore', 'RiskLevel']].head(10))

        st.download_button("⬇ Download Anomaly Scores", df_iso.to_csv(index=False), "anomaly_scores.csv")
        st.write("Anomaly Score Summary:")
        st.write(df_iso['AnomalyScore'].describe())

        st.subheader("📊 Score Distribution")
        if df_iso['AnomalyScore'].nunique() > 1:
            fig, ax = plt.subplots()
            sns.histplot(df_iso['AnomalyScore'], bins=30, kde=True, ax=ax, color='orange')
            ax.set_title("Anomaly Score Distribution")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Density")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough variation in Anomaly Scores to plot a histogram.")

