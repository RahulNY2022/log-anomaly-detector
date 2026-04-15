import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("🔐 Log Anomaly Detector")

contamination = st.slider("Sensitivity (higher = more anomalies flagged)", 
                           min_value=0.01, max_value=0.40, value=0.15)

uploaded_file = st.file_uploader("Upload a .log file", type=["log", "txt", "csv"])

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    records = []
    for line in lines:
        match = re.match(r'(\w+ \d+ \d+:\d+:\d+) \S+ \S+: (.+)', line)
        if match:
            _, message = match.groups()
            records.append({
                "failed": 1 if "Failed" in message or "Invalid" in message else 0,
                "accepted": 1 if "Accepted" in message else 0,
                "root_attempt": 1 if "root" in message.lower() else 0,
                "message": message
            })

    if not records:
        st.error("❌ No matching log entries found. Try a different file.")
    else:
        df = pd.DataFrame(records)

        # Train model
        features = df[["failed", "accepted", "root_attempt"]]
        model = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = model.fit_predict(features) == -1

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("📋 Total Logs", len(df))
        col2.metric("🚨 Anomalies Found", df["anomaly"].sum())

        # Chart
        st.subheader("📊 Anomaly Timeline")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df.index, df["failed"], alpha=0.4, label="Failed Logins")
        anomalies = df[df["anomaly"] == True]
        ax.scatter(anomalies.index, anomalies["failed"], 
                   color="red", label="Anomaly", zorder=5)
        ax.legend()
        st.pyplot(fig)

        # Table
        st.subheader("🔍 Flagged Log Entries")
        st.dataframe(df[df["anomaly"] == True][["message"]].reset_index(drop=True))