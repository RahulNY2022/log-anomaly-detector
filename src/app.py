import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("🔐 Log Anomaly Detector")
st.markdown("Supports **SSH**, **Apache**, and **Windows Event** logs")

contamination = st.slider("Sensitivity (higher = more anomalies flagged)",
                           min_value=0.01, max_value=0.40, value=0.15)

uploaded_file = st.file_uploader("Upload a log file", type=["log", "txt", "csv"])

# ── Log Parsers ───────────────────────────────────────────────────

def parse_ssh(lines):
    records = []
    for line in lines:
        match = re.match(r'(\w+ \d+ \d+:\d+:\d+) \S+ \S+: (.+)', line)
        if match:
            _, message = match.groups()
            records.append({
                "failed":       1 if "Failed" in message or "Invalid" in message else 0,
                "accepted":     1 if "Accepted" in message else 0,
                "root_attempt": 1 if "root" in message.lower() else 0,
                "error":        0,
                "message":      message
            })
    return records

def parse_apache(lines):
    records = []
    # Apache combined log format:
    # 127.0.0.1 - - [01/Jan/2024:00:00:01 +0000] "GET /index.html HTTP/1.1" 200 1234
    pattern = re.compile(
        r'(\S+) \S+ \S+ \[(.+?)\] "(\S+) (\S+) \S+" (\d+) (\S+)'
    )
    for line in lines:
        match = pattern.match(line)
        if match:
            ip, time, method, path, status, size = match.groups()
            status = int(status)
            records.append({
                "failed":       1 if status >= 400 else 0,
                "accepted":     1 if status == 200 else 0,
                "root_attempt": 1 if "admin" in path.lower() or "root" in path.lower() else 0,
                "error":        1 if status >= 500 else 0,
                "message":      f"{method} {path} → {status}"
            })
    return records

def parse_windows(lines):
    records = []
    for i, line in enumerate(lines):
        # Skip header
        if i == 0:
            continue
        # Handle both comma and tab separated
        parts = re.split(r'[,\t]', line)
        if len(parts) >= 4:
            try:
                level    = parts[0].strip().lower()
                date     = parts[1].strip()
                source   = parts[2].strip()
                event_id = int(re.search(r'\d+', parts[3]).group())
                message  = ",".join(parts[4:]).strip() if len(parts) > 4 else ""
                records.append({
                    "failed":       1 if event_id == 4625 or "fail" in level or "error" in level else 0,
                    "accepted":     1 if event_id == 4624 else 0,
                    "root_attempt": 1 if event_id == 4672 else 0,
                    "error":        1 if "critical" in level or "error" in level else 0,
                    "message":      f"EventID {event_id} [{level}] - {source} at {date}"
                })
            except:
                continue
    return records

def detect_log_type(lines):
    sample = " ".join(lines[:5])
    if re.search(r'Event ID|EventID|4624|4625', sample, re.IGNORECASE):
        return "Windows Event"
    elif re.search(r'\d+\.\d+\.\d+\.\d+.*\[.*\].*HTTP', sample):
        return "Apache"
    else:
        return "SSH"

# ── Main App ──────────────────────────────────────────────────────

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()

    # Auto-detect log type
    log_type = detect_log_type(lines)
    st.info(f"📁 Detected log type: **{log_type}**")

    # Parse based on type
    if log_type == "Apache":
        records = parse_apache(lines)
    elif log_type == "Windows Event":
        records = parse_windows(lines)
    else:
        records = parse_ssh(lines)

    if not records:
        st.error("❌ No matching log entries found. Try a different file.")
    else:
        df = pd.DataFrame(records)

        # Train Isolation Forest
        features = df[["failed", "accepted", "root_attempt", "error"]]
        model = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = model.fit_predict(features) == -1

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("📋 Total Logs", len(df))
        col2.metric("🚨 Anomalies Found", df["anomaly"].sum())
        col3.metric("⚠️ Failed Events", df["failed"].sum())

        # Chart
        st.subheader("📊 Anomaly Timeline")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df.index, df["failed"], alpha=0.4, label="Failed Events")
        anomalies = df[df["anomaly"] == True]
        ax.scatter(anomalies.index, anomalies["failed"],
                   color="red", label="Anomaly", zorder=5)
        ax.set_xlabel("Log Entry #")
        ax.legend()
        st.pyplot(fig)

        # Anomaly Table
        st.subheader("🔍 Flagged Log Entries")
        st.dataframe(
            df[df["anomaly"] == True][["message"]].reset_index(drop=True)
        )

        # Download flagged logs
        st.subheader("💾 Export Anomalies")
        csv = df[df["anomaly"] == True][["message"]].to_csv(index=False)
        st.download_button(
            label="Download Anomalies as CSV",
            data=csv,
            file_name="anomalies.csv",
            mime="text/csv"
        )