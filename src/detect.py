import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ── 1. Load & Parse Logs ──────────────────────────────────────────
with open("data/SSH.log", "r") as f:
    lines = f.readlines()

records = []
for line in lines:
    # Extract: Month Day Time Host Process Message
    match = re.match(
        r'(\w+ \d+ \d+:\d+:\d+) \S+ \S+: (.+)', line
    )
    if match:
        timestamp_str, message = match.groups()
        failed = 1 if "Failed" in message or "Invalid" in message else 0
        accepted = 1 if "Accepted" in message else 0
        records.append({
            "timestamp": timestamp_str,
            "failed": failed,
            "accepted": accepted,
            "raw": message
        })

df = pd.DataFrame(records)
print(f"✅ Loaded {len(df)} log entries")

# ── 2. Feature Engineering ────────────────────────────────────────
features = df[["failed", "accepted"]]

# ── 3. Train Isolation Forest ─────────────────────────────────────
model = IsolationForest(contamination=0.15, random_state=42)
df["anomaly_score"] = model.fit_predict(features)
df["is_anomaly"] = df["anomaly_score"] == -1

anomalies = df[df["is_anomaly"] == True]
print(f"🚨 Anomalies detected: {len(anomalies)}")
print(anomalies["raw"].head(10))

# ── 4. Visualize ──────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["failed"], label="Failed Logins", alpha=0.5)
plt.scatter(
    anomalies.index,
    anomalies["failed"],
    color="red", label="Anomaly", zorder=5
)
plt.title("Log Anomaly Detection")
plt.xlabel("Log Entry #")
plt.ylabel("Failed Login Flag")
plt.legend()
plt.tight_layout()
plt.savefig("anomaly_chart.png")
plt.show()
print("📊 Chart saved as anomaly_chart.png")