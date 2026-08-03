"""
Diabetes Risk Assessment - Beginner Data Science Project
-----------------------------------------------------------
Intern ID   : CITS6971
Intern Name : Suhaan Gupta
Duration    : 4 Weeks
Project     : Diabetes Risk Assessment

Pipeline:
  1. Data Cleaning      -> fills missing values, removes duplicates
  2. Visualization       -> feature distributions, correlation heatmap,
                             glucose vs outcome, BMI vs age
  3. Analysis             -> summary stats, risk-group comparisons,
                             high-risk factor identification
  4. Prediction Model    -> simple Logistic Regression risk classifier
                             (beginner-friendly, no complex ML)
  5. Dashboard Creation  -> one combined 4-panel summary image
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_IN = "data/diabetes_raw.csv"
DATA_OUT = "data/diabetes_clean.csv"
OUTPUT_DIR = "output"

print("=" * 60)
print("DIABETES RISK ASSESSMENT - BEGINNER PROJECT")
print("Intern ID: CITS6971 | Suhaan Gupta | 4 Weeks")
print("=" * 60)

# -------------------------------------------------------------
# 1. DATA CLEANING
# -------------------------------------------------------------
print("\n[1/5] DATA CLEANING")
df = pd.read_csv(DATA_IN)
print(f"Raw shape: {df.shape}")

before_dupes = len(df)
df = df.drop_duplicates()
print(f"Removed {before_dupes - len(df)} duplicate rows")

missing_before = df.isnull().sum().sum()
# Some fields use 0 as a stand-in for missing (common real-world issue
# in this style of medical dataset) - treat 0 in SkinThickness as missing
zero_as_missing_cols = ["SkinThickness"]
for col in zero_as_missing_cols:
    df[col] = df[col].replace(0, np.nan)

# Fill missing numeric values with the column median (robust to outliers,
# beginner-friendly approach)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print(f"Filled {missing_before} missing/placeholder values using column medians")
print(f"Clean shape: {df.shape}")

df.to_csv(DATA_OUT, index=False)
print(f"Saved cleaned dataset -> {DATA_OUT}")

# -------------------------------------------------------------
# 2. VISUALIZATION
# -------------------------------------------------------------
print("\n[2/5] VISUALIZATION")

plt.figure(figsize=(8, 5))
df["Glucose"].hist(bins=25, color="#4C72B0", edgecolor="white")
plt.title("Distribution of Glucose Levels")
plt.xlabel("Glucose")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/glucose_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(8, 5))
corr = df.corr(numeric_only=True)
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(im, label="Correlation")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

print("Saved: glucose_distribution.png, correlation_heatmap.png")

# -------------------------------------------------------------
# 3. ANALYSIS
# -------------------------------------------------------------
print("\n[3/5] ANALYSIS")

risk_counts = df["Outcome"].value_counts()
risk_pct = df["Outcome"].value_counts(normalize=True) * 100
print(f"Low-risk (0): {risk_counts.get(0,0)} ({risk_pct.get(0,0):.1f}%)")
print(f"High-risk (1): {risk_counts.get(1,0)} ({risk_pct.get(1,0):.1f}%)")

group_means = df.groupby("Outcome")[
    ["Glucose", "BMI", "Age", "BloodPressure", "Insulin"]
].mean().round(1)
print("\nAverage values by risk group:")
print(group_means)

# Simple correlation-based "top risk factors"
outcome_corr = df.corr(numeric_only=True)["Outcome"].drop("Outcome").sort_values(ascending=False)
print("\nTop factors correlated with diabetes risk:")
print(outcome_corr.round(3))

analysis_summary = {
    "low_risk_count": int(risk_counts.get(0, 0)),
    "high_risk_count": int(risk_counts.get(1, 0)),
    "top_risk_factor": outcome_corr.index[0],
    "top_risk_corr": round(float(outcome_corr.iloc[0]), 3),
}

# -------------------------------------------------------------
# 4. PREDICTION MODEL (beginner-friendly Logistic Regression)
# -------------------------------------------------------------
print("\n[4/5] PREDICTION MODEL")

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[features]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Model: Logistic Regression (beginner-friendly, no complex ML)")
print(f"Test Accuracy: {acc*100:.1f}%")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# Prediction chart: predicted risk probability vs actual outcome (test set)
proba = model.predict_proba(X_test_s)[:, 1]
plt.figure(figsize=(8, 5))
colors = np.where(y_test == 1, "#C44E52", "#55A868")
order = np.argsort(proba)
plt.scatter(range(len(proba)), np.array(proba)[order], c=np.array(colors)[order], s=25)
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
plt.title("Predicted Diabetes Risk Probability (Test Set)")
plt.xlabel("Patient (sorted by predicted risk)")
plt.ylabel("Predicted Probability of High Risk")
plt.legend(["Decision threshold", "Actual: High Risk", "Actual: Low Risk"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/prediction_chart.png", dpi=120)
plt.close()

print("Saved: prediction_chart.png")

# -------------------------------------------------------------
# 5. DASHBOARD CREATION (single combined 4-panel image)
# -------------------------------------------------------------
print("\n[5/5] DASHBOARD CREATION")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Diabetes Risk Assessment Dashboard", fontsize=16, fontweight="bold")

# Panel 1: Glucose distribution by outcome
ax = axes[0, 0]
df[df.Outcome == 0]["Glucose"].hist(ax=ax, bins=20, alpha=0.6, label="Low Risk", color="#55A868")
df[df.Outcome == 1]["Glucose"].hist(ax=ax, bins=20, alpha=0.6, label="High Risk", color="#C44E52")
ax.set_title("Glucose Distribution by Risk Group")
ax.set_xlabel("Glucose")
ax.set_ylabel("Frequency")
ax.legend()

# Panel 2: BMI vs Age scatter colored by outcome
ax = axes[0, 1]
ax.scatter(df["Age"], df["BMI"], c=np.where(df.Outcome == 1, "#C44E52", "#55A868"), alpha=0.5, s=15)
ax.set_title("BMI vs Age (colored by risk)")
ax.set_xlabel("Age")
ax.set_ylabel("BMI")

# Panel 3: Risk group counts
ax = axes[1, 0]
ax.bar(["Low Risk", "High Risk"], [risk_counts.get(0, 0), risk_counts.get(1, 0)],
       color=["#55A868", "#C44E52"])
ax.set_title("Risk Group Counts")
ax.set_ylabel("Number of Patients")
for i, v in enumerate([risk_counts.get(0, 0), risk_counts.get(1, 0)]):
    ax.text(i, v + 3, str(v), ha="center", fontweight="bold")

# Panel 4: Top correlated risk factors (horizontal bar)
ax = axes[1, 1]
top5 = outcome_corr.head(5)
ax.barh(top5.index[::-1], top5.values[::-1], color="#4C72B0")
ax.set_title("Top 5 Risk Factors (Correlation with Outcome)")
ax.set_xlabel("Correlation")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/dashboard.png", dpi=120)
plt.close()

print("Saved: dashboard.png")

# -------------------------------------------------------------
# Save terminal-style output log
# -------------------------------------------------------------
with open(f"{OUTPUT_DIR}/terminal_output.txt", "w") as f:
    f.write("DIABETES RISK ASSESSMENT - RUN LOG\n")
    f.write(f"Rows after cleaning: {len(df)}\n")
    f.write(f"Low-risk: {risk_counts.get(0,0)} | High-risk: {risk_counts.get(1,0)}\n")
    f.write(f"Model: Logistic Regression | Test Accuracy: {acc*100:.1f}%\n")
    f.write(f"Top risk factor: {analysis_summary['top_risk_factor']} "
            f"(corr={analysis_summary['top_risk_corr']})\n")

print("\n" + "=" * 60)
print("PROJECT COMPLETE - all outputs saved in /output")
print("=" * 60)
