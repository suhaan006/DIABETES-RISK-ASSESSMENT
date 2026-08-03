"""
generate_data.py
Generates a synthetic diabetes risk dataset (Pima-Indians-style) for the
Diabetes Risk Assessment intern project.

Since no real patient data is used, this creates a realistic synthetic
dataset with the same structure as the well-known public
'Pima Indians Diabetes' dataset, then intentionally introduces a few
missing values and duplicate rows so the Data Cleaning step has real
work to do.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 768  # same size as the classic Pima Indians Diabetes dataset

# Simulate two underlying groups (lower-risk / higher-risk) so the
# generated data has a believable relationship between features and outcome
outcome = np.random.choice([0, 1], size=N, p=[0.65, 0.35])

def sample(mean_low, mean_high, sd, low_clip=0):
    means = np.where(outcome == 1, mean_high, mean_low)
    vals = np.random.normal(means, sd)
    return np.clip(vals, low_clip, None)

df = pd.DataFrame({
    "Pregnancies": np.random.poisson(lam=np.where(outcome == 1, 4.5, 2.8)),
    "Glucose": sample(110, 145, 20, 40),
    "BloodPressure": sample(68, 75, 12, 30),
    "SkinThickness": sample(20, 27, 10, 0),
    "Insulin": sample(70, 130, 60, 0),
    "BMI": sample(28, 34, 6, 15),
    "DiabetesPedigreeFunction": np.round(sample(0.35, 0.55, 0.2, 0.05), 3),
    "Age": sample(30, 40, 10, 21).round().astype(int),
    "Outcome": outcome
})

df["Glucose"] = df["Glucose"].round(1)
df["BloodPressure"] = df["BloodPressure"].round(1)
df["SkinThickness"] = df["SkinThickness"].round(1)
df["Insulin"] = df["Insulin"].round(1)
df["BMI"] = df["BMI"].round(1)

# --- Intentionally dirty the data a bit for the Data Cleaning step ---
# 1) Insert some missing values (as 0, common real-world artifact in this
#    dataset, plus a few true NaNs)
zero_idx = np.random.choice(df.index, size=25, replace=False)
df.loc[zero_idx, "SkinThickness"] = 0

nan_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[nan_idx, "BMI"] = np.nan

nan_idx2 = np.random.choice(df.index, size=10, replace=False)
df.loc[nan_idx2, "BloodPressure"] = np.nan

# 2) Insert duplicate rows
dupes = df.sample(12, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("/home/claude/diabetes_project/data/diabetes_raw.csv", index=False)
print(f"Generated raw dataset with {len(df)} rows -> data/diabetes_raw.csv")
