# Diabetes Risk Assessment

**Intern ID:** CITS6971
**Intern Name:** Suhaan Gupta
**Duration:** 4 Weeks

## Overview
This project uses a synthetic, Pima-Indians-style dataset (public-dataset
format, no real patient data) to walk through a complete beginner data
science workflow: cleaning, visualization, analysis, prediction, and a
final dashboard summary.

## What was performed
- ✅ **Data Cleaning** — filled missing values (median imputation), removed
  duplicate rows, treated invalid 0-values in `SkinThickness` as missing
- ✅ **Visualization** — glucose distribution, feature correlation heatmap,
  BMI vs age, risk group comparisons
- ✅ **Analysis** — risk-group summary statistics, high/low-risk breakdown,
  top correlated risk factors
- ✅ **Prediction Model** — Logistic Regression risk classifier
  (beginner-friendly, no complex ML)
- ✅ **Dashboard Creation** — one combined 4-panel summary image

## Folder Structure
```
diabetes_project/
├── data/
│   ├── diabetes_raw.csv        # synthetic raw dataset (with dirt for cleaning)
│   └── diabetes_clean.csv      # cleaned dataset
├── src/
│   ├── generate_data.py        # generates the synthetic dataset
│   └── main.py                 # full pipeline: clean -> visualize -> analyze -> predict -> dashboard
├── output/
│   ├── glucose_distribution.png
│   ├── correlation_heatmap.png
│   ├── prediction_chart.png
│   ├── dashboard.png
│   └── terminal_output.txt
├── screenshots/
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/generate_data.py   # generates data/diabetes_raw.csv
python src/main.py            # runs full pipeline, saves outputs to /output
```

## Results Summary
- Dataset: 768 patient records (after cleaning)
- Risk split: ~66% Low Risk, ~34% High Risk
- Top risk factors (by correlation with outcome): Glucose, Insulin,
  Pregnancies, Diabetes Pedigree Function, BMI
- Model: Logistic Regression — **~91.6% test accuracy**

## Notes
- Dataset is synthetically generated to resemble the well-known public
  Pima Indians Diabetes dataset structure (no real patient data used).
- Model is intentionally kept simple (Logistic Regression) as this is a
  beginner-level project — no complex ML/deep learning involved.
