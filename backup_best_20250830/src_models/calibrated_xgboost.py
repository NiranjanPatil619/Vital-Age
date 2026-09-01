"""
VitalAge - XGBoost Age-Bias Calibration

Pipeline:

70% -> XGBoost training
15% -> Calibration
15% -> Final untouched test

The calibration model is learned ONLY from the calibration set.

Final test set is used only once for evaluation.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import pearsonr


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bioage_final_clean.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "blood"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

FIGURE_DIR = (
    REPORT_DIR
    / "figures"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FEATURES
# ============================================================

FEATURES = [
    "LBXSAL",
    "LBXSCR",
    "LBXGLU",
    "CRP",
    "LBXLYPCT",
    "LBXMCVSI",
    "LBXRDW",
    "LBXSAPSI",
    "LBXWBCSI",
    "LBXGH",
    "LBDHDD",
    "LBXTC",
]

TARGET = "Age"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("VITALAGE - XGBOOST AGE-BIAS CALIBRATION")
print("=" * 70)

print("\nLoading dataset:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=FEATURES + [TARGET]
)

X = df[FEATURES]
y = df[TARGET]

print(
    f"\nTotal samples: {len(df):,}"
)


# ============================================================
# 4. CREATE 70 / 15 / 15 SPLIT
# ============================================================

# First:
# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Then split the temporary 30%:
# 15% calibration
# 15% final test

X_calibration, X_test, y_calibration, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Training     : {len(X_train):,}"
)

print(
    f"Calibration  : {len(X_calibration):,}"
)

print(
    f"Final Test   : {len(X_test):,}"
)


# ============================================================
# 5. LOAD YOUR TUNED XGBOOST MODEL
# ============================================================

tuned_model_path = (
    MODEL_DIR
    / "xgboost_bioage_tuned.pkl"
)

if not tuned_model_path.exists():

    raise FileNotFoundError(
        f"\nTuned XGBoost model not found:\n"
        f"{tuned_model_path}\n\n"
        "Run tune_xgboost.py first."
    )

print("\nLoading tuned XGBoost model...")

model = joblib.load(
    tuned_model_path
)

print("Tuned model loaded.")


# ============================================================
# IMPORTANT
# ============================================================
#
# Your tuned model was originally trained on the 70% training
# data from the same random_state=42 split.
#
# We therefore use that model for the calibration experiment.
#
# The calibration and final test sets are untouched by training.
#
# ============================================================


# ============================================================
# 6. PREDICTIONS ON CALIBRATION SET
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION SET")
print("=" * 70)

calibration_pred = model.predict(
    X_calibration
)

calibration_actual = (
    y_calibration.to_numpy()
)


# ============================================================
# 7. FIT CALIBRATION MODEL
# ============================================================

print("\nLearning age-bias correction...")

calibration_model = LinearRegression()

calibration_model.fit(
    calibration_pred.reshape(-1, 1),
    calibration_actual
)

calibration_slope = (
    calibration_model.coef_[0]
)

calibration_intercept = (
    calibration_model.intercept_
)

print(
    "\nCalibration equation:"
)

print(
    f"Calibrated BioAge = "
    f"{calibration_slope:.4f} × "
    f"Raw XGBoost Age + "
    f"{calibration_intercept:.4f}"
)


# ============================================================
# 8. RAW CALIBRATION PERFORMANCE
# ============================================================

raw_calibration_mae = mean_absolute_error(
    calibration_actual,
    calibration_pred
)

raw_calibration_rmse = np.sqrt(
    mean_squared_error(
        calibration_actual,
        calibration_pred
    )
)

raw_calibration_r2 = r2_score(
    calibration_actual,
    calibration_pred
)


# ============================================================
# 9. FINAL TEST - RAW XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SET")
print("=" * 70)

raw_test_pred = model.predict(
    X_test
)

test_actual = (
    y_test.to_numpy()
)


# ============================================================
# 10. APPLY CALIBRATION
# ============================================================

calibrated_test_pred = (
    calibration_model.predict(
        raw_test_pred.reshape(-1, 1)
    )
)


# ============================================================
# 11. RAW TEST METRICS
# ============================================================

raw_mae = mean_absolute_error(
    test_actual,
    raw_test_pred
)

raw_rmse = np.sqrt(
    mean_squared_error(
        test_actual,
        raw_test_pred
    )
)

raw_r2 = r2_score(
    test_actual,
    raw_test_pred
)

raw_r, raw_p = pearsonr(
    test_actual,
    raw_test_pred
)


# ============================================================
# 12. CALIBRATED TEST METRICS
# ============================================================

calibrated_mae = mean_absolute_error(
    test_actual,
    calibrated_test_pred
)

calibrated_rmse = np.sqrt(
    mean_squared_error(
        test_actual,
        calibrated_test_pred
    )
)

calibrated_r2 = r2_score(
    test_actual,
    calibrated_test_pred
)

calibrated_r, calibrated_p = pearsonr(
    test_actual,
    calibrated_test_pred
)


# ============================================================
# 13. CALIBRATION OF FINAL TEST PREDICTIONS
# ============================================================

raw_slope, raw_intercept = np.polyfit(
    test_actual,
    raw_test_pred,
    1
)

calibrated_slope, calibrated_intercept = np.polyfit(
    test_actual,
    calibrated_test_pred,
    1
)


# ============================================================
# 14. BIOAGE GAP
# ============================================================

raw_gap = (
    raw_test_pred
    - test_actual
)

calibrated_gap = (
    calibrated_test_pred
    - test_actual
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RAW XGBOOST vs CALIBRATED XGBOOST")
print("=" * 70)

print(
    "\nRAW XGBOOST"
)

print(
    f"MAE       : {raw_mae:.3f} years"
)

print(
    f"RMSE      : {raw_rmse:.3f} years"
)

print(
    f"R²        : {raw_r2:.4f}"
)

print(
    f"Pearson r : {raw_r:.4f}"
)

print(
    f"Calibration slope: {raw_slope:.4f}"
)


print(
    "\nCALIBRATED XGBOOST"
)

print(
    f"MAE       : {calibrated_mae:.3f} years"
)

print(
    f"RMSE      : {calibrated_rmse:.3f} years"
)

print(
    f"R²        : {calibrated_r2:.4f}"
)

print(
    f"Pearson r : {calibrated_r:.4f}"
)

print(
    f"Calibration slope: "
    f"{calibrated_slope:.4f}"
)


# ============================================================
# 16. BIOAGE GAP RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BIOAGE GAP")
print("=" * 70)

print("\nRAW XGBOOST GAP")

print(
    f"Mean Gap   : "
    f"{np.mean(raw_gap):.3f}"
)

print(
    f"Median Gap : "
    f"{np.median(raw_gap):.3f}"
)

print(
    f"Gap MAE    : "
    f"{np.mean(np.abs(raw_gap)):.3f}"
)


print("\nCALIBRATED GAP")

print(
    f"Mean Gap   : "
    f"{np.mean(calibrated_gap):.3f}"
)

print(
    f"Median Gap : "
    f"{np.median(calibrated_gap):.3f}"
)

print(
    f"Gap MAE    : "
    f"{np.mean(np.abs(calibrated_gap)):.3f}"
)


# ============================================================
# 17. AGE GROUP ANALYSIS
# ============================================================

results = pd.DataFrame({

    "Actual_Age":
        test_actual,

    "Raw_BioAge":
        raw_test_pred,

    "Calibrated_BioAge":
        calibrated_test_pred
})

results["Raw_Gap"] = (
    results["Raw_BioAge"]
    - results["Actual_Age"]
)

results["Calibrated_Gap"] = (
    results["Calibrated_BioAge"]
    - results["Actual_Age"]
)

results["Raw_Absolute_Error"] = (
    np.abs(
        results["Raw_Gap"]
    )
)

results["Calibrated_Absolute_Error"] = (
    np.abs(
        results["Calibrated_Gap"]
    )
)


age_bins = [
    0,
    18,
    30,
    45,
    60,
    75,
    np.inf
]

age_labels = [
    "0-17",
    "18-29",
    "30-44",
    "45-59",
    "60-74",
    "75+"
]

results["Age_Group"] = pd.cut(
    results["Actual_Age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)


age_group_results = (
    results
    .groupby(
        "Age_Group",
        observed=False
    )
    .agg(

        N=(
            "Actual_Age",
            "size"
        ),

        Actual_Mean=(
            "Actual_Age",
            "mean"
        ),

        Raw_BioAge_Mean=(
            "Raw_BioAge",
            "mean"
        ),

        Calibrated_BioAge_Mean=(
            "Calibrated_BioAge",
            "mean"
        ),

        Raw_MAE=(
            "Raw_Absolute_Error",
            "mean"
        ),

        Calibrated_MAE=(
            "Calibrated_Absolute_Error",
            "mean"
        ),

        Raw_Mean_Gap=(
            "Raw_Gap",
            "mean"
        ),

        Calibrated_Mean_Gap=(
            "Calibrated_Gap",
            "mean"
        )
    )
    .reset_index()
)


print(
    "\n" +
    age_group_results.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.3f}"
    )
)


# ============================================================
# 18. SAVE CALIBRATION MODEL
# ============================================================

calibration_path = (
    MODEL_DIR
    / "xgboost_age_calibration.pkl"
)

joblib.dump(
    calibration_model,
    calibration_path
)

print(
    f"\nCalibration model saved:\n"
    f"{calibration_path}"
)


# ============================================================
# 19. SAVE PREDICTIONS
# ============================================================

prediction_path = (
    REPORT_DIR
    / "xgboost_calibrated_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)

print(
    f"Predictions saved:\n"
    f"{prediction_path}"
)


# ============================================================
# 20. SAVE AGE GROUP RESULTS
# ============================================================

age_group_path = (
    REPORT_DIR
    / "xgboost_calibrated_age_groups.csv"
)

age_group_results.to_csv(
    age_group_path,
    index=False
)


# ============================================================
# 21. PLOT RAW VS CALIBRATED
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    test_actual,
    raw_test_pred,
    alpha=0.20,
    s=15,
    label="Raw XGBoost"
)

plt.scatter(
    test_actual,
    calibrated_test_pred,
    alpha=0.20,
    s=15,
    label="Calibrated XGBoost"
)

min_value = min(
    test_actual.min(),
    raw_test_pred.min(),
    calibrated_test_pred.min()
)

max_value = max(
    test_actual.max(),
    raw_test_pred.max(),
    calibrated_test_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linewidth=2,
    label="Ideal"
)

plt.xlabel(
    "Chronological Age"
)

plt.ylabel(
    "Predicted Biological Age"
)

plt.title(
    "XGBoost Before vs After Age Calibration"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_calibration_comparison.png",
    dpi=200
)

plt.show()




plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    test_actual,
    calibrated_gap,
    alpha=0.25,
    s=15
)

plt.axhline(
    0,
    linewidth=2
)

plt.xlabel(
    "Chronological Age"
)

plt.ylabel(
    "Calibrated BioAge Gap"
)

plt.title(
    "Calibrated BioAge Gap vs Chronological Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_calibrated_gap.png",
    dpi=200
)

plt.show()



print("\n" + "=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)

print(
    f"\nRaw MAE:"
    f"        {raw_mae:.3f}"
)

print(
    f"Calibrated MAE:"
    f" {calibrated_mae:.3f}"
)

print(
    f"\nRaw RMSE:"
    f"        {raw_rmse:.3f}"
)

print(
    f"Calibrated RMSE:"
    f" {calibrated_rmse:.3f}"
)

print(
    f"\nRaw R²:"
    f"          {raw_r2:.4f}"
)

print(
    f"Calibrated R²:"
    f"   {calibrated_r2:.4f}"
)

print(
    f"\nRaw slope:"
    f"       {raw_slope:.4f}"
)

print(
    f"Calibrated slope:"
    f" {calibrated_slope:.4f}"
)

print(
    "\nCalibration completed successfully."
)