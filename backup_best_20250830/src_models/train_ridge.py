from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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

MODEL_DIR = PROJECT_ROOT / "models" / "blood"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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
print("VITALAGE - RIDGE REGRESSION")
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

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# 5. RIDGE MODEL
# ============================================================
#
# StandardScaler is important for Ridge because Ridge
# penalizes coefficients.
#
# Without scaling, biomarkers with different numerical
# ranges would not be treated appropriately.
#
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "ridge",
        Ridge(alpha=10.0)
    )
])


# ============================================================
# 6. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RIDGE")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 8. PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

pearson_r, pearson_p = pearsonr(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("TEST SET PERFORMANCE")
print("=" * 70)

print(f"MAE       : {mae:.3f} years")
print(f"RMSE      : {rmse:.3f} years")
print(f"R²        : {r2:.4f}")
print(f"Pearson r : {pearson_r:.4f}")


# ============================================================
# 9. BIOAGE GAP
# ============================================================

gap = (
    y_pred
    - y_test.to_numpy()
)

print("\n" + "=" * 70)
print("BIOAGE GAP")
print("=" * 70)

print(
    f"Mean Gap   : {np.mean(gap):.3f} years"
)

print(
    f"Median Gap : {np.median(gap):.3f} years"
)

print(
    f"Gap SD     : {np.std(gap):.3f} years"
)

print(
    f"Gap MAE    : {np.mean(np.abs(gap)):.3f} years"
)


# ============================================================
# 10. AGE CALIBRATION
# ============================================================

slope, intercept = np.polyfit(
    y_test,
    y_pred,
    1
)

print("\n" + "=" * 70)
print("AGE CALIBRATION")
print("=" * 70)

print(
    f"Predicted Age = "
    f"{slope:.3f} × Actual Age "
    f"+ {intercept:.3f}"
)


# ============================================================
# 11. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({
    "Actual_Age": y_test.to_numpy(),
    "Predicted_BioAge": y_pred
})

results["BioAge_Gap"] = (
    results["Predicted_BioAge"]
    - results["Actual_Age"]
)

results["Absolute_Error"] = (
    np.abs(
        results["BioAge_Gap"]
    )
)


# ============================================================
# 12. AGE GROUP PERFORMANCE
# ============================================================

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

        Predicted_Mean=(
            "Predicted_BioAge",
            "mean"
        ),

        MAE=(
            "Absolute_Error",
            "mean"
        ),

        Mean_BioAge_Gap=(
            "BioAge_Gap",
            "mean"
        )
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("PERFORMANCE BY AGE GROUP")
print("=" * 70)

print(
    age_group_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 13. RIDGE COEFFICIENTS
# ============================================================

ridge_model = model.named_steps["ridge"]

coefficients = pd.Series(
    ridge_model.coef_,
    index=FEATURES
)

coefficients = coefficients.sort_values(
    key=np.abs,
    ascending=False
)

print("\n" + "=" * 70)
print("RIDGE COEFFICIENTS")
print("=" * 70)

print(
    coefficients.to_string(
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "ridge_bioage_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print(
    f"\nModel saved to:\n{model_path}"
)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_path = (
    REPORT_DIR
    / "ridge_test_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)

print(
    f"Predictions saved to:\n"
    f"{prediction_path}"
)


# ============================================================
# 16. SAVE AGE GROUP RESULTS
# ============================================================

age_group_path = (
    REPORT_DIR
    / "ridge_age_groups.csv"
)

age_group_results.to_csv(
    age_group_path,
    index=False
)


# ============================================================
# 17. SAVE COEFFICIENTS
# ============================================================

coefficient_path = (
    REPORT_DIR
    / "ridge_coefficients.csv"
)

coefficients.rename(
    "coefficient"
).to_csv(
    coefficient_path
)


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics_path = (
    REPORT_DIR
    / "ridge_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "VitalAge Ridge Regression\n"
    )

    f.write(
        "=========================\n\n"
    )

    f.write(
        f"MAE: {mae:.6f} years\n"
    )

    f.write(
        f"RMSE: {rmse:.6f} years\n"
    )

    f.write(
        f"R2: {r2:.6f}\n"
    )

    f.write(
        f"Pearson r: {pearson_r:.6f}\n"
    )

    f.write(
        f"Mean BioAge Gap: "
        f"{np.mean(gap):.6f}\n"
    )

    f.write(
        f"Median BioAge Gap: "
        f"{np.median(gap):.6f}\n"
    )

    f.write(
        f"Calibration slope: "
        f"{slope:.6f}\n"
    )

    f.write(
        f"Calibration intercept: "
        f"{intercept:.6f}\n"
    )


# ============================================================
# 19. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.25,
    s=15
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linewidth=2
)

plt.xlabel(
    "Chronological Age"
)

plt.ylabel(
    "Predicted Biological Age"
)

plt.title(
    "Ridge - Actual vs Predicted Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "ridge_actual_vs_predicted.png",
    dpi=200
)

plt.show()


# ============================================================
# 20. BIOAGE GAP PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    gap,
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
    "BioAge Gap"
)

plt.title(
    "Ridge - BioAge Gap vs Chronological Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "ridge_bioage_gap.png",
    dpi=200
)

plt.show()


print("\n" + "=" * 70)
print("RIDGE TRAINING COMPLETED")
print("=" * 70)

print(
    "\nCurrent XGBoost benchmark:"
)

print(
    "MAE = 8.496 years"
)

print(
    "R²  = 0.6614"
)

print(
    "\nCompare Ridge against this benchmark."
)