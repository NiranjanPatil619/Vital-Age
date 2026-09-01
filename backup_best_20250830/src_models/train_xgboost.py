"""
VitalAge - XGBoost Biological Age Model

Input:
    data/processed/bioage_final_clean.csv

Target:
    Age

Features:
    12 clinical biomarkers selected by the existing preprocessing pipeline.

Outputs:
    models/blood/xgboost_bioage_model.pkl
    reports/xgboost_metrics.txt
    reports/xgboost_test_predictions.csv
    reports/xgboost_age_group_performance.csv
    reports/figures/xgboost_actual_vs_predicted.png
    reports/figures/xgboost_residuals_vs_age.png
    reports/figures/xgboost_feature_importance.png
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bioage_final_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "blood"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)



FEATURES = [
    "LBXSAL",      # Albumin
    "LBXSCR",      # Creatinine
    "LBXGLU",      # Glucose
    "CRP",         # C-reactive protein
    "LBXLYPCT",    # Lymphocyte %
    "LBXMCVSI",    # MCV
    "LBXRDW",      # RDW
    "LBXSAPSI",    # Alkaline phosphatase
    "LBXWBCSI",    # WBC
    "LBXGH",       # HbA1c
    "LBDHDD",      # HDL
    "LBXTC",       # Total cholesterol
]

TARGET = "Age"



print("=" * 70)
print("VITALAGE - XGBOOST BIOLOGICAL AGE MODEL")
print("=" * 70)

print(f"\nLoading dataset:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure bioage_final_clean.csv is inside:\n"
        "data/processed/"
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

# Check required columns
required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing columns in dataset:\n{missing_columns}"
    )


print("\nMissing values:")
missing = df[required_columns].isnull().sum()
print(missing)

if missing.sum() > 0:
    print("\nWARNING: Missing values detected.")
    print("Dropping incomplete rows for this experiment.")
    df = df.dropna(subset=required_columns)

print(f"\nRows after cleaning: {len(df):,}")

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nAge statistics:")
print(y.describe())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


model = XGBRegressor(
    objective="reg:squarederror",

    # Number of boosting trees
    n_estimators=1000,

    # Learning rate
    learning_rate=0.03,

    # Maximum tree depth
    max_depth=5,

    # Minimum number of samples needed in a child
    min_child_weight=3,

    # Row sampling
    subsample=0.8,

    # Feature sampling
    colsample_bytree=0.8,

    # Regularization
    reg_alpha=0.0,
    reg_lambda=1.0,

    random_state=42,

    # Use all CPU cores
    n_jobs=-1,

    # Faster histogram algorithm
    tree_method="hist",
)


print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print("Training completed.")


y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
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
print(f"p-value   : {pearson_p:.3e}")



bioage_gap = y_pred - y_test.to_numpy()

mean_gap = np.mean(bioage_gap)
median_gap = np.median(bioage_gap)
gap_sd = np.std(bioage_gap)
gap_mae = np.mean(np.abs(bioage_gap))

print("\n" + "=" * 70)
print("BIOAGE GAP")
print("=" * 70)

print(
    "BioAge Gap = Predicted Biological Age - Chronological Age"
)

print(f"Mean Gap   : {mean_gap:.3f} years")
print(f"Median Gap : {median_gap:.3f} years")
print(f"Gap SD     : {gap_sd:.3f} years")
print(f"Gap MAE    : {gap_mae:.3f} years")



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
    f"{slope:.3f} * Actual Age + {intercept:.3f}"
)

print("\nIdeal calibration would approximately be:")
print("Predicted Age = 1.000 * Actual Age + 0.000")




results = pd.DataFrame({
    "Actual_Age": y_test.to_numpy(),
    "Predicted_BioAge": y_pred,
})

results["BioAge_Gap"] = (
    results["Predicted_BioAge"]
    - results["Actual_Age"]
)

results["Prediction_Error"] = (
    results["Predicted_BioAge"]
    - results["Actual_Age"]
)


age_bins = [
    0,
    18,
    30,
    45,
    60,
    75,
    np.inf,
]

age_labels = [
    "0-17",
    "18-29",
    "30-44",
    "45-59",
    "60-74",
    "75+",
]

results["Age_Group"] = pd.cut(
    results["Actual_Age"],
    bins=age_bins,
    labels=age_labels,
    right=False,
)

age_group_results = (
    results
    .groupby("Age_Group", observed=False)
    .agg(
        N=("Actual_Age", "size"),
        Actual_Mean=("Actual_Age", "mean"),
        Predicted_Mean=("Predicted_BioAge", "mean"),

        MAE=(
            "Prediction_Error",
            lambda x: np.mean(np.abs(x))
        ),

        Mean_BioAge_Gap=(
            "BioAge_Gap",
            "mean"
        ),
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



print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)

cv_mae = -cv_scores

print("Fold MAE:")
print(cv_mae)

print(
    f"\nCV MAE: "
    f"{cv_mae.mean():.3f} "
    f"+/- "
    f"{cv_mae.std():.3f} years"
)



feature_importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)



model_path = (
    MODEL_DIR
    / "xgboost_bioage_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print(
    f"\nModel saved to:\n{model_path}"
)



prediction_path = (
    REPORT_DIR
    / "xgboost_test_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)

print(
    f"Predictions saved to:\n{prediction_path}"
)



age_group_path = (
    REPORT_DIR
    / "xgboost_age_group_performance.csv"
)

age_group_results.to_csv(
    age_group_path,
    index=False
)


importance_path = (
    REPORT_DIR
    / "xgboost_feature_importance.csv"
)

feature_importance.rename(
    "importance"
).to_csv(
    importance_path
)


report_path = (
    REPORT_DIR
    / "xgboost_metrics.txt"
)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("VITALAGE - XGBOOST MODEL EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write("Dataset\n")
    f.write("-" * 70 + "\n")
    f.write(f"Rows: {len(df):,}\n")
    f.write(f"Features: {len(FEATURES)}\n")
    f.write(f"Train rows: {len(X_train):,}\n")
    f.write(f"Test rows: {len(X_test):,}\n\n")

    f.write("Test Metrics\n")
    f.write("-" * 70 + "\n")
    f.write(f"MAE: {mae:.4f} years\n")
    f.write(f"RMSE: {rmse:.4f} years\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"Pearson r: {pearson_r:.4f}\n")
    f.write(f"Pearson p-value: {pearson_p:.4e}\n\n")

    f.write("5-Fold CV\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"CV MAE: "
        f"{cv_mae.mean():.4f} "
        f"+/- "
        f"{cv_mae.std():.4f} years\n\n"
    )

    f.write("BioAge Gap\n")
    f.write("-" * 70 + "\n")
    f.write(f"Mean Gap: {mean_gap:.4f} years\n")
    f.write(f"Median Gap: {median_gap:.4f} years\n")
    f.write(f"Gap SD: {gap_sd:.4f} years\n")
    f.write(f"Gap MAE: {gap_mae:.4f} years\n\n")

    f.write("Calibration\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Predicted Age = "
        f"{slope:.4f} * Actual Age + "
        f"{intercept:.4f}\n\n"
    )

    f.write("Feature Importance\n")
    f.write("-" * 70 + "\n")
    f.write(
        feature_importance.to_string()
    )

print(
    f"Metrics report saved to:\n{report_path}"
)



plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.25,
    s=15,
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
    linewidth=2,
)

plt.xlabel("Chronological Age")
plt.ylabel("Predicted Biological Age")

plt.title(
    "XGBoost - Actual vs Predicted Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

actual_predicted_path = (
    FIGURE_DIR
    / "xgboost_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=200,
)

plt.show()


residuals = y_pred - y_test.to_numpy()

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    residuals,
    alpha=0.25,
    s=15,
)

plt.axhline(
    0,
    linewidth=2,
)

plt.xlabel("Chronological Age")
plt.ylabel("Prediction Error")

plt.title(
    "XGBoost - Residuals vs Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

residual_path = (
    FIGURE_DIR
    / "xgboost_residuals_vs_age.png"
)

plt.savefig(
    residual_path,
    dpi=200,
)

plt.show()



plt.figure(figsize=(9, 6))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("XGBoost Importance")
plt.ylabel("Biomarker")

plt.title(
    "XGBoost Feature Importance"
)

plt.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()

importance_plot_path = (
    FIGURE_DIR
    / "xgboost_feature_importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=200,
)

plt.show()


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"MAE       : {mae:.3f} years")
print(f"RMSE      : {rmse:.3f} years")
print(f"R²        : {r2:.4f}")
print(f"Pearson r : {pearson_r:.4f}")
print(f"CV MAE    : {cv_mae.mean():.3f} +/- {cv_mae.std():.3f}")
print(f"Mean Gap  : {mean_gap:.3f} years")

print("\nTop 5 features:")

for feature, importance in feature_importance.head(5).items():
    print(
        f"{feature:<12} "
        f"{importance:.4f}"
    )

print("\nAll outputs generated successfully.")

print("\nModel:")
print(model_path)

print("\nReports:")
print(REPORT_DIR)

print("\nFigures:")
print(FIGURE_DIR)