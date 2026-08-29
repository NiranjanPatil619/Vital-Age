"""
VitalAge - XGBoost Hyperparameter Tuning

Goal:
Improve the baseline XGBoost model.

Baseline:
MAE  = 9.11 years
RMSE = 11.84 years
R²   = 0.647

Important:
The test set is NOT used during hyperparameter tuning.
"""

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold
)

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
print("VITALAGE - XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

print("\nLoading:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(
    f"\nDataset shape: {df.shape}"
)



required_columns = FEATURES + [TARGET]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df.dropna(
    subset=required_columns
)

X = df[FEATURES]
y = df[TARGET]

print(
    f"Rows used: {len(df):,}"
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Training: {len(X_train):,}"
)

print(
    f"Testing : {len(X_test):,}"
)



base_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)



param_grid = {

    "n_estimators": [
        300,
        500,
        700,
        1000,
        1500
    ],

    "learning_rate": [
        0.01,
        0.02,
        0.03,
        0.05,
        0.08
    ],

    "max_depth": [
        3,
        4,
        5,
        6,
        7,
        8
    ],

    "min_child_weight": [
        1,
        3,
        5,
        7,
        10
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "gamma": [
        0,
        0.1,
        0.3,
        0.5,
        1
    ],

    "reg_alpha": [
        0,
        0.01,
        0.1,
        0.5,
        1
    ],

    "reg_lambda": [
        0.5,
        1,
        2,
        5,
        10
    ]
}




cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)




print("\n" + "=" * 70)
print("STARTING HYPERPARAMETER SEARCH")
print("=" * 70)

print(
    "\nTesting 50 parameter combinations..."
)

search = RandomizedSearchCV(

    estimator=base_model,

    param_distributions=param_grid,

    n_iter=50,

    scoring="neg_mean_absolute_error",

    cv=cv,

    verbose=2,

    random_state=42,

    n_jobs=-1,

    return_train_score=True
)


search.fit(
    X_train,
    y_train
)



print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(
    json.dumps(
        search.best_params_,
        indent=4
    )
)

print(
    f"\nBest CV MAE: "
    f"{-search.best_score_:.4f} years"
)


best_model = search.best_estimator_




print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

y_pred = best_model.predict(
    X_test
)



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


print(
    f"\nMAE       : {mae:.3f} years"
)

print(
    f"RMSE      : {rmse:.3f} years"
)

print(
    f"R²        : {r2:.4f}"
)

print(
    f"Pearson r : {pearson_r:.4f}"
)


bioage_gap = (
    y_pred
    - y_test.to_numpy()
)

print("\n" + "=" * 70)
print("BIOAGE GAP")
print("=" * 70)

print(
    f"Mean Gap   : "
    f"{np.mean(bioage_gap):.3f} years"
)

print(
    f"Median Gap : "
    f"{np.median(bioage_gap):.3f} years"
)

print(
    f"Gap SD     : "
    f"{np.std(bioage_gap):.3f} years"
)

print(
    f"Gap MAE    : "
    f"{np.mean(np.abs(bioage_gap)):.3f} years"
)



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


results = pd.DataFrame({

    "Actual_Age":
        y_test.to_numpy(),

    "Predicted_BioAge":
        y_pred
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

print(
    "\n" +
    age_group_results.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.3f}"
    )
)



feature_importance = pd.Series(
    best_model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)



model_path = (
    MODEL_DIR
    / "xgboost_bioage_tuned.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print(
    f"\nBest model saved to:\n"
    f"{model_path}"
)


params_path = (
    REPORT_DIR
    / "xgboost_best_parameters.json"
)

with open(
    params_path,
    "w"
) as f:

    json.dump(
        search.best_params_,
        f,
        indent=4
    )



prediction_path = (
    REPORT_DIR
    / "xgboost_tuned_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)


age_group_path = (
    REPORT_DIR
    / "xgboost_tuned_age_groups.csv"
)

age_group_results.to_csv(
    age_group_path,
    index=False
)


importance_path = (
    REPORT_DIR
    / "xgboost_tuned_feature_importance.csv"
)

feature_importance.rename(
    "importance"
).to_csv(
    importance_path
)




metrics = {

    "model":
        "XGBoost Tuned",

    "dataset_rows":
        len(df),

    "train_rows":
        len(X_train),

    "test_rows":
        len(X_test),

    "MAE":
        float(mae),

    "RMSE":
        float(rmse),

    "R2":
        float(r2),

    "Pearson_r":
        float(pearson_r),

    "CV_MAE":
        float(
            -search.best_score_
        ),

    "Mean_BioAge_Gap":
        float(
            np.mean(bioage_gap)
        ),

    "Median_BioAge_Gap":
        float(
            np.median(bioage_gap)
        ),

    "Gap_MAE":
        float(
            np.mean(
                np.abs(bioage_gap)
            )
        ),

    "Calibration_Slope":
        float(slope),

    "Calibration_Intercept":
        float(intercept),

    "Best_Parameters":
        search.best_params_
}

metrics_path = (
    REPORT_DIR
    / "xgboost_tuned_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )



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
    "Tuned XGBoost - Actual vs Predicted Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_tuned_actual_vs_predicted.png",
    dpi=200
)

plt.show()



residuals = (
    y_pred
    - y_test.to_numpy()
)

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    residuals,
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
    "Prediction Error"
)

plt.title(
    "Tuned XGBoost - Residuals vs Age"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_tuned_residuals.png",
    dpi=200
)

plt.show()



plt.figure(
    figsize=(9, 6)
)

feature_importance.sort_values().plot(
    kind="barh"
)

plt.xlabel(
    "XGBoost Importance"
)

plt.ylabel(
    "Biomarker"
)

plt.title(
    "Tuned XGBoost Feature Importance"
)

plt.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_tuned_feature_importance.png",
    dpi=200
)

plt.show()



print("\n" + "=" * 70)
print("BASELINE VS TUNED XGBOOST")
print("=" * 70)

print(
    "\nBaseline:"
)

print(
    "MAE = 9.11"
)

print(
    "R²  = 0.647"
)

print(
    "\nTuned:"
)

print(
    f"MAE = {mae:.3f}"
)

print(
    f"R²  = {r2:.4f}"
)

print(
    "\nImprovement in MAE:"
)

print(
    f"{9.110 - mae:.3f} years"
)

print(
    "\nImprovement in R²:"
)

print(
    f"{r2 - 0.647:.4f}"
)

print("\n" + "=" * 70)
print("TUNING COMPLETED")
print("=" * 70)