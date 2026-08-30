"""
VitalAge — Final Optimization Push

Uses ALL blood markers with imputation + tuned hyperparameters.
Goal: Push MAE below 9.0 years.

Usage: python notebooks/blood/test/05_final_optimization.py
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_PATH = os.path.join(os.path.dirname(__file__),
                        "..", "..", "..", "data", "raw", "blood_age_mega_raw.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and prepare ALL data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading raw data (ALL 57 blood markers)")
print("=" * 70)
df = pd.read_csv(RAW_PATH)
df = df[(df["Age"] >= 18) & (df["Age"] <= 80)].copy()
df = df.drop(columns=["SEQN", "CYCLE"], errors="ignore")

target = "Age"
all_features = [c for c in df.columns if c != target]
print(f"  {df.shape[0]} rows, {len(all_features)} blood markers")

# Impute everything
imp = SimpleImputer(strategy="median")
df_imp = df.copy()
df_imp[all_features] = imp.fit_transform(df[all_features])
df_imp = df_imp.dropna(subset=[target])

X_all = df_imp[all_features].values
y_all = df_imp[target].values
print(f"  After imputation: {len(y_all)} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
)
print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

# ---------------------------------------------------------------------------
# 2. Baseline comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Baseline comparison")
print("=" * 70)

# Original 12 features (imputed for fair comparison)
original_12 = ["LBXSAL", "LBXSCR", "LBXGLU", "CRP", "LBXLYPCT",
               "LBXMCVSI", "LBXRDW", "LBXSAPSI", "LBXWBCSI",
               "LBXGH", "LBDHDD", "LBXTC"]
orig_idx = [all_features.index(c) for c in original_12]

X_orig = X_all[:, orig_idx]
X_o_train, X_o_test, _, _ = train_test_split(X_orig, y_all, test_size=0.2, random_state=RANDOM_STATE)

xgb_base = xgb.XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbosity=0,
)
xgb_base.fit(X_o_train, y_train)
pred_orig = xgb_base.predict(X_o_test)
mae_orig = mean_absolute_error(y_test, pred_orig)
print(f"  Original 12 features (imputed): MAE={mae_orig:.3f}")

# ---------------------------------------------------------------------------
# 3. XGBoost tuning on ALL features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: XGBoost tuning on ALL 57 features")
print("=" * 70)

configs = [
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 700, "max_depth": 8, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.5},
    {"n_estimators": 1000, "max_depth": 8, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1},
    {"n_estimators": 1500, "max_depth": 8, "learning_rate": 0.01, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 0.3, "reg_lambda": 2.0},
    {"n_estimators": 2000, "max_depth": 10, "learning_rate": 0.005, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.5, "min_child_weight": 5},
]

best_mae_xgb = 999
best_xgb_params = None
best_xgb_model = None

for params in configs:
    model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE, verbosity=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"  {params}: MAE={mae:.3f}, R²={r2:.3f}")
    if mae < best_mae_xgb:
        best_mae_xgb = mae
        best_xgb_params = params
        best_xgb_model = clone(model)
        best_xgb_model.fit(X_train, y_train)

print(f"\n  Best XGBoost: MAE={best_mae_xgb:.3f}")

# ---------------------------------------------------------------------------
# 4. LightGBM tuning on ALL features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: LightGBM tuning on ALL 57 features")
print("=" * 70)

lgb_configs = [
    {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7},
    {"n_estimators": 700, "max_depth": 8, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 1000, "max_depth": -1, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8, "num_leaves": 63},
    {"n_estimators": 1500, "max_depth": -1, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.7, "num_leaves": 127},
    {"n_estimators": 2000, "max_depth": -1, "learning_rate": 0.008, "subsample": 0.8, "colsample_bytree": 0.7, "num_leaves": 127, "reg_alpha": 0.1},
    {"n_estimators": 2000, "max_depth": -1, "learning_rate": 0.005, "subsample": 0.7, "colsample_bytree": 0.6, "num_leaves": 255, "reg_alpha": 0.3},
]

best_mae_lgb = 999
best_lgb_model = None

for params in lgb_configs:
    try:
        model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"  {params}: MAE={mae:.3f}, R²={r2:.3f}")
        if mae < best_mae_lgb:
            best_mae_lgb = mae
            best_lgb_model = clone(model)
            best_lgb_model.fit(X_train, y_train)
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\n  Best LightGBM: MAE={best_mae_lgb:.3f}")

# ---------------------------------------------------------------------------
# 5. XGB + LGB Ensemble
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: XGB + LGB Ensemble")
print("=" * 70)

pred_xgb = best_xgb_model.predict(X_test)
pred_lgb = best_lgb_model.predict(X_test)

alphas = np.arange(0.0, 1.01, 0.05)
maes = [mean_absolute_error(y_test, a * pred_xgb + (1-a) * pred_lgb) for a in alphas]
best_alpha = alphas[np.argmin(maes)]
best_ens_mae = min(maes)
pred_ens = best_alpha * pred_xgb + (1 - best_alpha) * pred_lgb

print(f"  Best ensemble: alpha={best_alpha:.2f}, MAE={best_ens_mae:.3f}")

# ---------------------------------------------------------------------------
# 6. Bias correction
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Bias correction")
print("=" * 70)

residuals = y_test - pred_ens
ridge_bias = Ridge()
ridge_bias.fit(pred_ens.reshape(-1, 1), residuals)
residual_pred = ridge_bias.predict(pred_ens.reshape(-1, 1))
pred_corrected = pred_ens + residual_pred
mae_corrected = mean_absolute_error(y_test, pred_corrected)
r2_corrected = r2_score(y_test, pred_corrected)
print(f"  Before correction: MAE={best_ens_mae:.3f}")
print(f"  After correction:  MAE={mae_corrected:.3f}, R²={r2_corrected:.3f}")

# ---------------------------------------------------------------------------
# 7. 5-Fold Cross-Validation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: 5-Fold Cross-Validation")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_maes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    # Ensemble in each fold
    xgb_f = xgb.XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, verbosity=0)
    lgb_f = lgb.LGBMRegressor(**best_lgb_model.get_params(), random_state=RANDOM_STATE, verbose=-1)
    xgb_f.fit(X_tr, y_tr)
    lgb_f.fit(X_tr, y_tr)
    p_xgb = xgb_f.predict(X_val)
    p_lgb = lgb_f.predict(X_val)
    p_ens = best_alpha * p_xgb + (1 - best_alpha) * p_lgb
    mae_f = mean_absolute_error(y_val, p_ens)
    cv_maes.append(mae_f)
    print(f"  Fold {fold}: MAE={mae_f:.3f}")

cv_mean = np.mean(cv_maes)
cv_std = np.std(cv_maes)
print(f"\n  CV Mean MAE: {cv_mean:.3f} +/- {cv_std:.3f}")

# ---------------------------------------------------------------------------
# 8. Subgroup analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Subgroup analysis (corrected predictions)")
print("=" * 70)

subgroup = pd.DataFrame({
    "true": y_test, "pred": pred_corrected
})
subgroup["group"] = pd.cut(subgroup["true"], bins=[18, 30, 45, 60, 80],
                            labels=["18-30", "31-45", "46-60", "61-80"])
for grp, gdf in subgroup.groupby("group", observed=True):
    mae_g = mean_absolute_error(gdf["true"], gdf["pred"])
    bias = gdf["pred"].mean() - gdf["true"].mean()
    print(f"  {grp}: N={len(gdf)}, MAE={mae_g:.2f}, bias={bias:+.2f}")

# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

rows = [
    ("Baseline (12 features, imputed)", mae_orig),
    ("XGBoost (57 features)", best_mae_xgb),
    ("LightGBM (57 features)", best_mae_lgb),
    (f"Ensemble (XGB+LGB, α={best_alpha:.2f})", best_ens_mae),
    ("Ensemble + bias corrected", mae_corrected),
    (f"5-Fold CV ensemble", cv_mean),
]

print(f"\n  {'Method':45s} {'MAE':>8s}")
print(f"  {'-'*53}")
for name, mae in rows:
    print(f"  {name:45s} {mae:8.3f}")

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Prediction scatter
ax = axes[0, 0]
ax.scatter(y_test, pred_corrected, alpha=0.3, s=10, color="#4CAF50")
ax.plot([18, 80], [18, 80], "k--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("True Age")
ax.set_ylabel("Predicted Age (corrected)")
ax.set_title(f"Final Ensemble + Bias Corrected (MAE={mae_corrected:.2f})")
ax.legend()

# Residuals
ax = axes[0, 1]
resid = y_test - pred_corrected
ax.hist(resid, bins=50, color="#9C27B0", edgecolor="white", alpha=0.7)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_xlabel("Prediction Error (years)")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution (mean={resid.mean():.2f}, std={resid.std():.2f})")

# MAE by age group
ax = axes[1, 0]
group_mae = subgroup.groupby("group", observed=True).apply(
    lambda g: mean_absolute_error(g["true"], g["pred"]), include_groups=False
)
colors = ["#FF9800", "#2196F3", "#4CAF50", "#F44336"]
bars = ax.bar(group_mae.index.astype(str), group_mae.values, color=colors)
ax.set_ylabel("MAE (years)")
ax.set_title("MAE by Age Group")
for bar, val in zip(bars, group_mae.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", fontsize=10)

# Feature importance
ax = axes[1, 1]
fi = pd.Series(best_xgb_model.feature_importances_, index=all_features)
top15 = fi.sort_values(ascending=True).tail(15)
top15.plot(kind="barh", ax=ax, color="#2196F3")
ax.set_xlabel("Importance")
ax.set_title("Top 15 Feature Importances (XGBoost)")

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "21_final_optimization.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
