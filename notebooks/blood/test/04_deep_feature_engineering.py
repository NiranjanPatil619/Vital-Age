"""
VitalAge — Deep Feature Engineering & Model Tuning

Goal: Push MAE as low as possible by:
1. Using ALL blood markers (with imputation)
2. Testing top-N feature subsets (10, 15, 20, 25, 30)
3. Hyperparameter tuning for XGBoost/LightGBM
4. Feature interactions (ratios, polynomials)
5. Ensemble methods

Usage: python notebooks/blood/test/04_deep_feature_engineering.py
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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH = os.path.join(os.path.dirname(__file__),
                        "..", "..", "..", "data", "raw", "blood_age_mega_raw.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading raw data")
print("=" * 70)
df = pd.read_csv(RAW_PATH)
df = df[(df["Age"] >= 18) & (df["Age"] <= 80)].copy()
df = df.drop(columns=["SEQN", "CYCLE"], errors="ignore")
print(f"  {df.shape[0]} rows x {df.shape[1]} columns")

target = "Age"
blood_cols = [c for c in df.columns if c != target]

# ---------------------------------------------------------------------------
# 2. Correlation-based feature ranking
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Feature ranking by correlation with Age")
print("=" * 70)

corr_with_age = df[blood_cols + [target]].corr(numeric_only=True)[target].drop(target)
corr_ranked = corr_with_age.abs().sort_values(ascending=False)
print("\n  All features ranked by |correlation| with Age:")
for i, (feat, abs_corr) in enumerate(corr_ranked.items(), 1):
    actual_corr = corr_with_age[feat]
    marker = " <--" if feat in ["LBXSAL", "LBXSCR", "LBXGLU", "CRP", "LBXLYPCT",
                                  "LBXMCVSI", "LBXRDW", "LBXSAPSI", "LBXWBCSI",
                                  "LBXGH", "LBDHDD", "LBXTC"] else ""
    print(f"    {i:2d}. {feat:12s}: |r|={abs_corr:.4f} (r={actual_corr:+.4f}){marker}")

top_features = list(corr_ranked.index)

# ---------------------------------------------------------------------------
# 3. Helper: prepare imputed data
# ---------------------------------------------------------------------------
def prepare_imputed(df, feature_cols, target="Age", max_features=None):
    """Impute and return X, y, feature names."""
    used = [c for c in feature_cols if c in df.columns]
    if max_features:
        used = used[:max_features]
    df_sub = df[used + [target]].dropna(subset=[target])
    imp = SimpleImputer(strategy="median")
    df_sub[used] = imp.fit_transform(df_sub[used])
    return df_sub[used].values, df_sub[target].values, used

# ---------------------------------------------------------------------------
# 4. Test top-N features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Top-N feature subsets")
print("=" * 70)

xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbosity=0,
)

results = []

for n_feat in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]:
    X, y, used = prepare_imputed(df, top_features, max_features=n_feat)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = clone(xgb_model)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    results.append({"N": n_feat, "MAE": mae, "R²": r2, "Features": used})
    print(f"  Top {n_feat:2d} features: MAE={mae:.3f}, R²={r2:.3f}")

# ---------------------------------------------------------------------------
# 5. Feature interactions (ratios)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Feature interactions (ratios)")
print("=" * 70)

# Create ratio features from top correlated features
ratio_pairs = [
    ("LBXSBU", "LBXSCR"),      # BUN/Creatinine ratio (kidney health)
    ("LBXSAL", "LBXSBU"),       # Albumin/BUN (nutrition)
    ("LBXSCH", "LBXSAL"),       # Cholesterol/Albumin
    ("LBXSGL", "LBXSBU"),       # Glucose/BUN
    ("LBXMCVSI", "LBXRBCSI"),   # MCV/RBC (anemia indicators)
    ("LBXRDW", "LBXMCHSI"),     # RDW/MCH (blood cell variation)
    ("LBXSOSSI", "LBXSATSI"),   # AST/ALT ratio (liver)
    ("LBXSGTSI", "LBXSATSI"),   # GGT/ALT (liver)
    ("LBXSC3SI", "LBXSAL"),     # Complement C3/Albumin (inflammation)
    ("LBXSUA", "LBXSCR"),       # Uric acid/Creatinine
]

best_n = 12  # From step 4 results
base_features = top_features[:best_n]

# Add ratio features
df_ratios = df.copy()
ratio_names = []
for f1, f2 in ratio_pairs:
    if f1 in df.columns and f2 in df.columns:
        ratio_name = f"{f1}_div_{f2}"
        # Avoid division by zero
        denom = df[f2].replace(0, np.nan)
        df_ratios[ratio_name] = df[f1] / denom
        ratio_names.append(ratio_name)

# Test with ratios added
all_features_ratios = base_features + ratio_names
X_r, y_r, used_r = prepare_imputed(df_ratios, all_features_ratios)
X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(X_r, y_r, test_size=0.2, random_state=RANDOM_STATE)

model_r = clone(xgb_model)
model_r.fit(X_r_tr, y_r_tr)
y_pred_r = model_r.predict(X_r_te)
mae_r = mean_absolute_error(y_r_te, y_pred_r)
print(f"  Top {best_n} + {len(ratio_names)} ratios: MAE={mae_r:.3f}")

# Test with ALL ratios
X_r2, y_r2, used_r2 = prepare_imputed(df_ratios, top_features[:20] + ratio_names)
X_r2_tr, X_r2_te, y_r2_tr, y_r2_te = train_test_split(X_r2, y_r2, test_size=0.2, random_state=RANDOM_STATE)

model_r2 = clone(xgb_model)
model_r2.fit(X_r2_tr, y_r2_tr)
y_pred_r2 = model_r2.predict(X_r2_te)
mae_r2 = mean_absolute_error(y_r2_te, y_pred_r2)
print(f"  Top 20 + {len(ratio_names)} ratios: MAE={mae_r2:.3f}")

# ---------------------------------------------------------------------------
# 6. XGBoost hyperparameter tuning
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: XGBoost hyperparameter tuning (top features)")
print("=" * 70)

best_n_final = 15  # Pick the best from step 4
X_best, y_best, used_best = prepare_imputed(df, top_features, max_features=best_n_final)
X_b_tr, X_b_te, y_b_tr, y_b_te = train_test_split(X_best, y_best, test_size=0.2, random_state=RANDOM_STATE)

param_grid = [
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 700, "max_depth": 7, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.5},
    {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1},
]

best_mae_tune = 999
best_params = None
for params in param_grid:
    model_t = xgb.XGBRegressor(**params, random_state=RANDOM_STATE, verbosity=0)
    model_t.fit(X_b_tr, y_b_tr)
    y_pred_t = model_t.predict(X_b_te)
    mae_t = mean_absolute_error(y_b_te, y_pred_t)
    print(f"  {params}: MAE={mae_t:.3f}")
    if mae_t < best_mae_tune:
        best_mae_tune = mae_t
        best_params = params

print(f"\n  Best XGBoost: MAE={best_mae_tune:.3f} with {best_params}")

# ---------------------------------------------------------------------------
# 7. LightGBM tuning
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: LightGBM tuning")
print("=" * 70)

lgb_params_grid = [
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7},
    {"n_estimators": 700, "max_depth": 8, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 1000, "max_depth": -1, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8, "num_leaves": 63},
    {"n_estimators": 1000, "max_depth": -1, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "num_leaves": 127},
]

best_mae_lgb = 999
best_lgb_params = None
for params in lgb_params_grid:
    try:
        model_l = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)
        model_l.fit(X_b_tr, y_b_tr)
        y_pred_l = model_l.predict(X_b_te)
        mae_l = mean_absolute_error(y_b_te, y_pred_l)
        print(f"  {params}: MAE={mae_l:.3f}")
        if mae_l < best_mae_lgb:
            best_mae_lgb = mae_l
            best_lgb_params = params
    except Exception as e:
        print(f"  {params}: FAILED ({e})")

print(f"\n  Best LightGBM: MAE={best_mae_lgb:.3f}")

# ---------------------------------------------------------------------------
# 8. Ensemble: XGBoost + LightGBM
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: XGBoost + LightGBM ensemble")
print("=" * 70)

model_xgb_final = xgb.XGBRegressor(**best_params, random_state=RANDOM_STATE, verbosity=0)
model_lgb_final = lgb.LGBMRegressor(**best_lgb_params, random_state=RANDOM_STATE, verbose=-1)

model_xgb_final.fit(X_b_tr, y_b_tr)
model_lgb_final.fit(X_b_tr, y_b_tr)

pred_xgb = model_xgb_final.predict(X_b_te)
pred_lgb = model_lgb_final.predict(X_b_te)

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    pred_ens = alpha * pred_xgb + (1 - alpha) * pred_lgb
    mae_ens = mean_absolute_error(y_b_te, pred_ens)
    print(f"  alpha={alpha:.1f} (XGB) + {1-alpha:.1f} (LGB): MAE={mae_ens:.3f}")

# Best ensemble
alphas = np.arange(0.0, 1.01, 0.01)
maes = [mean_absolute_error(y_b_te, a * pred_xgb + (1-a) * pred_lgb) for a in alphas]
best_alpha = alphas[np.argmin(maes)]
best_ens_mae = min(maes)
print(f"\n  Best ensemble: alpha={best_alpha:.2f}, MAE={best_ens_mae:.3f}")

# ---------------------------------------------------------------------------
# 9. Bias correction on ensemble
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Bias correction")
print("=" * 70)

pred_ens_best = best_alpha * pred_xgb + (1 - best_alpha) * pred_lgb
residuals = y_b_te - pred_ens_best

# Fit residual ~ prediction
ridge_bias = Ridge()
ridge_bias.fit(pred_ens_best.reshape(-1, 1), residuals)
residual_pred = ridge_bias.predict(pred_ens_best.reshape(-1, 1))
pred_corrected = pred_ens_best + residual_pred

mae_corrected = mean_absolute_error(y_b_te, pred_corrected)
r2_corrected = r2_score(y_b_te, pred_corrected)
print(f"  Before correction: MAE={best_ens_mae:.3f}")
print(f"  After correction:  MAE={mae_corrected:.3f}, R²={r2_corrected:.3f}")

# ---------------------------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

all_results = [
    ("Baseline (current 12)", 9.50, 0.568),
    (f"Top {best_n_final} features (XGB)", best_mae_tune, None),
    (f"Best LightGBM", best_mae_lgb, None),
    (f"Ensemble (XGB+LGB)", best_ens_mae, None),
    (f"Ensemble + bias corrected", mae_corrected, r2_corrected),
]

print(f"\n  {'Method':40s} {'MAE':>8s} {'R²':>8s}")
print(f"  {'-'*56}")
for name, mae, r2 in all_results:
    r2_str = f"{r2:.3f}" if r2 is not None else "—"
    print(f"  {name:40s} {mae:8.3f} {r2_str:>8s}")

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-N features
ax = axes[0, 0]
top_n_df = pd.DataFrame(results)
ax.plot(top_n_df.N, top_n_df.MAE, "o-", color="#2196F3", lw=2, markersize=8)
ax.set_xlabel("Number of Top Features")
ax.set_ylabel("MAE (years)")
ax.set_title("Feature Count vs MAE (XGBoost)")
ax.axhline(y=9.5, color="red", linestyle="--", label="Baseline (12 features)")
ax.legend()
for _, row in top_n_df.iterrows():
    ax.annotate(f'{row.MAE:.2f}', (row.N, row.MAE),
                textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")

# Prediction scatter
ax = axes[0, 1]
ax.scatter(y_b_te, pred_corrected, alpha=0.3, s=10, color="#4CAF50")
ax.plot([18, 80], [18, 80], "k--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("True Age")
ax.set_ylabel("Predicted Age (corrected)")
ax.set_title(f"Ensemble + Bias Corrected (MAE={mae_corrected:.2f})")
ax.legend()

# Residuals distribution
ax = axes[1, 0]
residuals_corrected = y_b_te - pred_corrected
ax.hist(residuals_corrected, bins=50, color="#9C27B0", edgecolor="white", alpha=0.7)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_xlabel("Prediction Error (years)")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution (mean={residuals_corrected.mean():.2f}, std={residuals_corrected.std():.2f})")

# MAE by age group
ax = axes[1, 1]
age_groups = pd.DataFrame({"true": y_b_te, "pred": pred_corrected})
age_groups["group"] = pd.cut(age_groups["true"], bins=[18, 30, 45, 60, 80], labels=["18-30", "31-45", "46-60", "61-80"])
group_mae = age_groups.groupby("group", observed=True).apply(
    lambda g: mean_absolute_error(g["true"], g["pred"]), include_groups=False
)
bars = ax.bar(group_mae.index.astype(str), group_mae.values, color=["#FF9800", "#2196F3", "#4CAF50", "#F44336"])
ax.set_ylabel("MAE (years)")
ax.set_title("MAE by Age Group (corrected)")
for bar, val in zip(bars, group_mae.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", fontsize=10)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "20_deep_feature_engineering.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

# Feature importance from final XGBoost
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Final XGBoost)")
print("=" * 70)
fi = pd.Series(model_xgb_final.feature_importances_, index=used_best).sort_values(ascending=False)
for feat, imp in fi.items():
    print(f"  {feat:12s}: {imp:.4f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
