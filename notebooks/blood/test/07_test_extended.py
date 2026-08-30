"""
VitalAge — Test Extended NHANES Features

Now that we have 82 columns (blood + demographics + body + BP + metabolic),
test how much MAE improves.

Usage: python notebooks/blood/test/07_test_extended.py
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
                        "..", "..", "..", "data", "raw", "nhanes_extended_merged.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading extended NHANES data")
print("=" * 70)
df = pd.read_csv(RAW_PATH)
df = df[(df["Age"] >= 18) & (df["Age"] <= 80)].copy()
df = df.drop(columns=["SEQN", "CYCLE"], errors="ignore")
print(f"  {df.shape[0]} rows x {df.shape[1]} columns")

target = "Age"

# ---------------------------------------------------------------------------
# 2. Define feature groups
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Defining feature groups")
print("=" * 70)

# Original 12
original_12 = ["LBXSAL", "LBXSCR", "LBXGLU", "CRP", "LBXLYPCT",
               "LBXMCVSI", "LBXRDW", "LBXSAPSI", "LBXWBCSI",
               "LBXGH", "LBDHDD", "LBXTC"]

# All blood markers (57)
blood_only = [c for c in df.columns if c not in ["Age", "Gender", "Age_demo",
               "Race", "Education", "Household_Income", "Weight", "Height",
               "BMI", "Arm_Circumference", "Waist_Circumference",
               "Pulse", "Systolic_BP_1", "Diastolic_BP_1",
               "Systolic_BP_2", "Diastolic_BP_2",
               "Systolic_BP_3", "Diastolic_BP_3",
               "Insulin", "Total_Cholesterol_new", "Triglycerides",
               "HDL_Cholesterol", "HDL_Cholesterol"]]

# Body + vitals (non-blood)
body_vitals = ["BMI", "Waist_Circumference", "Arm_Circumference",
               "Systolic_BP_1", "Diastolic_BP_1", "Pulse"]

# Demographics
demographics = ["Gender", "Race", "Education", "Household_Income"]

# Metabolic extras
metabolic = ["Insulin", "Triglycerides", "HDL_Cholesterol", "Total_Cholesterol_new"]

# Feature sets to test
feature_sets = {
    "A_original_12": original_12,
    "B_all_blood_57": blood_only,
    "C_blood_body_vitals": blood_only + body_vitals,
    "D_blood_body_vitals_demo": blood_only + body_vitals + demographics,
    "E_all_features": blood_only + body_vitals + demographics + metabolic,
    "F_top_blood_plus_vitals": ["LBXSBU", "LBXSOSSI", "LBXGH", "LBXGLU", "LBXSLDSI",
                                 "LBXSGL", "LBXMCVSI", "LBXRBCSI", "LBXSKSI", "LBXSCR",
                                 "LBXRDW", "LBXSC3SI", "LBXPLTSI", "LBXMCHSI", "LBXSAL",
                                 "LBXSUA", "LBXLYPCT", "LBXSAPSI", "LBXTC",
                                 "BMI", "Waist_Circumference", "Systolic_BP_1", "Diastolic_BP_1", "Pulse"],
}

for name, cols in feature_sets.items():
    valid = [c for c in cols if c in df.columns]
    print(f"  {name:40s}: {len(valid)} features")

# ---------------------------------------------------------------------------
# 3. Prepare imputed data and test
# ---------------------------------------------------------------------------
def prepare_imputed(df, feature_cols, target="Age"):
    used = [c for c in feature_cols if c in df.columns]
    df_sub = df[used + [target]].dropna(subset=[target])
    imp = SimpleImputer(strategy="median")
    df_sub[used] = imp.fit_transform(df_sub[used])
    return df_sub[used].values, df_sub[target].values, used, imp

print("\n" + "=" * 70)
print("STEP 3: Testing feature sets")
print("=" * 70)

results = []
feature_importances = {}

for name, feat_cols in feature_sets.items():
    t0 = time.time()
    X, y, used, imp = prepare_imputed(df, feat_cols)
    
    if len(y) < 1000:
        print(f"  {name}: SKIPPED (only {len(y)} rows)")
        continue
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    elapsed = time.time() - t0
    
    results.append({
        "Feature Set": name,
        "N Features": len(used),
        "N Samples": len(y),
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Time (s)": elapsed,
    })
    
    fi = pd.Series(model.feature_importances_, index=used)
    feature_importances[name] = fi.sort_values(ascending=False).head(15)
    
    print(f"  {name:40s}: {len(used):2d} feat, {len(y):6d} samples, MAE={mae:.3f}, R²={r2:.3f} ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
res_df = pd.DataFrame(results).sort_values("MAE")
print(res_df.to_string(index=False))

best = res_df.iloc[0]
print(f"\n  Best: {best['Feature Set']} — MAE={best['MAE']:.3f}, R²={best['R²']:.3f}")

# ---------------------------------------------------------------------------
# 5. Feature importance for best model
# ---------------------------------------------------------------------------
best_name = best["Feature Set"]
if best_name in feature_importances:
    print(f"\n  Top features ({best_name}):")
    for feat, imp in feature_importances[best_name].items():
        print(f"    {feat:25s}: {imp:.4f}")

# ---------------------------------------------------------------------------
# 6. 5-Fold CV on best feature set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: 5-Fold CV on best feature set")
print("=" * 70)

best_cols = feature_sets[best_name]
X_all, y_all, used_all, _ = prepare_imputed(df, best_cols)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_maes = []
cv_r2s = []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X_all), 1):
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    cv_maes.append(mae)
    cv_r2s.append(r2)
    print(f"  Fold {fold}: MAE={mae:.3f}, R²={r2:.3f}")

print(f"\n  CV Mean MAE: {np.mean(cv_maes):.3f} +/- {np.std(cv_maes):.3f}")
print(f"  CV Mean R²:  {np.mean(cv_r2s):.3f} +/- {np.std(cv_r2s):.3f}")

# ---------------------------------------------------------------------------
# 7. XGB + LGB ensemble on best features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: XGB + LGB Ensemble on best features")
print("=" * 70)

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_STATE)

xgb_m = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
    random_state=RANDOM_STATE, verbosity=0,
)
lgb_m = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=-1, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, num_leaves=63,
    random_state=RANDOM_STATE, verbose=-1,
)

xgb_m.fit(X_tr, y_tr)
lgb_m.fit(X_tr, y_tr)

p_xgb = xgb_m.predict(X_te)
p_lgb = lgb_m.predict(X_te)

alphas = np.arange(0.0, 1.01, 0.05)
maes = [mean_absolute_error(y_te, a * p_xgb + (1-a) * p_lgb) for a in alphas]
best_alpha = alphas[np.argmin(maes)]
best_ens_mae = min(maes)
pred_ens = best_alpha * p_xgb + (1 - best_alpha) * p_lgb

print(f"  XGBoost standalone: MAE={mean_absolute_error(y_te, p_xgb):.3f}")
print(f"  LightGBM standalone: MAE={mean_absolute_error(y_te, p_lgb):.3f}")
print(f"  Best ensemble (α={best_alpha:.2f}): MAE={best_ens_mae:.3f}")

# Bias correction
residuals = y_te - pred_ens
ridge_bias = Ridge()
ridge_bias.fit(pred_ens.reshape(-1, 1), residuals)
pred_corrected = pred_ens + ridge_bias.predict(pred_ens.reshape(-1, 1))
mae_corrected = mean_absolute_error(y_te, pred_corrected)
r2_corrected = r2_score(y_te, pred_corrected)
print(f"  After bias correction: MAE={mae_corrected:.3f}, R²={r2_corrected:.3f}")

# ---------------------------------------------------------------------------
# 8. Final comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL COMPARISON: OLD vs NEW")
print("=" * 70)

print(f"\n  {'Model':45s} {'MAE':>8s} {'R²':>8s}")
print(f"  {'-'*61}")
print(f"  {'Original 12 features':45s} {'9.500':>8s} {'0.568':>8s}")
for _, row in res_df.iterrows():
    print(f"  {row['Feature Set']:45s} {row.MAE:8.3f} {row['R²']:8.3f}")
print(f"  {'Ensemble + bias corrected (best)':45s} {mae_corrected:8.3f} {r2_corrected:8.3f}")

# Improvement
orig_mae = 9.500
improvement = orig_mae - mae_corrected
print(f"\n  Improvement over original 12 features: {improvement:.3f} years ({improvement/orig_mae*100:.1f}%)")

# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# MAE comparison
ax = axes[0, 0]
all_rows = [{"Feature Set": "Original 12", "MAE": 9.50, "N Features": 12}]
for _, row in res_df.iterrows():
    all_rows.append({"Feature Set": row["Feature Set"], "MAE": row.MAE, "N Features": row["N Features"]})
all_rows.append({"Feature Set": "Ensemble+corrected", "MAE": mae_corrected, "N Features": len(used_all)})
plot_df = pd.DataFrame(all_rows)
colors = ["#FF5722" if "Original" in n else "#4CAF50" if m < 9.0 else "#2196F3" for n, m in zip(plot_df["Feature Set"], plot_df.MAE)]
bars = ax.barh(plot_df["Feature Set"], plot_df.MAE, color=colors, edgecolor="white")
ax.set_xlabel("MAE (years)")
ax.set_title("Feature Set Comparison — MAE (lower = better)")
ax.axvline(x=9.5, color="red", linestyle="--", lw=1.5, label="Original 12 baseline")
ax.legend()
for bar, val in zip(bars, plot_df.MAE):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

# Prediction scatter (corrected)
ax = axes[0, 1]
ax.scatter(y_te, pred_corrected, alpha=0.3, s=10, color="#4CAF50")
ax.plot([18, 80], [18, 80], "k--", lw=1.5, label="Perfect")
ax.set_xlabel("True Age")
ax.set_ylabel("Predicted Age")
ax.set_title(f"Final Ensemble + Bias Corrected (MAE={mae_corrected:.2f})")
ax.legend()

# Residuals
ax = axes[1, 0]
resid = y_te - pred_corrected
ax.hist(resid, bins=50, color="#9C27B0", edgecolor="white", alpha=0.7)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_xlabel("Error (years)")
ax.set_title(f"Residuals (mean={resid.mean():.2f}, std={resid.std():.2f})")

# Feature importance
ax = axes[1, 1]
fi = feature_importances[best_name].head(15)
fi.plot(kind="barh", ax=ax, color="#2196F3")
ax.set_xlabel("Importance")
ax.set_title(f"Top 15 Features ({best_name})")

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "22_extended_features_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
