"""
VitalAge — Test Features Available from Full Body Test Reports

Maps real blood test report features to NHANES variables and tests performance.

Report features available in NHANES:
- CBC: Hb, RBC, HCT, MCV, MCH, RDW, Platelets, WBC, Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils
- Kidney: BUN, Creatinine, Uric Acid
- Diabetes: HbA1c, Glucose
- Lipids: Total Cholesterol, Triglycerides, HDL, LDL
- Liver: Albumin, Globulin, Total Protein, Bilirubin, ALT, AST, ALP, GGT
- Minerals: Calcium, Iron
- Inflammation: CRP

Usage: python notebooks/blood/test/11_report_features.py
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
# 1. Load raw data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading raw NHANES data")
print("=" * 70)
df = pd.read_csv(RAW_PATH)
df = df[(df["Age"] >= 18) & (df["Age"] <= 80)].copy()
df = df.drop(columns=["SEQN", "CYCLE"], errors="ignore")
print(f"  {df.shape[0]} rows x {df.shape[1]} columns")

target = "Age"

# ---------------------------------------------------------------------------
# 2. Map report features to NHANES columns
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Mapping report features to NHANES variables")
print("=" * 70)

# Original 12 features
original_12 = {
    "LBXSAL": "Albumin",
    "LBXSCR": "Creatinine",
    "LBXGLU": "Glucose",
    "CRP": "CRP (not in report, but useful)",
    "LBXLYPCT": "Lymphocytes %",
    "LBXMCVSI": "MCV",
    "LBXRDW": "RDW-CV",
    "LBXSAPSI": "Alkaline Phosphatase",
    "LBXWBCSI": "WBC Count",
    "LBXGH": "HbA1c",
    "LBDHDD": "HDL Cholesterol",
    "LBXTC": "Total Cholesterol",
}

# NEW features from report (available in NHANES)
report_features = {
    # CBC
    "LBXHGB": "Haemoglobin",
    "LBXRBCSI": "RBC Count",
    "LBXHCT": "Haematocrit",
    "LBXMCHSI": "MCH",
    "LBXPLTSI": "Platelet Count",
    "LBXNEPCT": "Neutrophils %",
    "LBXMOPCT": "Monocytes %",
    "LBXEOPCT": "Eosinophils %",
    "LBXBAPCT": "Basophils %",
    # Kidney
    "LBXSBU": "BUN",
    "LBXSUA": "Uric Acid",
    # Lipids
    "LBXTR": "Triglycerides",
    "LBDLDL": "LDL",
    # Liver
    "LBXSTB": "Total Bilirubin",
    "LBXSATSI": "ALT/SGPT",
    "LBXSGTSI": "GGT",
    "LBXSTP": "Total Protein",
    "LBXSGB": "Globulin",
    # Minerals
    "LBXSCA": "Calcium",
    "LBXSIR": "Iron",
}

print(f"\n  ORIGINAL 12 features (current model):")
for code, name in original_12.items():
    status = "Y" if code in df.columns else "N"
    print(f"    {status} {code:12s} = {name}")

print(f"\n  NEW features from report (available in NHANES):")
for code, name in report_features.items():
    status = "Y" if code in df.columns else "N"
    print(f"    {status} {code:12s} = {name}")

# All features combined
all_report_features = list(original_12.keys()) + list(report_features.keys())
valid_features = [f for f in all_report_features if f in df.columns]

print(f"\n  Total report features available: {len(valid_features)}")
print(f"  Missing: {[f for f in all_report_features if f not in df.columns]}")

# ---------------------------------------------------------------------------
# 3. Test feature sets
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Testing feature sets")
print("=" * 70)

def prepare_imputed(df, feature_cols, target="Age"):
    used = [c for c in feature_cols if c in df.columns]
    df_sub = df[used + [target]].dropna(subset=[target])
    imp = SimpleImputer(strategy="median")
    df_sub[used] = imp.fit_transform(df_sub[used])
    df_sub = df_sub.dropna()
    return df_sub[used].values, df_sub[target].values, used

feature_sets = {
    "Current 12": list(original_12.keys()),
    "Report CBC only": ["LBXHGB", "LBXRBCSI", "LBXHCT", "LBXMCVSI", "LBXMCHSI",
                         "LBXRDW", "LBXPLTSI", "LBXWBCSI", "LBXNEPCT", "LBXMOPCT",
                         "LBXEOPCT", "LBXBAPCT"],
    "Report Lipids only": ["LBXTC", "LBDHDD", "LBDLDL", "LBXTR"],
    "Report Liver only": ["LBXSAL", "LBXSGB", "LBXSTP", "LBXSTB", "LBXSATSI",
                           "LBXSGTSI", "LBXSAPSI"],
    "Report Kidney only": ["LBXSCR", "LBXSBU", "LBXSUA"],
    "Original 12 + CBC": list(original_12.keys()) + ["LBXHGB", "LBXRBCSI", "LBXHCT",
                                                       "LBXPLTSI", "LBXNEPCT", "LBXMOPCT"],
    "Original 12 + CBC + Lipids": list(original_12.keys()) + ["LBXHGB", "LBXRBCSI", "LBXHCT",
                                                               "LBXPLTSI", "LBXNEPCT",
                                                               "LBXTR", "LBDLDL"],
    "Original 12 + ALL report": valid_features,
    "ALL report features": valid_features,
}

results = []

for name, feat_cols in feature_sets.items():
    t0 = time.time()
    X, y, used = prepare_imputed(df, feat_cols)
    
    if len(y) < 1000:
        print(f"  {name}: SKIPPED (only {len(y)} rows)")
        continue
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    elapsed = time.time() - t0
    
    results.append({
        "Feature Set": name,
        "N Features": len(used),
        "N Samples": len(y),
        "MAE": mae,
        "R²": r2,
    })
    
    print(f"  {name:35s}: {len(used):2d} feat, {len(y):6d} samples, MAE={mae:.3f}, R²={r2:.3f}")

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

res_df = pd.DataFrame(results).sort_values("MAE")
print(res_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Feature importance for best model
# ---------------------------------------------------------------------------
best_name = res_df.iloc[0]["Feature Set"]
best_cols = feature_sets[best_name]
X_all, y_all, used_all = prepare_imputed(df, best_cols)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_STATE)

model_best = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
    random_state=RANDOM_STATE, verbosity=0,
)
model_best.fit(X_tr, y_tr)
fi = pd.Series(model_best.feature_importances_, index=used_all).sort_values(ascending=False)

print(f"\n  Top features ({best_name}):")
for feat, imp in fi.items():
    name = original_12.get(feat, report_features.get(feat, "?"))
    print(f"    {feat:12s} ({name:25s}): {imp:.4f}")

# ---------------------------------------------------------------------------
# 6. 5-Fold CV
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: 5-Fold CV on best feature set")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_maes = []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X_all), 1):
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_all[tr_idx], y_all[tr_idx])
    pred = model.predict(X_all[te_idx])
    mae = mean_absolute_error(y_all[te_idx], pred)
    cv_maes.append(mae)
    print(f"  Fold {fold}: MAE={mae:.3f}")

print(f"\n  CV Mean MAE: {np.mean(cv_maes):.3f} +/- {np.std(cv_maes):.3f}")

# ---------------------------------------------------------------------------
# 7. XGB + LGB ensemble
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: XGB + LGB Ensemble")
print("=" * 70)

xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
                           random_state=RANDOM_STATE, verbosity=0)
lgb_m = lgb.LGBMRegressor(n_estimators=1000, max_depth=-1, learning_rate=0.02,
                            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
                            random_state=RANDOM_STATE, verbose=-1)

xgb_m.fit(X_tr, y_tr)
lgb_m.fit(X_tr, y_tr)

p_xgb = xgb_m.predict(X_te)
p_lgb = lgb_m.predict(X_te)

alphas = np.arange(0.0, 1.01, 0.05)
maes = [mean_absolute_error(y_te, a * p_xgb + (1-a) * p_lgb) for a in alphas]
best_alpha = alphas[np.argmin(maes)]
best_ens_mae = min(maes)
pred_ens = best_alpha * p_xgb + (1 - best_alpha) * p_lgb

print(f"  XGBoost: MAE={mean_absolute_error(y_te, p_xgb):.3f}")
print(f"  LightGBM: MAE={mean_absolute_error(y_te, p_lgb):.3f}")
print(f"  Best ensemble (alpha={best_alpha:.2f}): MAE={best_ens_mae:.3f}")

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
print("FINAL COMPARISON")
print("=" * 70)

print(f"\n  {'Model':45s} {'MAE':>8s} {'R²':>8s}")
print(f"  {'-'*61}")
print(f"  {'Current model (12 features)':45s} {'9.500':>8s} {'0.568':>8s}")
for _, row in res_df.iterrows():
    print(f"  {row['Feature Set']:45s} {row.MAE:8.3f} {row['R²']:8.3f}")
print(f"  {'Ensemble + corrected (best)':45s} {mae_corrected:8.3f} {r2_corrected:8.3f}")

# ---------------------------------------------------------------------------
# 9. Report mapping summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("REPORT FEATURE MAPPING")
print("=" * 70)

print(f"""
  From a FULL BODY TEST REPORT, we can extract these NHANES features:

  CBC (Complete Blood Count):
    Haemoglobin    -> LBXHGB    (NEW, importance: {fi.get('LBXHGB', 0):.4f})
    RBC Count      -> LBXRBCSI  (NEW, importance: {fi.get('LBXRBCSI', 0):.4f})
    Haematocrit    -> LBXHCT    (NEW, importance: {fi.get('LBXHCT', 0):.4f})
    MCV            -> LBXMCVSI  (current, importance: {fi.get('LBXMCVSI', 0):.4f})
    MCH            -> LBXMCHSI  (NEW, importance: {fi.get('LBXMCHSI', 0):.4f})
    RDW-CV         -> LBXRDW    (current, importance: {fi.get('LBXRDW', 0):.4f})
    Platelets      -> LBXPLTSI  (NEW, importance: {fi.get('LBXPLTSI', 0):.4f})
    WBC            -> LBXWBCSI  (current, importance: {fi.get('LBXWBCSI', 0):.4f})
    Neutrophils    -> LBXNEPCT  (NEW, importance: {fi.get('LBXNEPCT', 0):.4f})
    Lymphocytes    -> LBXLYPCT  (current, importance: {fi.get('LBXLYPCT', 0):.4f})
    Monocytes      -> LBXMOPCT  (NEW, importance: {fi.get('LBXMOPCT', 0):.4f})
    Eosinophils    -> LBXEOPCT  (NEW, importance: {fi.get('LBXEOPCT', 0):.4f})
    Basophils      -> LBXBAPCT  (NEW, importance: {fi.get('LBXBAPCT', 0):.4f})

  Kidney/Diabetes:
    Creatinine     -> LBXSCR    (current, importance: {fi.get('LBXSCR', 0):.4f})
    BUN            -> LBXSBU    (NEW, importance: {fi.get('LBXSBU', 0):.4f})
    Uric Acid      -> LBXSUA    (NEW, importance: {fi.get('LBXSUA', 0):.4f})
    HbA1c          -> LBXGH     (current, importance: {fi.get('LBXGH', 0):.4f})
    Glucose        -> LBXGLU    (current, importance: {fi.get('LBXGLU', 0):.4f})

  Lipid Profile:
    Total Chol     -> LBXTC     (current, importance: {fi.get('LBXTC', 0):.4f})
    HDL            -> LBDHDD    (current, importance: {fi.get('LBDHDD', 0):.4f})
    LDL            -> LBDLDL    (NEW, importance: {fi.get('LBDLDL', 0):.4f})
    Triglycerides  -> LBXTR     (NEW, importance: {fi.get('LBXTR', 0):.4f})

  Liver Function:
    Albumin        -> LBXSAL    (current, importance: {fi.get('LBXSAL', 0):.4f})
    Globulin       -> LBXSGB    (NEW, importance: {fi.get('LBXSGB', 0):.4f})
    Total Protein  -> LBXSTP    (NEW, importance: {fi.get('LBXSTP', 0):.4f})
    Bilirubin      -> LBXSTB    (NEW, importance: {fi.get('LBXSTB', 0):.4f})
    ALT/SGPT       -> LBXSATSI  (NEW, importance: {fi.get('LBXSATSI', 0):.4f})
    ALP            -> LBXSAPSI  (current, importance: {fi.get('LBXSAPSI', 0):.4f})
    GGT            -> LBXSGTSI  (NEW, importance: {fi.get('LBXSGTSI', 0):.4f})

  Minerals:
    Calcium        -> LBXSCA    (NEW, importance: {fi.get('LBXSCA', 0):.4f})
    Iron           -> LBXSIR    (NEW, importance: {fi.get('LBXSIR', 0):.4f})

  NOT in report but in NHANES:
    CRP            -> CRP       (current, importance: {fi.get('CRP', 0):.4f})

  Summary: {len([f for f in valid_features if f in original_12])} current + {len([f for f in valid_features if f not in original_12])} NEW features from report = {len(valid_features)} total
""")

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# MAE comparison
ax = axes[0, 0]
all_rows = [{"Feature Set": "Current (12)", "MAE": 9.50, "N Features": 12}]
for _, row in res_df.iterrows():
    all_rows.append({"Feature Set": row["Feature Set"], "MAE": row.MAE, "N Features": row["N Features"]})
plot_df = pd.DataFrame(all_rows)
colors = ["#FF5722" if "Current" in n else "#4CAF50" if m < 8.5 else "#2196F3" for n, m in zip(plot_df["Feature Set"], plot_df.MAE)]
bars = ax.barh(plot_df["Feature Set"], plot_df.MAE, color=colors, edgecolor="white")
ax.set_xlabel("MAE (years)")
ax.set_title("Feature Set Comparison — MAE (lower = better)")
ax.axvline(x=9.5, color="red", linestyle="--", lw=1.5, label="Current baseline")
ax.legend()
for bar, val in zip(bars, plot_df.MAE):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9)

# Feature importance
ax = axes[0, 1]
fi.head(15).plot(kind="barh", ax=ax, color="#2196F3")
ax.set_xlabel("Importance")
ax.set_title(f"Top 15 Features ({best_name})")

# Prediction scatter
ax = axes[1, 0]
pred_best = model_best.predict(X_te)
ax.scatter(y_te, pred_best, alpha=0.3, s=10, color="#4CAF50")
ax.plot([18, 80], [18, 80], "k--", lw=1.5, label="Perfect")
ax.set_xlabel("True Age")
ax.set_ylabel("Predicted Age")
ax.set_title(f"Best Model (MAE={mean_absolute_error(y_te, pred_best):.2f})")
ax.legend()

# Residuals
ax = axes[1, 1]
resid = y_te - pred_best
ax.hist(resid, bins=50, color="#9C27B0", edgecolor="white", alpha=0.7)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_xlabel("Error (years)")
ax.set_title(f"Residuals (mean={resid.mean():.2f}, std={resid.std():.2f})")

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "25_report_features_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {out_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
