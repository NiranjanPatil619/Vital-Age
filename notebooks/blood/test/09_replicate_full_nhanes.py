"""
VitalAge — Download ALL NHANES 2017-2018 files and replicate YGhobara's approach

Downloads every available NHANES 2017-2018 file, merges them,
and trains the same stacked model.

Usage: python notebooks/blood/test/09_replicate_full_nhanes.py
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
import urllib.request

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
EXT_DIR = os.path.join(os.path.dirname(__file__), "external_models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EXT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Download ALL NHANES 2017-2018 files
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Download ALL NHANES 2017-2018 files")
print("=" * 70)

# All file codes for 2017-2018
NHANES_FILES = {
    # Demographics
    "DEMO_J": "Demographics",
    # Examination
    "BMX_J": "Body Measures",
    "BPX_J": "Blood Pressure",
    "LUX_J": "Liver Ultrasound",
    # Laboratory
    "ALB_CR_J": "Albumin/Creatinine Urine",
    "BIOPRO_J": "Standard Biochemistry",
    "CBC_J": "Complete Blood Count",
    "GHB_J": "Glycohemoglobin",
    "GLU_J": "Glucose",
    "HDL_J": "HDL Cholesterol",
    "HSCRP_J": "High-Sensitivity CRP",
    "INS_J": "Insulin",
    "TCHOL_J": "Total Cholesterol",
    "TRIGLY_J": "Triglycerides",
    # Questionnaire
    "ALQ_J": "Alcohol",
    "PAQ_J": "Physical Activity",
    "SMQ_J": "Smoking",
}

downloaded = {}
for file_code, desc in NHANES_FILES.items():
    filename = f"{file_code}.xpt"
    save_path = os.path.join(RAW_DIR, filename)
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"  {filename} exists ({os.path.getsize(save_path)} bytes)")
        downloaded[file_code] = save_path
        continue
    
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{filename}"
    print(f"  Downloading {filename} ({desc})... ", end="", flush=True)
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        with open(save_path, 'wb') as f:
            f.write(response.read())
        size = os.path.getsize(save_path)
        print(f"OK ({size} bytes)")
        downloaded[file_code] = save_path
    except Exception as e:
        print(f"FAILED ({e})")

print(f"\n  Downloaded: {len(downloaded)} files")

# ---------------------------------------------------------------------------
# 2. Load and merge all files
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Load and merge all files")
print("=" * 70)

all_dfs = {}
for file_code, path in downloaded.items():
    try:
        df = pd.read_sas(path, format='xport')
        df["SEQN"] = df["SEQN"].astype("Int64")
        all_dfs[file_code] = df
        print(f"  {file_code}: {df.shape[0]} rows x {df.shape[1]} cols")
    except Exception as e:
        print(f"  {file_code}: FAILED ({e})")

# Merge all on SEQN
merged = list(all_dfs.values())[0].copy()
for file_code, df in list(all_dfs.items())[1:]:
    try:
        merged = merged.merge(df, on="SEQN", how="outer", suffixes=("", f"_{file_code}"))
    except Exception as e:
        print(f"  Merge {file_code}: FAILED ({e})")

print(f"\n  Merged: {merged.shape[0]} rows x {merged.shape[1]} columns")

# ---------------------------------------------------------------------------
# 3. Replicate YGhobara's feature selection
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Feature selection (YGhobara's approach)")
print("=" * 70)

# Their selected features
their_features = [
    "BMXWT", "BMXLEG", "BMXARMC", "BMXWAIST", "BMXHIP",
    "BPXSY1", "BPXDI1",
    "LBXGLU", "LBXIN", "LBXSCR", "LBXSGB", "LBXHGB", "LBXHCT",
    "LBXTC", "LBDLDL", "LBDHDD", "LBXWBCSI",
    "ALQ130", "PAQ605", "SMQ020"
]

# Check which features exist in our merged data
available = [f for f in their_features if f in merged.columns]
missing = [f for f in their_features if f not in merged.columns]

print(f"  Their features: {len(their_features)}")
print(f"  Available in merged: {len(available)}")
print(f"  Missing: {missing}")

# Filter to only rows with age
df_clean = merged.copy()
if "RIDAGEYR" in df_clean.columns:
    df_clean = df_clean[df_clean["RIDAGEYR"].notna() & (df_clean["RIDAGEYR"] > 0)]
    print(f"  After age filter: {len(df_clean)} rows")
else:
    print("  WARNING: RIDAGEYR not found!")
    # Try to find age column
    age_cols = [c for c in df_clean.columns if "AGE" in c.upper() or "RIDAGE" in c.upper()]
    print(f"  Possible age columns: {age_cols}")

# Use available features + age
target = "RIDAGEYR"
use_cols = [c for c in available if c in df_clean.columns] + [target]
df_model = df_clean[use_cols].dropna()
print(f"  After dropping NaN: {len(df_model)} rows x {df_model.shape[1]} columns")

# ---------------------------------------------------------------------------
# 4. Train and test stacked model (YGhobara's approach)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Train stacked model (YGhobara's approach)")
print("=" * 70)

X = df_model.drop(columns=[target])
y = df_model[target]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Base models
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                           min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)
xgb_m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5,
                           subsample=0.7, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0)

t0 = time.time()
rf.fit(X_tr, y_tr)
xgb_m.fit(X_tr, y_tr)
t_train = time.time() - t0

# Meta-learner
rf_pred_tr = rf.predict(X_tr)
xgb_pred_tr = xgb_m.predict(X_tr)
meta_X_tr = np.column_stack((rf_pred_tr, xgb_pred_tr))

rf_pred_te = rf.predict(X_te)
xgb_pred_te = xgb_m.predict(X_te)
meta_X_te = np.column_stack((rf_pred_te, xgb_pred_te))

meta_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3,
                               subsample=0.7, colsample_bytree=0.8, reg_lambda=5, reg_alpha=5,
                               random_state=RANDOM_STATE, verbosity=0)
meta_model.fit(meta_X_tr, y_tr)
stacked_pred = meta_model.predict(meta_X_te)

# Results
mae_rf = mean_absolute_error(y_te, rf_pred_te)
mae_xgb = mean_absolute_error(y_te, xgb_pred_te)
mae_stacked = mean_absolute_error(y_te, stacked_pred)
r2_stacked = r2_score(y_te, stacked_pred)
rmse_stacked = np.sqrt(mean_squared_error(y_te, stacked_pred))

print(f"  Features used: {list(X.columns)}")
print(f"  Training samples: {len(y_tr)}")
print(f"  Test samples: {len(y_te)}")
print(f"  Training time: {t_train:.1f}s")
print(f"\n  Results:")
print(f"    RF standalone:  MAE={mae_rf:.3f}")
print(f"    XGB standalone: MAE={mae_xgb:.3f}")
print(f"    Stacked:        MAE={mae_stacked:.3f}, RMSE={rmse_stacked:.3f}, R²={r2_stacked:.3f}")

# ---------------------------------------------------------------------------
# 5. Age group breakdown
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Age group breakdown")
print("=" * 70)

groups = pd.DataFrame({"true": y_te.values, "pred": stacked_pred})
groups["group"] = pd.cut(groups["true"], bins=[18, 30, 45, 60, 80], labels=["18-30", "31-45", "46-60", "61-80"])
for grp, gdf in groups.groupby("group", observed=True):
    mae_g = mean_absolute_error(gdf["true"], gdf["pred"])
    bias = gdf["pred"].mean() - gdf["true"].mean()
    print(f"  {grp}: N={len(gdf)}, MAE={mae_g:.2f}, bias={bias:+.2f}")

# ---------------------------------------------------------------------------
# 6. Now test: what if we use ALL available features?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Test with ALL available features (our approach)")
print("=" * 70)

# Use all numeric columns except SEQN and target
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
exclude = ["SEQN", target, "SDDSRVYR", "RIDSTATR", "RIDRETH1", "RIDRETH3",
           "RIDEXMON", "RIDEXAGM", "SDMVPSU", "SDMVSTRA"]
use_all = [c for c in numeric_cols if c not in exclude and c in df_clean.columns]

df_all = df_clean[use_all + [target]].dropna()
print(f"  Using {len(use_all)} features, {len(df_all)} complete rows")

X_all = df_all[use_all]
y_all = df_all[target]

X_all_tr, X_all_te, y_all_tr, y_all_te = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_STATE)

# XGBoost with all features
xgb_all = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
                             random_state=RANDOM_STATE, verbosity=0)
xgb_all.fit(X_all_tr, y_all_tr)
pred_all = xgb_all.predict(X_all_te)
mae_all = mean_absolute_error(y_all_te, pred_all)
r2_all = r2_score(y_all_te, pred_all)
print(f"  XGBoost (all features): MAE={mae_all:.3f}, R²={r2_all:.3f}")

# Feature importance
fi = pd.Series(xgb_all.feature_importances_, index=use_all).sort_values(ascending=False)
print(f"\n  Top 20 features:")
for feat, imp in fi.head(20).items():
    print(f"    {feat:25s}: {imp:.4f}")

# ---------------------------------------------------------------------------
# 7. Final comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(f"\n  {'Model':45s} {'MAE':>8s} {'R²':>8s} {'Features':>8s}")
print(f"  {'-'*69}")
print(f"  {'Our current (12 blood markers)':45s} {'9.500':>8s} {'0.568':>8s} {'12':>8s}")
print(f"  {'YGhobara replicated (20 features)':45s} {mae_stacked:8.3f} {r2_stacked:8.3f} {'20':>8s}")
print(f"  {'All NHANES features (XGBoost)':45s} {mae_all:8.3f} {r2_all:8.3f} {len(use_all):>8d}")

# Feature importance comparison
print(f"\n  Top 10 features that matter most:")
for i, (feat, imp) in enumerate(fi.head(10).items(), 1):
    in_theirs = "✓" if feat in their_features else " "
    print(f"    {i:2d}. {feat:25s} imp={imp:.4f} [{in_theirs}]")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Comparison bar chart
ax = axes[0, 0]
models = ["Current\n(12 features)", "YGhobara\n(20 features)", "All Features\n(XGBoost)"]
maes = [9.50, mae_stacked, mae_all]
colors = ["#FF5722", "#2196F3", "#4CAF50"]
bars = ax.bar(models, maes, color=colors, edgecolor="white")
ax.set_ylabel("MAE (years)")
ax.set_title("Model Comparison — MAE (lower = better)")
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")

# Prediction scatter
ax = axes[0, 1]
ax.scatter(y_all_te, pred_all, alpha=0.3, s=10, color="#4CAF50")
ax.plot([18, 80], [18, 80], "k--", lw=1.5, label="Perfect")
ax.set_xlabel("True Age")
ax.set_ylabel("Predicted Age")
ax.set_title(f"All Features XGBoost (MAE={mae_all:.2f})")
ax.legend()

# Feature importance
ax = axes[1, 0]
fi.head(15).plot(kind="barh", ax=ax, color="#2196F3")
ax.set_xlabel("Importance")
ax.set_title("Top 15 Feature Importances (XGBoost)")

# Residuals
ax = axes[1, 1]
resid = y_all_te - pred_all
ax.hist(resid, bins=50, color="#9C27B0", edgecolor="white", alpha=0.7)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_xlabel("Error (years)")
ax.set_title(f"Residuals (mean={resid.mean():.2f}, std={resid.std():.2f})")

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "23_full_nhanes_replication.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
