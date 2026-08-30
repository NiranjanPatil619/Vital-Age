"""
VitalAge — Practical Feature Set Test

Test a model using ONLY features a real user can provide:
- Blood markers (from lab report)
- Demographics (user input)
- Body measures (scale + tape)
- Blood pressure (home monitor)
- Lifestyle (questionnaire)

Usage: python notebooks/blood/test/10_practical_features.py
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Download missing NHANES files for 2017-2018
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Ensure all NHANES 2017-2018 files are downloaded")
print("=" * 70)

MISSING_FILES = {
    "LUX_J": "Liver Ultrasound",
    "ALQ_J": "Alcohol",
    "PAQ_J": "Physical Activity",
    "SMQ_J": "Smoking",
}

for file_code, desc in MISSING_FILES.items():
    filename = f"{file_code}.xpt"
    save_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"  {filename} exists")
        continue
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{filename}"
    print(f"  Downloading {filename} ({desc})... ", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        with open(save_path, 'wb') as f:
            f.write(response.read())
        print(f"OK ({os.path.getsize(save_path)} bytes)")
    except Exception as e:
        print(f"FAILED ({e})")

# ---------------------------------------------------------------------------
# 2. Load ALL NHANES 2017-2018 files
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Load ALL NHANES 2017-2018 files")
print("=" * 70)

NHANES_FILES = {
    "DEMO_J": "Demographics",
    "BMX_J": "Body Measures",
    "BPX_J": "Blood Pressure",
    "LUX_J": "Liver Ultrasound",
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
    "ALQ_J": "Alcohol",
    "PAQ_J": "Physical Activity",
    "SMQ_J": "Smoking",
}

all_dfs = {}
for file_code, desc in NHANES_FILES.items():
    path = os.path.join(RAW_DIR, f"{file_code}.xpt")
    if not os.path.exists(path):
        print(f"  {file_code}: NOT FOUND")
        continue
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
# 3. Define PRACTICAL feature groups
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Define practical feature groups")
print("=" * 70)

target = "RIDAGEYR"

# Group 1: Blood markers only (from lab report)
blood_markers = [
    "LBXSAL",   # Albumin
    "LBXSCR",   # Creatinine
    "LBXGLU",   # Glucose
    "LBXHSCRP", # High-sensitivity CRP (if available)
    "LBXSGB",   # Globulin
    "LBXHGB",   # Hemoglobin
    "LBXHCT",   # Hematocrit
    "LBXTC",    # Total Cholesterol
    "LBDLDL",   # LDL
    "LBDHDD",   # HDL
    "LBXWBCSI", # White Blood Cells
    "LBXIN",    # Insulin
]

# Group 2: Demographics (user input)
demographics = [
    "RIAGENDR",  # Gender
]

# Group 3: Body measures (scale + tape)
body_measures = [
    "BMXWT",    # Weight
    "BMXBMI",   # BMI
    "BMXWAIST", # Waist circumference
]

# Group 4: Blood pressure (home monitor)
blood_pressure = [
    "BPXSY1",   # Systolic BP
    "BPXDI1",   # Diastolic BP
]

# Group 5: Lifestyle (questionnaire)
lifestyle = [
    "ALQ130",   # Alcohol drinks per day
    "PAQ605",   # Physical activity
    "SMQ020",   # Smoking
]

# Feature sets to test
feature_sets = {
    "A_blood_only": blood_markers,
    "B_blood_demo": blood_markers + demographics,
    "C_blood_body": blood_markers + body_measures,
    "D_blood_bp": blood_markers + blood_pressure,
    "E_blood_lifestyle": blood_markers + lifestyle,
    "F_blood_demo_body_bp": blood_markers + demographics + body_measures + blood_pressure,
    "G_blood_all_easy": blood_markers + demographics + body_measures + blood_pressure + lifestyle,
}

for name, cols in feature_sets.items():
    valid = [c for c in cols if c in merged.columns]
    print(f"  {name:30s}: {len(valid)} features")

# ---------------------------------------------------------------------------
# 4. Prepare data and test each feature set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Test each practical feature set")
print("=" * 70)

results = []

for name, feat_cols in feature_sets.items():
    t0 = time.time()
    
    # Get available features
    valid = [c for c in feat_cols if c in merged.columns]
    
    # Filter to rows with target
    df_sub = merged[valid + [target]].copy()
    df_sub = df_sub[df_sub[target].notna() & (df_sub[target] > 0)]
    
    # Impute missing values
    imp = SimpleImputer(strategy="median")
    df_sub[valid] = imp.fit_transform(df_sub[valid])
    
    # Drop rows where target is still NaN
    df_sub = df_sub.dropna()
    
    if len(df_sub) < 500:
        print(f"  {name}: SKIPPED (only {len(df_sub)} rows)")
        continue
    
    X = df_sub[valid].values
    y = df_sub[target].values
    
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
    r2 = r2_score(y_te, pred)
    elapsed = time.time() - t0
    
    results.append({
        "Feature Set": name,
        "N Features": len(valid),
        "N Samples": len(df_sub),
        "MAE": mae,
        "R²": r2,
        "Time (s)": elapsed,
    })
    
    print(f"  {name:30s}: {len(valid):2d} feat, {len(df_sub):5d} samples, MAE={mae:.3f}, R²={r2:.3f}")

# ---------------------------------------------------------------------------
# 5. Results summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

res_df = pd.DataFrame(results).sort_values("MAE")
print(res_df.to_string(index=False))

best = res_df.iloc[0]
print(f"\n  Best: {best['Feature Set']} — MAE={best['MAE']:.3f}, R²={best['R²']:.3f}")

# ---------------------------------------------------------------------------
# 6. Add our current 12 features comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT MODEL")
print("=" * 70)

print(f"\n  {'Model':40s} {'MAE':>8s} {'R²':>8s} {'Features':>8s}")
print(f"  {'-'*64}")
print(f"  {'Our current (12 blood, NHANES 2005-2023)':40s} {'9.50':>8s} {'0.568':>8s} {'12':>8s}")
for _, row in res_df.iterrows():
    print(f"  {row['Feature Set']:40s} {row.MAE:8.3f} {row['R²']:8.3f} {int(row['N Features']):>8d}")

# ---------------------------------------------------------------------------
# 7. Feature importance for best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Best Model)")
print("=" * 70)

best_cols = feature_sets[best["Feature Set"]]
valid_best = [c for c in best_cols if c in merged.columns]
df_best = merged[valid_best + [target]].copy()
df_best = df_best[df_best[target].notna() & (df_best[target] > 0)]
imp_best = SimpleImputer(strategy="median")
df_best[valid_best] = imp_best.fit_transform(df_best[valid_best])
df_best = df_best.dropna()

X_best = df_best[valid_best].values
y_best = df_best[target].values

X_b_tr, X_b_te, y_b_tr, y_b_te = train_test_split(X_best, y_best, test_size=0.2, random_state=RANDOM_STATE)
model_best = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3,
    random_state=RANDOM_STATE, verbosity=0,
)
model_best.fit(X_b_tr, y_b_tr)
fi = pd.Series(model_best.feature_importances_, index=valid_best).sort_values(ascending=False)

for feat, imp in fi.items():
    print(f"  {feat:15s}: {imp:.4f}")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# MAE comparison
ax = axes[0, 0]
all_rows = [{"Feature Set": "Current (12 blood)", "MAE": 9.50, "N Features": 12}]
for _, row in res_df.iterrows():
    all_rows.append({"Feature Set": row["Feature Set"], "MAE": row.MAE, "N Features": row["N Features"]})
plot_df = pd.DataFrame(all_rows)
colors = ["#FF5722" if "Current" in n else "#4CAF50" if m < 4.0 else "#2196F3" for n, m in zip(plot_df["Feature Set"], plot_df.MAE)]
bars = ax.barh(plot_df["Feature Set"], plot_df.MAE, color=colors, edgecolor="white")
ax.set_xlabel("MAE (years)")
ax.set_title("Practical Feature Sets — MAE (lower = better)")
ax.axvline(x=9.5, color="red", linestyle="--", lw=1.5, label="Current baseline")
ax.legend()
for bar, val in zip(bars, plot_df.MAE):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9)

# Prediction scatter (best model)
ax = axes[0, 1]
pred_best = model_best.predict(X_b_te)
ax.scatter(y_b_te, pred_best, alpha=0.3, s=10, color="#4CAF50")
ax.plot([18, 80], [18, 80], "k--", lw=1.5, label="Perfect")
ax.set_xlabel("True Age")
ax.set_ylabel("Predicted Age")
ax.set_title(f"Best Practical Model (MAE={mean_absolute_error(y_b_te, pred_best):.2f})")
ax.legend()

# Feature importance
ax = axes[1, 0]
fi.plot(kind="barh", ax=ax, color="#2196F3")
ax.set_xlabel("Importance")
ax.set_title(f"Feature Importance ({best['Feature Set']})")

# Residuals
ax = axes[1, 1]
resid = y_b_te - pred_best
ax.hist(resid, bins=50, color="#9C27B0", edgecolor="white", alpha=0.7)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_xlabel("Error (years)")
ax.set_title(f"Residuals (mean={resid.mean():.2f}, std={resid.std():.2f})")

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "24_practical_features_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 9. Demo recommendation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DEMO RECOMMENDATION")
print("=" * 70)

blood_only_mae = res_df[res_df['Feature Set']=='A_blood_only']['MAE'].values[0]
all_easy_mae = res_df[res_df['Feature Set']=='G_blood_all_easy']['MAE'].values[0]

print(f"""
  For a real-world demo, we recommend:

  1. BLOOD TEST REPORT (OCR extraction):
     - Albumin, Creatinine, Glucose, CRP
     - Hemoglobin, Hematocrit
     - Total Cholesterol, HDL, LDL
     - White Blood Cell Count
     - Insulin (if available)

  2. USER INPUT FORM:
     - Gender (M/F)
     - Weight (kg)
     - Height (cm) -> BMI calculated
     - Waist circumference (cm)
     - Systolic BP (mmHg)
     - Diastolic BP (mmHg)
     - Smoking status (Never/Former/Current)
     - Alcohol drinks per day
     - Physical activity (Yes/No)

  3. EXPECTED ACCURACY:
     - Blood only: MAE ~{blood_only_mae:.1f} years
     - Blood + all easy features: MAE ~{all_easy_mae:.1f} years
""")

print("=" * 70)
print("DONE")
print("=" * 70)
