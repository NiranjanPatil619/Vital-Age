"""
VitalAge — Expanded Feature Engineering & Testing

Goal: Find more predictive features beyond the 12 currently used.
Tests: all 48 blood markers, demographics, body measures, blood pressure.

NHANES data sources used:
- DEMO (demographics): gender, race, education, income
- BMX (body measures): BMI, waist circumference, arm circumference
- BPX (blood pressure): systolic, diastolic
- All available blood chemistry (60 columns)

Usage: python notebooks/blood/test/03_expanded_features.py
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH = os.path.join(os.path.dirname(__file__),
                        "..", "..", "..", "data", "raw", "blood_age_mega_raw.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
TEST_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw data with ALL columns
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading raw data (ALL columns)")
print("=" * 70)
df_raw = pd.read_csv(RAW_PATH)
print(f"  Raw data: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

# Filter age 18-80
df_raw = df_raw[(df_raw["Age"] >= 18) & (df_raw["Age"] <= 80)].copy()
print(f"  After age filter (18-80): {len(df_raw)} rows")

# Drop SEQN and CYCLE (IDs, not features)
df_raw = df_raw.drop(columns=["SEQN", "CYCLE"], errors="ignore")

# Count missingness
missing_pct = df_raw.isnull().mean().sort_values(ascending=False)
print(f"\n  Missingness summary:")
print(f"  Columns with 0% missing: {(missing_pct == 0).sum()}")
print(f"  Columns with <10% missing: {(missing_pct < 0.10).sum()}")
print(f"  Columns with <25% missing: {(missing_pct < 0.25).sum()}")
print(f"  Columns with <50% missing: {(missing_pct < 0.50).sum()}")
print(f"\n  Top 15 most missing:")
for col, pct in missing_pct.head(15).items():
    print(f"    {col:12s}: {pct*100:.1f}%")

# ---------------------------------------------------------------------------
# 2. Define feature sets to test
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Defining feature sets")
print("=" * 70)

# Current 12 features (our baseline)
current_12 = ["LBXSAL", "LBXSCR", "LBXGLU", "CRP", "LBXLYPCT",
              "LBXMCVSI", "LBXRDW", "LBXSAPSI", "LBXWBCSI",
              "LBXGH", "LBDHDD", "LBXTC"]

# All blood markers that exist in raw data (excluding Age)
all_blood = [c for c in df_raw.columns if c != "Age"]

# Group by category based on NHANES naming conventions
# Liver enzymes
liver = ["LBXSASSI", "LBXSATSI", "LBXSGTSI", "LBXSLDSI", "LBXSTB"]
# Kidney
kidney = ["LBXSCR", "LBXSBU", "URXCRS", "URXUCR"]
# Lipids
lipids = ["LBDHDD", "LBDLDL", "LBXTC", "LBXTR"]
# Blood count
blood_count = ["LBXHGB", "LBXHCT", "LBXRBCSI", "LBXPLTSI", "LBXRDW",
               "LBXMCVSI", "LBXMCHSI", "LBXMPSI", "LBXWBCSI",
               "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT",
               "LBDMONO", "LBDNENO", "LBDLYMNO", "LBDBANO", "LBDEONO", "LBXNRBC"]
# Metabolic
metabolic = ["LBXGLU", "LBXGH", "LBXSCH", "LBXSCLSI", "LBXSBU"]
# Minerals
minerals = ["LBXSCA", "LBXSCLSI", "LBXSNASI", "LBXSKSI", "LBXMAGN", "LBXSPH"]
# Enzymes
enzymes = ["LBXSAPSI", "LBXSCK", "LBXSC3SI", "LBXSOSSI", "LBXSGTSI"]
# Proteins
proteins = ["LBXSAL", "LBXSGB", "LBXSTP"]
# Iron / TIBC
iron = ["LBXSIR", "LBXMC"]
# Uric acid
uric = ["LBXSUA"]
# Thyroid
thyroid = ["LBXGH"]  # gamma-glutamyl transferase (not thyroid but metabolic)
# Urine
urine = ["URXUMA", "URXUMS", "URDACT"]
# Inflammation
inflammation = ["CRP"]

feature_sets = {
    "A_current_12": current_12,
    "B_all_blood_all": all_blood,
    "C_low_missing": [c for c in all_blood if missing_pct.get(c, 1) < 0.10],
    "D_liver_kidney": liver + kidney + current_12,
    "E_blood_count_extra": blood_count + current_12,
    "F_metabolic_extra": metabolic + iron + current_12,
    "G_all_no_urine": [c for c in all_blood if c not in urine],
    "H_top30_low_missing": [c for c in all_blood if missing_pct.get(c, 1) < 0.30][:30],
}

for name, cols in feature_sets.items():
    valid = [c for c in cols if c in df_raw.columns]
    print(f"  {name:30s}: {len(valid)} features")

# ---------------------------------------------------------------------------
# 3. Helper: prepare complete-case data for a feature set
# ---------------------------------------------------------------------------
def prepare_data(df, feature_cols, target="Age", impute=False):
    """Prepare X, y. If impute=True, use median imputation instead of dropping rows."""
    used = [c for c in feature_cols if c in df.columns]
    if impute:
        df_sub = df[used + [target]].copy()
        # Impute with median for features, drop rows only if target is missing
        df_sub = df_sub.dropna(subset=[target])
        imp = SimpleImputer(strategy="median")
        df_sub[used] = imp.fit_transform(df_sub[used])
        X = df_sub[used].values
        y = df_sub[target].values
        return X, y, used, imp
    else:
        cols = used + [target]
        df_sub = df[cols].dropna()
        X = df_sub[used].values
        y = df_sub[target].values
        return X, y, used, None

# ---------------------------------------------------------------------------
# 4. Test each feature set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Testing feature sets with XGBoost")
print("=" * 70)

xgb_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbosity=0,
)

results = []
feature_importances = {}

for name, feat_cols in feature_sets.items():
    t0 = time.time()
    use_impute = len(feat_cols) > 20  # Impute for large feature sets
    result = prepare_data(df_raw, feat_cols, impute=use_impute)
    X, y, used_cols, imp = result

    if len(y) < 1000:
        print(f"  {name}: SKIPPED (only {len(y)} complete rows)")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = clone(xgb_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    elapsed = time.time() - t0

    method = "imputed" if use_impute else "complete-case"
    results.append({
        "Feature Set": name,
        "N Features": len(used_cols),
        "N Samples": len(y),
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Time (s)": elapsed,
        "Method": method,
    })

    # Store top 10 feature importances
    fi = model.feature_importances_
    if len(fi) == len(used_cols):
        importances = pd.Series(fi, index=used_cols)
        feature_importances[name] = importances.sort_values(ascending=False).head(10)

    print(f"  {name:30s}: {len(used_cols):2d} features, {len(y):6d} samples, "
          f"MAE={mae:.3f}, R²={r2:.3f} [{method}] ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 5. Results table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

res_df = pd.DataFrame(results).sort_values("MAE")
print(res_df.to_string(index=False))

# Highlight best
best = res_df.iloc[0]
print(f"\n  Best: {best['Feature Set']} — MAE={best['MAE']:.3f}, R²={best['R²']:.3f} "
      f"({int(best['N Features'])} features, {int(best['N Samples'])} samples)")

# ---------------------------------------------------------------------------
# 6. Feature importance analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP FEATURES (Best Model)")
print("=" * 70)

best_name = best["Feature Set"]
if best_name in feature_importances:
    for feat, imp in feature_importances[best_name].items():
        print(f"  {feat:12s}: {imp:.4f}")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# MAE comparison
ax = axes[0]
colors = ["#2ecc71" if mae < 9.465 else "#e74c3c" for mae in res_df.MAE]
bars = ax.barh(res_df["Feature Set"], res_df.MAE, color=colors, edgecolor="white")
ax.set_xlabel("MAE (years)")
ax.set_title("Feature Set Comparison — XGBoost MAE (lower = better)")
ax.axvline(x=9.465, color="black", linestyle="--", lw=1.5, label="Current 12 features (9.465)")
ax.legend()
for bar, val in zip(bars, res_df.MAE):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

# Sample size vs MAE
ax = axes[1]
ax.scatter(res_df["N Samples"], res_df["MAE"], s=100, c=colors, edgecolors="black")
for _, row in res_df.iterrows():
    ax.annotate(row["Feature Set"], (row["N Samples"], row["MAE"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("Number of Complete Samples")
ax.set_ylabel("MAE (years)")
ax.set_title("Sample Size vs MAE (more features → fewer complete rows)")
ax.axhline(y=9.465, color="black", linestyle="--", lw=1)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "19_expanded_features_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 8. Detailed correlation analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CORRELATION WITH AGE (all features)")
print("=" * 70)

corr_with_age = df_raw.corr(numeric_only=True)["Age"].drop("Age").sort_values(key=abs, ascending=False)
print("\n  Top 20 features most correlated with Age:")
for feat, corr in corr_with_age.head(20).items():
    marker = " <--" if feat in current_12 else ""
    print(f"    {feat:12s}: r={corr:+.4f}{marker}")

print("\n  Features in current_12 that are NOT in top 20:")
for feat in current_12:
    if feat in corr_with_age.index:
        rank = list(corr_with_age.index).index(feat) + 1
        corr = corr_with_age[feat]
        if rank > 20:
            print(f"    {feat:12s}: rank={rank}, r={corr:+.4f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
