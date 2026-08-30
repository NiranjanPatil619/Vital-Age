"""
VitalAge — Test YGhobara's Pre-trained Models

Downloads and tests the stacked model from:
https://github.com/YGhobara/Biological-Age-Prediction

Their features: weight, leg, arm_circ, waist, hip, SBP, DBP,
glucose, insulin, creatinine, globulin, hemoglobin, hematocrit,
total_chol, LDL, HDL, WBC, alcohol, exercise, smoking

Usage: python notebooks/blood/test/08_test_external_models.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
EXT_DIR = os.path.join(os.path.dirname(__file__), "external_models")
EXT_DATA = os.path.join(os.path.dirname(__file__), "external_models")

# ---------------------------------------------------------------------------
# 1. Load YGhobara's final_dataset.csv from GitHub
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading YGhobara's dataset and models")
print("=" * 70)

# Download their final dataset
import urllib.request
final_csv_url = "https://raw.githubusercontent.com/YGhobara/Biological-Age-Prediction/main/reports/final_dataset.csv"
final_csv_path = os.path.join(EXT_DIR, "final_dataset.csv")

if not os.path.exists(final_csv_path):
    print("  Downloading final_dataset.csv...")
    urllib.request.urlretrieve(final_csv_url, final_csv_path)

df_theirs = pd.read_csv(final_csv_path)
print(f"  Their dataset: {df_theirs.shape[0]} rows x {df_theirs.shape[1]} columns")
print(f"  Their columns: {list(df_theirs.columns)}")
print(f"  Their target (RIDAGEYR): min={df_theirs['RIDAGEYR'].min()}, max={df_theirs['RIDAGEYR'].max()}")
print(f"  Their missing values:\n{df_theirs.isnull().sum()}")

# Load models
rf_model = joblib.load(os.path.join(EXT_DIR, "stacked_random_forest.pkl"))
xgb_model = joblib.load(os.path.join(EXT_DIR, "stacked_xgboost.pkl"))
meta_model = joblib.load(os.path.join(EXT_DIR, "stacked_meta_model.pkl"))

print(f"\n  Models loaded:")
print(f"    RF: {type(rf_model).__name__}")
print(f"    XGB: {type(xgb_model).__name__}")
print(f"    Meta: {type(meta_model).__name__}")

# ---------------------------------------------------------------------------
# 2. Test on their own data (reproduce their results)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Test on their own data (reproduce results)")
print("=" * 70)

X_theirs = df_theirs.drop(columns=["RIDAGEYR"])
y_theirs = df_theirs["RIDAGEYR"]

# Check for any remaining NaN
nan_cols = X_theirs.columns[X_theirs.isnull().any()].tolist()
if nan_cols:
    print(f"  Warning: NaN in columns: {nan_cols}")
    X_theirs = X_theirs.fillna(X_theirs.median())

X_tr, X_te, y_tr, y_te = train_test_split(X_theirs, y_theirs, test_size=0.2, random_state=42)

# Stacked prediction
rf_pred = rf_model.predict(X_te)
xgb_pred = xgb_model.predict(X_te)
meta_input = np.column_stack((rf_pred, xgb_pred))
stacked_pred = meta_model.predict(meta_input)

mae_theirs = mean_absolute_error(y_te, stacked_pred)
r2_theirs = r2_score(y_te, stacked_pred)
rmse_theirs = np.sqrt(mean_squared_error(y_te, stacked_pred))

print(f"  On their own test set (20% split):")
print(f"    MAE:  {mae_theirs:.3f} years")
print(f"    RMSE: {rmse_theirs:.3f} years")
print(f"    R²:   {r2_theirs:.3f}")

# Individual model performance
rf_mae = mean_absolute_error(y_te, rf_pred)
xgb_mae = mean_absolute_error(y_te, xgb_pred)
print(f"\n    RF standalone:  MAE={rf_mae:.3f}")
print(f"    XGB standalone: MAE={xgb_mae:.3f}")
print(f"    Stacked:        MAE={mae_theirs:.3f}")

# ---------------------------------------------------------------------------
# 3. Age group breakdown
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Age group breakdown (their model on their data)")
print("=" * 70)

groups = pd.DataFrame({"true": y_te.values, "pred": stacked_pred})
groups["group"] = pd.cut(groups["true"], bins=[18, 30, 45, 60, 80], labels=["18-30", "31-45", "46-60", "61-80"])
for grp, gdf in groups.groupby("group", observed=True):
    mae_g = mean_absolute_error(gdf["true"], gdf["pred"])
    bias = gdf["pred"].mean() - gdf["true"].mean()
    print(f"  {grp}: N={len(gdf)}, MAE={mae_g:.2f}, bias={bias:+.2f}")

# ---------------------------------------------------------------------------
# 4. Try to apply to our NHANES data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Apply their model to our NHANES extended data")
print("=" * 70)

EXTENDED_PATH = os.path.join(RAW_DIR, "nhanes_extended_merged.csv")

if os.path.exists(EXTENDED_PATH):
    df_ours = pd.read_csv(EXTENDED_PATH)
    df_ours = df_ours[(df_ours["Age"] >= 18) & (df_ours["Age"] <= 80)].copy()
    print(f"  Our extended data: {df_ours.shape[0]} rows x {df_ours.shape[1]} columns")

    # Map their features to our columns
    feature_map = {
        "BMXWT": "Weight",
        "BMXHT": "Height",
        "BMXBMI": "BMI",
        "BMXLEG": None,  # We don't have leg length
        "BMXARML": None,  # We don't have arm length
        "BMXARMC": "Arm_Circumference",
        "BMXWAIST": "Waist_Circumference",
        "BMXHIP": None,  # We don't have hip circumference
        "BPXSY1": "Systolic_BP_1",
        "BPXDI1": "Diastolic_BP_1",
        "LBXGLU": "LBXGLU",
        "LBXIN": "Insulin",
        "LBXSCR": "LBXSCR",
        "LBXSGB": "LBXSGB",
        "LBXHGB": "LBXHGB",
        "LBXHCT": "LBXHCT",
        "LBXTC": "LBXTC",
        "LBDLDL": "LBDLDL",
        "LBDHDD": "LBDHDD",
        "LBXWBCSI": "LBXWBCSI",
        "ALQ130": None,  # We don't have alcohol
        "PAQ605": None,  # We don't have physical activity
        "SMQ020": None,  # We don't have smoking
    }

    # Check which features we have
    their_features = list(X_theirs.columns)
    our_available = []
    missing = []

    for feat in their_features:
        our_col = feature_map.get(feat)
        if our_col and our_col in df_ours.columns:
            our_available.append((feat, our_col))
        else:
            missing.append(feat)

    print(f"\n  Their features: {len(their_features)}")
    print(f"  We can map: {len(our_available)}")
    print(f"  Missing: {len(missing)} -> {missing}")

    if len(missing) > 0:
        print(f"\n  Cannot test their model on our data (missing {len(missing)} features)")
        print(f"  Missing features: {missing}")
        print(f"\n  We need to download: LUX (liver ultrasound), ALQ (alcohol), PAQ (exercise), SMQ (smoking)")
        print(f"  Also missing: BMXLEG (leg length), BMXARML (arm length), BMXHIP (hip circumference)")
else:
    print(f"  Extended data not found at {EXTENDED_PATH}")
    print(f"  Run 06_download_nhanes.py first")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  Their claimed MAE: ~3 years")
print(f"  Reproduced MAE:   {mae_theirs:.3f} years")
print(f"  Our current MAE:   9.50 years")
print(f"\n  Key differences:")
print(f"    - They use 20 features (body + BP + labs + lifestyle)")
print(f"    - We use 12 features (blood only)")
print(f"    - They have: weight, waist, hip, arm, leg, BP, insulin, smoking, alcohol, exercise")
print(f"    - We are missing: {missing}")
print(f"\n  To match their performance, we need to download:")
print(f"    1. LUX_J.xpt (liver ultrasound) - may have more biochemistry")
print(f"    2. ALQ_J.xpt (alcohol consumption)")
print(f"    3. PAQ_J.xpt (physical activity)")
print(f"    4. SMQ_J.xpt (smoking)")
print(f"    5. BMX body measures with leg/arm/hip")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
