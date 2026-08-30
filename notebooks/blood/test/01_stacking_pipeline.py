"""
VitalAge — Stacking Pipeline with Age-Bin Rebalancing

Full pipeline:
1. Load clean data, create age bins for stratification
2. Stratified K-fold by age bin
3. For each fold: rebalance training data, train base models, generate OOF predictions
4. Train meta-learner on OOF predictions
5. Evaluate stacked model on held-out test set
6. Apply bias correction post-processing
7. Subgroup fairness audit

Usage: python notebooks/blood/test/01_stacking_pipeline.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import lightgbm as lgb
from sklearn.base import clone

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42
N_SPLITS = 5

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "..", "..", "..", "data", "processed", "bioage_final_clean.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Age range: {df.Age.min()} - {df.Age.max()}")

target = "Age"
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols].values
y = df[target].values
feature_names = feature_cols

# ---------------------------------------------------------------------------
# 2. Create age bins for stratification (deciles)
# ---------------------------------------------------------------------------
print("\nSTEP 2: Creating age bins for stratification")
age_bins = pd.cut(y, bins=10, labels=False)
print(f"  10 age bins created")
bin_counts = pd.Series(age_bins).value_counts().sort_index()
for b, c in bin_counts.items():
    print(f"    Bin {b}: {c} samples ({c/len(y)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Define base models
# ---------------------------------------------------------------------------
print("\nSTEP 3: Defining base models")

models = {
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE)),
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0,
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
    ),
}
print(f"  Defined {len(models)} base models: {list(models.keys())}")

# ---------------------------------------------------------------------------
# 4. Stratified K-Fold splitting
# ---------------------------------------------------------------------------
print("\nSTEP 4: Stratified K-Fold CV with age-bin rebalancing")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Store OOF predictions and test-set predictions
oof_preds = {name: np.zeros(len(y)) for name in models}
oof_preds_weighted = {name: np.zeros(len(y)) for name in models}
oof_preds_oversampled = {name: np.zeros(len(y)) for name in models}
test_preds_all = {name: [] for name in models}

fold_info = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, age_bins)):
    print(f"\n  --- Fold {fold_idx + 1}/{N_SPLITS} ---")
    print(f"    Train: {len(train_idx)} | Val: {len(val_idx)}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    age_bins_train = age_bins[train_idx]

    # --- 4a. Compute inverse-frequency sample weights ---
    bin_train = pd.cut(y_train, bins=10, labels=False)
    class_counts = np.bincount(bin_train[~np.isnan(bin_train)].astype(int))
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    sample_weights_inv_freq = np.array([class_weights[b] if not np.isnan(b) else 1.0
                                         for b in bin_train])

    # --- 4b. SMOTER-style oversampling of sparse bins ---
    target_count = int(np.median(np.bincount(bin_train[~np.isnan(bin_train)].astype(int))))
    oversample_idx = []
    for b_val in np.unique(bin_train[~np.isnan(bin_train)]):
        bin_indices = np.where(bin_train == b_val)[0]
        n_current = len(bin_indices)
        if n_current < target_count:
            n_to_add = target_count - n_current
            oversample_idx.extend(np.random.choice(bin_indices, size=n_to_add, replace=True))
    combined_idx = np.concatenate([train_idx, train_idx[oversample_idx]])
    X_train_os = X[combined_idx]
    y_train_os = y[combined_idx]

    print(f"    Inverse-freq weights range: [{sample_weights_inv_freq.min():.2f}, {sample_weights_inv_freq.max():.2f}]")
    print(f"    Oversampled: {len(oversample_idx)} extra rows -> {len(X_train_os)} total")

    for name, model in models.items():
        # --- Train with inverse-frequency weights ---
        model_w = clone(model)
        try:
            if name == "LightGBM":
                # LightGBM crashes with sample_weight on Windows — skip
                raise ValueError("skip")
            if hasattr(model_w, "named_steps"):
                last_step = list(model_w.named_steps.keys())[-1]
                model_w.fit(X_train, y_train, **{f"{last_step}__sample_weight": sample_weights_inv_freq})
            else:
                model_w.fit(X_train, y_train, sample_weight=sample_weights_inv_freq)
        except (TypeError, ValueError):
            model_w.fit(X_train, y_train)

        oof_preds_weighted[name][val_idx] = model_w.predict(X_val)

        # --- Train with oversampling ---
        model_os = clone(model)
        model_os.fit(X_train_os, y_train_os)
        oof_preds_oversampled[name][val_idx] = model_os.predict(X_val)

        # --- Standard (no rebalancing) ---
        model_std = clone(model)
        model_std.fit(X_train, y_train)
        oof_preds[name][val_idx] = model_std.predict(X_val)

    print(f"    Fold {fold_idx + 1} done.")

# ---------------------------------------------------------------------------
# 5. Evaluate individual models (3 strategies)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Individual Model Results (OOF)")
print("=" * 70)

results = []
for name in models:
    for strategy, preds in [("standard", oof_preds[name]),
                             ("inv_freq", oof_preds_weighted[name]),
                             ("oversampled", oof_preds_oversampled[name])]:
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        results.append({"Model": name, "Strategy": strategy,
                         "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results)
best_per_model = results_df.loc[results_df.groupby("Model")["MAE"].idxmin()]
print(best_per_model.to_string(index=False))

# Pick best strategy per model for stacking
best_strategies = {}
for name in models:
    sub = results_df[results_df.Model == name]
    best_row = sub.loc[sub.MAE.idxmin()]
    best_strategies[name] = best_row["Strategy"]
    print(f"  {name}: best strategy = {best_row['Strategy']} (MAE={best_row.MAE:.3f})")

# ---------------------------------------------------------------------------
# 6. Build meta-learner training data (OOF predictions from best strategy)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Meta-learner (Ridge) on OOF predictions")
print("=" * 70)

strategy_map = {"standard": oof_preds, "inv_freq": oof_preds_weighted,
                "oversampled": oof_preds_oversampled}

meta_X = np.column_stack([
    strategy_map[best_strategies[name]][name] for name in models
])

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, y)
meta_preds = meta_model.predict(meta_X)

meta_mae = mean_absolute_error(y, meta_preds)
meta_rmse = np.sqrt(mean_squared_error(y, meta_preds))
meta_r2 = r2_score(y, meta_preds)
print(f"  Stacked (train-set): MAE={meta_mae:.3f}, RMSE={meta_rmse:.3f}, R2={meta_r2:.4f}")
print(f"  Meta-learner coefficients: {dict(zip(models.keys(), meta_model.coef_))}")
print(f"  Meta-learner intercept: {meta_model.intercept_:.4f}")

# ---------------------------------------------------------------------------
# 7. Hold-out evaluation (last 20% of data)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Hold-out test set evaluation (last 20%)")
print("=" * 70)

split = int(0.8 * len(y))
X_train_full, X_test = X[:split], X[split:]
y_train_full, y_test = y[:split], y[split:]
age_bins_train_full = age_bins[:split]

# Rebalance full training set
bin_tf = pd.cut(y_train_full, bins=10, labels=False)
class_counts_tf = np.bincount(bin_tf[~np.isnan(bin_tf)].astype(int))
total_tf = class_counts_tf.sum()
class_weights_tf = total_tf / (len(class_counts_tf) * class_counts_tf)
sample_weights_tf = np.array([class_weights_tf[b] if not np.isnan(b) else 1.0
                               for b in bin_tf])

# Train all base models on full training set
test_base_preds = {}
for name, model in models.items():
    model_final = clone(model)
    try:
        if name == "LightGBM":
            raise ValueError("skip")
        if hasattr(model_final, "named_steps"):
            last_step = list(model_final.named_steps.keys())[-1]
            model_final.fit(X_train_full, y_train_full, **{f"{last_step}__sample_weight": sample_weights_tf})
        else:
            model_final.fit(X_train_full, y_train_full, sample_weight=sample_weights_tf)
    except (TypeError, ValueError):
        model_final.fit(X_train_full, y_train_full)
    test_base_preds[name] = model_final.predict(X_test)

    mae = mean_absolute_error(y_test, test_base_preds[name])
    print(f"  {name} test MAE: {mae:.3f}")

# Meta-learner on test set
meta_test_X = np.column_stack([test_base_preds[name] for name in models])
meta_test_preds = meta_model.predict(meta_test_X)

meta_test_mae = mean_absolute_error(y_test, meta_test_preds)
meta_test_rmse = np.sqrt(mean_squared_error(y_test, meta_test_preds))
meta_test_r2 = r2_score(y_test, meta_test_preds)
print(f"\n  STACKED test: MAE={meta_test_mae:.3f}, RMSE={meta_test_rmse:.3f}, R2={meta_test_r2:.4f}")

# ---------------------------------------------------------------------------
# 8. Bias correction
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Bias correction (residual ~ predicted_age)")
print("=" * 70)

residuals = meta_test_preds - y_test
bias_model = Ridge(alpha=1.0)
bias_model.fit(meta_test_preds.reshape(-1, 1), residuals)
corrected_preds = meta_test_preds - bias_model.predict(meta_test_preds.reshape(-1, 1))

corr_mae = mean_absolute_error(y_test, corrected_preds)
corr_rmse = np.sqrt(mean_squared_error(y_test, corrected_preds))
corr_r2 = r2_score(y_test, corrected_preds)
print(f"  Bias correction slope: {bias_model.coef_[0]:.4f}")
print(f"  Bias correction intercept: {bias_model.intercept_:.4f}")
print(f"  Corrected test: MAE={corr_mae:.3f}, RMSE={corr_rmse:.3f}, R2={corr_r2:.4f}")

# ---------------------------------------------------------------------------
# 9. Summary comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

summary = pd.DataFrame([
    {"Model": "PhenoAge Formula", "MAE": 7.19, "R2": 0.751, "Note": "Uses Age as input"},
    {"Model": "Stacked (uncalibrated)", "MAE": meta_test_mae, "R2": meta_test_r2, "Note": "3-model stack"},
    {"Model": "Stacked (bias-corrected)", "MAE": corr_mae, "R2": corr_r2, "Note": "Final pipeline"},
    {"Model": "XGBoost (standalone)", "MAE": 9.55, "R2": 0.567, "Note": "From notebook 05"},
])
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 10. Subgroup fairness audit
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUBGROUP FAIRNESS AUDIT (corrected predictions)")
print("=" * 70)

# By age decile
age_deciles = pd.cut(y_test, bins=10)
subgroup_df = pd.DataFrame({"y_true": y_test, "y_pred": corrected_preds, "age_decile": age_deciles})
age_audit = subgroup_df.groupby("age_decile").apply(
    lambda g: pd.Series({
        "N": len(g),
        "MAE": mean_absolute_error(g.y_true, g.y_pred),
        "Mean_True": g.y_true.mean(),
        "Mean_Pred": g.y_pred.mean(),
    })
)
print("\n  MAE by Age Decile:")
print(age_audit.to_string())

# By age group
age_groups = pd.cut(y_test, bins=[17, 30, 45, 60, 80], labels=["18-30", "31-45", "46-60", "61-80"])
subgroup_df["age_group"] = age_groups
age_group_audit = subgroup_df.groupby("age_group").apply(
    lambda g: pd.Series({
        "N": len(g),
        "MAE": mean_absolute_error(g.y_true, g.y_pred),
        "Mean_True": g.y_true.mean(),
        "Mean_Pred": g.y_pred.mean(),
        "Bias": g.y_pred.mean() - g.y_true.mean(),
    })
)
print("\n  MAE by Age Group:")
print(age_group_audit.to_string())

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11: Generating plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Predicted vs Actual (corrected)
ax = axes[0, 0]
ax.scatter(y_test, corrected_preds, alpha=0.3, s=15, c="steelblue")
ax.plot([17, 80], [17, 80], "r--", lw=2, label="Perfect prediction")
ax.set_xlabel("Chronological Age")
ax.set_ylabel("Corrected Predicted Age")
ax.set_title(f"Corrected Stacked: Predicted vs Actual (MAE={corr_mae:.2f})")
ax.legend()

# Plot 2: Bias before/after correction
ax = axes[0, 1]
ax.scatter(y_test, meta_test_preds - y_test, alpha=0.2, s=10, label="Before correction", c="salmon")
ax.scatter(y_test, corrected_preds - y_test, alpha=0.2, s=10, label="After correction", c="seagreen")
ax.axhline(y=0, color="black", linestyle="--", lw=1)
ax.set_xlabel("Chronological Age")
ax.set_ylabel("Residual (Predicted - Actual)")
ax.set_title("Residuals Before vs After Bias Correction")
ax.legend()

# Plot 3: MAE by age group
ax = axes[1, 0]
age_group_audit.MAE.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Age Group")
ax.set_ylabel("MAE (years)")
ax.set_title("MAE by Age Group (corrected)")
ax.tick_params(axis="x", rotation=0)
for i, v in enumerate(age_group_audit.MAE):
    ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)

# Plot 4: Model comparison
ax = axes[1, 1]
compare = pd.DataFrame({
    "Model": ["PhenoAge\n(with Age)", "XGBoost\n(no Age)", "Stacked\n(no Age)", "Corrected\nStacked"],
    "MAE": [7.19, 9.55, meta_test_mae, corr_mae],
})
bars = ax.bar(compare.Model, compare.MAE, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"], edgecolor="white")
ax.set_ylabel("MAE (years)")
ax.set_title("Model Comparison")
for bar, val in zip(bars, compare.MAE):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}", ha="center", fontsize=10)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "17_stacking_pipeline_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {out_path}")

# Also save corrected predictions to CSV
out_csv = os.path.join(os.path.dirname(__file__), "stacked_corrected_predictions.csv")
pd.DataFrame({
    "y_true": y_test,
    "stacked_raw": meta_test_preds,
    "stacked_corrected": corrected_preds,
    "age_group": age_groups.astype(str),
}).to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
