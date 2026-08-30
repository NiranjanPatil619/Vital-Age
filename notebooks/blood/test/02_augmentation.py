"""
VitalAge — Data Augmentation Comparison

Compare augmentation strategies against baseline:
1. Bootstrap resampling (with replacement)
2. Gaussian Copula (manual implementation, no torch)
3. Gaussian noise injection

Each strategy generates synthetic training data, then we evaluate
XGBoost on the augmented training set using a held-out test set.

Usage: python notebooks/blood/test/02_augmentation.py
"""

import sys
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
from scipy.stats import norm
from scipy.linalg import cholesky

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

import xgboost as xgb

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Paths
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
print(f"  {df.shape[0]} rows x {df.shape[1]} columns")

target = "Age"
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols].values
y = df[target].values

# Hold-out test set (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"  Train: {len(y_train_full)} | Test: {len(y_test)}")

# ---------------------------------------------------------------------------
# 2. Baseline (no augmentation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Baseline — no augmentation")
print("=" * 70)

xgb_base = xgb.XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbosity=0,
)
xgb_base.fit(X_train_full, y_train_full)
y_pred_base = xgb_base.predict(X_test)
base_mae = mean_absolute_error(y_test, y_pred_base)
print(f"  Baseline MAE: {base_mae:.4f}")

# ---------------------------------------------------------------------------
# 3. Bootstrap augmentation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Bootstrap augmentation")
print("=" * 70)

results = {"Method": ["Baseline (no aug)"], "MAE": [base_mae],
           "Train Size": [len(y_train_full)]}

for multiplier in [2, 3, 4, 5]:
    t0 = time.time()
    n_target = len(y_train_full) * multiplier
    idx = np.random.choice(len(y_train_full), size=n_target, replace=True)
    X_boot = X_train_full[idx]
    y_boot = y_train_full[idx]

    model = clone(xgb_base)
    model.fit(X_boot, y_boot)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - t0

    results["Method"].append(f"Bootstrap {multiplier}x")
    results["MAE"].append(mae)
    results["Train Size"].append(len(y_boot))
    print(f"  {multiplier}x ({len(y_boot)} rows): MAE={mae:.4f} ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 4. Gaussian Copula augmentation (manual)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Gaussian Copula augmentation (manual)")
print("=" * 70)

# Stack features + target for copula
all_cols = feature_cols + [target]
data_all = np.column_stack([X_train_full, y_train_full])
n_features = data_all.shape[1]

# Step 1: Fit marginals (normal CDF -> normal PPF)
# Transform each column to uniform via CDF, then to Gaussian via PPF
from scipy.stats import rankdata

def fit_copula(data):
    """Fit Gaussian copula: return transformed data + fitted params."""
    n, d = data.shape
    ranks = np.zeros_like(data)
    for j in range(d):
        ranks[:, j] = rankdata(data[:, j]) / (n + 1)
    gaussian_data = norm.ppf(np.clip(ranks, 1e-6, 1 - 1e-6))
    mu = gaussian_data.mean(axis=0)
    cov = np.cov(gaussian_data, rowvar=False)
    return gaussian_data, mu, cov, data.min(axis=0), data.max(axis=0)

def sample_copula(mu, cov, mins, maxs, n_samples):
    """Sample from fitted Gaussian copula."""
    d = len(mu)
    L = cholesky(cov, lower=True)
    z = np.random.randn(n_samples, d) @ L.T + mu
    # Uniform via CDF
    u = norm.cdf(z)
    # Scale to original ranges (quantile mapping)
    samples = np.zeros_like(u)
    return u  # Return uniforms, we'll map back

def copula_generate(data_train, n_synth):
    """Generate synthetic data using Gaussian copula."""
    n, d = data_train.shape

    # Rank-based inverse CDF approach
    ranks = np.zeros_like(data_train)
    for j in range(d):
        ranks[:, j] = rankdata(data_train[:, j]) / (n + 1)

    # Fit normal to each column's ranks
    mu = np.zeros(d)
    sigma = np.ones(d)
    for j in range(d):
        mu[j] = ranks[:, j].mean()
        sigma[j] = ranks[:, j].std()

    # Fit copula correlation on Gaussian-transformed ranks
    gaussian = norm.ppf(np.clip(ranks, 1e-6, 1 - 1e-6))
    cov_matrix = np.cov(gaussian, rowvar=False)

    # Sample from multivariate normal
    L = cholesky(cov_matrix + np.eye(d) * 1e-6, lower=True)
    z = np.random.randn(n_synth, d) @ L.T + gaussian.mean(axis=0)

    # Transform back to uniform via CDF
    u = norm.cdf(z)

    # Inverse rank mapping: for each synthetic row, find nearest rank in real data
    # and use the actual values (this preserves marginal distributions exactly)
    synthetic = np.zeros((n_synth, d))
    for j in range(d):
        sorted_vals = np.sort(data_train[:, j])
        # Map uniform to index
        indices = (u[:, j] * n).astype(int)
        indices = np.clip(indices, 0, n - 1)
        synthetic[:, j] = sorted_vals[indices]

    return synthetic

for multiplier in [2, 3, 4]:
    t0 = time.time()
    n_synth = len(y_train_full) * (multiplier - 1)
    synth = copula_generate(data_all, n_synth)

    X_aug = np.vstack([X_train_full, synth[:, :-1]])
    y_aug = np.concatenate([y_train_full, synth[:, -1]])

    model = clone(xgb_base)
    model.fit(X_aug, y_aug)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - t0

    results["Method"].append(f"GaussianCopula {multiplier}x")
    results["MAE"].append(mae)
    results["Train Size"].append(len(y_aug))
    print(f"  {multiplier}x ({len(y_aug)} rows): MAE={mae:.4f} ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 5. Gaussian noise injection
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Gaussian noise injection")
print("=" * 70)

for noise_frac in [0.01, 0.02, 0.05, 0.10]:
    for multiplier in [2, 4]:
        t0 = time.time()
        n_target = len(y_train_full) * multiplier
        idx = np.random.choice(len(y_train_full), size=n_target - len(y_train_full), replace=True)
        X_noisy = X_train_full[idx].copy()
        y_noisy = y_train_full[idx].copy()

        # Add noise proportional to feature std
        feat_std = X_train_full.std(axis=0)
        noise = np.random.randn(*X_noisy.shape) * feat_std * noise_frac
        X_noisy += noise

        X_aug = np.vstack([X_train_full, X_noisy])
        y_aug = np.concatenate([y_train_full, y_noisy])

        model = clone(xgb_base)
        model.fit(X_aug, y_aug)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0

        results["Method"].append(f"Noise {noise_frac:.0%} {multiplier}x")
        results["MAE"].append(mae)
        results["Train Size"].append(len(y_aug))
        print(f"  Noise {noise_frac:.0%} {multiplier}x ({len(y_aug)} rows): MAE={mae:.4f} ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

res_df = pd.DataFrame(results).sort_values("MAE")
res_df["Delta_vs_Baseline"] = res_df["MAE"] - base_mae
print(res_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

ax = axes[0]
colors = ["#2ecc71" if d < -0.01 else "#e74c3c" if d > 0.01 else "#95a5a6"
          for d in res_df.Delta_vs_Baseline]
bars = ax.barh(res_df.Method, res_df.MAE, color=colors, edgecolor="white")
ax.set_xlabel("MAE (years)")
ax.set_title("Augmentation Comparison — XGBoost MAE (lower = better)")
ax.axvline(x=base_mae, color="black", linestyle="--", lw=1.5, label=f"Baseline ({base_mae:.3f})")
ax.legend()
for bar, val in zip(bars, res_df.MAE):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)

ax = axes[1]
# Show only unique multipliers (skip noise for scatter)
scatter_df = res_df[~res_df.Method.str.contains("Noise")].copy()
scatter_df["Multiplier"] = scatter_df.Method.str.extract(r'(\d+)x').astype(float)
for method_type in scatter_df.Method.str.split(" ").str[0].unique():
    sub = scatter_df[scatter_df.Method.str.startswith(method_type)]
    ax.scatter(sub["Train Size"], sub["MAE"], s=100, label=method_type, edgecolors="black")
    for _, row in sub.iterrows():
        mult = row["Multiplier"]
        if not np.isnan(mult):
            ax.annotate(f'{int(mult)}x', (row["Train Size"], row["MAE"]),
                        textcoords="offset points", xytext=(8, 3), fontsize=9)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("MAE (years)")
ax.set_title("Training Size vs MAE")
ax.legend()

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "18_augmentation_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 8. Copula synthetic data quality
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SYNTHETIC DATA QUALITY (Gaussian Copula 3x)")
print("=" * 70)

n_synth_check = len(y_train_full) * 2
synth_check = copula_generate(data_all, n_synth_check)

for j, col in enumerate(all_cols):
    real = data_all[:, j]
    synth = synth_check[:, j]
    print(f"  {col:12s}: real({real.mean():7.2f} +/- {real.std():6.2f}) "
          f"synth({synth.mean():7.2f} +/- {synth.std():6.2f}) "
          f"diff={abs(real.mean()-synth.mean()):.4f}")

# Correlation comparison
print("\n  Correlation matrix differences (|real - synth|):")
corr_real = np.corrcoef(data_all.T)
corr_synth = np.corrcoef(synth_check.T)
corr_diff = np.abs(corr_real - corr_synth)
avg_diff = corr_diff[np.triu_indices_from(corr_diff, k=1)].mean()
max_diff = corr_diff[np.triu_indices_from(corr_diff, k=1)].max()
print(f"  Mean abs diff: {avg_diff:.4f}")
print(f"  Max abs diff: {max_diff:.4f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
