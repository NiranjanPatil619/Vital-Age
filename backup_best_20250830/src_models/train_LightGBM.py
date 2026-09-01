import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()

markdown_intro = """# Model Comparison and Training

This notebook compares the Baseline (PhenoAge formula), Ridge Regression, LightGBM, and a Neural Network (MLP) for predicting chronological age (as a proxy for biological age) from clinical blood biomarkers.

We will use 5-fold cross-validation and evaluate models based on MAE, RMSE, and R²."""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import pickle
import os

import warnings
warnings.filterwarnings('ignore')"""

code_load_data = """# Load the model-ready dataset
df = pd.read_csv('../../data/processed/bioage_final_clean.csv')

# PhenoAge formula features (as baseline)
phenoage_features = ['LBXSAL', 'LBXSCR', 'LBXGLU', 'CRP', 'LBXLYPCT', 
                     'LBXMCVSI', 'LBXRDW', 'LBXSAPSI', 'LBXWBCSI']

# All features (excluding log transforms and age_group since they are in the other file)
# The config states 12 features.
features = ['LBXSAL', 'LBXSCR', 'LBXGLU', 'CRP', 'LBXLYPCT', 'LBXMCVSI', 'LBXRDW',
            'LBXSAPSI', 'LBXWBCSI', 'LBXGH', 'LBDHDD', 'LBXTC']

X = df[features]
y = df['Age']

print(f"Dataset shape: {df.shape}")
print(f"Features used: {len(features)}")
"""

markdown_baseline = """## 1. Baseline: PhenoAge Formula
The original PhenoAge formula is a specific linear combination of 9 biomarkers and Age, but here we can train a simple Ridge regression on just the 9 core features to serve as our baseline."""

code_baseline = """# Baseline using only the 9 core PhenoAge features
X_baseline = df[phenoage_features]

baseline_model = Ridge(alpha=1.0)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results_baseline = cross_validate(baseline_model, X_baseline, y, cv=cv,
                                     scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'))

print("Baseline (Ridge on 9 Core Features) CV Results:")
print(f"MAE:  {-cv_results_baseline['test_neg_mean_absolute_error'].mean():.3f} +/- {cv_results_baseline['test_neg_mean_absolute_error'].std():.3f}")
print(f"RMSE: {-cv_results_baseline['test_neg_root_mean_squared_error'].mean():.3f} +/- {cv_results_baseline['test_neg_root_mean_squared_error'].std():.3f}")
print(f"R²:   {cv_results_baseline['test_r2'].mean():.3f} +/- {cv_results_baseline['test_r2'].std():.3f}")"""


markdown_ridge = """## 2. Ridge Regression (All Features)"""

code_ridge = """# Ridge regression on all features
ridge_model = Ridge(alpha=1.0)

cv_results_ridge = cross_validate(ridge_model, X, y, cv=cv,
                                  scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'))

print("Ridge Regression CV Results:")
print(f"MAE:  {-cv_results_ridge['test_neg_mean_absolute_error'].mean():.3f} +/- {cv_results_ridge['test_neg_mean_absolute_error'].std():.3f}")
print(f"RMSE: {-cv_results_ridge['test_neg_root_mean_squared_error'].mean():.3f} +/- {cv_results_ridge['test_neg_root_mean_squared_error'].std():.3f}")
print(f"R²:   {cv_results_ridge['test_r2'].mean():.3f} +/- {cv_results_ridge['test_r2'].std():.3f}")"""


markdown_lgb = """## 3. LightGBM (Gradient Boosting) with Hyperparameter Tuning
To minimize the MAE specifically, we will change LightGBM's objective function to 'mae' (L1 loss) and use RandomizedSearchCV to find the best hyperparameters."""

code_lgb = """from sklearn.model_selection import RandomizedSearchCV

# Base LightGBM model optimizing for MAE
base_lgb = lgb.LGBMRegressor(objective='mae', random_state=42, verbose=-1, n_jobs=-1)

# Parameter grid to search
param_distributions = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 7, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Randomized search for best parameters
print("Starting Hyperparameter Tuning for LightGBM (Optimizing for MAE)...")
random_search = RandomizedSearchCV(
    estimator=base_lgb, 
    param_distributions=param_distributions, 
    n_iter=10, 
    scoring='neg_mean_absolute_error', 
    cv=3, # 3-fold to speed up tuning
    random_state=42,
    verbose=1
)
random_search.fit(X, y)

best_lgb = random_search.best_estimator_
print(f"Best Parameters found: {random_search.best_params_}")

# Evaluate the tuned model with 5-fold CV to compare fairly with others
cv_results_lgb = cross_validate(best_lgb, X, y, cv=cv,
                                scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'))

print("\\nTuned LightGBM CV Results:")
print(f"MAE:  {-cv_results_lgb['test_neg_mean_absolute_error'].mean():.3f} +/- {cv_results_lgb['test_neg_mean_absolute_error'].std():.3f}")
print(f"RMSE: {-cv_results_lgb['test_neg_root_mean_squared_error'].mean():.3f} +/- {cv_results_lgb['test_neg_root_mean_squared_error'].std():.3f}")
print(f"R²:   {cv_results_lgb['test_r2'].mean():.3f} +/- {cv_results_lgb['test_r2'].std():.3f}")"""


markdown_nn = """## 4. Neural Network (MLP)
We include a simple Multi-Layer Perceptron to see if a neural network architecture can outperform tree-based methods on this tabular dataset. MLP often requires careful tuning and scaling."""

code_nn = """# Multi-Layer Perceptron (Neural Network)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True))
])

cv_results_nn = cross_validate(mlp_pipeline, X, y, cv=cv,
                               scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'))

print("Neural Network (MLP) CV Results:")
print(f"MAE:  {-cv_results_nn['test_neg_mean_absolute_error'].mean():.3f} +/- {cv_results_nn['test_neg_mean_absolute_error'].std():.3f}")
print(f"RMSE: {-cv_results_nn['test_neg_root_mean_squared_error'].mean():.3f} +/- {cv_results_nn['test_neg_root_mean_squared_error'].std():.3f}")
print(f"R²:   {cv_results_nn['test_r2'].mean():.3f} +/- {cv_results_nn['test_r2'].std():.3f}")"""

markdown_final = """## 5. Final Model: Ensemble (Stacking)
To get the absolute best performance (lower MAE, lower RMSE, and higher R²), we will combine our models into a **Stacking Regressor**. 
This "super model" uses the tuned LightGBM, the Neural Network, and Ridge Regression as base learners, and a final meta-model to intelligently blend their predictions."""

code_final = """# Define the base models for the ensemble
estimators = [
    ('lgb', best_lgb),
    ('mlp', mlp_pipeline),
    ('ridge', Ridge(alpha=1.0))
]

# Create the Stacking Regressor
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=-1
)

print("Evaluating the Ensemble Model...")
cv_results_stack = cross_validate(stacking_model, X, y, cv=cv,
                                  scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'))

print("\\nEnsemble CV Results:")
print(f"MAE:  {-cv_results_stack['test_neg_mean_absolute_error'].mean():.3f} +/- {cv_results_stack['test_neg_mean_absolute_error'].std():.3f}")
print(f"RMSE: {-cv_results_stack['test_neg_root_mean_squared_error'].mean():.3f} +/- {cv_results_stack['test_neg_root_mean_squared_error'].std():.3f}")
print(f"R²:   {cv_results_stack['test_r2'].mean():.3f} +/- {cv_results_stack['test_r2'].std():.3f}")

# Train final ensemble on full dataset
print("\\nTraining final ensemble on 100% of the data...")
final_model = stacking_model
final_model.fit(X, y)

# Predict Biological Age
predicted_bioage = final_model.predict(X)

# Compute BioAge Gap
df['Predicted_BioAge'] = predicted_bioage
df['BioAge_Gap'] = df['Predicted_BioAge'] - df['Age']

# Save the model
os.makedirs('../../models/blood', exist_ok=True)
model_path = '../../models/blood/bioage_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)

print(f"Model successfully saved to {model_path}")

# Display a few examples
df[['Age', 'Predicted_BioAge', 'BioAge_Gap']].head(10)"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_markdown_cell(markdown_baseline),
    nbf.v4.new_code_cell(code_baseline),
    nbf.v4.new_markdown_cell(markdown_ridge),
    nbf.v4.new_code_cell(code_ridge),
    nbf.v4.new_markdown_cell(markdown_lgb),
    nbf.v4.new_code_cell(code_lgb),
    nbf.v4.new_markdown_cell(markdown_nn),
    nbf.v4.new_code_cell(code_nn),
    nbf.v4.new_markdown_cell(markdown_final),
    nbf.v4.new_code_cell(code_final)
]

os.makedirs('notebooks/blood', exist_ok=True)
with open('notebooks/blood/05_model_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
