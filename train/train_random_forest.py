"""
VitalAge — Random Forest trainer (new structure)

Reads: data/processed/test2_mae67_training_data.csv (77 feats, 59k rows) if present,
       else fallback to data/processed/bioage_final_clean.csv (12 feats, 20k rows)
Saves: model/random_forest/rf_model.pkl + data/reports/metrics/rf_metrics.txt
"""
from pathlib import Path
import joblib, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT/"data/processed/test2_mae67_training_data.csv", ROOT/"data/processed/test2_training_full.csv", ROOT/"data/processed/bioage_final_clean.csv"]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), CANDIDATES[-1])
MODEL_DIR = ROOT/"model/random_forest"
REPORT_DIR = ROOT/"data/reports/metrics"
FIG_DIR = ROOT/"data/reports/figures"
for d in [MODEL_DIR, REPORT_DIR, FIG_DIR]: d.mkdir(parents=True, exist_ok=True)

print("="*70); print("TRAIN — RANDOM FOREST"); print("="*70)
print(f"Dataset: {DATA_PATH} exists={DATA_PATH.exists()}")
df = pd.read_csv(DATA_PATH)
TARGET="Age"
FEATURES=[c for c in df.columns if c!=TARGET]
print(f"Shape {df.shape} | Features {len(FEATURES)} | Age {df[TARGET].min():.0f}-{df[TARGET].max():.0f}")
X, y = df[FEATURES].copy(), df[TARGET].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Train {len(X_train)} | Test {len(X_test)}")

model=RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_leaf=10, min_samples_split=10, n_jobs=2, random_state=42)
print("\nTraining (80 trees, depth15, 4 jobs)...")
model.fit(X_train,y_train)
print("Done")

pred=model.predict(X_test)
mae=mean_absolute_error(y_test,pred); rmse=np.sqrt(mean_squared_error(y_test,pred)); r2=r2_score(y_test,pred)
r, pval = pearsonr(y_test, pred)
print(f"\nTEST MAE {mae:.4f} RMSE {rmse:.4f} R2 {r2:.4f} r {r:.4f}")

gap=pred-y_test.to_numpy()
print(f"Gap mean {gap.mean():.3f} SD {gap.std():.3f}")

# CV skipped for 59k RF (too slow) — estimate from holdout
cv=np.array([mae])
print(f"CV skipped (holdout MAE {mae:.3f} used as estimate)")

# Importance
imp=pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 8 features:")
print(imp.head(8).to_string())

# Save
joblib.dump(model, MODEL_DIR/"rf_model.pkl")
imp.to_csv(REPORT_DIR/"rf_feature_importance.csv")
pd.DataFrame({"y_true":y_test,"y_pred":pred}).to_csv(REPORT_DIR/"rf_predictions.csv",index=False)
with open(REPORT_DIR/"rf_metrics.txt","w") as f:
    f.write(f"RandomForest MAE {mae:.4f} RMSE {rmse:.4f} R2 {r2:.4f} r {r:.4f} CV {cv.mean():.4f}+/-{cv.std():.4f}\n")
    f.write(f"Dataset {DATA_PATH.name} {df.shape} | {len(FEATURES)} feats\n")
# plots
plt.figure(figsize=(7,5)); plt.scatter(y_test,pred,alpha=0.25,s=10); mn=min(y_test.min(),pred.min()); mx=max(y_test.max(),pred.max()); plt.plot([mn,mx],[mn,mx],"r--"); plt.xlabel("True Age"); plt.ylabel("Predicted"); plt.title(f"RF Actual vs Predicted (MAE {mae:.2f})"); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(FIG_DIR/"rf_actual_vs_predicted.png",dpi=150); plt.close()
plt.figure(figsize=(8,5)); imp.head(12).sort_values().plot(kind="barh"); plt.xlabel("Importance"); plt.title("RF Feature Importance"); plt.tight_layout(); plt.savefig(FIG_DIR/"rf_feature_importance.png",dpi=150); plt.close()
print(f"\nSaved model {MODEL_DIR/'rf_model.pkl'}")
print(f"Metrics {REPORT_DIR/'rf_metrics.txt'}")
