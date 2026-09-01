"""
VitalAge — XGBoost trainer (new structure)

Reads: data/processed/test2_mae67_training_data.csv (77 feats, 59k) else bioage_final_clean.csv (12 feats)
Saves: model/xgboost/xgb_model.pkl + data/reports/metrics/xgb_metrics.txt
GPU: device=cuda if available else hist
"""
from pathlib import Path
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES=[ROOT/"data/processed/test2_mae67_training_data.csv", ROOT/"data/processed/test2_training_full.csv", ROOT/"data/processed/bioage_final_clean.csv"]
DATA_PATH=next((p for p in CANDIDATES if p.exists()), CANDIDATES[-1])
MODEL_DIR=ROOT/"model/xgboost"
REPORT_DIR=ROOT/"data/reports/metrics"
FIG_DIR=ROOT/"data/reports/figures"
for d in [MODEL_DIR, REPORT_DIR, FIG_DIR]: d.mkdir(parents=True, exist_ok=True)

print("="*70); print("TRAIN — XGBOOST"); print("="*70)
print(f"Dataset {DATA_PATH} -> {DATA_PATH.exists()}")
df=pd.read_csv(DATA_PATH)
TARGET="Age"
FEATURES=[c for c in df.columns if c!=TARGET]
print(f"Shape {df.shape} feats {len(FEATURES)} Age {df[TARGET].min():.0f}-{df[TARGET].max():.0f}")
X,y=df[FEATURES].copy(), df[TARGET].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Train {len(X_train)} Test {len(X_test)}")

# GPU detection
try:
    import subprocess
    subprocess.check_output(["nvidia-smi"],stderr=subprocess.DEVNULL)
    has_cuda=True
except: has_cuda=False

base=dict(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0, tree_method="hist", device="cuda" if has_cuda else "cpu")
if not has_cuda: base.pop("device",None)
model=xgb.XGBRegressor(**base)
print(f"\nTraining {base} ...")
model.fit(X_train,y_train, verbose=False)
print("Done")
pred=model.predict(X_test)
mae=mean_absolute_error(y_test,pred); rmse=np.sqrt(mean_squared_error(y_test,pred)); r2=r2_score(y_test,pred)
r,pval=pearsonr(y_test,pred)
print(f"TEST MAE {mae:.4f} RMSE {rmse:.4f} R2 {r2:.4f} r {r:.4f}")

kf=KFold(5,shuffle=True,random_state=42)
cv=-cross_val_score(model,X_train,y_train,cv=kf,scoring="neg_mean_absolute_error",n_jobs=1)
print(f"CV MAE {cv.mean():.4f} +/- {cv.std():.4f}")

imp=pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 8:")
print(imp.head(8).to_string())

# Save
joblib.dump(model, MODEL_DIR/"xgb_model.pkl")
try: model.save_model(str(MODEL_DIR/"xgb_model.json"))
except: pass
imp.to_csv(REPORT_DIR/"xgb_feature_importance.csv")
pd.DataFrame({"y_true":y_test,"y_pred":pred}).to_csv(REPORT_DIR/"xgb_predictions.csv",index=False)
with open(REPORT_DIR/"xgb_metrics.txt","w") as f:
    f.write(f"XGBoost MAE {mae:.4f} RMSE {rmse:.4f} R2 {r2:.4f} r {r:.4f} CV {cv.mean():.4f}+/-{cv.std():.4f}\n")
    f.write(f"Params {base}\nDataset {DATA_PATH.name} {df.shape}\n")

plt.figure(figsize=(7,5)); plt.scatter(y_test,pred,alpha=0.25,s=10); mn=min(y_test.min(),pred.min()); mx=max(y_test.max(),pred.max()); plt.plot([mn,mx],[mn,mx],"r--"); plt.xlabel("True Age"); plt.ylabel("Predicted"); plt.title(f"XGB Actual vs Pred (MAE {mae:.2f})"); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(FIG_DIR/"xgb_actual_vs_predicted.png",dpi=150); plt.close()
plt.figure(figsize=(8,5)); imp.head(12).sort_values().plot(kind="barh"); plt.xlabel("Importance"); plt.title("XGB Feature Importance"); plt.tight_layout(); plt.savefig(FIG_DIR/"xgb_feature_importance.png",dpi=150); plt.close()
print(f"Saved {MODEL_DIR/'xgb_model.pkl'}")
