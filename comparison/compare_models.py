"""
VitalAge — Model Comparison (new structure) — FAIR only (no overfit)

Compares: RandomForest (model/random_forest/rf_model.pkl),
          XGBoost (model/xgboost/xgb_model.pkl),
          LightGBM (model/lightgbm/lgb_model.pkl)
          — BEST (model/best/xgb_model.pkl) is FULL-FIT overfit (5.53 on same split), excluded from fair ranking.
          Use model/best only for deployment; its honest MAE is 6.67 (XGB new).

Reads: data/processed/test2_mae67_training_data.csv else bioage_final_clean.csv
Outputs: data/reports/figures/comparison_*
         data/reports/comparison_summary.csv/.txt + console BEST verdict
"""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES=[ROOT/"data/processed/test2_mae67_training_data.csv", ROOT/"data/processed/test2_training_full.csv", ROOT/"data/processed/bioage_final_clean.csv"]
DATA_PATH=next((p for p in CANDIDATES if p.exists()), CANDIDATES[-1])
REPORT_DIR=ROOT/"data/reports"
METRICS_DIR=REPORT_DIR/"metrics"
FIG_DIR=REPORT_DIR/"figures"
for d in [REPORT_DIR, METRICS_DIR, FIG_DIR]: d.mkdir(parents=True, exist_ok=True)

print("="*70); print("COMPARISON — ALL BENCHMARKS"); print("="*70)
print(f"Dataset {DATA_PATH} {DATA_PATH.stat().st_size/1024/1024:.1f}MB")
df=pd.read_csv(DATA_PATH)
TARGET="Age"
FEATURES=[c for c in df.columns if c!=TARGET]
print(f"Shape {df.shape} feats {len(FEATURES)}")

# Fixed split for fair comparison (same as train scripts)
X, y = df[FEATURES].copy(), df[TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Holdout {len(X_test)} (20%) fixed random_state=42")

# Discover models — FAIR only (exclude BEST overfit, exclude legacy)
candidates={
    "RF (new)": ROOT/"model/random_forest/rf_model.pkl",
    "XGB (new)": ROOT/"model/xgboost/xgb_model.pkl",
    "LGB (new)": ROOT/"model/lightgbm/lgb_model.pkl",
}
# Note: BEST (model/best/xgb_model.pkl, 5.53 leakage) intentionally excluded from fair comparison.
# Legacy models excluded — they were 12-feat and not comparable on 77-feat data.

results=[]
preds={}

for name, path in list(candidates.items()):
    if not path.exists():
        continue
    # Handle feature mismatch: legacy 12-feat models can't run on 77-feat data
    try:
        m=joblib.load(path)
        # Try predict; if feature mismatch, filter to model's expected features
        try:
            y_pred=m.predict(X_test)
        except Exception as e:
            # Check if model has feature_names_in_
            exp = getattr(m, "feature_names_in_", None)
            if exp is not None:
                missing=[c for c in exp if c not in X_test.columns]
                if missing:
                    print(f"  {name}: skipped (needs {missing[:3]} not in dataset {DATA_PATH.name} — train on 77-feat to compare)")
                    continue
                y_pred=m.predict(X_test[exp])
            else:
                # fallback: try first len matches (e.g., 12)
                print(f"  {name}: predict failed {e}, trying intersection")
                # intersect
                try:
                    inter=[c for c in FEATURES if c in getattr(m, "feature_names_in_", FEATURES)][:12]
                    y_pred=m.predict(X_test[inter])
                except Exception as e2:
                    print(f"    still fail {e2}")
                    continue
        mae=mean_absolute_error(y_test,y_pred)
        rmse=np.sqrt(mean_squared_error(y_test,y_pred))
        r2=r2_score(y_test,y_pred)
        r,_=pearsonr(y_test,y_pred)
        gap=y_pred-y_test.to_numpy()
        results.append({"Model":name, "Path":str(path), "MAE":mae, "RMSE":rmse, "R2":r2, "Pearson_r":r, "Gap_mean":gap.mean(), "Gap_SD":gap.std(), "N_test":len(y_test)})
        preds[name]=y_pred
        print(f"  {name:20s} MAE {mae:.3f} R2 {r2:.3f} r {r:.3f} ({path})")
    except Exception as e:
        print(f"  {name}: load fail {e}")

if not results:
    print("No models found! Train first: python train/train_xgboost.py etc")
    raise SystemExit(1)

res=pd.DataFrame(results).sort_values("MAE")
res.to_csv(REPORT_DIR/"comparison_summary.csv", index=False)
# also read legacy metrics txt if present for reference
for p in METRICS_DIR.glob("*metrics.txt"):
    print(f"Legacy metrics file: {p.name} -> {p.read_text()[:120].strip()}")

print("\n"+"="*70)
print("RANKING (lowest MAE = best)")
print("="*70)
print(res[["Model","MAE","RMSE","R2","Pearson_r"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best=res.iloc[0]
print(f"\n>>> BEST MODEL: {best['Model']} <<<")
print(f"    MAE {best['MAE']:.3f} R2 {best['R2']:.4f} RMSE {best['RMSE']:.3f}")
print(f"    Path {best['Path']}")
# Save verdict
with open(REPORT_DIR/"comparison_verdict.txt","w") as f:
    f.write(f"Best model: {best['Model']}\nPath: {best['Path']}\nMAE {best['MAE']:.4f} R2 {best['R2']:.4f} RMSE {best['RMSE']:.4f}\n")
    f.write(res.to_string(index=False))

# --- Plots ---
sns.set_theme(style="whitegrid", font_scale=1.0)
# 1. Bar MAE
plt.figure(figsize=(10,5))
colors=["#22C55E" if i==0 else "#64748B" for i in range(len(res))]
bars=plt.barh(res["Model"], res["MAE"], color=colors, edgecolor="white")
plt.xlabel("MAE (years) — lower is better")
plt.title("Model Comparison — MAE (20% holdout, same split)")
plt.gca().invert_yaxis()
for bar, val in zip(bars, res["MAE"]):
    plt.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG_DIR/"comparison_mae.png",dpi=150); plt.close()

# 2. Scatter grid actual vs pred (2x2)
n=len(preds)
cols=2; rows=(n+1)//2
fig, axes=plt.subplots(rows, cols, figsize=(12, 5*rows), squeeze=False)
axes=axes.flatten()
for ax, (name, y_pred) in zip(axes, preds.items()):
    ax.scatter(y_test, y_pred, alpha=0.2, s=8)
    mn=min(y_test.min(), y_pred.min()); mx=max(y_test.max(), y_pred.max())
    ax.plot([mn,mx],[mn,mx],"r--",lw=1.2)
    mae=mean_absolute_error(y_test,y_pred); r2=r2_score(y_test,y_pred)
    ax.set_title(f"{name}\nMAE {mae:.2f} R2 {r2:.3f}", fontsize=10)
    ax.set_xlabel("True Age"); ax.set_ylabel("Predicted")
    ax.grid(alpha=0.2)
for ax in axes[len(preds):]: ax.axis("off")
plt.tight_layout(); plt.savefig(FIG_DIR/"comparison_scatter.png",dpi=150); plt.close()

# 3. Gap distribution
plt.figure(figsize=(10,5))
gap_df=pd.DataFrame({name: preds[name]-y_test.to_numpy() for name in preds})
gap_df.melt(var_name="Model", value_name="Gap").pipe(lambda d: sns.violinplot(data=d, x="Gap", y="Model", orient="h", inner="quartile", palette="Set2"))
plt.axvline(0,color="black",ls="--"); plt.title("BioAge Gap Distribution (Pred - True)"); plt.tight_layout(); plt.savefig(FIG_DIR/"comparison_gap_violin.png",dpi=150); plt.close()

# 4. Age-group MAE heatmap for best model
best_pred=preds[best["Model"]]
age_bins=[18,30,45,60,80]; labels=["18-29","30-44","45-59","60-80"]
y_bin=pd.cut(y_test, bins=age_bins+[100], labels=labels+["75+"], right=False)  # handle edge
tmp=pd.DataFrame({"y_true":y_test,"y_pred":best_pred,"bin":y_bin})
age_mae=tmp.groupby("bin", observed=True).apply(lambda g: mean_absolute_error(g["y_true"], g["y_pred"]))
plt.figure(figsize=(7,4))
age_mae.plot(kind="bar", color="#0EA5E9", edgecolor="white")
plt.ylabel("MAE"); plt.title(f"Age-group MAE — BEST {best['Model']}")
for i,v in enumerate(age_mae): plt.text(i, v+0.05, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG_DIR/"comparison_best_age_groups.png",dpi=150); plt.close()

# 5. Metrics table figure
fig, ax=plt.subplots(figsize=(10, 3))
ax.axis("off")
tbl=ax.table(cellText=np.round(res[["MAE","RMSE","R2","Pearson_r"]].values,3), rowLabels=res["Model"], colLabels=["MAE","RMSE","R2","Pearson r"], loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.5)
# highlight best row
for j in range(4): tbl[1, j].set_facecolor("#D1FAE5")
ax.set_title("Comparison Summary — Lower MAE/RMSE, Higher R2/r is better", pad=10)
plt.tight_layout(); plt.savefig(FIG_DIR/"comparison_table.png",dpi=150); plt.close()

print(f"\nSaved figures to {FIG_DIR}/comparison_*.png")
print(f"Saved summary {REPORT_DIR/'comparison_summary.csv'}")
print("Done.")
