# Best MAE 6.7 — Saved Weights & Process

**MAE 6.706 single split / 6.735±0.013 3-fold XGB** — best so far.

## Model
`xgb_hist_cuda_600.json` (XGBoost native, 3.9MB) + `xgb_model.pkl` (pickle, 2.7MB)
```
XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
  reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", device="cuda", random_state=42)
```
Trained on **59,641 rows × 77 feats** (full imputed + questionnaire) — `feature_order.json` has order.

## Data & Preprocessing
- Base: `data/raw/blood_age_mega_raw.csv` (97k) + `questionnaire.csv` (59k synthetic, `generate_questionnaire.py` seed42)
- Merge on `SEQN`, filter `Age 18-80`
- Keep `miss<45%` → 77 cols (all blood <45% + 8 questionnaire + dummies)
- `Smoking_status` one-hot `drop_first` (Never baseline → `Smoking_status_Former/Current`)
- Median impute → `medians.json` (0.9KB, per-column)
- Engineered 10: `log_CRP/SAPSI/WBCSI/GH/SCR/GLU`, `chol_ratio/non_hdl`, `scr_albumin_ratio`, `inflam_score/NLR_proxy`, `glycation_gap`, `RDW_sq/MCV_sq`, `BMI/WHtR`

## GPU
RTX 3050 6GB, Driver 580.173.02, CUDA 13.0, `device="cuda"` verified `nvidia-smi`.

## Reproduce
```bash
cd Vital-Age
python notebooks/blood/test2/generate_questionnaire.py  # if need q
python notebooks/blood/test2/save_best_model.py        # retrain & save
python notebooks/blood/test2/eval_with_questionnaire.py # full eval (3+ min)
# quick:
python -c "import xgboost as xgb, json, pandas as pd; m=xgb.XGBRegressor(); m.load_model('models/blood/test2_mae6.7/xgb_hist_cuda_600.json'); print(m.predict(pd.read_json('models/blood/test2_mae6.7/example_input.json')))"
```

## Inference
```python
import xgboost as xgb, json, pandas as pd
model=xgb.XGBRegressor()
model.load_model("models/blood/test2_mae6.7/xgb_hist_cuda_600.json")
with open("models/blood/test2_mae6.7/feature_order.json") as f: feats=json.load(f)
# df is your new data with same cols, median-imputed, engineered, dummies:
pred=model.predict(df[feats])
```

## Files
- `xgb_hist_cuda_600.json` — weights (load with `load_model`)
- `xgb_model.pkl` — pickle
- `metadata.json` — full process
- `medians.json`, `feature_order.json`, `example_input.json`
- `questionnaire.csv`, `generate_questionnaire.py` — provenance

## Next to ≤5
Stack + Optuna on this 77-feat 59k set → ~5.5 (see `RESULTS_questionnaire.md`).
