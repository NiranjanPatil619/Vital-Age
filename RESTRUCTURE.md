# VitalAge — Restructured (4 folders)

**Backup of former best (6.7 MAE) preserved at `backup_best_20250830/` (164MB, entire prior `models/blood/test2_mae6.7`, `reports`, `data/processed`, `synthetic_patients`). Do not delete until new structure validated.**

## New Layout (as requested)
```
VitalAge/
├── data/              # raw + cleaned + reports (single source of truth)
│   ├── raw/
│   │   ├── blood_age_mega_raw.csv        # 97k NHANES
│   │   ├── questionnaire.csv             # synthetic 59k (SEQN-keyed)
│   │   └── synthetic_profiles.csv        # 12 demo profiles
│   ├── processed/
│   │   ├── bioage_final_clean.csv        # 20k ×13 (12 feats + Age)
│   │   ├── bioage_model_ready.csv        # 20k ×18 (+logs+age_group)
│   │   ├── test2_mae67_training_data.csv # 59k ×78 (77 feats + Age) — BEST training set
│   │   └── test2_training_full.csv       # duplicate for provenance
│   └── reports/
│       ├── caveats_and_limitations.md
│       ├── BIAS_REPORT.md / RESULTS_questionnaire.md / patient_Sonali_mapping.md
│       ├── metrics/ (ridge/xgb *.txt/*.csv/*.json)
│       ├── figures/ (01_*.png … comparison_*.png)
│       ├── synthetic_reports/ (12 realistic PDFs)
│       ├── comparison_summary.csv / comparison_verdict.txt
│       └── ...
│
├── train/             # 3 training scripts (same split 80/20, random_state=42)
│   ├── train_random_forest.py  # RF 300 trees → model/random_forest/rf_model.pkl
│   ├── train_xgboost.py        # XGB 600 depth6 gpu→hist → model/xgboost/xgb_model.pkl
│   └── train_lightgbm.py       # LGB 800 leaves31 gpu → model/lightgbm/lgb_model.pkl
│
├── model/             # benchmark weights (all from prior runs)
│   ├── best/          # CURRENT BEST — XGB hist cuda 600, 59k 77 feats, MAE 6.70 holdout / 5.53 train-overlap
│   │   ├── xgb_model.pkl / xgb_hist_cuda_600.json
│   │   ├── feature_order.json, medians.json, metadata.json (mae_single 6.706, 3-fold 6.735±0.013)
│   │   ├── training_data.csv (33MB, auditable)
│   │   └── questionnaire.csv
│   ├── random_forest/ # rf_model.pkl (new) + ridge_bioage_model.pkl + bioage_model.pkl (legacy)
│   ├── xgboost/       # xgb_model.pkl (new) + xgboost_bioage_model.pkl / tuned / calibration
│   └── lightgbm/      # lgb_model.pkl (new) + LightGBM_model.pkl (legacy)
│
├── comparison/        # visual benchmark
│   └── compare_models.py  # loads models/best + model/*/ + legacy, fixed holdout, 5 plots, declares BEST
│
├── backup_best_20250830/  # FULL SNAPSHOT before restructure (keep)
│   ├── models/test2_mae6.7, reports, data_raw, data_processed, synthetic_patients …
│   └── ...
│
├── src/ notebooks/ app/ reports/ models/  # LEGACY (still present, now superseded by new 4 folders)
│   └── see backup_best_20250830 for authoritative prior state
└── train/ comparison/ data/ model/ backup...  # NEW
```

## Quick Start (new structure)
```bash
# Train any benchmark (auto picks 59k 77-feat if present, else 20k 12-feat)
python train/train_random_forest.py
python train/train_xgboost.py
python train/train_lightgbm.py

# Compare all benchmarks (fixed 20% holdout)
python comparison/compare_models.py
# -> data/reports/comparison_summary.csv + data/reports/figures/comparison_*.png
# -> console: BEST MODEL: BEST (6.7) MAE 5.53 train-overlap / 6.70 proper holdout

# Streamlit (unchanged, now reads model/best)
streamlit run notebooks/blood/test2/streamlit_app.py
# or: python -m streamlit run notebooks/blood/test2/streamlit_app.py

# Data
ls data/raw/ data/processed/ data/reports/figures/comparison*
```

## Notes
- `model/best` is unchanged from `models/blood/test2_mae6.7` (verified `xgb_model.pkl` 2.8MB, `xgb_hist_cuda_600.json` 3.9MB, `training_data.csv` 33MB).
- `comparison/compare_models.py` handles 12-feat legacy models on 77-feat data via `feature_names_in_` filter; skipped if missing features (correct).
- `*.csv` gitignored (`.gitignore:5`); `data/processed/*.csv` etc will not show in `git status` — `git add -f` to track.
- Legacy `src/models/blood/train_xgboost.py` etc remain for reference; new `train/` scripts are thin wrappers with GPU auto-detect.

