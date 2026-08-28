# VitalAge — Project Reference Document

> **Purpose**: This document provides a complete reference for anyone taking over this project. It covers what exists, how it works, and what needs to be built next.

---

## 1. Project Overview

**VitalAge** is a healthcare AI project that predicts a person's **biological age** from clinical blood biomarkers, then computes a **BioAge Gap**:

```
BioAge Gap = Predicted Biological Age - Chronological Age
```

- **Positive gap** = person's biology looks older than their real age (early warning for chronic disease risk)
- **Negative gap** = healthy/slower aging

**Domain**: Health-tech hackathon (company hiring evaluation)
**Data source**: NHANES (US CDC National Health and Nutrition Examination Survey), 9 survey cycles merged (2005-2023)
**Raw data**: 97,683 participants, 60 blood biomarker columns

The project also has a **separate face-image branch** (CNN predicting "visual apparent age") built by a different sub-team. This document covers only the **blood/clinical data side**.

---

## 2. How to Run Everything

### One command (recommended)
```bash
python run_pipeline.py
```
This runs the entire pipeline in ~100 seconds: cleaning → features → 4 notebooks → 13 charts.

### Manual step-by-step
```bash
pip install -r requirements.txt
python -c "from src.data.clean_blood_data import run_cleaning_pipeline; run_cleaning_pipeline(config_path='config.yaml')"
python -c "from src.features.build_features import build_features; build_features()"
jupyter nbconvert --to notebook --execute notebooks/blood/01_data_overview.ipynb --output 01_data_overview.ipynb
jupyter nbconvert --to notebook --execute notebooks/blood/02_missingness_and_cleaning.ipynb --output 02_missingness_and_cleaning.ipynb
jupyter nbconvert --to notebook --execute notebooks/blood/03_eda.ipynb --output 03_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/blood/04_feature_selection.ipynb --output 04_feature_selection.ipynb
```

### Output files generated
| File | Description |
|---|---|
| `data/processed/bioage_final_clean.csv` | 22,094 rows × 13 columns (12 features + Age) |
| `data/processed/bioage_model_ready.csv` | 22,094 rows × 18 columns (+ log transforms + age_group) |
| `reports/figures/*.png` | 13 EDA charts |
| `reports/caveats_and_limitations.md` | All cleaning decisions documented |

---

## 3. Directory Structure

```
Vital-Age/
├── config.yaml                  # All paths, column definitions, biological ranges, NHANES metadata
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # ONE-CLICK runner (cleans + features + notebooks)
│
├── src/
│   ├── data/
│   │   ├── load_data.py         # load_raw(), profile_missingness(), profile_dtypes()
│   │   └── clean_blood_data.py  # select_columns(), flag_topcoded_age(), clip_outliers(), run_cleaning_pipeline()
│   ├── features/
│   │   └── build_features.py    # build_features() — log transforms, age group bins
│   ├── models/                  # EMPTY — model training goes here
│   │   ├── blood/
│   │   │   ├── train_model.py
│   │   │   ├── evaluate_model.py
│   │   │   └── phenoage_formula.py
│   │   ├── face/
│   │   └── fusion/
│   └── explainability/          # EMPTY — SHAP goes here
│
├── notebooks/
│   ├── blood/
│   │   ├── 01_data_overview.ipynb              # DONE — raw profiling
│   │   ├── 02_missingness_and_cleaning.ipynb   # DONE — cleaning pipeline
│   │   ├── 03_eda.ipynb                        # DONE — full EDA
│   │   ├── 04_feature_selection.ipynb          # DONE — feature ranking
│   │   └── 05_model_comparison.ipynb           # EMPTY — model training
│   └── face/                                   # Separate sub-team
│
├── data/
│   ├── raw/
│   │   ├── blood_age_mega_raw.csv   # 97,683 × 60 (NOT in git — too large)
│   │   └── face_images/
│   ├── interim/                     # Intermediate processing files
│   └── processed/
│       ├── bioage_final_clean.csv   # 22,094 × 13 (NOT in git)
│       └── bioage_model_ready.csv   # 22,094 × 18 (NOT in git)
│
├── reports/
│   ├── caveats_and_limitations.md   # DONE — documented decisions
│   └── figures/
│       ├── 01_missingness_all_columns.png
│       ├── 02_age_distribution.png
│       ├── 03_cycle_breakdown.png
│       ├── 04_missingness_heatmap.png
│       ├── 05_candidate_missingness.png
│       ├── 06_distributions.png
│       ├── 07_correlation_heatmap.png
│       ├── 08_boxplots.png
│       ├── 09_age_group_breakdowns.png
│       ├── 10_feature_vs_age.png
│       ├── 11_correlation_with_age.png
│       ├── 12_mutual_information.png
│       └── 13_clustermap.png
│
├── models/                         # Trained model artifacts (.pkl, .h5)
├── app/                            # Streamlit app (separate team)
└── tests/                          # Test files
```

---

## 4. The 12 Features (What the Model Uses)

### PhenoAge Core (9 features) — from the published scientific formula
| Config Key | Column | Biomarker | What it measures | Typical Unit |
|---|---|---|---|---|
| `LBXSAL` | Albumin | Liver/nutritional status | g/dL |
| `LBXSCR` | Creatinine | Kidney function | mg/dL |
| `LBXGLU` | Fasting glucose | Blood sugar (69% missing, retained because essential) | mg/dL |
| `CRP` | C-reactive protein | Inflammation | mg/L |
| `LBXLYPCT` | Lymphocyte percent | Immune function | % |
| `LBXMCVSI` | Mean cell volume | Red blood cell size | fL |
| `LBXRDW` | Red cell distribution width | Red blood cell variation | % |
| `LBXSAPSI` | Alkaline phosphatase | Liver enzyme | U/L |
| `LBXWBCSI` | White blood cell count | Immune/inflammation | 1000 cells/uL |

### Secondary (3 features) — lipid/metabolic panel
| Config Key | Column | Biomarker | What it measures |
|---|---|---|---|
| `LBXGH` | HbA1c | Long-term blood sugar | % |
| `LBDHDD` | HDL cholesterol | "Good" cholesterol | mg/dL |
| `LBXTC` | Total cholesterol | Overall lipid level | mg/dL |

### Dropped from secondary (too much missing data)
- `LBDLDL` (LDL cholesterol) — 70% missing → dropped
- `LBXTR` (Triglycerides) — 70% missing → dropped

### What the model-ready CSV contains (18 columns)
```
Age, LBXSAL, LBXSCR, LBXGLU, CRP, LBXLYPCT, LBXMCVSI, LBXRDW,
LBXSAPSI, LBXWBCSI, LBXGH, LBDHDD, LBXTC,
log_CRP, log_LBXSAPSI, log_LBXWBCSI, log_LBXGH, age_group
```

---

## 5. What Has Been Done (Completed)

### Preprocessing Pipeline
1. **Column selection** — Reduced 60 raw columns to 12 features (9 PhenoAge + 3 secondary)
2. **Age top-coding** — Dropped 3,635 rows at age 80 (NHANES privacy cap); 170 rows at age 85 from 2005-2006 cycle retained
3. **High-missingness columns dropped** — 6 columns with >60% missing removed
4. **Duplicate alt-codes removed** — LBXSCH (alt cholesterol), LBXSGL (alt glucose) dropped
5. **Outlier clipping** — Values clipped to biological plausible ranges defined in config.yaml
6. **Complete-case filtering** — All rows with any NaN dropped (97,683 → 22,094 rows)
7. **Log transforms** — log_CRP, log_LBXSAPSI, log_LBXWBCSI, log_LBXGH created for right-skewed features
8. **Age groups** — Categorical bins added (0-17, 18-29, 30-44, 45-59, 60-74, 75+)

### EDA (Exploratory Data Analysis)
- 13 charts generated covering: missingness, distributions, correlations, boxplots, age-group breakdowns, feature importance rankings
- All saved to `reports/figures/`

### Documentation
- `reports/caveats_and_limitations.md` — Documents every cleaning decision and known limitation
- `config.yaml` — Self-documenting configuration with column descriptions and biological ranges

---

## 6. What Needs to Be Built Next (Future Features)

### Priority 1: Model Training (notebooks/blood/05_model_comparison.ipynb)
- [ ] Baseline: PhenoAge formula (weighted linear combination)
- [ ] Ridge / Lasso regression
- [ ] Random Forest
- [ ] XGBoost / LightGBM
- [ ] Neural network (optional)
- [ ] 5-fold cross-validation
- [ ] Metrics: MAE, RMSE, R² for age prediction
- [ ] BioAge Gap computation

### Priority 2: Model Explainability (src/explainability/)
- [ ] SHAP values for feature importance
- [ ] Partial dependence plots
- [ ] Individual prediction explanations

### Priority 3: Streamlit App (app/)
- [ ] Upload lab report PDF → extract 12 features → predict BioAge Gap
- [ ] PDF parser (pdfplumber + Tesseract OCR)
- [ ] Test name → feature mapping dictionary
- [ ] Visualization of results (gauge chart, feature contributions)

### Priority 4: Model Serving
- [ ] Save trained model as .pkl
- [ ] Load model in Streamlit app
- [ ] API endpoint (optional)

---

## 7. Key Decisions Documented

| Decision | Rationale |
|---|---|
| Complete-case filtering (no imputation) | Honest for hackathon; ~22K rows sufficient |
| Drop age==80 rows | NHANES top-code creates artificial spike |
| Drop LBDLDL and LBXTR | >69% missing, not usable |
| Keep LBXGLU despite 69% missing | Essential PhenoAge component, scientific requirement |
| Log transform skewed features | CRP, Alk Phos, WBC, HbA1c are right-skewed |
| Clip outliers to biological ranges | Preserve rows while limiting implausible values |

---

## 8. Raw Data Details

**File**: `data/raw/blood_age_mega_raw.csv`
**Join key**: `SEQN` (participant ID)
**Cycles**: 2005-2006, 2007-2008, 2009-2010, 2011-2012, 2013-2014, 2015-2016, 2017-2018, 2017-2020, 2021-2023

**Column naming**: NHANES standard codes (e.g., `LBXSAL` = Serum Albumin, `LBXSCR` = Serum Creatinine)

**Missingness ranges**:
- Near 0%: SEQN, CYCLE, Age (3.8%)
- 20-40%: Most core biomarkers (MCV, RDW, WBC, Lymphocyte %, etc.)
- 60-94%: Some secondary markers (LDL, Triglycerides, Magnesium)

---

## 9. Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
jupyter>=1.0
openpyxl>=3.1
scipy>=1.10
scikit-learn>=1.3  (for model training phase)
```

---

## 10. GitHub Repository

```
https://github.com/NiranjanPatil619/Vital-Age
```

**Branch**: `main`
**Data files** are gitignored (too large). Run `python run_pipeline.py` to regenerate them from the raw CSV.

---

## 11. Tips for the Next AI

1. **Start with model training** in `notebooks/blood/05_model_comparison.ipynb` — the clean data is ready
2. **Read config.yaml first** — it documents every column and its meaning
3. **Read reports/caveats_and_limitations.md** — understand the data limitations before modeling
4. **The PhenoAge formula** uses specific published weights — check `src/models/blood/phenoage_formula.py` for reference
5. **For the Streamlit app** — the PDF parser needs a test name → feature mapping dictionary; see Section 4 of this document for the mapping
6. **Don't modify the cleaning pipeline** unless you have a strong reason — the decisions are documented and tested
