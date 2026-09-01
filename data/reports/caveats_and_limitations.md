# Caveats & Limitations — VitalAge Blood Biomarker Pipeline

## Data Source
- **NHANES** (US CDC National Health and Nutrition Examination Survey), merged across 9 survey cycles from 2005-2006 through 2021-2023.
- Total raw participants: **97,683 rows × 60 columns**.

## Age Top-Coding
- NHANES top-codes participant ages for privacy protection, but the ceiling varies by cycle:
  - **2005-2006 cycle**: top-codes at **85 years** (170 rows at exactly 85, all from this cycle)
  - **All other cycles (2007-2023)**: top-codes at **80 years** (3,635 rows at exactly 80)
- **Decision**: All rows with Age == 80 were **dropped** from the clean dataset. This removes ~3.7% of raw rows but prevents the model from learning a distorted age-ceiling effect.
- The 170 rows at age 85 (2005-2006 cycle only) are **retained** — they represent a small fraction and the 2005-2006 cycle's true ceiling is 85, so these are valid data points for that cycle.
- **Limitation**: The model cannot predict biological age for individuals aged 80+ (except the 2005-2006 cohort up to 85). This is an acceptable constraint for the target population.

## Missing Data
- Missingness ranges from **0%** (SEQN, CYCLE) to **93.5%** (LBXMAGN) across the 60 raw columns.
- **6 columns** had >60% missingness and were dropped entirely: LBXMAGN (93.5%), LBXNRBC (81.8%), LBDLDL (70.3%), LBXTR (69.9%), LBXGLU (69.2%), LBDBANO (65.7%).
  - Note: LBDLDL and LBXTR (lipids) are in the secondary feature set but were excluded due to high missingness.
  - LBXGLU (fasting glucose) is a PhenoAge core marker but has 69.2% missingness. It is **retained** because it is scientifically essential, but its effective sample size is smaller.
- **Handling method**: Complete-case analysis (rows with any NaN in the final 13-feature set were dropped).
- **Result**: 97,683 raw rows → **22,094 clean rows** (22.6% retention). This is a known trade-off: fewer rows but zero imputation assumptions.

## Duplicate / Alt-Code Columns
- `LBXSCH` (total cholesterol, alt code) and `LBXTC` (total cholesterol, primary) correlate at **r = 0.989**. `LBXSCH` was dropped.
- `LBXSGL` (glucose, alt code) and `LBXGLU` (glucose, primary) correlate at **r = 0.985**. `LBXSGL` was dropped.
- Keeping only one measurement per biomarker avoids multicollinearity artifacts.

## Cross-Cycle Measurement Drift
- Data spans 18 years (2005–2023) across 9 NHANES cycles.
- Lab assay methods, calibration standards, and equipment may have shifted between cycles.
- This is **not corrected** in the current pipeline. It is a known limitation that may add noise to biomarker-age relationships.
- Future work could include cycle-fixed effects or standardization.

## Biological Plausible Range Clipping
- Outliers were clipped to biologically plausible ranges defined in `config.yaml`.
- Only a small number of values were affected (most data already falls within expected ranges).
- Clipping preserves rows while preventing extreme implausible values from dominating model training.

## Feature Set
- **13 features** in the final clean dataset:
  - 9 PhenoAge core biomarkers: LBXSAL, LBXSAL, LBXSCR, LBXGLU, CRP, LBXLYPCT, LBXMCVSI, LBXRDW, LBXSAPSI, LBXWBCSI
  - 3 secondary biomarkers: LBXGH, LBDHDD, LBXTC
  - Target: Age
- The PhenoAge formula was originally validated on a different population (NHANES III, pre-2000). Generalizability to the modern merged NHANES dataset is assumed but not guaranteed.
- **LBDLDL** (LDL) and **LBXTR** (triglycerides) were excluded despite being in the original secondary list due to >69% missingness.

## Model-Ready Dataset
- Output: `data/processed/bioage_model_ready.csv` (22,094 rows × 18 columns)
- Includes 4 log-transformed features for right-skewed biomarkers: `log_CRP`, `log_LBXSAPSI`, `log_LBXWBCSI`, `log_LBXGH`
- Includes `age_group` categorical bin for stratified analysis

## Charts Generated
All EDA charts are saved in `reports/figures/`:
1. `01_missingness_all_columns.png` — Missingness across all 60 raw columns
2. `02_age_distribution.png` — Age histogram with top-code spike at 80
3. `03_cycle_breakdown.png` — Participants per NHANES cycle
4. `04_missingness_heatmap.png` — Missingness pattern for core + secondary features
5. `05_candidate_missingness.png` — Missingness in candidate features only
6. `06_distributions.png` — Histograms of all 12 biomarker features
7. `07_correlation_heatmap.png` — Full correlation matrix
8. `08_boxplots.png` — Boxplots per biomarker
9. `09_age_group_breakdowns.png` — Top features stratified by age group
10. `10_feature_vs_age.png` — Scatter plots with regression lines
11. `11_correlation_with_age.png` — Feature-Age correlation ranking
12. `12_mutual_information.png` — Mutual information importance
13. `13_clustermap.png` — Clustered correlation heatmap

## Recommendations for Next Pipeline Stage
1. Consider **regularized regression** (Ridge/Lasso) given multicollinearity among CBC-derived features.
2. Evaluate **gradient boosting** (XGBoost/LightGBM) which handles non-linear age-biomarker relationships.
3. The PhenoAge formula uses a specific weighted combination — compare model predictions against the original PhenoAge formula as a baseline.
4. Consider adding **cycle-year** as a feature to capture temporal drift.
