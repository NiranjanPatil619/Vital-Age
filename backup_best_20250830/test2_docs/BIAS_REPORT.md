# Attributes & Age-Group Bias — XGB 6.7 MAE Model

**Model:** `models/blood/test2_mae6.7/xgb_hist_cuda_600.json` (XGB hist cuda, 600 trees, 77 feats, 59,641 rows)

## 77 Training Attributes

**A. Raw blood/urine kept (miss<45%, 51 cols):**
`CRP, LBDEONO, LBDHDD, LBDLYMNO, LBDMONO, LBDNENO, LBXBAPCT, LBXEOPCT, LBXGH, LBXHCT, LBXHGB, LBXLYPCT, LBXMC, LBXMCHSI, LBXMCVSI, LBXMOPCT, LBXMPSI, LBXNEPCT, LBXPLTSI, LBXRBCSI, LBXRDW, LBXSAL, LBXSAPSI, LBXSASSI, LBXSATSI, LBXSBU, LBXSC3SI, LBXSCA, LBXSCH, LBXSCLSI, LBXSCR, LBXSGB, LBXSGL, LBXSGTSI, LBXSIR, LBXSKSI, LBXSLDSI, LBXSNASI, LBXSOSSI, LBXSPH, LBXSTB, LBXSTP, LBXSUA, LBXTC, LBXWBCSI, URXCRS, URXUCR, URXUMA, URXUMS, URDACT, LBXSCK, LBXGLU`

**B. Questionnaire (8 → 8 cols after one-hot):**
`Gender (0F/1M), Weight kg, Height cm, Waist cm, Systolic_BP, Alcohol_days 0-7, Exercise_days 0-7, Smoking_status_Former, Smoking_status_Never` (Current = baseline)

**C. Engineered (12):**
`log_CRP, log_LBXSAPSI, log_LBXWBCSI, log_LBXGH, log_LBXSCR, log_LBXGLU, chol_ratio (TC/HDL), non_hdl (TC-HDL), scr_albumin_ratio, inflam_score (logCRP*logWBC), NLR_proxy ((100-Lymph)/Lymph), glycation_gap (GH - (GLU+46.7)/28.7), LBXRDW_sq, LBXMCVSI_sq, BMI (Weight/(Height/100)^2), WHtR (Waist/Height)`

> Full ordered list: `models/blood/test2_mae6.7/feature_order.json`

## Bias Analysis (proper holdout: train 80% → test 20% unseen, n=11,929, overall MAE 6.674)

| Age group | n | MAE | Bias (pred - true) |
|---|---|---|---|
| 18-29 | 2473 | 6.79 | **+5.86** (over-predict) |
| 30-44 | 2734 | 6.39 | +1.84 |
| 45-59 | 2713 | 6.82 | -0.58 |
| 60-74 | 2745 | 6.34 | -3.32 |
| 75-80 | 1264 | 7.47 | **-7.06** (under-predict) |

Classic **regression-to-the-mean** (young old tails shrink to ~45-60). After Ridge bias corrector (`pred_corrected = pred + Ridge(pred)`): overall 6.651, 18-29 +5.15, 75-80 -6.23 — marginal.

### Why not uniform?
- Blood markers weakly correlate with age (|r|≤0.29); model relies on SBP/BMI/Waist (r~0.3-0.5) which plateau at extremes
- Training data imbalance: 75-80 only 10.6% of holdout

### Mitigations (not yet in saved weights, to add for production)
1. **Sample weights:** `weight = 1 / count_per_bin` in `XGB.fit(sample_weight=...)` or `StratifiedKFold` on age bins
2. **Isotonic/Platt calibration:** Fit `IsotonicRegression` on OOF preds → true
3. **Balanced resampling / SMOGN** for tails
4. **Report stratified MAE** alongside overall

Saved weights are **not age-debiased** — add corrector layer before app deployment. See `/tmp/bias_proper.py` to reproduce.
