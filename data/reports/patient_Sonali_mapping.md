# Patient Sonali Patil — Feature Coverage vs 77 Model Features

Report date 03-May-2024 Age 45 Female — CBC + LFT + Lipid + Kidney etc

Clean CSV: `patient_Sonali_Patil_clean.csv` (1 row, 78 cols: Age + 77 feats)

Covered **49/77** (64%):
- LBDEONO: 0.16
- LBDHDD: 52
- LBDLYMNO: 1.58
- LBDMONO: 0.24
- LBDNENO: 5.93
- LBXBAPCT: 0
- LBXEOPCT: 2
- LBXGH: 5.1
- LBXHCT: 37.3
- LBXHGB: 12.2
- LBXLYPCT: 20
- LBXMC: 32.8
- LBXMCHSI: 30.7
- LBXMCVSI: 94
- LBXMOPCT: 3
- LBXNEPCT: 75
- LBXPLTSI: 208
- LBXRBCSI: 3.98
- LBXRDW: 12.2
- LBXSAL: 3.9
- LBXSAPSI: 75.1
- LBXSASSI: 19.1
- LBXSATSI: 15.6
- LBXSBU: 9.24
- LBXSCA: 9.3
- LBXSCR: 0.62
- LBXSGB: 3.4
- LBXSGL: 99.67
- LBXSGTSI: 15.1
- LBXSIR: 85
- LBXSTB: 0.79
- LBXSTP: 7.3
- LBXSUA: 4.0
- LBXTC: 153
- LBXWBCSI: 7.9
- Gender: 0
- LBXGLU: 99.67
- log_LBXSAPSI: 4.33204826486764
- log_LBXWBCSI: 2.186051276738094
- log_LBXGH: 1.8082887711792655
- log_LBXSCR: 0.4824261492442927
- log_LBXGLU: 4.611847840741332
- chol_ratio: 2.9423076923076925
- non_hdl: 101
- scr_albumin_ratio: 0.15897435897435896
- NLR_proxy: 4.0
- glycation_gap: -8.881784197001252e-16
- LBXRDW_sq: 148.83999999999997
- LBXMCVSI_sq: 8836

Missing **28/77** (needs imputation/median):
- CRP
- LBXMPSI
- LBXSC3SI
- LBXSCH
- LBXSCLSI
- LBXSKSI
- LBXSLDSI
- LBXSNASI
- LBXSOSSI
- LBXSPH
- URXCRS
- URXUCR
- URXUMA
- URXUMS
- URDACT
- LBXSCK
- Weight
- Height
- Waist
- Systolic_BP
- Alcohol_days
- Exercise_days
- Smoking_status_Former
- Smoking_status_Never
- log_CRP
- inflam_score
- BMI
- WHtR

## Notes
- Direct bilirubin 0.22 slightly >0.2 but total bilirubin used (LBXSTB 0.79)
- LDL 80 / Trig 105 / VLDL 21 not in 77 (miss<45 excludes LBDLDL/LBXTR) → ratio derived via chol_ratio/non_hdl
- TSH/T3/T4, Amylase/Lipase not in blood panel → missing (not predictive for BioAge in this model)
- Weight/Height/Waist/SBP not in lab report → left NaN (will impute median before predict; for real app collect via questionnaire)
- CRP missing (not tested) → inflam_score NaN → median impute
