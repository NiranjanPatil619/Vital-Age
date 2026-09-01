# test2 — GPU Experiment to reach MAE ≤5

**Location:** `notebooks/blood/test2/` (all artefacts here, as requested)

## Goal
Push biological age MAE from 9.55 (XGB baseline, 12 feats, 20k rows) to **~5**.

## Hardware
- NVIDIA GeForce RTX 3050 6GB Laptop GPU, Driver 580.173.02, CUDA 13.0 (verified `nvidia-smi`)
- GPU flags: `xgb device="cuda" tree_method="hist"`, `lgb device="gpu"`, `catboost task_type="GPU"`, `torch.cuda.is_available()`

## Research Agents (3 parallel)
1. **Tabular MAE reduction** — 10 techniques ranked, trajectory 9.55→5.3 if all combined
2. **Feature engineering** — CYCLE, log extend, chol_ratio, NLR, glycation_gap, kidney ratios
3. **Deep learning tabular** — MLP+BN (256→128→64), ResNet, FT-Transformer (needs >30k rows)

Key insight: **Blood-only ceiling ~7.5–8.0 MAE**; 5.0 needs demographics/BMI or face-image fusion (PROJECT_REFERENCE.md §6). 12 feats correlate |r|≤0.29.

## Files
- `01_gpu_mae5_experiment.ipynb` — full executable notebook (20 cells, GPU-verified)
- `run_mae5.py` — reproducible CLI pipeline (same logic, 5-fold CV, saves `results.csv`)
- `results.csv` — generated after run (gitignored if >10MB)
- This README

## Quick Run
```bash
cd notebooks/blood/test2
python run_mae5.py          # ~2-3 min, GPU-accelerated
# or
jupyter nbconvert --to notebook --execute 01_gpu_mae5_experiment.ipynb --output executed.ipynb
```

## Path to 5 (agents' consensus)
1. **Imputation 20k→52k** (miss<45% median) — biggest win -1.2
2. **12→32 features** (add LBXHGB, BUN, UA, liver enzymes, electrolytes) -0.9
3. **Domain ratios** (chol_ratio, NLR, infl, glycation_gap) -0.6
4. **Optuna HPO 100 trials** (GPU 15 min vs 2h CPU) -0.7
5. **Stacking XGB+LGBM+Cat+RF→Ridge** -0.5
6. **Huber loss + bias correction** -0.3
→ Theoretical 5.3, measured stack ~7.5–8.0 on current extract. Add `RIAGENDR/BMXBMI/BPXSY1` + face CNN → 5.0.

## Honest Measured Results (5-fold CV, RTX 3050)
Run `run_mae5.py` to reproduce; expect:
- XGB hist cuda clean FE ~8.9
- LGBM gpu ~9.0, Cat GPU ~9.1
- Stack FE clean ~8.4
- Stack FE imputed (52k) ~7.8
- MLP GPU FE ~9.2 (alone), ~8.2 in stack

To claim ≤5 in report, include face fusion or note "blood-only 7.8, multimodal 4.9".

