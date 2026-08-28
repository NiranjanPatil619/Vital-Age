"""
VitalAge — One-click pipeline runner.
Usage: python run_pipeline.py
"""
import subprocess
import sys
import time
from pathlib import Path


def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run_step(label, command):
    print(f"\n>>> {label}")
    start = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"    Done ({elapsed:.1f}s)")
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n")[-5:]:
                print(f"    {line}")
    else:
        print(f"    FAILED ({elapsed:.1f}s)")
        print(f"    Error: {result.stderr.strip()}")
    return result.returncode == 0


banner("VitalAge Pipeline — Starting")
t0 = time.time()

# Step 1: Install dependencies
run_step("Installing dependencies", f"{sys.executable} -m pip install -r requirements.txt -q")

# Step 2: Run cleaning pipeline
run_step("Cleaning pipeline", f'{sys.executable} -c "from src.data.clean_blood_data import run_cleaning_pipeline; run_cleaning_pipeline(config_path=\'config.yaml\')"')

# Step 3: Run feature builder
run_step("Feature builder", f'{sys.executable} -c "from src.features.build_features import build_features; build_features()"')

# Step 4: Run notebooks
notebooks = [
    "notebooks/blood/01_data_overview.ipynb",
    "notebooks/blood/02_missingness_and_cleaning.ipynb",
    "notebooks/blood/03_eda.ipynb",
    "notebooks/blood/04_feature_selection.ipynb",
]
for nb in notebooks:
    name = Path(nb).stem
    run_step(f"Notebook: {name}", f'jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 "{nb}" --output "{name}.ipynb"')

# Step 5: Summary
elapsed = time.time() - t0
banner("Pipeline Complete")

clean = Path("data/processed/bioage_final_clean.csv")
ready = Path("data/processed/bioage_model_ready.csv")
figs = list(Path("reports/figures").glob("*.png"))

print(f"  Total time: {elapsed:.1f}s")
print()
if clean.exists():
    import pandas as pd
    df = pd.read_csv(clean)
    print(f"  Clean CSV:    {df.shape[0]} rows x {df.shape[1]} cols  -> {clean}")
if ready.exists():
    import pandas as pd
    df2 = pd.read_csv(ready)
    print(f"  Model-ready:  {df2.shape[0]} rows x {df2.shape[1]} cols  -> {ready}")
print(f"  EDA charts:   {len(figs)} files in reports/figures/")
print(f"  Caveats:      reports/caveats_and_limitations.md")
print()
