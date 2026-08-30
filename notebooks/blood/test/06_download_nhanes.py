"""
VitalAge — NHANES Extended Data Download & Merge

Downloads additional NHANES data files:
- DEMO: Demographics (age, gender, race, education, income)
- BMX: Body measures (BMI, waist, arm circumference)
- BPX: Blood pressure (systolic, diastolic)
- INS: Insulin
- GHB: HbA1c (glycohemoglobin)
- TCHOL: Total cholesterol
- TRIGLY: Triglycerides
- HDL: HDL cholesterol
- ALB_CR: Albumin/Creatinine ratio (urine)

Merges all with existing blood_age_mega_raw.csv

Usage: python notebooks/blood/test/06_download_nhanes.py
"""

import os
import io
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
RAW_BLOOD = os.path.join(RAW_DIR, "blood_age_mega_raw.csv")
os.makedirs(RAW_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# NHANES file URLs (2017-2018 cycle as primary, fallback to other cycles)
# ---------------------------------------------------------------------------
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"

# Cycle mappings: cycle_name -> (year_start, year_end, suffix)
CYCLES = {
    "2017-2018": ("2017", "2018", "J"),
    "2015-2016": ("2015", "2016", "I"),
    "2013-2014": ("2013", "2014", "H"),
    "2011-2012": ("2011", "2012", "G"),
    "2009-2010": ("2009", "2010", "F"),
    "2007-2008": ("2007", "2008", "E"),
    "2005-2006": ("2005", "2006", "D"),
}

# Files to download: (nhanes_filename_base, suffix_pattern, description)
# Format: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/YYYY/DataFiles/FILENAME.xpt
FILES_TO_DOWNLOAD = [
    # Demographics
    ("DEMO", "J", "Demographics"),
    ("DEMO", "I", "Demographics"),
    ("DEMO", "H", "Demographics"),
    ("DEMO", "G", "Demographics"),
    ("DEMO", "F", "Demographics"),
    ("DEMO", "E", "Demographics"),
    ("DEMO", "D", "Demographics"),
    # Body measures
    ("BMX", "J", "Body Measures"),
    ("BMX", "I", "Body Measures"),
    ("BMX", "H", "Body Measures"),
    ("BMX", "G", "Body Measures"),
    ("BMX", "F", "Body Measures"),
    ("BMX", "E", "Body Measures"),
    ("BMX", "D", "Body Measures"),
    # Blood pressure
    ("BPX", "J", "Blood Pressure"),
    ("BPX", "I", "Blood Pressure"),
    ("BPX", "H", "Blood Pressure"),
    ("BPX", "G", "Blood Pressure"),
    ("BPX", "F", "Blood Pressure"),
    ("BPX", "E", "Blood Pressure"),
    ("BPX", "D", "Blood Pressure"),
    # Insulin
    ("INS", "J", "Insulin"),
    ("INS", "I", "Insulin"),
    ("INS", "H", "Insulin"),
    ("INS", "G", "Insulin"),
    ("INS", "F", "Insulin"),
    ("INS", "E", "Insulin"),
    ("INS", "D", "Insulin"),
    # Glycohemoglobin (HbA1c)
    ("GHB", "J", "HbA1c"),
    ("GHB", "I", "HbA1c"),
    ("GHB", "H", "HbA1c"),
    ("GHB", "G", "HbA1c"),
    ("GHB", "F", "HbA1c"),
    ("GHB", "E", "HbA1c"),
    ("GHB", "D", "HbA1c"),
    # Total cholesterol
    ("TCHOL", "J", "Total Cholesterol"),
    ("TCHOL", "I", "Total Cholesterol"),
    ("TCHOL", "H", "Total Cholesterol"),
    ("TCHOL", "G", "Total Cholesterol"),
    ("TCHOL", "F", "Total Cholesterol"),
    ("TCHOL", "E", "Total Cholesterol"),
    ("TCHOL", "D", "Total Cholesterol"),
    # Triglycerides
    ("TRIGLY", "J", "Triglycerides"),
    ("TRIGLY", "I", "Triglycerides"),
    ("TRIGLY", "H", "Triglycerides"),
    ("TRIGLY", "G", "Triglycerides"),
    ("TRIGLY", "F", "Triglycerides"),
    ("TRIGLY", "E", "Triglycerides"),
    ("TRIGLY", "D", "Triglycerides"),
    # HDL
    ("HDL", "J", "HDL Cholesterol"),
    ("HDL", "I", "HDL Cholesterol"),
    ("HDL", "H", "HDL Cholesterol"),
    ("HDL", "G", "HDL Cholesterol"),
    ("HDL", "F", "HDL Cholesterol"),
    ("HDL", "E", "HDL Cholesterol"),
    ("HDL", "D", "HDL Cholesterol"),
]

def download_xpt(url, save_path):
    """Download XPT file from NHANES."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        with open(save_path, 'wb') as f:
            f.write(response.read())
        return True
    except Exception as e:
        print(f"    Failed: {e}")
        return False

def read_xpt_safe(path):
    """Read XPT file, handling version differences."""
    try:
        return pd.read_sas(path, format='xport')
    except Exception:
        try:
            return pd.read_sas(path, format='xport', encoding='latin1')
        except Exception as e:
            print(f"    Error reading {path}: {e}")
            return None

# ---------------------------------------------------------------------------
# 1. Download files
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Downloading NHANES data files")
print("=" * 70)

suffix_to_cycle = {
    "J": "2017-2018",
    "I": "2015-2016",
    "H": "2013-2014",
    "G": "2011-2012",
    "F": "2009-2010",
    "E": "2007-2008",
    "D": "2005-2006",
}

suffix_to_years = {
    "J": ("2017", "2018"),
    "I": ("2015", "2016"),
    "H": ("2013", "2014"),
    "G": ("2011", "2012"),
    "F": ("2009", "2010"),
    "E": ("2007", "2008"),
    "D": ("2005", "2006"),
}

downloaded = {}
for file_base, suffix, desc in FILES_TO_DOWNLOAD:
    filename = f"{file_base}_{suffix}.xpt"
    save_path = os.path.join(RAW_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"  {filename} already exists, skipping")
        downloaded[f"{file_base}_{suffix}"] = save_path
        continue
    
    years = suffix_to_years[suffix]
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{years[0]}/DataFiles/{filename}"
    print(f"  Downloading {filename} ({desc})... ", end="", flush=True)
    
    if download_xpt(url, save_path):
        print("OK")
        downloaded[f"{file_base}_{suffix}"] = save_path
    else:
        print("FAILED")

print(f"\n  Downloaded: {len(downloaded)} files")

# ---------------------------------------------------------------------------
# 2. Load and merge all data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Loading and merging data")
print("=" * 70)

# Load existing blood data
df_blood = pd.read_csv(RAW_BLOOD)
df_blood["SEQN"] = df_blood["SEQN"].astype("Int64")
print(f"  Blood data: {len(df_blood)} rows")

# Collect all demo, bmx, bpx files
all_demo = []
all_bmx = []
all_bpx = []
all_ins = []
all_ghb = []
all_tchol = []
all_trigly = []
all_hdl = []

for key, path in downloaded.items():
    df = read_xpt_safe(path)
    if df is None:
        continue
    
    if "DEMO" in key:
        all_demo.append(df)
    elif "BMX" in key:
        all_bmx.append(df)
    elif "BPX" in key:
        all_bpx.append(df)
    elif "INS" in key:
        all_ins.append(df)
    elif "GHB" in key:
        all_ghb.append(df)
    elif "TCHOL" in key:
        all_tchol.append(df)
    elif "TRIGLY" in key:
        all_trigly.append(df)
    elif "HDL" in key:
        all_hdl.append(df)

# Concatenate and deduplicate
def merge_files(file_list, name):
    if not file_list:
        print(f"  {name}: No files loaded")
        return pd.DataFrame()
    df = pd.concat(file_list, ignore_index=True)
    df["SEQN"] = df["SEQN"].astype("Int64")
    df = df.drop_duplicates(subset=["SEQN"])
    print(f"  {name}: {len(df)} unique participants")
    return df

df_demo = merge_files(all_demo, "Demographics")
df_bmx = merge_files(all_bmx, "Body Measures")
df_bpx = merge_files(all_bpx, "Blood Pressure")
df_ins = merge_files(all_ins, "Insulin")
df_ghb = merge_files(all_ghb, "HbA1c")
df_tchol = merge_files(all_tchol, "Total Cholesterol")
df_trigly = merge_files(all_trigly, "Triglycerides")
df_hdl = merge_files(all_hdl, "HDL")

# ---------------------------------------------------------------------------
# 3. Select relevant columns from each file
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Selecting relevant columns")
print("=" * 70)

# Demographics: gender, race, education, income, marital status
demo_cols = ["SEQN"]
demo_rename = {}
for col in df_demo.columns:
    if col in ["RIDAGEYR", "RIDREY3"]:
        demo_cols.append(col)
        demo_rename[col] = "Age_demo"
    elif col == "RIAGENDR":
        demo_cols.append(col)
        demo_rename[col] = "Gender"
    elif col == "RIDRETH3":
        demo_cols.append(col)
        demo_rename[col] = "Race"
    elif col == "DMDEDUC2":
        demo_cols.append(col)
        demo_rename[col] = "Education"
    elif col == "INDHHIN2":
        demo_cols.append(col)
        demo_rename[col] = "Household_Income"
    elif col == "DMDMILTT":
        demo_cols.append(col)
        demo_rename[col] = "Marital_Status"

df_demo_sel = df_demo[[c for c in demo_cols if c in df_demo.columns]].rename(columns=demo_rename)
print(f"  Demographics: {list(df_demo_sel.columns)}")

# Body measures: BMI, waist, arm circumference
bmx_cols = ["SEQN"]
bmx_rename = {}
for col in df_bmx.columns:
    if col == "BMXBMI":
        bmx_cols.append(col)
        bmx_rename[col] = "BMI"
    elif col == "BMXWAIST":
        bmx_cols.append(col)
        bmx_rename[col] = "Waist_Circumference"
    elif col == "BMXARMC":
        bmx_cols.append(col)
        bmx_rename[col] = "Arm_Circumference"
    elif col == "BMXHT":
        bmx_cols.append(col)
        bmx_rename[col] = "Height"
    elif col == "BMXWT":
        bmx_cols.append(col)
        bmx_rename[col] = "Weight"

df_bmx_sel = df_bmx[[c for c in bmx_cols if c in df_bmx.columns]].rename(columns=bmx_rename)
print(f"  Body measures: {list(df_bmx_sel.columns)}")

# Blood pressure
bpx_cols = ["SEQN"]
bpx_rename = {}
for col in df_bpx.columns:
    if col == "BPXSY1":
        bpx_cols.append(col)
        bpx_rename[col] = "Systolic_BP_1"
    elif col == "BPXDI1":
        bpx_cols.append(col)
        bpx_rename[col] = "Diastolic_BP_1"
    elif col == "BPXSY2":
        bpx_cols.append(col)
        bpx_rename[col] = "Systolic_BP_2"
    elif col == "BPXDI2":
        bpx_cols.append(col)
        bpx_rename[col] = "Diastolic_BP_2"
    elif col == "BPXSY3":
        bpx_cols.append(col)
        bpx_rename[col] = "Systolic_BP_3"
    elif col == "BPXDI3":
        bpx_cols.append(col)
        bpx_rename[col] = "Diastolic_BP_3"
    elif col == "BPXPULS":
        bpx_cols.append(col)
        bpx_rename[col] = "Pulse"

df_bpx_sel = df_bpx[[c for c in bpx_cols if c in df_bpx.columns]].rename(columns=bpx_rename)
print(f"  Blood pressure: {list(df_bpx_sel.columns)}")

# Insulin
ins_cols = ["SEQN"]
ins_rename = {}
for col in df_ins.columns:
    if "INS" in col.upper() and "SEQN" not in col:
        ins_cols.append(col)
        ins_rename[col] = "Insulin"
df_ins_sel = df_ins[[c for c in ins_cols if c in df_ins.columns]].rename(columns=ins_rename)
print(f"  Insulin: {list(df_ins_sel.columns)}")

# HbA1c
ghb_cols = ["SEQN"]
ghb_rename = {}
for col in df_ghb.columns:
    if "GHB" in col.upper() or "HBA1C" in col.upper():
        ghb_cols.append(col)
        ghb_rename[col] = "HbA1c"
df_ghb_sel = df_ghb[[c for c in ghb_cols if c in df_ghb.columns]].rename(columns=ghb_rename)
print(f"  HbA1c: {list(df_ghb_sel.columns)}")

# Total cholesterol
tchol_cols = ["SEQN"]
tchol_rename = {}
for col in df_tchol.columns:
    if "TCHOL" in col.upper() or "LBXTC" in col.upper():
        tchol_cols.append(col)
        tchol_rename[col] = "Total_Cholesterol_new"
df_tchol_sel = df_tchol[[c for c in tchol_cols if c in df_tchol.columns]].rename(columns=tchol_rename)
print(f"  Total Cholesterol: {list(df_tchol_sel.columns)}")

# Triglycerides
trigly_cols = ["SEQN"]
trigly_rename = {}
for col in df_trigly.columns:
    if "TRIGLY" in col.upper() or "LBXTR" in col.upper():
        trigly_cols.append(col)
        trigly_rename[col] = "Triglycerides"
df_trigly_sel = df_trigly[[c for c in trigly_cols if c in df_trigly.columns]].rename(columns=trigly_rename)
print(f"  Triglycerides: {list(df_trigly_sel.columns)}")

# HDL
hdl_cols = ["SEQN"]
hdl_rename = {}
for col in df_hdl.columns:
    if "HDL" in col.upper() or "LBDHDD" in col.upper():
        hdl_cols.append(col)
        hdl_rename[col] = "HDL_Cholesterol"
df_hdl_sel = df_hdl[[c for c in hdl_cols if c in df_hdl.columns]].rename(columns=hdl_rename)
print(f"  HDL: {list(df_hdl_sel.columns)}")

# ---------------------------------------------------------------------------
# 4. Merge all data on SEQN
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Merging all data on SEQN")
print("=" * 70)

# Start with blood data
df_merged = df_blood.copy()

# Merge each dataset
for df_add, name in [
    (df_demo_sel, "Demographics"),
    (df_bmx_sel, "Body Measures"),
    (df_bpx_sel, "Blood Pressure"),
    (df_ins_sel, "Insulin"),
    (df_ghb_sel, "HbA1c"),
    (df_tchol_sel, "Total Cholesterol"),
    (df_trigly_sel, "Triglycerides"),
    (df_hdl_sel, "HDL"),
]:
    if df_add.empty:
        print(f"  Skipping {name} (empty)")
        continue
    before = len(df_merged)
    df_merged = df_merged.merge(df_add, on="SEQN", how="left")
    new_cols = [c for c in df_add.columns if c != "SEQN" and c not in df_blood.columns]
    print(f"  Merged {name}: added {len(new_cols)} columns ({list(new_cols)})")

print(f"\n  Final merged data: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")

# ---------------------------------------------------------------------------
# 5. Save merged data
# ---------------------------------------------------------------------------
out_path = os.path.join(RAW_DIR, "nhanes_extended_merged.csv")
df_merged.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

# Summary
print("\n" + "=" * 70)
print("MERGED DATA SUMMARY")
print("=" * 70)
print(f"  Total columns: {df_merged.shape[1]}")
print(f"  Blood biomarkers: {len([c for c in df_blood.columns if c not in ['SEQN', 'CYCLE']])}")
print(f"  New columns added: {df_merged.shape[1] - df_blood.shape[1]}")

# Missingness of new columns
new_cols = [c for c in df_merged.columns if c not in df_blood.columns and c != "SEQN"]
print(f"\n  Missingness of new columns:")
for col in new_cols:
    pct = df_merged[col].isnull().mean() * 100
    print(f"    {col:25s}: {pct:.1f}% missing")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
