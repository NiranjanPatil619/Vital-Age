"""
Blood biomarker data cleaning pipeline for VitalAge.

Handles: column selection, age top-coding, outlier clipping,
missing value filtering, and export of a clean model-ready CSV.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional

from .load_data import load_raw, load_config


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

def select_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only: target (Age) + phenoage_core + secondary biomarkers.

    Excludes any columns listed in drop_high_missing or drop_duplicate,
    even if they appear in the secondary list.
    """
    target = config["columns"]["target"]
    phenoage = config["columns"]["phenoage_core"]
    secondary = config["columns"].get("secondary", [])
    drop_high = set(config["columns"].get("drop_high_missing", []))
    drop_dup = set(config["columns"].get("drop_duplicate", []))
    excluded = drop_high | drop_dup

    keep = [target] + phenoage + secondary
    keep = [c for c in keep if c in df.columns and c not in excluded]

    dropped_from_config = [c for c in (phenoage + secondary) if c in excluded]
    if dropped_from_config:
        print(f"  [select] Excluded (high-missing/duplicate): {dropped_from_config}")

    df = df[keep].copy()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found after selection.")

    return df


# ---------------------------------------------------------------------------
# Age filtering
# ---------------------------------------------------------------------------

def filter_age_range(df: pd.DataFrame, min_age: int = 18, max_age: int = 80) -> pd.DataFrame:
    """Filter to only include rows within the specified age range (inclusive).

    Parameters
    ----------
    min_age : int
        Minimum age to include (default 18).
    max_age : int
        Maximum age to include (default 80).
    """
    before = len(df)
    df = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)].copy()
    dropped = before - len(df)
    print(f"  [age] Filtered to ages {min_age}-{max_age}: dropped {dropped} rows, kept {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def clip_outliers(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """Clip each column to its biological plausible range.

    Parameters
    ----------
    ranges : dict
        Mapping of column_name -> [min, max] from config['outlier_ranges'].
    """
    df = df.copy()
    total_clipped = 0
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        before_null = df[col].isnull().sum()
        mask = df[col].notna()
        n_below = ((df[col] < lo) & mask).sum()
        n_above = ((df[col] > hi) & mask).sum()
        n_clip = n_below + n_above
        if n_clip > 0:
            df[col] = df[col].clip(lower=lo, upper=hi)
            total_clipped += n_clip
            print(f"  [outlier] {col}: clipped {n_clip} values ({n_below} below {lo}, {n_above} above {hi})")
    print(f"  [outlier] Total values clipped: {total_clipped}")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    config_path: str = "config.yaml",
    drop_topcoded: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline.

    Returns
    -------
    clean_df : pd.DataFrame
        The cleaned DataFrame with target + selected features, no missing values.
    log : dict
        Dictionary documenting every decision (row counts, columns kept, etc.).
    """
    config = load_config(config_path)
    log: dict = {}

    # 1. Load raw
    df = load_raw(config=config)
    log["raw_shape"] = df.shape
    if verbose:
        print(f"[1/6] Loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols")

    # 2. Column selection
    df = select_columns(df, config)
    log["after_column_selection"] = df.shape
    log["kept_columns"] = list(df.columns)
    if verbose:
        print(f"[2/7] After column selection: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"       Kept: {list(df.columns)}")

    # 3. Age range filter
    min_age = config["age"].get("min_age", 18)
    max_age = config["age"].get("max_age", 80)
    df = filter_age_range(df, min_age=min_age, max_age=max_age)
    log["min_age"] = min_age
    log["max_age"] = max_age
    log["after_age_filter"] = len(df)

    # 4. Outlier clipping
    outlier_ranges = config.get("outlier_ranges", {})
    if outlier_ranges:
        if verbose:
            print("[4/7] Clipping outliers...")
        df = clip_outliers(df, outlier_ranges)

    # 5. Drop rows with missing values in the remaining columns (complete-case)
    before_dropna = len(df)
    df = df.dropna()
    dropped = before_dropna - len(df)
    log["before_dropna"] = before_dropna
    log["after_dropna"] = len(df)
    log["rows_dropped_missing"] = dropped
    if verbose:
        print(f"[5/7] Dropped {dropped} rows with missing values")
        print(f"[6/7] Complete-case shape: {df.shape}")

    # 6. Export
    from .load_data import _resolve_path
    base = config.get("_base_dir", Path("."))
    out_path = _resolve_path(config["paths"]["processed_data"], base)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log["output_path"] = str(out_path)
    if verbose:
        print(f"[7/7] Saved clean data to {out_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    return df, log
