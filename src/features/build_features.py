"""
Feature engineering for VitalAge.

Takes the cleaned CSV from the preprocessing pipeline and produces
a model-ready DataFrame with optional log-transforms for skewed features.
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Features known to be right-skewed in blood biomarker data
SKEWED_FEATURES = ["CRP", "LBXSAPSI", "LBXWBCSI", "LBXTR", "LBXGH"]


def build_features(
    clean_path: str = "data/processed/bioage_final_clean.csv",
    output_path: str = "data/processed/bioage_model_ready.csv",
    log_transform_skewed: bool = True,
    add_age_group: bool = True,
) -> pd.DataFrame:
    """Build model-ready features from the cleaned CSV.

    Steps
    -----
    1. Load cleaned data.
    2. Optionally log-transform right-skewed biomarkers (log1p).
    3. Optionally add an age_group categorical bin.
    4. Export final CSV.

    Parameters
    ----------
    clean_path : str
        Path to the clean CSV produced by the cleaning pipeline.
    output_path : str
        Where to save the model-ready CSV.
    log_transform_skewed : bool
        If True, create log1p versions of skewed features.
    add_age_group : bool
        If True, add an 'age_group' column with decade bins.

    Returns
    -------
    pd.DataFrame — the model-ready feature set.
    """
    df = pd.read_csv(clean_path)
    print(f"Loaded clean data: {df.shape[0]} rows x {df.shape[1]} cols")

    # 1. Log transforms for skewed features
    if log_transform_skewed:
        for col in SKEWED_FEATURES:
            if col in df.columns and (df[col] >= 0).all():
                new_col = f"log_{col}"
                df[new_col] = np.log1p(df[col])
                print(f"  Created log transform: {new_col}")

    # 2. Age group bins
    if add_age_group and "Age" in df.columns:
        bins = [0, 18, 30, 45, 60, 75, 100]
        labels = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
        df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
        print(f"  Created age_group column with bins: {labels}")

    # 3. Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved model-ready data to {output_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    return df
