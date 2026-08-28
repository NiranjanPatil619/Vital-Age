"""
Data loading and initial profiling utilities for VitalAge.
"""
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the project YAML configuration.

    Returns
    -------
    dict with config content plus '_base_dir' key pointing to the
    directory containing the config file (for resolving relative paths).
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["_base_dir"] = config_path.parent
    return config


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path relative to the config file's directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_raw(path: str = None, config: dict = None) -> pd.DataFrame:
    """Load the raw blood biomarker CSV.

    Parameters
    ----------
    path : str, optional
        Direct path to the CSV. If None, uses config['paths']['raw_data'].
    config : dict, optional
        Loaded config dict. Required if path is None.

    Returns
    -------
    pd.DataFrame with SEQN as string dtype.
    """
    if path is None:
        if config is None:
            config = load_config()
        base = config.get("_base_dir", Path("."))
        path = _resolve_path(config["paths"]["raw_data"], base)

    df = pd.read_csv(path)
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype("Int64").astype(str)
    return df


def profile_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of column name, missing count, missing percent, sorted descending."""
    total = len(df)
    miss = df.isnull().sum()
    pct = (miss / total * 100).round(2)
    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": miss.values,
        "missing_pct": pct.values,
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)
    return result


def profile_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of column name, dtype, non-null count, unique count."""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "non_null": df.notnull().sum().values,
        "n_unique": df.nunique().values,
    })
