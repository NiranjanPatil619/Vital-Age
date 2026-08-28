"""Generate the Jupyter notebooks for the VitalAge blood analysis pipeline."""
import json
from pathlib import Path


def make_notebook(cells: list) -> dict:
    """Create a notebook dict from a list of (cell_type, source) tuples."""
    nb_cells = []
    for cell_type, source in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source if isinstance(source, list) else [source],
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.3",
            },
        },
        "cells": nb_cells,
    }


# =========================================================================
# Notebook 01: Data Overview
# =========================================================================
nb01 = make_notebook([
    ("markdown", [
        "# 01 — Data Overview\n",
        "Initial profiling of `blood_age_mega_raw.csv`: shape, dtypes, missingness, age distribution."
    ]),
    ("code", [
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from src.data.load_data import load_raw, load_config, profile_missingness, profile_dtypes\n",
        "\n",
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
        "config = load_config('../../config.yaml')\n",
        "print('Config loaded.')"
    ]),
    ("markdown", ["## 1. Load raw data"]),
    ("code", [
        "df = load_raw(config=config)\n",
        "print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')\n",
        "df.head()"
    ]),
    ("code", ["df.dtypes.to_frame('dtype')"]),
    ("markdown", ["## 2. Missingness profile"]),
    ("code", [
        "miss = profile_missingness(df)\n",
        "miss.head(20)"
    ]),
    ("code", [
        "fig, ax = plt.subplots(figsize=(12, 8))\n",
        "# Only plot columns with >0% missing\n",
        "plot_data = miss[miss['missing_pct'] > 0].copy()\n",
        "colors = ['#e74c3c' if p > 60 else '#f39c12' if p > 30 else '#2ecc71' for p in plot_data['missing_pct']]\n",
        "ax.barh(plot_data['column'], plot_data['missing_pct'], color=colors)\n",
        "ax.set_xlabel('Missing %')\n",
        "ax.set_title('Missingness by Column')\n",
        "ax.axvline(x=60, color='red', linestyle='--', alpha=0.7, label='60% threshold')\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/01_missingness_all_columns.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/01_missingness_all_columns.png')"
    ]),
    ("markdown", ["## 3. Age distribution & top-coding"]),
    ("code", [
        "ceiling = config['age']['top_code_ceiling']\n",
        "topcoded_count = (df['Age'] == ceiling).sum()\n",
        "print(f'Top-code ceiling: {ceiling}')\n",
        "print(f'Rows at exactly {ceiling}: {topcoded_count} ({topcoded_count/len(df)*100:.1f}%)')\n",
        "print(f'Max age in data: {df[\"Age\"].max()}')\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "age_data = df['Age'].dropna()\n",
        "ax.hist(age_data, bins=86, edgecolor='white', alpha=0.8)\n",
        "ax.axvline(x=ceiling, color='red', linestyle='--', linewidth=2, label=f'Top-code at {ceiling}')\n",
        "ax.set_xlabel('Age (years)')\n",
        "ax.set_ylabel('Count')\n",
        "ax.set_title('Age Distribution with NHANES Top-Coding')\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/02_age_distribution.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/02_age_distribution.png')"
    ]),
    ("markdown", ["## 4. Survey cycle breakdown"]),
    ("code", [
        "cycle_counts = df['CYCLE'].value_counts().sort_index()\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "cycle_counts.plot(kind='bar', ax=ax, color='steelblue')\n",
        "ax.set_ylabel('Count')\n",
        "ax.set_title('Participants per NHANES Cycle')\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/03_cycle_breakdown.png', dpi=150)\n",
        "plt.show()"
    ]),
    ("markdown", ["## 5. Column classification"]),
    ("code", [
        "target = config['columns']['target']\n",
        "identifiers = config['columns']['identifiers']\n",
        "phenoage = config['columns']['phenoage_core']\n",
        "secondary = config['columns']['secondary']\n",
        "drop_hm = config['columns']['drop_high_missing']\n",
        "drop_dup = config['columns']['drop_duplicate']\n",
        "\n",
        "classification = []\n",
        "for col in df.columns:\n",
        "    if col == target:\n",
        "        role = 'TARGET'\n",
        "    elif col in identifiers:\n",
        "        role = 'IDENTIFIER'\n",
        "    elif col in phenoage:\n",
        "        role = 'PHENOAGE_CORE'\n",
        "    elif col in secondary:\n",
        "        role = 'SECONDARY'\n",
        "    elif col in drop_hm:\n",
        "        role = 'DROP_HIGH_MISSING'\n",
        "    elif col in drop_dup:\n",
        "        role = 'DROP_DUPLICATE'\n",
        "    else:\n",
        "        role = 'OTHER'\n",
        "    miss_pct = df[col].isnull().mean() * 100\n",
        "    classification.append({'column': col, 'role': role, 'missing_pct': round(miss_pct, 1)})\n",
        "\n",
        "class_df = pd.DataFrame(classification)\n",
        "class_df"
    ]),
    ("code", [
        "print('Column counts by role:')\n",
        "print(class_df['role'].value_counts().to_string())"
    ]),
    ("markdown", ["## 6. Summary statistics for candidate features"]),
    ("code", [
        "candidate_cols = phenoage + secondary + [target]\n",
        "available = [c for c in candidate_cols if c in df.columns]\n",
        "df[available].describe().round(2)"
    ]),
    ("markdown", ["## 7. Duplicate column check"]),
    ("code", [
        "for dup, primary in [('LBXSCH', 'LBXTC'), ('LBXSGL', 'LBXGLU')]:\n",
        "    mask = df[dup].notna() & df[primary].notna()\n",
        "    if mask.sum() > 0:\n",
        "        corr = df.loc[mask, dup].corr(df.loc[mask, primary])\n",
        "        print(f'{dup} vs {primary}: r={corr:.4f} (n={mask.sum()} overlapping rows)')\n",
        "    else:\n",
        "        print(f'{dup} vs {primary}: no overlapping rows')"
    ]),
    ("markdown", [
        "## Summary\n",
        "- **97,683 rows × 60 columns** in the raw data.\n",
        "- **Age top-coding**: spike at 80 years (NHANES privacy cap).\n",
        "- **6 columns** have >60% missingness and will be dropped.\n",
        "- **2 duplicate alt-code columns** (LBXSCH, LBXSGL) will be dropped.\n",
        "- **9 PhenoAge core + 5 secondary** biomarkers are the candidate features.\n",
        "- Next notebook: missingness visualization + cleaning pipeline."
    ]),
])

Path("notebooks/blood/01_data_overview.ipynb").write_text(
    json.dumps(nb01, indent=1), encoding="utf-8"
)
print("Created 01_data_overview.ipynb")


# =========================================================================
# Notebook 02: Missingness & Cleaning
# =========================================================================
nb02 = make_notebook([
    ("markdown", [
        "# 02 — Missingness & Cleaning\n",
        "Visualize missingness patterns, run the cleaning pipeline, and document every decision."
    ]),
    ("code", [
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from src.data.load_data import load_raw, load_config\n",
        "from src.data.clean_blood_data import run_cleaning_pipeline\n",
        "\n",
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
        "config = load_config('../../config.yaml')\n",
        "print('Ready.')"
    ]),
    ("markdown", ["## 1. Load raw + visualize missingness"]),
    ("code", [
        "df_raw = load_raw(config=config)\n",
        "print(f'Raw shape: {df_raw.shape}')"
    ]),
    ("code", [
        "# Missingness heatmap for core + secondary features\n",
        "target = config['columns']['target']\n",
        "core = config['columns']['phenoage_core']\n",
        "secondary = config['columns']['secondary']\n",
        "features = [c for c in core + secondary + [target] if c in df_raw.columns]\n",
        "\n",
        "# Sort by missingness\n",
        "order = df_raw[features].isnull().mean().sort_values(ascending=False).index.tolist()\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(14, 8))\n",
        "sns.heatmap(df_raw[order].isnull().T, cbar_kws={'label': 'Missing'}, cmap='YlOrRd', ax=ax)\n",
        "ax.set_title('Missingness Pattern — Core + Secondary Biomarkers')\n",
        "ax.set_xlabel('Participant (sorted)')\n",
        "ax.set_ylabel('Column')\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/04_missingness_heatmap.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/04_missingness_heatmap.png')"
    ]),
    ("code", [
        "# Bar chart: missingness for core + secondary only\n",
        "miss_pct = df_raw[features].isnull().mean().sort_values(ascending=True) * 100\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "colors = ['#e74c3c' if p > 60 else '#f39c12' if p > 30 else '#2ecc71' for p in miss_pct.values]\n",
        "ax.barh(miss_pct.index, miss_pct.values, color=colors)\n",
        "ax.set_xlabel('Missing %')\n",
        "ax.set_title('Missingness in Candidate Features')\n",
        "ax.axvline(x=60, color='red', linestyle='--', alpha=0.7)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/05_candidate_missingness.png', dpi=150)\n",
        "plt.show()"
    ]),
    ("markdown", ["## 2. Run cleaning pipeline"]),
    ("code", [
        "clean_df, log = run_cleaning_pipeline(\n",
        "    config_path='../../config.yaml',\n",
        "    drop_topcoded=True,\n",
        "    verbose=True,\n",
        ")\n",
        "print('\\n--- Cleaning log ---')\n",
        "for k, v in log.items():\n",
        "    print(f'  {k}: {v}')"
    ]),
    ("markdown", ["## 3. Before / after comparison"]),
    ("code", [
        "print(f'Raw rows:               {log[\"raw_shape\"][0]:,}')\n",
        "print(f'After column selection: {log[\"after_column_selection\"][0]:,} rows x {log[\"after_column_selection\"][1]} cols')\n",
        "print(f'Top-coded dropped:      {log.get(\"topcoded_count\", \"N/A\")}')\n",
        "print(f'After dropna:           {log[\"after_dropna\"]:,} rows')\n",
        "print(f'Rows lost total:        {log[\"raw_shape\"][0] - log[\"after_dropna\"]:,} '\n",
        "      f'({(log[\"raw_shape\"][0] - log[\"after_dropna\"]) / log[\"raw_shape\"][0] * 100:.1f}%)')"
    ]),
    ("code", [
        "clean_df.head(10)"
    ]),
    ("code", [
        "clean_df.describe().round(2)"
    ]),
    ("code", [
        "# Verify zero missing in clean data\n",
        "print('Missing values in clean data:')\n",
        "print(clean_df.isnull().sum().to_string())\n",
        "print(f'\\nTotal missing: {clean_df.isnull().sum().sum()}')"
    ]),
    ("markdown", [
        "## Decision log\n",
        "| Step | Decision | Rationale |\n",
        "|------|----------|-----------|\n",
        "| Column selection | Drop 6 high-missing + 2 duplicate-alt columns | >60% missing is unusable; alt-codes correlate >0.98 with primaries |\n",
        "| Age top-coding | Drop rows at age 80 | Artificial spike from NHANES privacy cap; distorts age regression |\n",
        "| Outliers | Clip to biological bounds | Preserve rows while limiting implausible values |\n",
        "| Missing values | Complete-case (drop rows with any NaN) | Honest for hackathon; ~23K rows is sufficient for modeling |\n"
    ]),
    ("markdown", [
        "## Summary\n",
        "- Raw: **97,683 × 60** → Clean: **~23,253 × 15**\n",
        "- All missing values eliminated via complete-case filtering.\n",
        "- Clean CSV exported to `data/processed/bioage_final_clean.csv`.\n",
        "- Next notebook: full EDA on the clean data."
    ]),
])

Path("notebooks/blood/02_missingness_and_cleaning.ipynb").write_text(
    json.dumps(nb02, indent=1), encoding="utf-8"
)
print("Created 02_missingness_and_cleaning.ipynb")


# =========================================================================
# Notebook 03: EDA
# =========================================================================
nb03 = make_notebook([
    ("markdown", [
        "# 03 — Exploratory Data Analysis\n",
        "Distributions, correlations, outlier boxplots, age-group breakdowns."
    ]),
    ("code", [
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "\n",
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
        "print('Ready.')"
    ]),
    ("markdown", ["## 1. Load clean data"]),
    ("code", [
        "df = pd.read_csv('../../data/processed/bioage_final_clean.csv')\n",
        "print(f'Clean data: {df.shape[0]} rows x {df.shape[1]} columns')\n",
        "df.head()"
    ]),
    ("markdown", ["## 2. Feature distributions"]),
    ("code", [
        "feature_cols = [c for c in df.columns if c != 'Age']\n",
        "n_features = len(feature_cols)\n",
        "n_cols_plot = 3\n",
        "n_rows_plot = (n_features + n_cols_plot - 1) // n_cols_plot\n",
        "\n",
        "fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(14, n_rows_plot * 3.5))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for i, col in enumerate(feature_cols):\n",
        "    ax = axes[i]\n",
        "    data = df[col].dropna()\n",
        "    ax.hist(data, bins=40, edgecolor='white', alpha=0.8, color='steelblue')\n",
        "    ax.set_title(col, fontsize=11)\n",
        "    skew = data.skew()\n",
        "    ax.text(0.95, 0.95, f'skew={skew:.1f}', transform=ax.transAxes,\n",
        "            ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
        "\n",
        "# Hide unused subplots\n",
        "for j in range(i + 1, len(axes)):\n",
        "    axes[j].set_visible(False)\n",
        "\n",
        "fig.suptitle('Feature Distributions (Clean Data)', fontsize=14, y=1.01)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/06_distributions.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: reports/figures/06_distributions.png')"
    ]),
    ("markdown", ["## 3. Correlation heatmap"]),
    ("code", [
        "corr = df.corr(numeric_only=True)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 10))\n",
        "mask = np.triu(np.ones_like(corr, dtype=bool), k=1)\n",
        "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,\n",
        "            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,\n",
        "            cbar_kws={'label': 'Pearson r'})\n",
        "ax.set_title('Correlation Matrix — All Features + Age')\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/07_correlation_heatmap.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/07_correlation_heatmap.png')"
    ]),
    ("code", [
        "# Top correlations with Age\n",
        "age_corr = corr['Age'].drop('Age').sort_values(key=abs, ascending=False)\n",
        "print('Correlations with Age (sorted by |r|):')\n",
        "print(age_corr.to_string())"
    ]),
    ("markdown", ["## 4. Boxplots per biomarker"]),
    ("code", [
        "fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 5))\n",
        "if len(feature_cols) == 1:\n",
        "    axes = [axes]\n",
        "\n",
        "for i, col in enumerate(feature_cols):\n",
        "    ax = axes[i]\n",
        "    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,\n",
        "               boxprops=dict(facecolor='lightblue', color='navy'),\n",
        "               medianprops=dict(color='red'))\n",
        "    ax.set_title(col, fontsize=9, rotation=45)\n",
        "    ax.set_xticklabels([])\n",
        "\n",
        "fig.suptitle('Boxplots — Candidate Features', fontsize=14)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/08_boxplots.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: reports/figures/08_boxplots.png')"
    ]),
    ("markdown", ["## 5. Age-group breakdowns"]),
    ("code", [
        "bins = [0, 18, 30, 45, 60, 75, 100]\n",
        "labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75+']\n",
        "df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)\n",
        "\n",
        "# Top 4 features most correlated with Age\n",
        "top4 = age_corr.head(4).index.tolist()\n",
        "print(f'Top 4 Age-correlated features: {top4}')\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "for i, col in enumerate(top4):\n",
        "    ax = axes[i // 2][i % 2]\n",
        "    order = labels\n",
        "    sns.boxplot(data=df, x='age_group', y=col, order=order, ax=ax,\n",
        "                palette='viridis', fliersize=2)\n",
        "    ax.set_title(f'{col} by Age Group')\n",
        "    ax.set_xlabel('Age Group')\n",
        "\n",
        "fig.suptitle('Top Biomarkers Stratified by Age Group', fontsize=14)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/09_age_group_breakdowns.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/09_age_group_breakdowns.png')"
    ]),
    ("markdown", ["## 6. Feature vs Age scatter plots"]),
    ("code", [
        "n_feat = len(feature_cols)\n",
        "n_cols = 3\n",
        "n_rows = (n_feat + n_cols - 1) // n_cols\n",
        "\n",
        "fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))\n",
        "axes = np.array(axes).flatten() if n_feat > 1 else [axes]\n",
        "\n",
        "for i, col in enumerate(feature_cols):\n",
        "    ax = axes[i]\n",
        "    ax.scatter(df['Age'], df[col], alpha=0.15, s=5, color='steelblue')\n",
        "    # Add regression line\n",
        "    mask = df[col].notna() & df['Age'].notna()\n",
        "    if mask.sum() > 10:\n",
        "        slope, intercept, r, p, se = stats.linregress(df.loc[mask, 'Age'], df.loc[mask, col])\n",
        "        x_line = np.linspace(df['Age'].min(), df['Age'].max(), 100)\n",
        "        ax.plot(x_line, slope * x_line + intercept, color='red', linewidth=2,\n",
        "                label=f'r={r:.3f}')\n",
        "        ax.legend(fontsize=9)\n",
        "    ax.set_xlabel('Age')\n",
        "    ax.set_ylabel(col)\n",
        "    ax.set_title(col)\n",
        "\n",
        "for j in range(i + 1, len(axes)):\n",
        "    axes[j].set_visible(False)\n",
        "\n",
        "fig.suptitle('Features vs Age with Linear Fit', fontsize=14, y=1.01)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/10_feature_vs_age.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: reports/figures/10_feature_vs_age.png')"
    ]),
    ("markdown", ["## 7. Summary statistics table"]),
    ("code", [
        "summary = df.describe().round(3).T\n",
        "summary['skew'] = df[feature_cols + ['Age']].skew().round(2)\n",
        "summary['kurtosis'] = df[feature_cols + ['Age']].kurtosis().round(2)\n",
        "summary"
    ]),
    ("markdown", [
        "## EDA Summary\n",
        "See saved charts in `reports/figures/` for presentation. Key findings:\n",
        "1. **Age is top-coded at 80** — spike removed during cleaning.\n",
        "2. **Right-skewed features**: CRP, LBXSAPSI, LBXWBCSI — log transforms may help.\n",
        "3. **Strongest Age correlations**: [to be filled after running].\n",
        "4. **Multicollinearity**: some CBC-derived features are highly correlated (r > 0.8).\n",
        "5. **Age-group patterns**: biomarker distributions shift with age, some non-linearly.\n",
    ]),
])

Path("notebooks/blood/03_eda.ipynb").write_text(
    json.dumps(nb03, indent=1), encoding="utf-8"
)
print("Created 03_eda.ipynb")


# =========================================================================
# Notebook 04: Feature Selection
# =========================================================================
nb04 = make_notebook([
    ("markdown", [
        "# 04 — Feature Selection\n",
        "Rank features by importance, check multicollinearity, finalize feature set."
    ]),
    ("code", [
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "from sklearn.feature_selection import mutual_info_regression\n",
        "\n",
        "from src.features.build_features import build_features\n",
        "\n",
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
        "print('Ready.')"
    ]),
    ("markdown", ["## 1. Load clean data"]),
    ("code", [
        "df = pd.read_csv('../../data/processed/bioage_final_clean.csv')\n",
        "print(f'Clean data: {df.shape}')\n",
        "feature_cols = [c for c in df.columns if c not in ['Age', 'age_group']]\n",
        "X = df[feature_cols].values\n",
        "y = df['Age'].values"
    ]),
    ("markdown", ["## 2. Pearson correlation with Age"]),
    ("code", [
        "corr_with_age = df[feature_cols + ['Age']].corr()['Age'].drop('Age').sort_values(key=abs, ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr_with_age.values]\n",
        "ax.barh(corr_with_age.index, corr_with_age.values, color=colors)\n",
        "ax.set_xlabel('Pearson r with Age')\n",
        "ax.set_title('Feature–Age Correlation Ranking')\n",
        "ax.axvline(x=0, color='black', linewidth=0.5)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/11_correlation_with_age.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/11_correlation_with_age.png')"
    ]),
    ("markdown", ["## 3. Mutual Information with Age"]),
    ("code", [
        "mi = mutual_info_regression(X, y, random_state=42)\n",
        "mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "ax.barh(mi_series.index, mi_series.values, color='steelblue')\n",
        "ax.set_xlabel('Mutual Information')\n",
        "ax.set_title('Feature Importance — Mutual Information with Age')\n",
        "plt.tight_layout()\n",
        "plt.savefig('../../reports/figures/12_mutual_information.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: reports/figures/12_mutual_information.png')"
    ]),
    ("markdown", ["## 4. Multicollinearity check"]),
    ("code", [
        "corr_matrix = df[feature_cols].corr()\n",
        "\n",
        "# Find pairs with |r| > 0.8\n",
        "high_corr_pairs = []\n",
        "for i in range(len(feature_cols)):\n",
        "    for j in range(i + 1, len(feature_cols)):\n",
        "        r = corr_matrix.iloc[i, j]\n",
        "        if abs(r) > 0.8:\n",
        "            high_corr_pairs.append((feature_cols[i], feature_cols[j], r))\n",
        "\n",
        "if high_corr_pairs:\n",
        "    print('Highly correlated feature pairs (|r| > 0.8):')\n",
        "    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):\n",
        "        print(f'  {f1} <-> {f2}: r={r:.3f}')\n",
        "else:\n",
        "    print('No feature pairs with |r| > 0.8 found.')"
    ]),
    ("code", [
        "# Clustermap of feature correlations\n",
        "g = sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,\n",
        "                   vmin=-1, vmax=1, figsize=(12, 10), linewidths=0.5,\n",
        "                   dendrogram_ratio=0.15)\n",
        "g.fig.suptitle('Feature Correlation Cluster Map', y=1.02)\n",
        "plt.savefig('../../reports/figures/13_clustermap.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: reports/figures/13_clustermap.png')"
    ]),
    ("markdown", ["## 5. Final feature ranking"]),
    ("code", [
        "# Combined ranking: average of (correlation rank, MI rank)\n",
        "corr_rank = corr_with_age.abs().rank(ascending=False)\n",
        "mi_rank = mi_series.rank(ascending=False)\n",
        "\n",
        "ranking = pd.DataFrame({\n",
        "    'feature': feature_cols,\n",
        "    'pearson_r': [corr_with_age.get(f, 0) for f in feature_cols],\n",
        "    'mi_score': [mi_series.get(f, 0) for f in feature_cols],\n",
        "    'corr_rank': [corr_rank.get(f, len(feature_cols)) for f in feature_cols],\n",
        "    'mi_rank': [mi_rank.get(f, len(feature_cols)) for f in feature_cols],\n",
        "})\n",
        "ranking['avg_rank'] = (ranking['corr_rank'] + ranking['mi_rank']) / 2\n",
        "ranking = ranking.sort_values('avg_rank')\n",
        "ranking"
    ]),
    ("markdown", [
        "## Recommended feature set\n",
        "Based on correlation, mutual information, and multicollinearity analysis.\n",
        "Highly correlated duplicates (|r| > 0.8) should have one removed.\n",
        "Final feature list saved in the model-ready CSV."
    ]),
    ("code", [
        "# Build the model-ready dataset with log transforms\n",
        "model_df = build_features(\n",
        "    clean_path='../../data/processed/bioage_final_clean.csv',\n",
        "    output_path='../../data/processed/bioage_model_ready.csv',\n",
        "    log_transform_skewed=True,\n",
        "    add_age_group=True,\n",
        ")"
    ]),
])

Path("notebooks/blood/04_feature_selection.ipynb").write_text(
    json.dumps(nb04, indent=1), encoding="utf-8"
)
print("Created 04_feature_selection.ipynb")

print("\nAll notebooks generated successfully.")
