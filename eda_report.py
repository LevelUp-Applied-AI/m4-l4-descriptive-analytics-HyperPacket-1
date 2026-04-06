import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def profile_data(df: pd.DataFrame, output_dir: str = "output", columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    ensure_dir(output_dir)
    if columns is not None:
        df = df.loc[:, list(columns)]

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isnull().sum(),
        "missing_pct": df.isnull().mean() * 100,
    })
    numeric_stats = df.select_dtypes(include=[np.number]).describe().T
    profile_file = os.path.join(output_dir, "eda_profile.csv")
    profile.to_csv(profile_file)

    stats_file = os.path.join(output_dir, "eda_numeric_stats.csv")
    numeric_stats.to_csv(stats_file)

    return profile


def plot_numeric_distributions(
    df: pd.DataFrame,
    output_dir: str = "output",
    columns: Optional[Sequence[str]] = None,
    style: str = "whitegrid",
) -> None:
    ensure_dir(output_dir)
    sns.set_style(style)
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="steelblue", edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"), dpi=300)
        plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str = "output",
    columns: Optional[Sequence[str]] = None,
    style: str = "whitegrid",
) -> None:
    ensure_dir(output_dir)
    sns.set_style(style)
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.8)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()


def plot_missing_data(df: pd.DataFrame, output_dir: str = "output", style: str = "whitegrid") -> None:
    ensure_dir(output_dir)
    sns.set_style(style)
    missing_counts = df.isnull().sum()

    plt.figure(figsize=(10, 6))
    missing_summary = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if missing_summary.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        missing_summary.plot(kind="bar", color="salmon", edgecolor="black")
        plt.ylabel("Missing Count")
    plt.title("Missing Values by Column")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values_bar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull().astype(int), cbar=False, cmap="viridis")
    plt.title("Missing Data Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values_matrix.png"), dpi=300)
    plt.close()


def summarize_outliers(
    df: pd.DataFrame,
    output_dir: str = "output",
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    summary = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        summary.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": len(outliers),
                "outlier_pct": len(outliers) / len(series) * 100,
            }
        )

    outlier_df = pd.DataFrame(summary)
    outlier_df.to_csv(os.path.join(output_dir, "outlier_summary.csv"), index=False)
    return outlier_df


def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str = "output",
    columns: Optional[Sequence[str]] = None,
    style: str = "whitegrid",
) -> dict:
    ensure_dir(output_dir)
    profile = profile_data(df, output_dir=output_dir, columns=columns)
    plot_numeric_distributions(df, output_dir=output_dir, columns=columns, style=style)
    plot_correlation_heatmap(df, output_dir=output_dir, columns=columns, style=style)
    plot_missing_data(df, output_dir=output_dir, style=style)
    outliers = summarize_outliers(df, output_dir=output_dir, columns=columns)

    summary = {
        "profile_file": os.path.join(output_dir, "eda_profile.csv"),
        "numeric_stats_file": os.path.join(output_dir, "eda_numeric_stats.csv"),
        "outlier_summary_file": os.path.join(output_dir, "outlier_summary.csv"),
        "missing_bar": os.path.join(output_dir, "missing_values_bar.png"),
        "missing_matrix": os.path.join(output_dir, "missing_values_matrix.png"),
        "heatmap": os.path.join(output_dir, "eda_correlation_heatmap.png"),
        "distributions": [os.path.join(output_dir, f"distribution_{col}.png") for col in (columns if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist())],
        "outliers": outliers,
    }
    return summary


def load_dataframe_from_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reusable EDA report for any DataFrame.")
    parser.add_argument("filepath", help="Path to CSV data file")
    parser.add_argument("--output-dir", default="output", help="Directory to save report files")
    parser.add_argument("--columns", default=None, help="Comma-separated numeric columns to include")
    parser.add_argument("--style", default="whitegrid", help="Seaborn plot style")
    args = parser.parse_args()

    columns = args.columns.split(",") if args.columns else None
    df = load_dataframe_from_csv(args.filepath)
    generate_eda_report(df, output_dir=args.output_dir, columns=columns, style=args.style)
    print(f"EDA report generated in {args.output_dir}")
