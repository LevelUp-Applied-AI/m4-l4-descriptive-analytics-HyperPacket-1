import os
import pandas as pd
import numpy as np
from eda_report import generate_eda_report, profile_data, summarize_outliers


def test_profile_data_handles_missing_values(tmp_path):
    df = pd.DataFrame({
        "numeric": [1, 2, np.nan, 4],
        "categorical": ["a", None, "b", "c"],
    })
    output_dir = tmp_path / "report"
    profile = profile_data(df, output_dir=str(output_dir))
    assert "missing_count" in profile.columns
    assert profile.loc["numeric", "missing_count"] == 1
    assert (output_dir / "eda_profile.csv").exists()
    assert (output_dir / "eda_numeric_stats.csv").exists()


def test_generate_eda_report_creates_expected_files(tmp_path):
    df = pd.DataFrame({
        "a": np.random.normal(size=50),
        "b": np.random.normal(size=50),
        "cat": ["x"] * 25 + ["y"] * 25,
    })
    output_dir = tmp_path / "report"
    summary = generate_eda_report(df, output_dir=str(output_dir), style="whitegrid")
    assert os.path.exists(summary["profile_file"])
    assert os.path.exists(summary["numeric_stats_file"])
    assert os.path.exists(summary["heatmap"])
    assert os.path.exists(summary["missing_bar"])
    assert os.path.exists(summary["missing_matrix"])
    for path in summary["distributions"]:
        assert os.path.exists(path)


def test_summarize_outliers_reports_expected_columns(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3, 100],
        "y": [10, 12, 14, 16],
    })
    output_dir = tmp_path / "report"
    outliers = summarize_outliers(df, output_dir=str(output_dir))
    assert "outlier_count" in outliers.columns
    assert outliers.loc[outliers["column"] == "x", "outlier_count"].iloc[0] >= 1
    assert (output_dir / "outlier_summary.csv").exists()
