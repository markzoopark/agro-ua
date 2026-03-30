#!/usr/bin/env python3
"""QA check script for agrostats ML pipeline.

Verifies:
1. All required reports and figures exist
2. Test/lag_only metrics stay within revision-era guardrails
3. SHAP tables are non-empty
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Required files
REQUIRED_FILES = [
    "reports/validation.md",
    "reports/metrics.csv",
    "reports/predictions.csv",
    "reports/metrics_by_scenario.csv",
    "reports/metrics_leaderboard.csv",
]

# SHAP tables for each crop (lag_only scenario, lightgbm & xgboost)
SHAP_PATTERNS = [
    "reports/shap_top_lightgbm_{crop}_lag_only.csv",
    "reports/shap_top_xgboost_{crop}_lag_only.csv",
]

# ACF/PACF figures
ACF_PACF_PATTERNS = [
    "reports/figures/yield_acf_{crop}.png",
    "reports/figures/yield_pacf_{crop}.png",
]

# Correlation heatmaps
CORRELATION_PATTERNS = [
    "reports/figures/correlation_heatmap_{crop}.png",
]

CROPS = ["pshenytsia", "kukurudza", "sonyashnyk"]

# Revision-era guardrails for test/lag_only.
# These limits are intentionally one-sided: the CI should flag regressions,
# not improvements versus the previous publication snapshot.
KPI_THRESHOLDS = {
    "Кукурудза": {"mae_max": 0.95, "mape_max": 14.0, "n_years": 3},
    "Пшениця": {"mae_max": 0.60, "mape_max": 12.0, "n_years": 3},
    "Соняшник": {"mae_max": 0.20, "mape_max": 7.0, "n_years": 3},
}


def check_file_exists(path: Path) -> bool:
    """Check if a file exists."""
    return path.exists() and path.is_file()


def check_required_files() -> List[str]:
    """Check all required files exist."""
    errors = []
    for file_path in REQUIRED_FILES:
        path = Path(file_path)
        if not check_file_exists(path):
            errors.append(f"Missing required file: {file_path}")
    return errors


def check_shap_tables() -> List[str]:
    """Check SHAP tables exist and are non-empty."""
    errors = []
    for crop in CROPS:
        for pattern in SHAP_PATTERNS:
            file_path = pattern.format(crop=crop)
            path = Path(file_path)
            if not check_file_exists(path):
                errors.append(f"Missing SHAP table: {file_path}")
            else:
                try:
                    df = pd.read_csv(path)
                    if df.empty:
                        errors.append(f"Empty SHAP table: {file_path}")
                    elif len(df) < 5:
                        errors.append(f"SHAP table has fewer than 5 features: {file_path} (found {len(df)})")
                except Exception as exc:
                    errors.append(f"Error reading SHAP table {file_path}: {exc}")
    return errors


def check_acf_pacf_figures() -> List[str]:
    """Check ACF/PACF figures exist."""
    errors = []
    for crop in CROPS:
        for pattern in ACF_PACF_PATTERNS:
            file_path = pattern.format(crop=crop)
            path = Path(file_path)
            if not check_file_exists(path):
                errors.append(f"Missing ACF/PACF figure: {file_path}")
    return errors


def check_correlation_heatmaps() -> List[str]:
    """Check correlation heatmap figures exist."""
    errors = []
    for crop in CROPS:
        for pattern in CORRELATION_PATTERNS:
            file_path = pattern.format(crop=crop)
            path = Path(file_path)
            if not check_file_exists(path):
                errors.append(f"Missing correlation heatmap: {file_path}")
    return errors


def check_kpi_metrics() -> List[str]:
    """Check test/lag_only metrics meet KPI thresholds using best models from leaderboard."""
    errors = []
    leaderboard_path = Path("reports/metrics_leaderboard.csv")
    predictions_path = Path("reports/predictions.csv")

    if not check_file_exists(leaderboard_path):
        return ["Cannot check KPI metrics: metrics_leaderboard.csv not found"]

    if not check_file_exists(predictions_path):
        return ["Cannot check KPI metrics: predictions.csv not found"]

    try:
        leaderboard = pd.read_csv(leaderboard_path)
        predictions = pd.read_csv(predictions_path)
    except Exception as exc:
        return [f"Error reading CSV files: {exc}"]

    # Filter leaderboard for lag_only scenario
    leaderboard = leaderboard[leaderboard["scenario"] == "lag_only"]

    if leaderboard.empty:
        errors.append("No lag_only entries in leaderboard")
        return errors

    # For each crop, get the best model and check its predictions
    for crop, thresholds in KPI_THRESHOLDS.items():
        # Find best model for this crop
        crop_leader = leaderboard[leaderboard["crop"] == crop]

        if crop_leader.empty:
            errors.append(f"No leaderboard entry for {crop}")
            continue

        best_model = crop_leader.iloc[0]["model"]
        best_mae = crop_leader.iloc[0]["mae"]
        best_mape = crop_leader.iloc[0]["mape"]

        # Get predictions for best model
        crop_preds = predictions[
            (predictions["crop"] == crop)
            & (predictions["model"] == best_model)
            & (predictions["scenario"] == "lag_only")
            & (predictions["split"] == "test")
        ]

        if crop_preds.empty:
            errors.append(f"{crop}: No test/lag_only predictions for best model {best_model}")
            continue

        n_years = len(crop_preds["year"].unique())
        expected_years = thresholds["n_years"]

        if n_years != expected_years:
            errors.append(
                f"{crop} ({best_model}): Expected {expected_years} test years, found {n_years}"
            )

        # Check regression-oriented MAE ceiling
        if best_mae > thresholds["mae_max"]:
            errors.append(
                f"{crop} ({best_model}): MAE={best_mae:.3f} exceeds ceiling {thresholds['mae_max']}"
            )

        # Check MAPE bound
        if best_mape > thresholds["mape_max"]:
            errors.append(
                f"{crop} ({best_model}): MAPE={best_mape:.2f}% exceeds threshold {thresholds['mape_max']}%"
            )

    return errors


def main() -> int:
    """Run all QA checks."""
    print("=" * 70)
    print("QA CHECKS FOR AGROSTATS ML PIPELINE")
    print("=" * 70)
    print()

    all_errors: List[str] = []

    # Check 1: Required files
    print("[1/5] Checking required files...")
    errors = check_required_files()
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ All required files present")
    print()

    # Check 2: SHAP tables
    print("[2/5] Checking SHAP tables...")
    errors = check_shap_tables()
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ All SHAP tables valid")
    print()

    # Check 3: ACF/PACF figures
    print("[3/5] Checking ACF/PACF figures...")
    errors = check_acf_pacf_figures()
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ All ACF/PACF figures present")
    print()

    # Check 4: Correlation heatmaps
    print("[4/5] Checking correlation heatmaps...")
    errors = check_correlation_heatmaps()
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ All correlation heatmaps present")
    print()

    # Check 5: KPI metrics
    print("[5/5] Checking KPI metrics (test/lag_only)...")
    errors = check_kpi_metrics()
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ All KPI metrics within thresholds")
    print()

    # Summary
    print("=" * 70)
    if all_errors:
        print(f"QA CHECKS FAILED: {len(all_errors)} error(s)")
        print("=" * 70)
        print()
        print("Error summary:")
        for i, error in enumerate(all_errors, 1):
            print(f"{i}. {error}")
        return 1
    else:
        print("✅ ALL CHECKS PASSED")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
