"""Compute Excel-like baseline forecasts for comparison with ML models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FEATURES = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
REPORTS_DIR = BASE_DIR / "reports"
BASELINE_METRICS_PATH = REPORTS_DIR / "metrics_baselines.csv"
BASELINE_SUMMARY_PATH = REPORTS_DIR / "metrics_baselines_summary.csv"

TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
CROP_SLUG = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}

TRAIN_END = 2018
VAL_YEARS = [2019, 2020, 2021]
TEST_YEARS = [2022, 2023, 2024]

BASELINES = ("naive_lag1", "forecast_linear", "linest_lag_only")


def load_features() -> pd.DataFrame:
    if not PROCESSED_FEATURES.exists():
        raise FileNotFoundError("Processed feature file not found. Run training pipeline first.")
    df = pd.read_parquet(PROCESSED_FEATURES)
    return df


def lag_only_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in df.columns
        if c.endswith("_lag1") or c.startswith("ma5_")
    ]
    if "Yield_t_ha_lag1" not in cols and "Yield_t_ha_lag1" in df.columns:
        cols.append("Yield_t_ha_lag1")
    # remove columns that become NaN entirely
    result = [c for c in cols if df[c].notna().any()]
    return sorted(result)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.abs((y_true - y_pred) / y_true) * 100
        valid = perc[~np.isinf(perc) & ~np.isnan(perc)]
        return float(np.mean(valid)) if valid.size else float("nan")


def forecast_naive(crop_df: pd.DataFrame, year: int) -> float | None:
    prev = crop_df[crop_df["year"] == year - 1]
    if prev.empty:
        return None
    return float(prev["Yield_t_ha"].iloc[0])


def forecast_linear_trend(crop_df: pd.DataFrame, year: int) -> float | None:
    train_df = crop_df[crop_df["year"] <= year - 1]
    if len(train_df) < 2:
        return None
    X_train = train_df[["year"]].to_numpy()
    y_train = train_df["Yield_t_ha"].to_numpy()
    model = LinearRegression()
    model.fit(X_train, y_train)
    return float(model.predict(np.array([[year]]))[0])


def forecast_linest_lag(crop_df: pd.DataFrame, year: int, lag_columns: List[str]) -> float | None:
    train_df = crop_df[crop_df["year"] <= year - 1]
    target_row = crop_df[crop_df["year"] == year]
    if target_row.empty or train_df.empty:
        return None
    X_train = train_df[lag_columns].dropna()
    y_train = train_df.loc[X_train.index, "Yield_t_ha"]
    if len(X_train) < 2:
        return None
    # remove columns with zero variance
    std = X_train.std(axis=0)
    valid_cols = std[std > 0].index.tolist()
    if not valid_cols:
        return None
    X_train = X_train[valid_cols]
    X_target = target_row[valid_cols]
    if X_target.isna().any().any():
        return None
    model = LinearRegression()
    model.fit(X_train, y_train)
    return float(model.predict(X_target)[0])


def evaluate_baselines(features: pd.DataFrame) -> pd.DataFrame:
    lag_cols = lag_only_columns(features)
    records: List[Dict[str, float]] = []

    for crop in TARGET_CROPS:
        crop_df = features[features["group_or_crop"] == crop].sort_values("year").reset_index(drop=True)
        years = crop_df["year"].tolist()
        for year in years:
            if year <= TRAIN_END:
                continue
            split = "validation" if year in VAL_YEARS else "test" if year in TEST_YEARS else "train"
            y_true = float(crop_df.loc[crop_df["year"] == year, "Yield_t_ha"].iloc[0])

            for baseline in BASELINES:
                if baseline == "naive_lag1":
                    y_pred = forecast_naive(crop_df, year)
                elif baseline == "forecast_linear":
                    y_pred = forecast_linear_trend(crop_df, year)
                elif baseline == "linest_lag_only":
                    y_pred = forecast_linest_lag(crop_df, year, lag_cols)
                else:
                    continue
                if y_pred is None:
                    continue
                records.append(
                    {
                        "year": year,
                        "crop": crop,
                        "baseline": baseline,
                        "scenario": "lag_only" if baseline != "forecast_linear" else "trend",
                        "split": split,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "mae": abs(y_true - y_pred),
                        "rmse": (y_true - y_pred) ** 2,
                        "mape": abs((y_true - y_pred) / y_true) * 100 if y_true else float("nan"),
                    }
                )
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No baseline predictions were generated.")
    # Convert rmse column from squared error to value
    df["rmse"] = np.sqrt(df["rmse"])
    return df


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df[df["split"] == "test"]
        .groupby(["baseline", "crop"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
        )
    )
    return summary


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    results = evaluate_baselines(features)
    results.to_csv(BASELINE_METRICS_PATH, index=False)
    summary = summarise_metrics(results)
    summary.to_csv(BASELINE_SUMMARY_PATH, index=False)
    print("Saved baseline metrics to", BASELINE_METRICS_PATH)
    print("Saved baseline summary to", BASELINE_SUMMARY_PATH)


if __name__ == "__main__":
    main()
