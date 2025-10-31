"""Model training and evaluation pipeline for agrostats features."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
from rich.console import Console
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agrostats import utils, eda


console = Console()
app = typer.Typer(help="Training and evaluation commands for agrostats features.")

FEATURES_PATH = Path("data/processed/agrostats_poltava_features.parquet")
METRICS_PATH = Path("reports/metrics.csv")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")

TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
SPLIT_TRAIN_END = 2018
SPLIT_VAL_END = 2021
TEST_START = 2022
RANDOM_STATE = 42

MANUAL_SLUGS = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}

SCENARIOS = ("lag_only", "in_season")
LAG_FEATURE_SUFFIX = "_lag1"
MA_PREFIX = "ma5_"

CROP_NAME_TRANSLATIONS = {
    "uk": {
        "Пшениця": "Пшениця",
        "Кукурудза": "Кукурудза",
        "Соняшник": "Соняшник",
    },
    "en": {
        "Пшениця": "Wheat",
        "Кукурудза": "Corn",
        "Соняшник": "Sunflower",
    },
}

TRAIN_LANG = {
    "uk": {
        "legend_actual": "Факт",
        "legend_pred": "Прогноз",
        "year_label": "Рік",
        "yield_label": "Урожайність, т/га",
        "actual_title": "Факт vs прогноз (test) — {crop} ({model})",
        "scatter_x": "Факт (т/га)",
        "scatter_y": "Прогноз (т/га)",
        "scatter_title": "{crop} ({model}) — MAE={mae:.2f}, MAPE={mape:.1f}%",
        "shap_title": "SHAP топ-10 ознак — {crop} ({model})",
        "shap_xlabel": "Mean |SHAP|",
        "shap_ylabel": "Ознака",
    },
    "en": {
        "legend_actual": "Actual",
        "legend_pred": "Prediction",
        "year_label": "Year",
        "yield_label": "Yield, t/ha",
        "actual_title": "Actual vs forecast (test) — {crop} ({model})",
        "scatter_x": "Actual (t/ha)",
        "scatter_y": "Prediction (t/ha)",
        "scatter_title": "{crop} ({model}) — MAE={mae:.2f}, MAPE={mape:.1f}%",
        "shap_title": "SHAP top-10 features — {crop} ({model})",
        "shap_xlabel": "Mean |SHAP|",
        "shap_ylabel": "Feature",
    },
}


@dataclass
class CropDataset:
    crop: str
    years: pd.Series
    features: pd.DataFrame
    target: pd.Series


def slugify(value: str) -> str:
    if value in MANUAL_SLUGS:
        return MANUAL_SLUGS[value]
    normalised = unicodedata.normalize("NFKD", value)
    ascii_value = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value)
    slug = ascii_value.strip("_").lower()
    if slug:
        return slug
    return f"crop_{abs(hash(value)) % 10000}"


def classify_split(year: int) -> str:
    if year <= SPLIT_TRAIN_END:
        return "train"
    if year <= SPLIT_VAL_END:
        return "validation"
    return "test"


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Generate features before training.")
    df = pd.read_parquet(path)
    return df


def prepare_crop_dataset(df: pd.DataFrame, crop: str) -> Optional[CropDataset]:
    subset = df[df["group_or_crop"] == crop].copy()
    if subset.empty:
        console.log(f"[yellow]No data available for crop {crop}; skipping.[/yellow]")
        return None

    subset = subset.sort_values("year")
    subset["year"] = subset["year"].astype(int)
    numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()

    for col in subset.columns:
        if col not in numeric_cols and col not in {"region", "group_or_crop"}:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")

    feature_cols = [
        col
        for col in subset.columns
        if col
        not in {
            "region",
            "group_or_crop",
            "year",
            "Yield_t_ha",
            "Yield_anom",
        }
    ]

    X = subset[feature_cols].copy()
    # drop columns without data
    X = X.dropna(axis=1, how="all")
    # forward/backward fill within crop, then mean
    X = X.ffill().bfill()
    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="any")
    std = X.std(axis=0)
    non_constant = std[std > 0].index.tolist()
    X = X[non_constant]
    if X.shape[1] == 0:
        console.log(f"[yellow]All features are constant for {crop}; skipping.[/yellow]")
        return None

    target = subset["Yield_t_ha"].astype(float)
    if target.isna().all():
        console.log(f"[yellow]No target values for {crop}; skipping.[/yellow]")
        return None

    return CropDataset(
        crop=crop,
        years=subset["year"],
        features=X,
        target=target,
    )


def build_scenario_dataset(dataset: CropDataset, scenario: str) -> Optional[CropDataset]:
    features = dataset.features.copy()
    if scenario == "lag_only":
        allowed_cols = [
            col for col in features.columns if col.endswith(LAG_FEATURE_SUFFIX) or col.startswith(MA_PREFIX)
        ]
        if "Yield_t_ha_lag1" not in allowed_cols and "Yield_t_ha_lag1" in features.columns:
            allowed_cols.append("Yield_t_ha_lag1")
        features = features.loc[:, [col for col in allowed_cols if col in features.columns]]
    elif scenario == "in_season":
        # Use all available features.
        pass
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if features.empty or features.shape[1] == 0:
        console.log(f"[yellow]Skipping scenario {scenario} for {dataset.crop}: no features available.[/yellow]")
        return None

    return CropDataset(
        crop=dataset.crop,
        years=dataset.years.copy(),
        features=features,
        target=dataset.target.copy(),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_array = np.abs((y_true - y_pred) / y_true) * 100
        mape_array = mape_array[~np.isinf(mape_array)]
        mape = float(np.nanmean(mape_array)) if mape_array.size else float("nan")
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": mape}


def walk_forward_predictions(
    model_name: str,
    dataset: CropDataset,
    scenario: str,
) -> List[Dict[str, float]]:
    years_unique = sorted(dataset.years.unique())
    evaluation_years = [year for year in years_unique if classify_split(year) in {"validation", "test"}]
    results: List[Dict[str, float]] = []

    for year in evaluation_years:
        train_mask = dataset.years < year
        test_mask = dataset.years == year
        if not train_mask.any() or not test_mask.any():
            continue

        model = build_model(model_name)
        X_train = dataset.features.loc[train_mask]
        y_train = dataset.target.loc[train_mask]
        X_test = dataset.features.loc[test_mask]
        y_test = dataset.target.loc[test_mask]

        if X_train.empty or X_test.empty:
            continue

        train_std = X_train.std(axis=0)
        variable_columns = train_std[train_std > 0].index.tolist()
        if not variable_columns:
            console.log(f"[yellow]Skipping {model_name} {dataset.crop} {year}: no varying features.[/yellow]")
            continue
        X_train = X_train[variable_columns]
        X_test = X_test[variable_columns]
        feature_count = len(variable_columns)

        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to train {model_name} for crop {dataset.crop}: {exc}"
            ) from exc

        for actual, predicted in zip(y_test.to_numpy(), predictions):
            metrics = compute_metrics(np.array([actual]), np.array([predicted]))
            results.append(
                {
                    "model": model_name,
                    "crop": dataset.crop,
                    "year": int(year),
                    "split": classify_split(year),
                    "scenario": scenario,
                    "n_features": feature_count,
                    "actual": float(actual),
                    "predicted": float(predicted),
                    **metrics,
                }
            )

    return results


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    grouped = (
        metrics_df.groupby(["scenario", "model", "crop", "split"], as_index=False)[["MAE", "RMSE", "MAPE"]]
        .mean()
        .assign(year="avg")
    )
    return pd.concat([metrics_df, grouped], ignore_index=True, sort=False)


def plot_actual_vs_predicted(
    metrics_df: pd.DataFrame,
    crop: str,
    model: str,
    scenario: str,
    language: str,
) -> None:
    if language not in TRAIN_LANG:
        raise ValueError(f"Unsupported language: {language}")
    lang_cfg = TRAIN_LANG[language]
    crop_label = CROP_NAME_TRANSLATIONS.get(language, {}).get(crop, crop)
    subset = metrics_df[
        (metrics_df["crop"] == crop)
        & (metrics_df["model"] == model)
        & (metrics_df["split"] == "test")
        & (metrics_df["scenario"] == scenario)
    ]
    if subset.empty:
        return
    subset = subset.sort_values("year")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(subset["year"], subset["actual"], marker="o", label=lang_cfg["legend_actual"])
    ax.plot(subset["year"], subset["predicted"], marker="o", label=lang_cfg["legend_pred"])
    ax.set_title(lang_cfg["actual_title"].format(crop=crop_label, model=model))
    ax.set_xlabel(lang_cfg["year_label"])
    ax.set_ylabel(lang_cfg["yield_label"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_dir = FIGURES_DIR / language
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        fig_dir / f"{model}_{slugify(crop)}_{scenario}_actual_vs_pred.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_scatter_actual_pred(
    metrics_df: pd.DataFrame,
    crop: str,
    model: str,
    scenario: str,
    language: str,
) -> None:
    if language not in TRAIN_LANG:
        raise ValueError(f"Unsupported language: {language}")
    lang_cfg = TRAIN_LANG[language]
    crop_label = CROP_NAME_TRANSLATIONS.get(language, {}).get(crop, crop)
    subset = metrics_df[
        (metrics_df["crop"] == crop)
        & (metrics_df["model"] == model)
        & (metrics_df["split"] == "test")
        & (metrics_df["scenario"] == scenario)
    ]
    if subset.empty:
        return

    y_true = subset["actual"].to_numpy()
    y_pred = subset["predicted"].to_numpy()
    mae = subset["MAE"].mean()
    mape_values = subset["MAPE"].replace([np.inf, -np.inf], np.nan).dropna()
    mape = float(mape_values.mean()) if not mape_values.empty else float("nan")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    line = np.linspace(min_val, max_val, 100)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, color="#1f77b4", edgecolor="black")
    ax.plot(line, line, color="red", linestyle="--", label="y = x")
    ax.set_xlabel(lang_cfg["scatter_x"])
    ax.set_ylabel(lang_cfg["scatter_y"])
    ax.set_title(lang_cfg["scatter_title"].format(crop=crop_label, model=model, mae=mae, mape=mape))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_dir = FIGURES_DIR / language
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        fig_dir / f"scatter_{model}_{slugify(crop)}_{scenario}_test.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def train_for_shap(model_name: str, dataset: CropDataset) -> Optional[Tuple[object, pd.DataFrame]]:
    """Train model on full train set and return the model with test features for SHAP."""
    years_unique = sorted(dataset.years.unique())
    evaluation_years = [year for year in years_unique if classify_split(year) == "test"]
    if not evaluation_years:
        console.log(f"[yellow]SHAP skipped: no test period ({model_name}, {dataset.crop}).[/yellow]")
        return None
    min_test_year = min(evaluation_years)

    train_mask = dataset.years < min_test_year
    test_mask = dataset.years.isin(evaluation_years)
    if not train_mask.any() or not test_mask.any():
        console.log(f"[yellow]SHAP skipped: insufficient data ({model_name}, {dataset.crop}).[/yellow]")
        return None

    X_train = dataset.features.loc[train_mask]
    y_train = dataset.target.loc[train_mask]
    X_test = dataset.features.loc[test_mask]

    train_std = X_train.std(axis=0)
    variable_columns = train_std[train_std > 0].index.tolist()
    if not variable_columns:
        console.log(f"[yellow]SHAP skipped: features are constant ({model_name}, {dataset.crop}).[/yellow]")
        return None

    X_train = X_train[variable_columns]
    X_test = X_test[variable_columns]

    model = build_model(model_name)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to train {model_name} for SHAP (crop {dataset.crop}): {exc}"
        ) from exc
    return model, X_test


def plot_shap_importance(
    model: object,
    X_data: pd.DataFrame,
    model_name: str,
    crop: str,
    scenario: str,
    language: str,
) -> None:
    if language not in TRAIN_LANG:
        raise ValueError(f"Unsupported language: {language}")
    lang_cfg = TRAIN_LANG[language]
    crop_label = CROP_NAME_TRANSLATIONS.get(language, {}).get(crop, crop)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
    except Exception as exc:
        console.log(f"[yellow]SHAP could not be computed for {model_name} ({crop}): {exc}[/yellow]")
        return
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs, index=X_data.columns, name="mean_abs_shap")
    sorted_importance = feature_importance.sort_values(ascending=False)
    if sorted_importance.empty:
        return

    top_features = sorted_importance.head(10)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shap_csv = REPORTS_DIR / f"shap_top_{model_name}_{slugify(crop)}_{scenario}.csv"
    top_features.to_frame(name="mean_abs_shap").to_csv(shap_csv)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_features.sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_title(lang_cfg["shap_title"].format(crop=crop_label, model=model_name))
    ax.set_xlabel(lang_cfg["shap_xlabel"])
    ax.set_ylabel(lang_cfg["shap_ylabel"])
    ax.grid(True, axis="x", alpha=0.3)
    fig_dir = FIGURES_DIR / language
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        fig_dir / f"shap_{model_name}_{slugify(crop)}_{scenario}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_model(model_name: str):
    if model_name == "elasticnet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
                ),
            ]
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError(
                "XGBoost is unavailable (check that libomp is installed). "
                "Install libomp or remove xgboost from the model list."
            ) from exc

        try:
            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                objective="reg:squarederror",
                verbosity=0,
            )
        except Exception as exc:
            raise RuntimeError(
                "XGBoost could not be initialised (libxgboost missing). "
                "Install libomp or remove xgboost from the model list."
            ) from exc
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError(
                "LightGBM is unavailable (check that libomp is installed). "
                "Install libomp or remove lightgbm from the model list."
            ) from exc

        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
             min_child_samples=1,
             min_data_in_leaf=1,
             min_data_in_bin=1,
             verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_models(
    features_df: pd.DataFrame,
    models: Iterable[str] = ("elasticnet", "xgboost", "lightgbm"),
    scenarios: Iterable[str] = SCENARIOS,
    languages: Iterable[str] = ("uk", "en"),
) -> pd.DataFrame:
    all_records: List[Dict[str, float]] = []
    language_list = [lang for lang in languages]
    if not language_list:
        language_list = ["uk"]
    for language in language_list:
        if language not in TRAIN_LANG:
            raise ValueError(f"Unsupported language: {language}")

    for crop in TARGET_CROPS:
        base_dataset = prepare_crop_dataset(features_df, crop)
        if base_dataset is None:
            continue

        for scenario in scenarios:
            scenario_dataset = build_scenario_dataset(base_dataset, scenario)
            if scenario_dataset is None:
                continue

            for model_name in models:
                console.print(f"[cyan]Model {model_name} — crop {crop} — scenario {scenario}[/cyan]")
                try:
                    predictions = walk_forward_predictions(model_name, scenario_dataset, scenario)
                except RuntimeError as exc:
                    console.log(f"[yellow]{exc}[/yellow]")
                    continue
                all_records.extend(predictions)

                if predictions:
                    metrics_df = pd.DataFrame(predictions)
                    for language in language_list:
                        plot_actual_vs_predicted(metrics_df, crop, model_name, scenario, language)
                        plot_scatter_actual_pred(metrics_df, crop, model_name, scenario, language)

                if model_name in {"xgboost", "lightgbm"}:
                    try:
                        shap_artifacts = train_for_shap(model_name, scenario_dataset)
                    except RuntimeError as exc:
                        console.log(f"[yellow]{exc}[/yellow]")
                        shap_artifacts = None
                    if shap_artifacts:
                        shap_model, shap_X = shap_artifacts
                        for language in language_list:
                            plot_shap_importance(shap_model, shap_X, model_name, crop, scenario, language)

    metrics_df = pd.DataFrame(all_records)
    if not metrics_df.empty:
        ordered_cols = [
            "scenario",
            "model",
            "crop",
            "split",
            "year",
            "actual",
            "predicted",
            "MAE",
            "RMSE",
            "MAPE",
        ]
        existing_cols = [col for col in ordered_cols if col in metrics_df.columns]
        remaining = [col for col in metrics_df.columns if col not in existing_cols]
        metrics_df = metrics_df[existing_cols + remaining]
    return metrics_df


@app.command("poltava")
def poltava_command(
    features_path: Path = typer.Option(FEATURES_PATH, exists=True, help="Path to the feature parquet file."),
    languages: str = typer.Option("uk,en", help="Comma-separated list of languages for plot generation (uk,en)."),
) -> None:
    """Run model training/evaluation for the Poltava dataset."""
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    if not language_list:
        language_list = ["uk"]
    for language in language_list:
        if language not in TRAIN_LANG:
            raise ValueError(f"Unsupported language: {language}")
    features_df = load_features(features_path)
    metrics_df = train_models(features_df, languages=language_list)

    if metrics_df.empty:
        console.print("[red]Unable to compute metrics — please verify the data.[/red]")
        return

    utils.ensure_directories([METRICS_PATH.parent])

    predictions_export = metrics_df[
        ["year", "crop", "model", "scenario", "split", "actual", "predicted"]
    ].rename(
        columns={
            "actual": "y_true",
            "predicted": "y_pred",
        }
    )
    predictions_path = REPORTS_DIR / "predictions.csv"
    predictions_export.to_csv(predictions_path, index=False)
    console.print(f"[green]Predictions saved to {predictions_path}[/green]")

    metrics_export = metrics_df[
        ["year", "crop", "model", "scenario", "split", "MAE", "RMSE", "MAPE", "n_features"]
    ].rename(
        columns={
            "MAE": "mae",
            "RMSE": "rmse",
            "MAPE": "mape",
            "n_features": "n_features",
        }
    )
    metrics_export.columns = [col.lower() for col in metrics_export.columns]
    metrics_export.to_csv(METRICS_PATH, index=False)
    console.print(f"[green]Metrics saved to {METRICS_PATH}[/green]")

    try:
        eda.plot_trends(features_df, eda.TARGET_CROPS, language_list)
    except Exception as exc:
        console.log(f"[yellow]Failed to render trend plots: {exc}[/yellow]")

    test_predictions = predictions_export[predictions_export["split"] == "test"].copy()
    if not test_predictions.empty:
        test_predictions = test_predictions.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
        summary = (
            test_predictions.groupby(["scenario", "model", "crop"])
            .apply(
                lambda g: pd.Series(
                    {
                        "mae": float(np.abs(g["y_true"] - g["y_pred"]).mean()),
                        "rmse": float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean())),
                        "mape": float((np.abs((g["y_true"] - g["y_pred"]) / g["y_true"]) * 100).mean()),
                        "n": int(len(g)),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["scenario", "model", "crop", "mae", "rmse", "mape", "n"])

    summary_path = REPORTS_DIR / "metrics_by_scenario.csv"
    summary.to_csv(summary_path, index=False)
    console.print(f"[green]Scenario summary saved to {summary_path}[/green]")

    if not summary.empty:
        leaderboard = summary.loc[summary.groupby("crop")["mae"].idxmin()].reset_index(drop=True)
        leaderboard_path = REPORTS_DIR / "metrics_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        console.print(f"[green]Leaderboard saved to {leaderboard_path}[/green]")


if __name__ == "__main__":
    app()
