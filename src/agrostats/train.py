"""Training and backtesting pipelines for agrostats models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import typer
from pydantic import BaseModel, Field, validator
from rich.console import Console
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler


console = Console()
app = typer.Typer(help="Model training and backtesting commands.")


ModelName = Literal["xgboost", "lightgbm", "random_forest"]


class TrainConfig(BaseModel):
    """Configuration for a single training run."""

    data_path: Path = Field(..., description="Path to the CSV file with training data.")
    target: str = Field(..., description="Target column name.")
    features: Sequence[str] = Field(..., description="Feature column names.")
    model: ModelName = Field("xgboost", description="Model family to train.")
    date_column: Optional[str] = Field(None, description="Date column for backtesting splits.")
    test_size: float = Field(0.2, ge=0.05, le=0.5, description="Test set size for holdout split.")
    n_splits: int = Field(5, ge=2, description="Number of folds for backtesting.")
    scale_features: bool = Field(True, description="Whether to standardise features.")
    random_state: int = Field(42, description="Random seed.")

    @validator("features")
    def _ensure_features_present(cls, value: Sequence[str]) -> Sequence[str]:
        if not value:
            raise ValueError("At least one feature must be provided.")
        return value


def load_config(path: Path) -> TrainConfig:
    """Load configuration from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def load_dataset(
    path: Path,
    features: Sequence[str],
    target: str,
    *,
    date_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features/target."""
    df = pd.read_csv(path)
    missing_columns = set(features) | {target} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {sorted(missing_columns)}")
    X = df.loc[:, features].copy()
    if date_column and date_column in X.columns:
        X[date_column] = pd.to_datetime(X[date_column])
    y = df[target].copy()
    return X, y


def make_model(name: ModelName, random_state: int):
    """Instantiate a model by name."""
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def backtest(
    config: TrainConfig,
    X: pd.DataFrame,
    y: pd.Series,
) -> list[dict[str, float]]:
    """Perform time-series backtesting."""
    if not config.date_column:
        raise ValueError("Backtesting requires a date column in the dataset.")

    df = X.copy()
    df[config.target] = y
    df = df.sort_values(config.date_column)
    X_sorted = df.loc[:, X.columns]
    y_sorted = df[config.target]

    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    metrics: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_sorted), start=1):
        model = make_model(config.model, config.random_state)
        X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
        y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]

        if config.scale_features:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        fold_metrics = evaluate(y_test.to_numpy(), prediction)
        fold_metrics["fold"] = fold
        metrics.append(fold_metrics)
        console.print(f"[cyan]Fold {fold}[/cyan]: {fold_metrics}")

    return metrics


def train_holdout(config: TrainConfig, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Train a model with a simple train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    scaler = None
    if config.scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    model = make_model(config.model, config.random_state)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    metrics = evaluate(y_test.to_numpy(), prediction)
    console.print(f"[green]Holdout metrics[/green]: {metrics}")

    artefacts = {"model": model}
    if scaler is not None:
        artefacts["scaler"] = scaler
    return {"metrics": metrics, "artefacts": artefacts}


@app.command("run")
def run_command(config_path: Path, model_dir: Optional[Path] = typer.Option(Path("models"), help="Artefact dir.")) -> None:
    """Run training with configuration provided as JSON."""
    config = load_config(config_path)
    X, y = load_dataset(config.data_path, config.features, config.target, date_column=config.date_column)

    if config.date_column and config.date_column not in X.columns:
        raise ValueError(f"date_column '{config.date_column}' not found among feature columns.")

    if config.date_column:
        metrics = backtest(config, X, y)
        console.print(f"[bold green]Backtest complete[/bold green]")
    else:
        result = train_holdout(config, X, y)
        metrics = [result["metrics"]]
        if model_dir:
            model_dir = model_dir or Path("models")
            model_dir.mkdir(parents=True, exist_ok=True)
            artefact_path = model_dir / f"{config.model}_model.pkl"
            import joblib

            joblib.dump(result["artefacts"], artefact_path)
            console.print(f"[green]Saved artefacts to {artefact_path}[/green]")

    console.print({"metrics": metrics})


if __name__ == "__main__":
    app()
