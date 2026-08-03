import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrostats.baselines import summarise_metrics
from agrostats.modeling import aggregate_predictions, validation_selected_test_metrics
from agrostats.revision import _format_climate_comparison


def test_aggregate_predictions_uses_pooled_rmse():
    predictions = pd.DataFrame(
        {
            "crop": ["A", "A", "A"],
            "model": ["m", "m", "m"],
            "year": [2022, 2023, 2024],
            "actual": [1.0, 2.0, 3.0],
            "predicted": [1.0, 0.0, 4.0],
            "mae": [0.0, 2.0, 1.0],
            "rmse": [0.0, 2.0, 1.0],
            "mape": [0.0, 100.0, 100.0 / 3.0],
        }
    )

    result = aggregate_predictions(predictions, group_cols=["crop", "model"])

    assert result.iloc[0]["mae"] == pytest.approx(1.0)
    assert result.iloc[0]["rmse"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert result.iloc[0]["rmse"] != pytest.approx(result.iloc[0]["mae"])


def test_baseline_summary_uses_pooled_rmse():
    rows = pd.DataFrame(
        {
            "baseline": ["b", "b", "b", "b"],
            "crop": ["A", "A", "A", "A"],
            "split": ["validation", "test", "test", "test"],
            "y_true": [10.0, 1.0, 2.0, 3.0],
            "y_pred": [0.0, 1.0, 0.0, 4.0],
            "mae": [10.0, 0.0, 2.0, 1.0],
            "rmse": [10.0, 0.0, 2.0, 1.0],
            "mape": [100.0, 0.0, 100.0, 100.0 / 3.0],
        }
    )

    result = summarise_metrics(rows)

    assert result.iloc[0]["n"] == 3
    assert result.iloc[0]["mae"] == pytest.approx(1.0)
    assert result.iloc[0]["rmse"] == pytest.approx(math.sqrt(5.0 / 3.0))


def test_validation_selection_does_not_use_test_ranking():
    predictions = pd.DataFrame(
        {
            "crop": ["A"] * 8,
            "model": ["validation_winner"] * 4 + ["test_winner"] * 4,
            "split": ["validation", "validation", "test", "test"] * 2,
            "actual": [1.0, 1.0, 1.0, 1.0] * 2,
            "predicted": [1.1, 0.9, 3.0, 3.0, 1.3, 0.7, 1.0, 1.0],
        }
    )

    result = validation_selected_test_metrics(
        predictions,
        method_col="model",
        actual_col="actual",
        predicted_col="predicted",
    )

    assert result.iloc[0]["selected_method"] == "validation_winner"
    assert result.iloc[0]["validation_mae"] == pytest.approx(0.1)
    assert result.iloc[0]["test_mae"] == pytest.approx(2.0)


def test_climate_post_hoc_and_validation_selection_are_separate():
    rows = []
    for feature_set in ("agro_only", "agro_climate"):
        rows.extend(
            [
                {"crop": "Пшениця", "feature_set": feature_set, "model": "validation_winner", "validation_mae": 0.1, "validation_rmse": 0.1, "validation_mape": 1.0, "test_mae": 1.0, "test_rmse": 1.0, "test_mape": 10.0},
                {"crop": "Пшениця", "feature_set": feature_set, "model": "test_winner", "validation_mae": 0.3, "validation_rmse": 0.3, "validation_mape": 3.0, "test_mae": 0.2, "test_rmse": 0.2, "test_mape": 2.0},
            ]
        )
    candidates = pd.DataFrame(rows)

    strict = _format_climate_comparison(candidates, select_on="validation")
    post_hoc = _format_climate_comparison(candidates, select_on="test")

    assert strict.iloc[0]["base_model"] == "validation_winner"
    assert post_hoc.iloc[0]["base_model"] == "test_winner"
