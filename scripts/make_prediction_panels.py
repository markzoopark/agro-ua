"""Create a 3-panel figure with wheat, corn, and sunflower predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FIG = REPORTS_DIR / "figures_article" / "en" / "yield_predictions_panel.png"

CROP_ORDER = ["Пшениця", "Кукурудза", "Соняшник"]
CROP_EN = {
    "Пшениця": "Wheat",
    "Кукурудза": "Corn",
    "Соняшник": "Sunflower",
}


def select_best_models() -> dict[str, str]:
    """Return best ML model (lag_only) per crop."""
    metrics = pd.read_csv(REPORTS_DIR / "metrics_by_scenario.csv")
    lag_only = metrics[metrics["scenario"] == "lag_only"].copy()
    idx = lag_only.groupby("crop")["mae"].idxmin()
    best = lag_only.loc[idx, ["crop", "model"]]
    return {row["crop"]: row["model"] for _, row in best.iterrows()}


def select_best_baselines() -> dict[str, str]:
    """Return best Excel baseline per crop."""
    summary = pd.read_csv(REPORTS_DIR / "metrics_baselines_summary.csv")
    idx = summary.groupby("crop")["mae"].idxmin()
    best = summary.loc[idx, ["crop", "baseline"]]
    return {row["crop"]: row["baseline"] for _, row in best.iterrows()}


def load_predictions(crop: str, model: str) -> pd.DataFrame:
    """Load ML predictions for the specified crop/model."""
    df = pd.read_csv(REPORTS_DIR / "predictions.csv")
    subset = df[
        (df["crop"] == crop)
        & (df["model"] == model)
        & (df["scenario"] == "lag_only")
    ].copy()
    return subset.sort_values("year")


def load_baseline_predictions(crop: str, baseline: str) -> pd.DataFrame:
    """Load Excel baseline predictions."""
    df = pd.read_csv(REPORTS_DIR / "metrics_baselines.csv")
    subset = df[(df["crop"] == crop) & (df["baseline"] == baseline)].copy()
    return subset.sort_values("year")


def main() -> None:
    best_models = select_best_models()
    best_baselines = select_best_baselines()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    letters = ["(a)", "(b)", "(c)"]

    for ax, crop, letter in zip(axes, CROP_ORDER, letters):
        model = best_models.get(crop)
        baseline = best_baselines.get(crop)
        if not model or not baseline:
            ax.set_visible(False)
            continue

        ml_df = load_predictions(crop, model)
        base_df = load_baseline_predictions(crop, baseline)

        if ml_df.empty or base_df.empty:
            ax.set_visible(False)
            continue

        years = ml_df["year"].tolist()
        ax.plot(years, ml_df["y_true"], marker="o", color="black", linestyle="-", label="Actual")
        ax.plot(years, ml_df["y_pred"], marker="o", color="blue", linestyle="--", label="ML prediction")

        base_years = base_df["year"].tolist()
        ax.plot(
            base_years,
            base_df["y_pred"],
            marker="o",
            color="red",
            linestyle=":",
            label="Excel prediction",
        )

        ax.set_title(f"{CROP_EN.get(crop, crop)} ({model})")
        ax.set_ylabel("Yield (t/ha)")
        ax.set_xlabel("Year")
        ax.set_xticks(list(range(2018, 2025)))
        ax.set_xlim(2018, 2024)
        ax.legend()
        ax.text(
            0.02,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
