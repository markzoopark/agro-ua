"""Create MAE bar charts for Excel baselines vs ML per crop."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FIG = REPORTS_DIR / "figures_article" / "en" / "mae_by_method_bars.png"

CROPS = ["Пшениця", "Кукурудза", "Соняшник"]
CROP_EN = {
    "Пшениця": "Wheat",
    "Кукурудза": "Corn",
    "Соняшник": "Sunflower",
}

EXCEL_METHODS = ["naive_lag1", "forecast_linear", "linest_lag_only"]
EXCEL_LABELS = {
    "naive_lag1": "Naive (t-1)",
    "forecast_linear": "FORECAST.LINEAR",
    "linest_lag_only": "LINEST + lags",
}


def load_baseline_mae() -> pd.DataFrame:
    df = pd.read_csv(REPORTS_DIR / "metrics_baselines_summary.csv")
    return df


def load_ml_mae() -> pd.DataFrame:
    df = pd.read_csv(REPORTS_DIR / "metrics_by_scenario.csv")
    df = df[df["scenario"] == "lag_only"].copy()
    idx = df.groupby("crop")["mae"].idxmin()
    best = df.loc[idx, ["crop", "model", "mae"]]
    return best


def main() -> None:
    baseline_df = load_baseline_mae()
    ml_df = load_ml_mae()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    letters = ["(a)", "(b)", "(c)"]

    for ax, crop, letter in zip(axes, CROPS, letters):
        crop_baselines = baseline_df[baseline_df["crop"] == crop]
        crop_ml = ml_df[ml_df["crop"] == crop].iloc[0]

        bars = []
        values = []
        labels = []
        colors = []

        for method in EXCEL_METHODS:
            row = crop_baselines[crop_baselines["baseline"] == method]
            if row.empty:
                continue
            value = float(row["mae"].iloc[0])
            bars.append(EXCEL_LABELS[method])
            values.append(value)
            colors.append("#82B0D9")

        bars.append(f"ML ({crop_ml['model']})")
        values.append(float(crop_ml["mae"]))
        colors.append("#F28E2B")

        positions = range(len(bars))
        rects = ax.bar(positions, values, color=colors)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(bars, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("MAE, t/ha")
        ax.set_title(f"{CROP_EN.get(crop, crop)}")
        ax.grid(True, axis="y", alpha=0.3)
        # Ensure y-axis ticks are visible on all subplots
        ax.tick_params(axis="y", labelleft=True)

        # Add a bit of headroom so value labels are not clipped
        ax.margins(y=0.15)
        offset = 0.02
        for rect, value in zip(rects, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )


    # Arrange layout and add global legend
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    excel_patch = mpatches.Patch(color="#82B0D9", label="Excel baselines")
    ml_patch = mpatches.Patch(color="#F28E2B", label="ML model")
    fig.legend(
        handles=[excel_patch, ml_patch],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )

    # Add (a), (b), (c) labels in figure coordinates
    letters = ["(a)", "(b)", "(c)"]
    for ax, letter in zip(axes, letters):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.03,
            bbox.y1 + 0.01,
            letter,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
