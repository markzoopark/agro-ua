"""Create 2x3 panel with ACF and PACF for Wheat, Corn, Sunflower."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
OUTPUT_FIG = BASE_DIR / "reports" / "figures_article" / "en" / "acf_pacf_crops.png"

CROPS = ["Пшениця", "Кукурудза", "Соняшник"]
CROP_EN = {
    "Пшениця": "Wheat",
    "Кукурудза": "Corn",
    "Соняшник": "Sunflower",
}


def load_series() -> dict[str, pd.Series]:
    df = pd.read_parquet(FEATURES_PATH)
    series_by_crop: dict[str, pd.Series] = {}
    for crop in CROPS:
        subset = df[df["group_or_crop"] == crop].copy()
        subset = subset.sort_values("year")
        y = subset["Yield_t_ha"].dropna().astype(float)
        series_by_crop[crop] = y
    return series_by_crop


def main() -> None:
    series_by_crop = load_series()

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=False, sharey=False)
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # Top row: ACF (a–c)
    for idx, crop in enumerate(CROPS):
        ax = axes[0, idx]
        y = series_by_crop[crop]
        plot_acf(y, ax=ax, lags=5)
        ax.set_title(f"{CROP_EN[crop]} - ACF", fontsize=10)
        ax.set_xlabel("Lag", fontsize=9)
        ax.set_ylabel("Correlation", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    # Bottom row: PACF (d–f)
    for idx, crop in enumerate(CROPS):
        ax = axes[1, idx]
        y = series_by_crop[crop]
        plot_pacf(y, ax=ax, lags=5, method="yw")
        ax.set_title(f"{CROP_EN[crop]} - PACF", fontsize=10)
        ax.set_xlabel("Lag", fontsize=9)
        ax.set_ylabel("Correlation", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout(rect=[0.02, 0, 1, 0.95])

    # Add subplot labels (a)–(f) slightly outside axes to avoid overlap
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, letter in zip(axes.flatten(), letters):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.035,
            min(1.0, bbox.y1 + 0.015),
            letter,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
