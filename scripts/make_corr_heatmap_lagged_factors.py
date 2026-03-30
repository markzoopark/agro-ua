"""Create 3-panel heatmap of Pearson r between Yield_anom and lagged factors."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FIG = REPORTS_DIR / "figures_article" / "en" / "corr_heatmap_lagged_factors.png"

CROPS = ["Пшениця", "Кукурудза", "Соняшник"]
CROP_EN = {
    "Пшениця": "Wheat",
    "Кукурудза": "Corn",
    "Соняшник": "Sunflower",
}
CROP_SLUG = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}


def load_corr(crop: str) -> pd.DataFrame:
    slug = CROP_SLUG[crop]
    path = REPORTS_DIR / f"correlations_{slug}.csv"
    df = pd.read_csv(path)
    # Keep factor and Pearson for anomaly
    df = df[["factor", "pearson_anom"]].copy()
    df = df.dropna(subset=["pearson_anom"])
    df["abs"] = df["pearson_anom"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs")
    return df


def main() -> None:
    sns.set(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=False, sharey=False)
    letters = ["(a)", "(b)", "(c)"]

    for ax, crop, letter in zip(axes, CROPS, letters):
        df = load_corr(crop)
        # Matrix is (n_factors x 1)
        values = df["pearson_anom"].values[:, None]
        sns.heatmap(
            values,
            ax=ax,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            cbar=True,
            cbar_kws={"shrink": 0.7},
            annot=np.round(values, 2),
            fmt=".2f",
            yticklabels=df["factor"].tolist(),
            xticklabels=["Yield_anom"],
        )
        ax.set_xlabel("Yield_anom", fontsize=9)
        ax.set_ylabel("Feature", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(
            f"{CROP_EN[crop]} - Pearson r with Yield_anom",
            fontsize=10,
        )
        # Move colorbar to the right of each subplot (default position is fine)

        ax.text(
            -0.22,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
