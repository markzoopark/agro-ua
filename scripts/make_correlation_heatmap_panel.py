"""Create 3-panel Pearson correlation heatmaps for lag factors vs Yield_anom (en only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR_EN = BASE_DIR / "reports" / "figures_article" / "en"

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

OUTPUT = FIG_DIR_EN / "correlation_heatmap_panel.png"


def load_corr(crop: str) -> pd.DataFrame:
    slug = CROP_SLUG[crop]
    path = REPORTS_DIR / f"correlations_{slug}.csv"
    df = pd.read_csv(path)
    df = df[["factor", "pearson_anom"]].dropna(subset=["pearson_anom"]).copy()
    df["abs"] = df["pearson_anom"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs")
    return df


def main() -> None:
    sns.set(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, crop in zip(axes, CROPS):
        df = load_corr(crop)
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
            xticklabels=[""],
        )
        ax.set_xlabel("Yield_anom", fontsize=9)
        ax.set_ylabel("Feature", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(
            f"{CROP_EN[crop]} - Pearson r with Yield_anom",
            fontsize=10,
        )

    # Add (a), (b), (c) labels in axes coordinates, consistently across panels
    letters = ["(a)", "(b)", "(c)"]
    for ax, letter in zip(axes, letters):
        ax.text(
            -0.15,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
