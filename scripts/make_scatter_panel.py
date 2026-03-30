"""Combine single-crop scatter plots into a 3-panel figure (en only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR_EN = BASE_DIR / "reports" / "figures_article" / "en"

PANELS = [
    ("scatter_lightgbm_pshenytsia_lag_only.png", "(a)"),
    ("scatter_xgboost_kukurudza_lag_only.png", "(b)"),
    ("scatter_elasticnet_sonyashnyk_lag_only.png", "(c)"),
]

OUTPUT = FIG_DIR_EN / "scatter_panel.png"


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (filename, letter) in zip(axes, PANELS):
        path = FIG_DIR_EN / filename
        if not path.exists():
            ax.set_visible(False)
            continue
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.02,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
