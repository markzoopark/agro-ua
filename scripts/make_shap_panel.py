"""Create a 3-panel SHAP figure from SHAP tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures_article" / "en"
OUTPUT_PATH = FIG_DIR / "shap_panel.png"

PANEL_CONFIG = [
    ("shap_top_lightgbm_pshenytsia_lag_only.csv", "Wheat"),
    ("shap_top_xgboost_kukurudza_lag_only.csv", "Corn"),
    ("shap_top_lightgbm_sonyashnyk_lag_only.csv", "Sunflower"),
]


def load_shap_table(filename: str) -> pd.DataFrame:
    path = BASE_DIR / "reports" / filename
    if not path.exists():
        raise FileNotFoundError(f"SHAP table not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df.columns = ["value"]
    df = df.head(10).sort_values("value", ascending=True)
    return df


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    letters = ["(a)", "(b)", "(c)"]

    for ax, (csv_name, title), letter in zip(axes, PANEL_CONFIG, letters):
        try:
            df = load_shap_table(csv_name)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        ax.barh(df.index, df["value"], color="#1f77b4")
        ax.set_xlabel("Mean |SHAP| value", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2)
        ax.text(
            -0.22,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
