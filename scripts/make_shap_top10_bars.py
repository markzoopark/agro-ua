"""Generate horizontal bar charts of SHAP top-10 features for lag_only scenario."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FIG = REPORTS_DIR / "figures_article" / "en" / "shap_top10_bars.png"

PANEL_CONFIG = [
    ("shap_top_lightgbm_pshenytsia_lag_only.csv", "Wheat", "LightGBM", "(a)"),
    ("shap_top_xgboost_kukurudza_lag_only.csv", "Corn", "XGBoost", "(b)"),
    ("shap_top_lightgbm_sonyashnyk_lag_only.csv", "Sunflower", "LightGBM", "(c)"),
]


def load_shap_table(filename: str) -> pd.DataFrame:
    path = REPORTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"SHAP table not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df.columns = ["value"]
    df = df.head(10).sort_values("value", ascending=False)
    return df


def main() -> None:
    # Set total fig size ~14x4.5 so each subplot ~4.5 in wide
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, (csv_name, crop_label, model_label, letter) in zip(axes, PANEL_CONFIG):
        df = load_shap_table(csv_name)

        ax.barh(df.index, df["value"], color="#1f77b4")
        ax.invert_yaxis()  # highest at top
        ax.set_xlabel("Mean |SHAP|", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        title_text = f'SHAP top-10 features — {crop_label} ({model_label})'
        # Wrap long titles (e.g. Sunflower (LightGBM)) onto two lines to avoid clipping.
        if len(title_text) > 32:
            parts = title_text.split(' — ', 1)
            if len(parts) == 2:
                title_text = parts[0] + ' —\n' + parts[1]
        ax.set_title(title_text, fontsize=10, loc='center')
        ax.grid(True, axis="x", alpha=0.2)

    fig.subplots_adjust(left=0.34, right=0.98, top=0.9, bottom=0.18, wspace=0.5)
    fig.tight_layout(rect=[0.02, 0, 1.0, 0.95])

    # Add (a), (b), (c) labels slightly to the left of each subplot in figure coordinates
    # so that they do not overlap titles or bars.
    letters = ["(a)", "(b)", "(c)"]
    for ax, letter in zip(axes, letters):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.03,
            min(1.0, bbox.y1 + 0.02),
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
