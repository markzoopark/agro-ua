"""Generate publication-ready figures in Ukrainian and English."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures_article"
PROCESSED_FEATURES = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
METRICS_CSV = REPORTS_DIR / "metrics.csv"
PREDICTIONS_CSV = REPORTS_DIR / "predictions.csv"

TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
CROP_SLUG = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}

LANG_CONFIG: Dict[str, Dict[str, str]] = {
    "uk": {
        "crop_names": {
            "Пшениця": "Пшениця",
            "Кукурудза": "Кукурудза",
            "Соняшник": "Соняшник",
        },
        "legend": "Культура",
        "actual": "Факт",
        "pred": "Прогноз",
        "year": "Рік",
        "yield": "Урожайність, т/га",
        "scatter_x": "Факт (т/га)",
        "scatter_y": "Прогноз (т/га)",
        "scatter_title": "{crop} ({model}) — MAE={mae:.2f}, MAPE={mape:.1f}%",
        "trends": {
            "Yield_t_ha": ("Урожайність", "т/га"),
            "Area_ha": ("Посівна площа", "га"),
            "N_kg_ha": ("Азотні добрива", "кг/га"),
            "P2O5_kg_ha": ("Фосфорні добрива", "кг/га"),
            "K_kg_ha": ("Калійні добрива", "кг/га"),
            "Irrig_mm": ("Зрошення", "мм"),
        },
        "heatmap_title": "Кореляції лагових факторів — {crop}",
        "heatmap_cb": "Pearson r",
        "shap_title": "SHAP топ-10 ознак — {crop} ({model})",
        "shap_xlabel": "Середнє |SHAP|",
        "shap_ylabel": "Ознака",
    },
    "en": {
        "crop_names": {
            "Пшениця": "Wheat",
            "Кукурудза": "Corn",
            "Соняшник": "Sunflower",
        },
        "legend": "Crop",
        "actual": "Actual",
        "pred": "Forecast",
        "year": "Year",
        "yield": "Yield, t/ha",
        "scatter_x": "Actual (t/ha)",
        "scatter_y": "Predicted (t/ha)",
        "scatter_title": "{crop} ({model}) — MAE={mae:.2f}, MAPE={mape:.1f}%",
        "trends": {
            "Yield_t_ha": ("Yield", "t/ha"),
            "Area_ha": ("Sown area", "ha"),
            "N_kg_ha": ("Nitrogen fertilisers", "kg/ha"),
            "P2O5_kg_ha": ("Phosphorus fertilisers", "kg/ha"),
            "K_kg_ha": ("Potassium fertilisers", "kg/ha"),
            "Irrig_mm": ("Irrigation", "mm"),
        },
        "heatmap_title": "Lag-factor correlations — {crop}",
        "heatmap_cb": "Pearson r",
        "shap_title": "SHAP top-10 features — {crop} ({model})",
        "shap_xlabel": "Mean |SHAP|",
        "shap_ylabel": "Feature",
    },
}

TREND_COLUMNS = [
    "Yield_t_ha",
    "Area_ha",
    "N_kg_ha",
    "P2O5_kg_ha",
    "K_kg_ha",
    "Irrig_mm",
]


def ensure_language(lang: str) -> None:
    if lang not in LANG_CONFIG:
        raise ValueError(f"Unsupported language: {lang}")


def get_crop_label(crop: str, lang: str) -> str:
    ensure_language(lang)
    return LANG_CONFIG[lang]["crop_names"].get(crop, crop)


def style_matplotlib() -> None:
    style_name = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "classic"
    plt.style.use(style_name)
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 200,
    })


def ensure_dirs(languages: Iterable[str]) -> Dict[str, Path]:
    dirs = {}
    for lang in languages:
        path = FIGURES_DIR / lang
        path.mkdir(parents=True, exist_ok=True)
        dirs[lang] = path
    return dirs


def plot_trends(features: pd.DataFrame, languages: Iterable[str]) -> None:
    subset = features[features["group_or_crop"].isin(TARGET_CROPS)].copy()
    subset = subset.sort_values("year")
    dirs = ensure_dirs(languages)

    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        axes = axes.flatten()
        for ax, column in zip(axes, TREND_COLUMNS):
            title, unit = cfg["trends"][column]
            for crop in TARGET_CROPS:
                crop_df = subset[subset["group_or_crop"] == crop]
                ax.plot(
                    crop_df["year"],
                    crop_df[column],
                    marker="o",
                    label=get_crop_label(crop, lang),
                )
            ax.set_title(f"{title} ({unit})")
            ax.set_xlabel(cfg["year"])
            ax.set_ylabel(unit)
            ax.set_xlim(2010, 2024)
            ax.set_xticks(range(2010, 2025, 2))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(TARGET_CROPS), frameon=False, title=cfg["legend"])
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(dirs[lang] / "poltava_trends.png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def plot_prediction_series(predictions: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    crop_label = get_crop_label(crop, lang)
    subset = predictions[
        (predictions["crop"] == crop)
        & (predictions["model"] == model)
        & (predictions["scenario"] == "lag_only")
    ].sort_values("year")
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(subset["year"], subset["y_true"], marker="o", linewidth=2, label=cfg["actual"])
    ax.plot(subset["year"], subset["y_pred"], marker="o", linewidth=2, label=cfg["pred"], linestyle="--")
    ax.set_title(f"{crop_label} ({model})")
    ax.set_xlabel(cfg["year"])
    ax.set_ylabel(cfg["yield"])
    ax.set_xlim(2018, 2024)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    (FIGURES_DIR / lang).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / lang / f"line_{model}_{CROP_SLUG[crop]}_lag_only.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def plot_prediction_scatter(predictions: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    crop_label = get_crop_label(crop, lang)
    subset = predictions[
        (predictions["crop"] == crop)
        & (predictions["model"] == model)
        & (predictions["scenario"] == "lag_only")
        & (predictions["split"] == "test")
    ]
    if subset.empty:
        return
    y_true = subset["y_true"].to_numpy()
    y_pred = subset["y_pred"].to_numpy()
    mae = float(np.abs(y_true - y_pred).mean())
    mape = float((np.abs((y_true - y_pred) / y_true) * 100).mean())
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    line = np.linspace(min_val, max_val, 100)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(y_true, y_pred, color="#1f77b4", edgecolor="black", s=70)
    ax.plot(line, line, color="#d62728", linestyle="--", linewidth=1.5, label="y = x")
    ax.set_xlabel(cfg["scatter_x"])
    ax.set_ylabel(cfg["scatter_y"])
    ax.set_title(cfg["scatter_title"].format(crop=crop_label, model=model, mae=mae, mape=mape))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / lang / f"scatter_{model}_{CROP_SLUG[crop]}_lag_only.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def plot_shap(shap_df: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    crop_label = get_crop_label(crop, lang)
    if shap_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    shap_df.sort_values("mean_abs_shap").plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_title(cfg["shap_title"].format(crop=crop_label, model=model))
    ax.set_xlabel(cfg["shap_xlabel"])
    ax.set_ylabel(cfg["shap_ylabel"])
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / lang / f"shap_{model}_{CROP_SLUG[crop]}_lag_only.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def plot_heatmap(corr_df: pd.DataFrame, crop: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    crop_label = get_crop_label(crop, lang)
    pearson = corr_df.set_index("factor")["pearson_yield"].sort_values(key=lambda s: np.abs(s), ascending=False)
    fig, ax = plt.subplots(figsize=(5.2, max(3, len(pearson) * 0.4)))
    matrix = pearson.values[:, None]
    cax = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels([cfg["heatmap_cb"]])
    ax.set_yticks(range(len(pearson)))
    ax.set_yticklabels(pearson.index)
    for i, value in enumerate(pearson.values):
        ax.text(0, i, f"{value:.2f}", ha="center", va="center", color="black")
    ax.set_title(cfg["heatmap_title"].format(crop=crop_label))
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / lang / f"correlation_heatmap_{CROP_SLUG[crop]}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def load_shap_table(model: str, crop: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"shap_top_{model}_{CROP_SLUG[crop]}_lag_only.csv"
    if not path.exists():
        return pd.DataFrame(columns=["mean_abs_shap"])
    df = pd.read_csv(path, index_col=0)
    df = df.rename_axis("feature").rename(columns={df.columns[0]: "mean_abs_shap"})
    return df.head(10)


def main() -> None:
    languages = ("uk", "en")
    style_matplotlib()
    ensure_dirs(languages)

    features = pd.read_parquet(PROCESSED_FEATURES)
    metrics = pd.read_csv(METRICS_CSV)
    predictions = pd.read_csv(PREDICTIONS_CSV)

    plot_trends(features, languages)

    best_models = {
        "Кукурудза": "xgboost",
        "Пшениця": "lightgbm",
        "Соняшник": "elasticnet",
    }

    for crop, model in best_models.items():
        for lang in languages:
            plot_prediction_series(predictions[predictions["scenario"] == "lag_only"], crop, model, lang)
            plot_prediction_scatter(predictions, crop, model, lang)
            shap_df = load_shap_table(model, crop)
            plot_shap(shap_df, crop, model, lang)
        corr_path = REPORTS_DIR / f"correlations_{CROP_SLUG[crop]}.csv"
        if corr_path.exists():
            corr_df = pd.read_csv(corr_path)
            for lang in languages:
                plot_heatmap(corr_df, crop, lang)


if __name__ == "__main__":
    main()
