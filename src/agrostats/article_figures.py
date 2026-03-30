"""Generate publication-ready figures for the revised manuscript."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures_article"
FIGURES_BASE_DIR = REPORTS_DIR / "figures"
PROCESSED_FEATURES = BASE_DIR / "data" / "processed" / "agrostats_poltava_features.parquet"
PREDICTIONS_CSV = REPORTS_DIR / "predictions.csv"
METRICS_BY_SCENARIO_CSV = REPORTS_DIR / "metrics_by_scenario.csv"
BASELINE_SUMMARY_CSV = REPORTS_DIR / "metrics_baselines_summary.csv"
LAG_SENSITIVITY_CSV = REPORTS_DIR / "lag_sensitivity.csv"
CLIMATE_SENSITIVITY_CSV = REPORTS_DIR / "climate_sensitivity.csv"
CORR_TEMPLATE = REPORTS_DIR / "correlations_{crop}.csv"

TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")
CROP_SLUG = {
    "Пшениця": "pshenytsia",
    "Кукурудза": "kukurudza",
    "Соняшник": "sonyashnyk",
}

LANG_CONFIG: Dict[str, Dict[str, str | dict[str, str]]] = {
    "uk": {
        "crop_names": {"Пшениця": "Пшениця", "Кукурудза": "Кукурудза", "Соняшник": "Соняшник"},
        "year": "Рік",
        "yield": "Урожайність, т/га",
        "actual": "Факт",
        "pred": "Прогноз",
        "scatter_x": "Факт (т/га)",
        "scatter_y": "Прогноз (т/га)",
        "heatmap_title": "Кореляції лагових факторів — {crop}",
        "heatmap_cb": "Pearson r",
        "baseline_title": "Порівняння базових прогнозних методів і ML",
        "baseline_ylabel": "MAE, т/га",
        "lag_title": "Чутливість до структури лагів",
        "lag_ylabel": "MAE, т/га",
        "climate_title": "Експеримент чутливості до кліматичних ознак",
        "climate_ylabel": "MAE, т/га",
        "climate_labels": ("Лише агростатистика", "Агростатистика + клімат"),
        "legend": "Культура",
        "trends": {
            "Yield_t_ha": ("Урожайність", "т/га"),
            "Area_ha": ("Площа культури", "га"),
            "N_kg_ha": ("Азотні добрива", "кг/га"),
            "P2O5_kg_ha": ("Фосфорні добрива", "кг/га"),
            "K_kg_ha": ("Калійні добрива", "кг/га"),
            "Irrig_mm": ("Зрошення", "мм"),
        },
    },
    "en": {
        "crop_names": {"Пшениця": "Wheat", "Кукурудза": "Corn", "Соняшник": "Sunflower"},
        "year": "Year",
        "yield": "Yield, t/ha",
        "actual": "Actual",
        "pred": "Prediction",
        "scatter_x": "Actual (t/ha)",
        "scatter_y": "Prediction (t/ha)",
        "heatmap_title": "Lag-factor correlations — {crop}",
        "heatmap_cb": "Pearson r",
        "baseline_title": "Forecasting baselines versus tuned ML",
        "baseline_ylabel": "MAE, t/ha",
        "lag_title": "Sensitivity to lag structure",
        "lag_ylabel": "MAE, t/ha",
        "climate_title": "Climate sensitivity experiment",
        "climate_ylabel": "MAE, t/ha",
        "climate_labels": ("Agro-only", "Agro+climate"),
        "legend": "Crop",
        "trends": {
            "Yield_t_ha": ("Yield", "t/ha"),
            "Area_ha": ("Crop area", "ha"),
            "N_kg_ha": ("Nitrogen fertilisers", "kg/ha"),
            "P2O5_kg_ha": ("Phosphorus fertilisers", "kg/ha"),
            "K_kg_ha": ("Potassium fertilisers", "kg/ha"),
            "Irrig_mm": ("Irrigation", "mm"),
        },
    },
}

TREND_COLUMNS = ["Yield_t_ha", "Area_ha", "N_kg_ha", "P2O5_kg_ha", "K_kg_ha", "Irrig_mm"]
BASELINE_LABELS = {
    "naive_lag1": {"uk": "Naive (t-1)", "en": "Naive (t-1)"},
    "forecast_linear": {"uk": "Лінійний тренд", "en": "Linear trend"},
    "linest_lag_only": {"uk": "LINEST + лаги", "en": "LINEST + lags"},
    "arima": {"uk": "ARIMA", "en": "ARIMA"},
}
MODEL_LABELS = {"elasticnet": "ElasticNet", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
PANEL_LABELS = ("a", "b", "c")


def ensure_dirs(languages: Iterable[str]) -> dict[str, Path]:
    result = {}
    for lang in languages:
        out_dir = FIGURES_DIR / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        result[lang] = out_dir
    return result


def style_matplotlib() -> None:
    style_name = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "classic"
    plt.style.use(style_name)
    plt.rcParams.update(
        {
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 220,
        }
    )


def crop_label(crop: str, lang: str) -> str:
    return str(LANG_CONFIG[lang]["crop_names"][crop])


def plot_trends(features: pd.DataFrame, languages: Iterable[str]) -> None:
    dirs = ensure_dirs(languages)
    subset = features[features["group_or_crop"].isin(TARGET_CROPS)].sort_values("year")
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        for ax, column in zip(axes, TREND_COLUMNS):
            title, unit = cfg["trends"][column]
            for crop in TARGET_CROPS:
                crop_df = subset[subset["group_or_crop"] == crop]
                ax.plot(crop_df["year"], crop_df[column], marker="o", linewidth=2, label=crop_label(crop, lang))
            ax.set_title(f"{title} ({unit})")
            ax.set_xlabel(str(cfg["year"]))
            ax.set_ylabel(unit)
            ax.set_xticks(range(2010, 2025, 2))
            ax.grid(alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3, title=str(cfg["legend"]))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(dirs[lang] / "poltava_trends.png", bbox_inches="tight")
        plt.close(fig)


def plot_prediction_series(predictions: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    subset = predictions[
        (predictions["crop"] == crop)
        & (predictions["model"] == model)
        & (predictions["scenario"] == "lag_only")
        & (predictions["split"] == "test")
    ].sort_values("year")
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(subset["year"], subset["y_true"], marker="o", linewidth=2.2, label=str(cfg["actual"]))
    ax.plot(subset["year"], subset["y_pred"], marker="o", linewidth=2.2, linestyle="--", label=str(cfg["pred"]))
    ax.set_title(f"{crop_label(crop, lang)} ({MODEL_LABELS.get(model, model)})")
    ax.set_xlabel(str(cfg["year"]))
    ax.set_ylabel(str(cfg["yield"]))
    ax.set_xticks(subset["year"].tolist())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / lang / f"line_{model}_{CROP_SLUG[crop]}_lag_only.png", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(predictions: pd.DataFrame, crop: str, model: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
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
    line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(y_true, y_pred, color="#1f77b4", edgecolor="black", s=85)
    ax.plot(line, line, color="#d62728", linestyle="--", linewidth=1.4, label="y = x")
    ax.set_xlabel(str(cfg["scatter_x"]))
    ax.set_ylabel(str(cfg["scatter_y"]))
    ax.set_title(f"{crop_label(crop, lang)} ({MODEL_LABELS.get(model, model)})\nMAE={mae:.2f}, MAPE={mape:.1f}%")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / lang / f"scatter_{model}_{CROP_SLUG[crop]}_lag_only.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(corr_df: pd.DataFrame, crop: str, lang: str) -> None:
    cfg = LANG_CONFIG[lang]
    pearson = corr_df.set_index("factor")["pearson_yield"].sort_values(key=lambda s: np.abs(s), ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, max(3.2, len(pearson) * 0.45)))
    matrix = pearson.values[:, None]
    cax = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels([str(cfg["heatmap_cb"])])
    ax.set_yticks(range(len(pearson)))
    ax.set_yticklabels(pearson.index)
    for i, value in enumerate(pearson.values):
        ax.text(0, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title(str(cfg["heatmap_title"]).format(crop=crop_label(crop, lang)))
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / lang / f"correlation_heatmap_{CROP_SLUG[crop]}.png", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_ml(baselines: pd.DataFrame, leaderboard: pd.DataFrame, languages: Iterable[str]) -> None:
    dirs = ensure_dirs(languages)
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        max_val = 0.0
        for crop in TARGET_CROPS:
            crop_base = baselines[baselines["crop"] == crop]
            crop_ml = leaderboard[leaderboard["crop"] == crop]
            max_val = max(max_val, crop_base["mae"].max(), crop_ml["mae"].max())
        max_val *= 1.25

        for ax, crop in zip(axes, TARGET_CROPS):
            crop_base = baselines[baselines["crop"] == crop].copy()
            crop_base["label"] = crop_base["baseline"].map(lambda x: BASELINE_LABELS.get(x, {}).get(lang, x))
            crop_ml = leaderboard[leaderboard["crop"] == crop].copy()
            labels = crop_base["label"].tolist() + [f"ML ({MODEL_LABELS.get(crop_ml.iloc[0]['model'], crop_ml.iloc[0]['model'])})"]
            values = crop_base["mae"].tolist() + [float(crop_ml.iloc[0]["mae"])]
            colors = ["#9ecae1"] * len(crop_base) + ["#f28e2b"]
            x = np.arange(len(values))
            bars = ax.bar(x, values, color=colors, width=0.65)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + max_val * 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(crop_label(crop, lang))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=22, ha="right")
            ax.set_ylim(0, max_val)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(str(cfg["baseline_ylabel"]))
        fig.suptitle(str(cfg["baseline_title"]), fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(dirs[lang] / "mae_baselines_vs_ml.png", bbox_inches="tight")
        plt.close(fig)


def plot_lag_sensitivity(lag_df: pd.DataFrame, languages: Iterable[str]) -> None:
    best_per_lag = lag_df.sort_values(["crop", "lag_config", "validation_mae", "mae"]).groupby(["crop", "lag_config"], as_index=False).first()
    dirs = ensure_dirs(languages)
    lag_label_map = {"L1": "L1", "L1_L2": "L1+L2", "L1_L2_L3": "L1+L2+L3"}
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
        for ax, crop in zip(axes, TARGET_CROPS):
            subset = best_per_lag[best_per_lag["crop"] == crop]
            x = np.arange(len(subset))
            labels = [lag_label_map[item] for item in subset["lag_config"]]
            bars = ax.bar(x, subset["mae"], color="#4e79a7", width=0.65)
            for bar, value, model_name in zip(bars, subset["mae"], subset["model"]):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}\n{MODEL_LABELS.get(model_name, model_name)}", ha="center", va="bottom", fontsize=8)
            ax.set_title(crop_label(crop, lang))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(str(cfg["lag_ylabel"]))
        fig.suptitle(str(cfg["lag_title"]), fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(dirs[lang] / "lag_sensitivity.png", bbox_inches="tight")
        plt.close(fig)


def plot_climate_sensitivity(climate_df: pd.DataFrame, languages: Iterable[str]) -> None:
    dirs = ensure_dirs(languages)
    for lang in languages:
        cfg = LANG_CONFIG[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
        for ax, crop in zip(axes, TARGET_CROPS):
            subset = climate_df[climate_df["crop"] == crop]
            if subset.empty:
                ax.set_visible(False)
                continue
            row = subset.iloc[0]
            labels = list(cfg["climate_labels"])
            values = [float(row["base_mae"]), float(row["climate_mae"])]
            colors = ["#9ecae1", "#59a14f"]
            bars = ax.bar(np.arange(2), values, color=colors, width=0.62)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(crop_label(crop, lang))
            ax.set_xticks(np.arange(2))
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(str(cfg["climate_ylabel"]))
        fig.suptitle(str(cfg["climate_title"]), fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(dirs[lang] / "climate_sensitivity.png", bbox_inches="tight")
        plt.close(fig)


def copy_best_shap_figures(leaderboard: pd.DataFrame, languages: Iterable[str]) -> None:
    for lang in languages:
        out_dir = FIGURES_DIR / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, row in leaderboard.iterrows():
            crop = row["crop"]
            model = row["model"]
            src = FIGURES_BASE_DIR / lang / f"shap_{model}_{CROP_SLUG[crop]}_lag_only.png"
            dst = out_dir / f"shap_{model}_{CROP_SLUG[crop]}_lag_only.png"
            if src.exists():
                shutil.copy2(src, dst)


def _panel_paths_for_figure(leaderboard: pd.DataFrame, lang: str, figure_no: int) -> list[Path]:
    figure_paths: list[Path] = []
    for crop in TARGET_CROPS:
        row = leaderboard[leaderboard["crop"] == crop].iloc[0]
        model = row["model"]
        crop_slug = CROP_SLUG[crop]
        if figure_no == 2:
            figure_paths.append(FIGURES_DIR / lang / f"correlation_heatmap_{crop_slug}.png")
        elif figure_no == 3:
            figure_paths.append(FIGURES_DIR / lang / f"line_{model}_{crop_slug}_lag_only.png")
        elif figure_no == 4:
            figure_paths.append(FIGURES_DIR / lang / f"scatter_{model}_{crop_slug}_lag_only.png")
        elif figure_no == 6:
            figure_paths.append(FIGURES_DIR / lang / f"shap_{model}_{crop_slug}_lag_only.png")
        else:
            raise ValueError(f"Unsupported composite figure number: {figure_no}")
    return figure_paths


def build_composite_figure(
    input_paths: list[Path],
    output_path: Path,
    *,
    ncols: int = 1,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    fig, axes = plt.subplots(int(np.ceil(len(input_paths) / ncols)), ncols, figsize=figsize)
    axes_array = np.atleast_1d(axes).flatten()
    for ax, panel_label, path in zip(axes_array, PANEL_LABELS, input_paths):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.01,
            0.98,
            f"({panel_label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
        )
    for ax in axes_array[len(input_paths):]:
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_manuscript_figure_set(leaderboard: pd.DataFrame, languages: Iterable[str]) -> None:
    for lang in languages:
        out_dir = FIGURES_DIR / lang
        shutil.copy2(out_dir / "poltava_trends.png", out_dir / "manuscript_figure1.png")
        shutil.copy2(out_dir / "mae_baselines_vs_ml.png", out_dir / "manuscript_figure5.png")
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 2),
            out_dir / "manuscript_figure2.png",
            ncols=1,
            figsize=(9, 14),
        )
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 3),
            out_dir / "manuscript_figure3.png",
            ncols=1,
            figsize=(11, 12),
        )
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 4),
            out_dir / "manuscript_figure4.png",
            ncols=1,
            figsize=(9, 12),
        )
        build_composite_figure(
            _panel_paths_for_figure(leaderboard, lang, 6),
            out_dir / "manuscript_figure6.png",
            ncols=1,
            figsize=(10, 12),
        )


def main() -> None:
    languages = ("uk", "en")
    style_matplotlib()
    ensure_dirs(languages)

    features = pd.read_parquet(PROCESSED_FEATURES)
    predictions = pd.read_csv(PREDICTIONS_CSV)
    leaderboard = pd.read_csv(METRICS_BY_SCENARIO_CSV)
    leaderboard = leaderboard[leaderboard["scenario"] == "lag_only"].sort_values(["crop", "mae"]).groupby("crop", as_index=False).first()
    baselines = pd.read_csv(BASELINE_SUMMARY_CSV)
    lag_df = pd.read_csv(LAG_SENSITIVITY_CSV)
    climate_df = pd.read_csv(CLIMATE_SENSITIVITY_CSV)

    plot_trends(features, languages)
    plot_baseline_vs_ml(baselines, leaderboard, languages)
    plot_lag_sensitivity(lag_df, languages)
    plot_climate_sensitivity(climate_df, languages)

    for _, row in leaderboard.iterrows():
        crop = row["crop"]
        model = row["model"]
        for lang in languages:
            plot_prediction_series(predictions, crop, model, lang)
            plot_prediction_scatter(predictions, crop, model, lang)
            corr_path = str(CORR_TEMPLATE).format(crop=CROP_SLUG[crop])
            corr_df = pd.read_csv(corr_path)
            plot_heatmap(corr_df, crop, lang)

    copy_best_shap_figures(leaderboard, languages)
    export_manuscript_figure_set(leaderboard, languages)


if __name__ == "__main__":
    main()
