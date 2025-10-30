"""Exploratory data analysis toolkit for agrostats."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

FEATURES_PATH = Path("data/processed/agrostats_poltava_features.parquet")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")
TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")

LAG_FEATURES = [
    "N_kg_ha_lag1",
    "P2O5_kg_ha_lag1",
    "K_kg_ha_lag1",
    "Mineral_treated_share_lag1",
    "Org_kg_ha_or_share_lag1",
    "Irrig_mm_lag1",
    "Area_ha_lag1",
]

TREND_METRICS = [
    ("Yield_t_ha", "Урожайність", "t/ha"),
    ("Area_ha", "Посівна площа", "ha"),
    ("N_kg_ha", "Азотні добрива", "kg/ha"),
    ("P2O5_kg_ha", "Фосфорні добрива", "kg/ha"),
    ("K_kg_ha", "Калійні добрива", "kg/ha"),
    ("Irrig_mm", "Зрошення", "mm"),
]


def slugify(value: str) -> str:
    mapping = {
        "Пшениця": "pshenytsia",
        "Кукурудза": "kukurudza",
        "Соняшник": "sonyashnyk",
    }
    if value in mapping:
        return mapping[value]
    ascii_value = (
        value.encode("ascii", "ignore").decode("ascii") if value.isascii() else value
    )
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in ascii_value).strip("_").lower()
    return cleaned or f"crop_{abs(hash(value)) % 10000}"


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден parquet с фичами: {path}")
    return pd.read_parquet(path)


# -------------------------- ACF/PACF ----------------------------------------

def compute_acf_pacf(series: pd.Series) -> tuple[np.ndarray, np.ndarray, int, float]:
    ordered = series.dropna().astype(float)
    ordered = ordered.sort_index()
    y = ordered.values
    if len(y) < 3:
        raise ValueError("Ряд слишком короткий для расчета ACF/PACF.")
    max_lag = min(6, len(y) - 2)
    acf_vals = sm_acf(y, nlags=max_lag, fft=True, adjusted=False)
    pacf_vals = sm_pacf(y, nlags=max_lag, method="yw")
    conf = 1.96 / np.sqrt(len(y))
    return acf_vals[1:], pacf_vals[1:], max_lag, conf


def plot_acf_pacf(series: pd.Series, crop: str) -> None:
    try:
        acf_vals, pacf_vals, max_lag, conf = compute_acf_pacf(series)
    except ValueError as exc:
        print(f"[!] Пропуск ACF/PACF для {crop}: {exc}")
        return

    lags = np.arange(1, max_lag + 1)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ACF plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lags, acf_vals, color="#4B8BBE")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(conf, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(f"ACF урожайності — {crop}")
    ax.set_xlabel("Лаг (років)")
    ax.set_ylabel("Кореляція")
    ax.set_ylim(-1, 1)
    ax.grid(True, axis="y", alpha=0.3)
    acf_path = FIGURES_DIR / f"yield_acf_{slugify(crop)}.png"
    fig.tight_layout()
    fig.savefig(acf_path, dpi=200)
    plt.close(fig)

    # PACF plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lags, pacf_vals, color="#FF7F0E")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(conf, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(f"PACF урожайності — {crop}")
    ax.set_xlabel("Лаг (років)")
    ax.set_ylabel("Часткова кореляція")
    ax.set_ylim(-1, 1)
    ax.grid(True, axis="y", alpha=0.3)
    pacf_path = FIGURES_DIR / f"yield_pacf_{slugify(crop)}.png"
    fig.tight_layout()
    fig.savefig(pacf_path, dpi=200)
    plt.close(fig)

    print(f"[+] Збережено ACF/PACF для {crop}")


def generate_acf_pacf(features_df: pd.DataFrame, crops: Iterable[str]) -> None:
    for crop in crops:
        subset = features_df[features_df["group_or_crop"] == crop].copy()
        if subset.empty:
            print(f"[!] Немає даних для {crop}")
            continue
        subset = subset.sort_values("year")
        plot_acf_pacf(subset.set_index("year")["Yield_t_ha"], crop)


# ----------------------- Correlation analysis -------------------------------

def prepare_factor_table(subset: pd.DataFrame) -> pd.DataFrame:
    columns = ["year", "Yield_t_ha", "Yield_anom"] + LAG_FEATURES
    table = subset[columns].dropna(subset=["Yield_t_ha"])
    for col in columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna()
    # remove zero variance columns
    std = table.std(axis=0, numeric_only=True)
    keep = std[std > 0].index
    return table[list(keep)]


def corr_with_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)
    return {
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
    }


def generate_correlation_reports(subset: pd.DataFrame, crop: str) -> None:
    table = prepare_factor_table(subset)
    if table.empty:
        print(f"[!] Недостатньо даних для кореляцій ({crop})")
        return

    factors = [col for col in table.columns if col not in {"year", "Yield_t_ha", "Yield_anom"}]
    if not factors:
        print(f"[!] Немає факторів для {crop}")
        return

    pearson_matrix = []
    for factor in factors:
        stats = corr_with_stats(table[factor], table["Yield_t_ha"])
        pearson_matrix.append(
            {
                "factor": factor,
                "pearson_yield": stats["pearson"],
                "pearson_yield_p": stats["pearson_p"],
                "spearman_yield": stats["spearman"],
                "spearman_yield_p": stats["spearman_p"],
            }
        )

    anomaly_matrix = []
    for factor in factors:
        stats = corr_with_stats(table[factor], table["Yield_anom"])
        anomaly_matrix.append(
            {
                "factor": factor,
                "pearson_anom": stats["pearson"],
                "pearson_anom_p": stats["pearson_p"],
                "spearman_anom": stats["spearman"],
                "spearman_anom_p": stats["spearman_p"],
            }
        )

    corr_df = pd.DataFrame(pearson_matrix).merge(pd.DataFrame(anomaly_matrix), on="factor", how="outer")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"correlations_{slugify(crop)}.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"[+] Збережено таблицю кореляцій {csv_path}")

    # Heatmap (Pearson with Yield)
    pivot_data = corr_df.set_index("factor")["pearson_yield"]
    fig, ax = plt.subplots(figsize=(6, max(3, len(pivot_data) * 0.4)))
    cax = ax.imshow(pivot_data.values[:, None], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Pearson(Yield)"])
    ax.set_yticks(range(len(pivot_data)))
    ax.set_yticklabels(pivot_data.index)
    for i, value in enumerate(pivot_data.values):
        ax.text(0, i, f"{value:.2f}", ha="center", va="center", color="black")
    ax.set_title(f"Кореляції (лагові фактори) — {crop}")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = FIGURES_DIR / f"correlation_heatmap_{slugify(crop)}.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)


def correlation_pipeline(features_df: pd.DataFrame, crops: Sequence[str]) -> None:
    for crop in crops:
        subset = features_df[features_df["group_or_crop"] == crop].copy()
        if subset.empty:
            print(f"[!] Немає даних для {crop}")
            continue
        subset = subset.sort_values("year")
        generate_correlation_reports(subset, crop)


def plot_trends(features_df: pd.DataFrame, crops: Sequence[str]) -> None:
    subset = features_df[features_df["group_or_crop"].isin(crops)].copy()
    if subset.empty:
        print("[!] Не знайдено даних для графіка трендів.")
        return
    subset = subset.sort_values("year")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    for ax, (column, title, unit) in zip(axes, TREND_METRICS):
        for crop in crops:
            crop_df = subset[subset["group_or_crop"] == crop]
            if crop_df.empty:
                continue
            ax.plot(
                crop_df["year"],
                crop_df[column],
                marker="o",
                label=crop,
            )
        ax.set_title(f"{title} ({unit})")
        ax.set_xlabel("Рік")
        ax.set_ylabel(unit)
        ax.set_xlim(2010, 2024)
        ax.set_xticks(range(2010, 2025))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(crops), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "poltava_trends.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[+] Збережено графік трендів {output_path}")


# --------------------------- CLI entry point --------------------------------

def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EDA utilities for agrostats.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURES_PATH,
        help="Parquet файл з фічами (default: data/processed/agrostats_poltava_features.parquet)",
    )
    parser.add_argument(
        "--mode",
        choices=["acf", "correlations", "trends"],
        default="acf",
        help="Що згенерувати: acf (ACF/PACF) або correlations (кореляційний аналіз).",
    )
    parsed = parser.parse_args(args=args)
    df = load_features(parsed.features_path)
    if parsed.mode == "acf":
        generate_acf_pacf(df, TARGET_CROPS)
    elif parsed.mode == "correlations":
        correlation_pipeline(df, TARGET_CROPS)
    elif parsed.mode == "trends":
        plot_trends(df, TARGET_CROPS)


if __name__ == "__main__":
    main()
