# Agrostats: Low-Data Crop Yield Forecasting in Poltava Oblast (2010–2024)

This repository hosts a reproducible pipeline for forecasting crop yields in Poltava region (Ukraine) using AgroStats statistics from 2010 to 2024. The project focuses on three crops: wheat (`Пшениця`), corn (`Кукурудза`), and sunflower (`Соняшник`), and evaluates tuned machine-learning models against practical forecasting baselines under small-sample conditions.

---

## Project Overview

-   **Goal.** Predict annual yields (`t/ha`) for Poltava crops based on regional statistics and agronomic factors.
-   **Data source.** AgroStats regional exports (CSV). Place raw files inside `data/raw/agrostats/poltava/` (region slugs use Latin characters).
-   **Unit harmonisation.** Yields `ц/га → т/га`, areas `тис. га → га`, irrigation `млн м³ → мм`, fertiliser masses converted to kg/ha or shares as specified in `src/agrostats/normalize.py`.
-   **Feature engineering.** Leakage-safe lagged indicators (`t-1`, `t-2`, `t-3`) and 5-year moving averages are stored in `data/processed/agrostats_poltava_features.parquet`.
-   **Modelling scenarios.**
    -   `lag_only` – only historical predictors available before harvest; this is the main, leakage-safe scenario.
    -   `in_season` – includes current-season values and is kept only as a comparison scenario.
-   **Models.** ElasticNet, XGBoost, and LightGBM with constrained hyperparameter tuning on an origin-expanding split: train ≤ 2018, validation 2019–2021, test 2022–2024. For each forecast year, its yield is hidden; previously completed years enter the next expanding training window.
-   **Baselines.** Naive (`t-1`), linear trend (`FORECAST.LINEAR` analogue), `LINEST + lags`, and `ARIMA`.
-   **Evaluation outputs.** The pipeline keeps three stages separate: hyperparameters are tuned on validation, `metrics_leaderboard.csv` is a descriptive comparison on the observed Test window, and `validation_selected_test_metrics.csv` reports the stricter prospective choice made without Test-based model selection.
-   **Supplementary outputs.** The main pipeline exports lag and climate sensitivity, variable summaries, maize diagnostics, and publication-ready figures. The overlapping 2020–2024 robustness check is post-hoc descriptive and is generated only with `--include-post-hoc-robustness`.

---

## Data Preparation

1. **Raw CSVs**

    ```
    data/raw/agrostats/poltava/*.csv
    ```

2. **Ingestion / Normalisation / Validation**

    ```bash
    python -m src.agrostats.io load-folder
    python -m src.agrostats.normalize agrostats
    python -m src.agrostats.validate
    ```

---

## Running the Pipeline

### Quick start

```bash
python run_all.py --region poltava --languages uk,en
```

The command automatically loads raw CSVs from `data/raw/agrostats/poltava/`, normalises units, builds features, runs validation, trains both `lag_only`/`in_season` models, computes forecasting baselines, runs revision analyses, and exports publication figures (Ukrainian and English). All artefacts land in `reports/`.

### Manual steps (advanced)

```bash
python -m src.agrostats.validate
python -m src.agrostats.train --languages uk,en
```

-   `python -m src.agrostats.train` generates tuned ML metrics, predictions, multilingual figures, SHAP tables, and scenario summaries.
-   `python run_all.py --skip-figures` is the quickest way to refresh tabular artefacts without rebuilding publication figures.

> **Tip.** To regenerate EDA visuals manually:
>
> ```bash
> python -m src.agrostats.eda --mode acf --language uk,en
> python -m src.agrostats.eda --mode correlations --language uk,en
> python -m src.agrostats.eda --mode trends --language uk,en
> ```

---

## Generated Artefacts

### Tabular outputs (under `reports/`)

| File | Description |
| --- | --- |
| `metrics.csv` | Row-level metrics (`year`, `crop`, `model`, `scenario`, `split`, `mae`, `rmse`, `mape`, `n_features`). |
| `predictions.csv` | Actual vs predicted (`y_true`, `y_pred`) for all splits. |
| `metrics_by_scenario.csv` | Aggregated MAE/RMSE/MAPE and sample count `n` per `(scenario, model, crop)` on test. |
| `metrics_leaderboard.csv` | Publication snapshot: lowest observed Test MAE among the pre-specified ML candidates (descriptive, not prospective selection). |
| `metrics_baselines.csv` | Year-level predictions and errors for naive, linear trend, `LINEST + lags`, and `ARIMA`. |
| `metrics_baselines_summary.csv` | Test-window summary of baseline performance by crop. |
| `validation_selected_test_metrics.csv` | Separate ML and baseline methods selected only by 2019–2021 Validation MAE, with their untouched 2022–2024 Test metrics. |
| `metrics_arima.csv` | Extract of the year-level `ARIMA` results with selected `(p, d, q)` orders. |
| `tuned_hyperparameters.csv` | Selected hyperparameters for every `(crop, scenario, model)` combination. |
| `lag_sensitivity.csv` | Sensitivity analysis for `L1`, `L1+L2`, and `L1+L2+L3` lag structures. |
| `robustness_2020_2024.csv` | Optional post-hoc descriptive comparison over the overlapping 2020–2024 window; generated only with `--include-post-hoc-robustness`. |
| `climate_sensitivity.csv` | Post-hoc descriptive Test comparison of agro-only versus agro+climate candidates using NASA POWER seasonal aggregates. |
| `climate_sensitivity_validation_selected.csv` | Strict climate sensitivity: candidate model selected by Validation, then evaluated on Test. |
| `variable_summary.csv` | Descriptive statistics, units, and timing for the model input variables. |
| `maize_diagnostics.csv` | Year-level maize errors and feature-group ablation diagnostics. |
| `correlations_{crop}.csv` | Pearson/Spearman correlations (with p-values) of lag factors vs `Yield_t_ha` and `Yield_anom`. |
| `shap_top_{model}_{crop}_{scenario}.csv` | Top-10 features ranked by mean SHAP on the test holdout. |

### Figures (under `reports/figures/uk/` and `reports/figures/en/`)

| Category                                        | Notes                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| `yield_acf_{crop}.png`, `yield_pacf_{crop}.png` | ACF/PACF (lags 1..6, ±1.96/√n confidence bands, no lag 0).              |
| `correlation_heatmap_{crop}.png`                | Pearson correlations of lag factors per crop (single column heatmap).   |
| `poltava_trends.png`                            | Six time series (Yield, Area, N/P/K, Irrigation) across 2010–2024.      |
| `{model}_{crop}_{scenario}_actual_vs_pred.png`  | Year-by-year actual vs forecast on test.                                |
| `scatter_{model}_{crop}_{scenario}_test.png`    | y_pred vs y_true scatter with y = x reference line (MAE/MAPE in title). |
| `shap_{model}_{crop}_{scenario}.png`            | Signed-colour SHAP summary plots for the test holdout.                  |

Under `reports/figures_article/{uk,en}/`, the pipeline also exports manuscript-ready panels including `manuscript_figure1.png` … `manuscript_figure6.png`, plus supplementary `lag_sensitivity.png`, `climate_sensitivity.png`, and baseline comparison figures.

---

## Scenario Acceptance Targets (`lag_only`, test 2022–2024)

| Crop                 | Target MAE (t/ha) |
| -------------------- | ----------------- |
| Corn (Кукурудза)     | 0.60 – 0.95       |
| Wheat (Пшениця)      | 0.25 – 0.50       |
| Sunflower (Соняшник) | 0.05 – 0.15       |

These ranges are indicative only. The revision-ready workflow reports the actual model ranking directly from generated CSV artefacts, including cases where a baseline remains stronger than tuned ML on the narrow 2022–2024 window.

### How to read the evaluation files

`tuned_hyperparameters.csv` records hyperparameter choices made from the 2019–2021 Validation window. `metrics_leaderboard.csv` then preserves the accepted article's retrospective comparison of the pre-specified machine-learning candidates on 2022–2024; it is descriptive evidence, not a prospective deployment rule. For a Test-independent method choice, use `validation_selected_test_metrics.csv`. The same distinction applies to the two climate-sensitivity files.

The Test set is rolling-origin. The yield for the year being forecast is always withheld, while a completed earlier Test year becomes historical information for the next forecast. With only three Test years, the reports are a small-sample benchmark rather than a production forecasting guarantee.

---

## Reproducibility

1. **Create & activate virtual environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate      # Windows: .venv\Scripts\activate
    ```
2. **Install dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
3. **Use frozen snapshot (CI / publication)**
    ```bash
    pip install -r reports/requirements.freeze.txt
    ```

---

## Continuous Integration

![CI](https://github.com/markzoopark/agrostats-forecast/actions/workflows/ci.yml/badge.svg)

The `Agrostats CI` workflow performs:

1. Checkout repository & set up Python.
2. Cache dependencies.
3. Install system prerequisites.
4. Install Python dependencies (`requirements.txt`).
5. Verify raw data presence (`data/raw/agrostats/**/*.csv`).
6. Run ingestion & normalisation (`python -m src.agrostats.io load-folder`, `python -m src.agrostats.normalize agrostats`).
7. Validate normalised data (`python -m src.agrostats.validate`).
8. Train models and export reports (`python run_all.py --region poltava --languages uk,en`).
9. Run unit tests (`python -m pytest`).

---

## Licence & Citation

-   **Code licence:** MIT (see `LICENSE`).
-   **Data:** Derived from AgroStats regional statistics – cite the original source when publishing.
-   **Suggested citation:**

Feel free to adapt this template to local publication requirements. Contributions and issue reports are welcome!
