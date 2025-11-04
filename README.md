# Agrostats: Forecasting Harvest Yield in Poltava Oblast (2010–2024)

This repository hosts a reproducible pipeline for forecasting crop yields in Poltava region (Ukraine) using AgroStats statistics from 2010 to 2024. The project currently focuses on three key crops: wheat (`Пшениця`), corn (`Кукурудза`), and sunflower (`Соняшник`).

---

## Project Overview

-   **Goal.** Predict annual yields (`t/ha`) for Poltava crops based on regional statistics and agronomic factors.
-   **Data source.** AgroStats regional exports (CSV). Place raw files inside `data/raw/agrostats/poltava/` (region slugs use Latin characters).
-   **Unit harmonisation.** Yields `ц/га → т/га`, areas `тис. га → га`, irrigation `млн м³ → мм`, fertiliser masses converted to kg/ha or shares as specified in `src/agrostats/normalize.py`.
-   **Feature engineering.** Lagged (t−1) indicators and 5-year moving averages are stored in `data/processed/agrostats_poltava_features.parquet`.
-   **Modelling scenarios.**
    -   `lag_only` – only lagged (t−1) factors + moving averages (no leakage; recommended baseline).
    -   `in_season` – includes factors from the prediction year `t` (allows leakage; used for comparison).
-   **Models.** ElasticNet, XGBoost, LightGBM with a walk-forward, origin-expanding split: train ≤ 2018, validation 2019–2021, test 2022–2024.

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

The command automatically loads raw CSVs from `data/raw/agrostats/poltava/`, normalises units, builds features, runs validation, trains both `lag_only`/`in_season` models, computes Excel baselines, and exports publication figures (Ukrainian & English). All artefacts land in `reports/`.

### Manual steps (advanced)

```bash
python -m src.agrostats.validate
python -m src.agrostats.train --languages uk,en
python scripts/qa_check.py
```

-   `python -m src.agrostats.train` generates metrics, predictions, multilingual figures, SHAP tables, and scenario summaries (`lag_only` / `in_season`).
-   `scripts/qa_check.py` provides additional regression/QA checks (extend to enforce custom acceptance criteria).

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
| `metrics_leaderboard.csv` | Best model per crop (lowest MAE on test). |
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
| `shap_{model}_{crop}_{scenario}.png`            | SHAP top-10 feature importances (test holdout).                         |

All graphics are available in both Ukrainian (`uk/`) and English (`en/`) captioning folders.

---

## Scenario Acceptance Targets (`lag_only`, test 2022–2024)

| Crop                 | Target MAE (t/ha) |
| -------------------- | ----------------- |
| Corn (Кукурудза)     | 0.60 – 0.95       |
| Wheat (Пшениця)      | 0.25 – 0.50       |
| Sunflower (Соняшник) | 0.05 – 0.15       |

The latest run meets these KPI ranges (see `reports/metrics.csv`, split `test`, scenario `lag_only`). Use the ranges to validate future updates.

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
8. Train models (`python -m src.agrostats.train --languages uk,en`).
9. Execute QA script (`python scripts/qa_check.py`).
10. Run unit tests (`python -m pytest`).

---

## Licence & Citation

-   **Code licence:** MIT (see `LICENSE`).
-   **Data:** Derived from AgroStats regional statistics – cite the original source when publishing.
-   **Suggested citation:**

Feel free to adapt this template to local publication requirements. Contributions and issue reports are welcome!
