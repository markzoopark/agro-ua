"""Data quality validation helpers and CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import typer
from rich.console import Console

from agrostats import utils


console = Console()
app = typer.Typer(help="Validation CLI for agrostats datasets.")


VALIDATION_REPORT_PATH = Path("reports/validation.md")
EXPECTED_YEARS = list(range(2010, 2025))
KEY_SERIES = [
    ("Урожайність", "Пшениця"),
    ("Урожайність", "Кукурудза"),
    ("Урожайність", "Соняшник"),
    ("Посівна площа", "Всі культури"),
]
ALLOWED_UNITS = {"t/ha", "ha", "kg/ha", "share", "m3/ha", "mm"}
UNIT_CANONICAL = {
    None: None,
    "т/га": "t/ha",
    "t/ha": "t/ha",
    "га": "ha",
    "ha": "ha",
    "кг/га": "kg/ha",
    "кг N/га": "kg/ha",
    "кг P2O5/га": "kg/ha",
    "кг K2O/га": "kg/ha",
    "kg/ha": "kg/ha",
    "кг/га ": "kg/ha",
    "доля": "share",
    "share": "share",
    "м³/га": "m3/ha",
    "м3/га": "m3/ha",
    "m3/ha": "m3/ha",
    "мм": "mm",
    "mm": "mm",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise an error if mandatory columns are missing."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def check_year_coverage(df: pd.DataFrame) -> List[dict[str, object]]:
    """Ensure key series cover all expected years without gaps."""
    issues: List[dict[str, object]] = []
    for metric, crop in KEY_SERIES:
        subset = df[(df["metric"] == metric) & (df["group_or_crop"] == crop)]
        if subset.empty:
            issues.append(
                {
                    "metric": metric,
                    "group_or_crop": crop,
                    "reason": "series_missing",
                    "missing_years": EXPECTED_YEARS,
                }
            )
            continue
        years_present = set(int(year) for year in subset["year"].dropna().astype(int))
        missing_years = [year for year in EXPECTED_YEARS if year not in years_present]
        if missing_years:
            issues.append(
                {
                    "metric": metric,
                    "group_or_crop": crop,
                    "reason": "missing_years",
                    "missing_years": missing_years,
                }
            )
    return issues


def canonical_unit(value: Optional[str]) -> Optional[str]:
    """Map raw unit names to canonical representations."""
    if value is None:
        return None
    return UNIT_CANONICAL.get(value.strip(), value.strip())


def check_unit_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where unit_norm does not match the allowed set."""
    units = df["unit_norm"].apply(canonical_unit)
    invalid_mask = ~units.isin(ALLOWED_UNITS)
    invalid_rows = df.loc[invalid_mask, ["metric", "group_or_crop", "fert_type", "year", "unit_norm"]].copy()
    invalid_rows["unit_norm"] = invalid_rows["unit_norm"].fillna("∅")
    invalid_rows["canonical"] = units.loc[invalid_mask].fillna("∅")
    return invalid_rows


def detect_yield_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers in yield (value_norm, metric=Урожайність) using the IQR rule."""
    subset = df[(df["metric"] == "Урожайність") & (~df["value_norm"].isna())].copy()
    if subset.empty:
        return pd.DataFrame(columns=["group_or_crop", "year", "value_norm", "lower_bound", "upper_bound", "iqr"])

    records: List[dict[str, object]] = []
    for crop, crop_df in subset.groupby("group_or_crop"):
        values = crop_df["value_norm"].astype(float).dropna()
        if values.count() < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (crop_df["value_norm"] < lower) | (crop_df["value_norm"] > upper)
        for _, row in crop_df[mask].iterrows():
            records.append(
                {
                    "group_or_crop": crop,
                    "year": int(row["year"]),
                    "value_norm": float(row["value_norm"]),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "iqr": iqr,
                }
            )
    if not records:
        return pd.DataFrame(columns=["group_or_crop", "year", "value_norm", "lower_bound", "upper_bound", "iqr"])
    return pd.DataFrame(records).sort_values(["group_or_crop", "year"])


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a dataframe to a markdown-friendly table."""
    if df.empty:
        return "_Нет записей._"
    return df.to_string(index=False)


def build_report(
    year_issues: Sequence[dict[str, object]],
    invalid_units: pd.DataFrame,
    outliers: pd.DataFrame,
    success: bool,
) -> str:
    """Compose a markdown report from validation results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "✅ PASSED" if success else "❌ FAILED"

    lines: List[str] = [
        f"# Validation Report ({timestamp})",
        "",
        f"Result: **{status}**",
        "",
        "## Year Coverage",
    ]
    if year_issues:
        for issue in year_issues:
            missing_years = ", ".join(str(year) for year in issue["missing_years"])
            lines.append(
                f"- **{issue['metric']} / {issue['group_or_crop']}** — отсутствуют года: {missing_years}."
            )
    else:
        lines.append("- Все ключевые серии покрывают период 2010–2024 без пропусков.")

    lines.extend(
        [
            "",
            "## Unit Consistency",
        ]
    )
    if invalid_units.empty:
        lines.append("- Все значения unit_norm соответствуют ожидаемым единицам.")
    else:
        lines.append("- Обнаружены строки с неподдерживаемыми единицами:")
        lines.append("")
        lines.append("```")
        lines.append(dataframe_to_markdown(invalid_units))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Yield Outliers (IQR)",
        ]
    )
    if outliers.empty:
        lines.append("- Выбросы по урожайности не обнаружены.")
    else:
        lines.append("- Найдены потенциальные выбросы:")
        lines.append("")
        lines.append("```")
        lines.append(dataframe_to_markdown(outliers))
        lines.append("```")

    return "\n".join(lines) + "\n"


def write_report(report: str, path: Path = VALIDATION_REPORT_PATH) -> None:
    """Persist the validation report to disk."""
    utils.ensure_directories([path.parent])
    path.write_text(report, encoding="utf-8")


def validate_agrostats(df: pd.DataFrame) -> Tuple[bool, str]:
    """Run the full agrostats validation suite."""
    required_columns = [
        "year",
        "region",
        "metric",
        "group_or_crop",
        "fert_type",
        "value_norm",
        "unit_norm",
    ]
    require_columns(df, required_columns)

    year_issues = check_year_coverage(df)
    invalid_units = check_unit_values(df)
    outliers = detect_yield_outliers(df)

    success = not year_issues and invalid_units.empty
    report = build_report(year_issues, invalid_units, outliers, success)
    write_report(report)
    return success, report


@app.command("agrostats")
def validate_agrostats_command(
    path: Path = typer.Argument(Path("data/interim/agrostats_norm.parquet"), exists=True, file_okay=True),
) -> None:
    """Validate the normalized AgroStats dataset and persist a report."""
    df = pd.read_parquet(path)
    success, report = validate_agrostats(df)
    console.print(report)
    status = "green" if success else "red"
    console.print(f"[{status}]Validation {'passed' if success else 'failed'}[/]")
    console.print(f"Отчет сохранен в {VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    app()
