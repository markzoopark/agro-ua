"""End-to-end orchestration script for the agrostats pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
# Ensure src/ is on sys.path so `agrostats` package is importable.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import typer
from rich.console import Console

from agrostats.features import OUTPUT_PARQUET
from agrostats import features, io, normalize, train, validate


app = typer.Typer(help="Run the full AgroStats pipeline with a single command.")
console = Console()

DEFAULT_REGION = "poltava"
RAW_ROOT = Path("data/raw/agrostats")


def ensure_raw_dir(region_slug: str) -> Path:
    raw_dir = RAW_ROOT / region_slug
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw directory '{raw_dir}' не знайдено. "
            "Скопіюйте CSV у відповідний підкаталог або вкажіть інший slug через --region."
        )
    return raw_dir


def run_subprocess(command: list[str], description: str) -> None:
    console.rule(f"[bold blue]{description}")
    console.print(f"[cyan]$ {' '.join(command)}[/cyan]")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Команда '{' '.join(command)}' завершилась з помилкою (код {exc.returncode}).") from exc


def parse_languages(languages: str) -> list[str]:
    langs = [lang.strip() for lang in languages.split(",") if lang.strip()]
    return langs or ["uk"]


def pipeline(
    region: str,
    languages: str,
    run_baselines: bool,
    run_figures: bool,
) -> None:
    raw_dir = ensure_raw_dir(region)
    console.rule(f"[bold green]1. Завантаження CSV з {raw_dir}")
    raw_df = io.load_folder(raw_dir)
    console.print(f"[green]Прочитано {len(raw_df)} рядків з {raw_dir}[/green]")

    console.rule("[bold green]2. Нормалізація одиниць")
    norm_df = normalize.normalize_units(raw_df)
    console.print(f"[green]Нормалізовано {len(norm_df)} рядків → data/interim/agrostats_norm.parquet[/green]")

    console.rule("[bold green]3. Побудова фіч")
    features_df = features.build_features(norm_df)
    console.print(
        "[green]Фічі збережено в data/processed/agrostats_poltava_features.parquet "
        f"({len(features_df)} рядків)[/green]"
    )

    console.rule("[bold green]4. Перевірка цілісності")
    success, report, feature_message = validate.validate_agrostats(norm_df)
    console.print(report)
    if not success:
        raise RuntimeError("Валідація не пройдена. Див. reports/validation.md для подробиць.")
    if feature_message:
        raise RuntimeError(feature_message)

    console.rule("[bold green]5. Навчання моделей")
    language_list = parse_languages(languages)
    train_languages = ",".join(language_list)
    train.poltava_command(features_path=OUTPUT_PARQUET, languages=train_languages)

    if run_baselines:
        console.rule("[bold green]6. Excel-бейзлайни")
        run_subprocess([sys.executable, "scripts/baseline_excel.py"], "Excel baselines")

    if run_figures:
        console.rule("[bold green]7. Публікаційні графіки")
        run_subprocess([sys.executable, "scripts/export_article_figures.py"], "Export figures")

    console.rule("[bold green]Готово")
    console.print("[bold green]Усі артефакти сформовано в каталозі reports/[/bold green]")


@app.command()
def main(
    region: str = typer.Option(
        DEFAULT_REGION,
        "--region",
        "-r",
        help="Region slug (Latin characters, e.g. 'poltava').",
    ),
    languages: str = typer.Option(
        "uk,en",
        "--languages",
        "-l",
        help="Кома-розділений список мов для графіків/звіту (наприклад, 'uk,en').",
    ),
    skip_baselines: bool = typer.Option(
        False,
        "--skip-baselines",
        help="Не запускати Excel-порівняння (scripts/baseline_excel.py).",
    ),
    skip_figures: bool = typer.Option(
        False,
        "--skip-figures",
        help="Не перевідмалювати фінальні графіки (scripts/export_article_figures.py).",
    ),
) -> None:
    """Запустити весь конвеєр (інжест → нормалізація → фічі → валідація → моделі → звіти)."""
    pipeline(
        region=region,
        languages=languages,
        run_baselines=not skip_baselines,
        run_figures=not skip_figures,
    )


if __name__ == "__main__":
    app()
