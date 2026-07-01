from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a Markdown comparison table for YOLO runs.")
    parser.add_argument("runs", nargs="+", help="Run directories containing results.csv.")
    parser.add_argument("--labels", nargs="*", help="Optional labels matching the run directories.")
    return parser.parse_args()


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [clean_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {results_path}")
    return rows


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def fmt(value: str | float) -> str:
    number = value if isinstance(value, float) else to_float(value)
    return f"{number:.5f}" if number == number else ""


def main() -> int:
    args = parse_args()
    run_dirs = [Path(run).expanduser().resolve() for run in args.runs]
    labels = args.labels if args.labels else [path.name for path in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("--labels must have the same number of items as runs")

    print("| Run | Epochs | Precision | Recall | mAP@50 | mAP@50:95 | Best mAP@50 | Best mAP@50:95 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for label, run_dir in zip(labels, run_dirs):
        rows = read_rows(run_dir)
        last = rows[-1]
        best_map50 = max(rows, key=lambda row: to_float(row.get(METRICS["map50"], "nan")))
        best_map5095 = max(rows, key=lambda row: to_float(row.get(METRICS["map5095"], "nan")))
        print(
            "| "
            f"{label} | "
            f"{last.get('epoch', '')} | "
            f"{fmt(last.get(METRICS['precision'], 'nan'))} | "
            f"{fmt(last.get(METRICS['recall'], 'nan'))} | "
            f"{fmt(last.get(METRICS['map50'], 'nan'))} | "
            f"{fmt(last.get(METRICS['map5095'], 'nan'))} | "
            f"{fmt(best_map50.get(METRICS['map50'], 'nan'))} | "
            f"{fmt(best_map5095.get(METRICS['map5095'], 'nan'))} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
