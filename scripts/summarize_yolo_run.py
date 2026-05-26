from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "val/box_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact summary for an Ultralytics YOLO run.")
    parser.add_argument("run_dir", help="Run directory, for example runs/baselines/yolo11n_full_960_server")
    return parser.parse_args()


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def fmt(value: str) -> str:
    number = to_float(value)
    return f"{number:.5f}" if number == number else value


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")

    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [clean_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {results_path}")

    last = rows[-1]
    best_map50 = max(rows, key=lambda row: to_float(row.get("metrics/mAP50(B)", "nan")))
    best_map5095 = max(rows, key=lambda row: to_float(row.get("metrics/mAP50-95(B)", "nan")))

    print(f"Run: {run_dir}")
    print(f"Epochs: {last.get('epoch', 'unknown')}")
    print("Final metrics:")
    for column in METRIC_COLUMNS:
        if column in last:
            print(f"  {column}: {fmt(last[column])}")
    print("Best metrics:")
    print(
        "  best mAP50: "
        f"{fmt(best_map50.get('metrics/mAP50(B)', 'nan'))} "
        f"at epoch {best_map50.get('epoch', 'unknown')}"
    )
    print(
        "  best mAP50-95: "
        f"{fmt(best_map5095.get('metrics/mAP50-95(B)', 'nan'))} "
        f"at epoch {best_map5095.get('epoch', 'unknown')}"
    )

    useful_files = [
        "results.png",
        "confusion_matrix_normalized.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_pred.jpg",
        "weights/best.pt",
    ]
    print("Files:")
    for relative in useful_files:
        path = run_dir / relative
        print(f"  {'exists' if path.exists() else 'missing'}: {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

