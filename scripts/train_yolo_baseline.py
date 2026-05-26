from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "pcb-defect-yolo" / "pcb-defect.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO baseline on PCB-Defect.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or path.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="YOLO dataset YAML.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="yolo11n_full_640")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", default="False")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def parse_bool(value: str) -> bool | str:
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return value


def main() -> int:
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=parse_bool(args.cache),
        exist_ok=args.exist_ok,
        plots=True,
        save=True,
        val=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
