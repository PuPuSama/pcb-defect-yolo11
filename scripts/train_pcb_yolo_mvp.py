from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from pcb_yolo_mvp_modules import ReplacementStats, apply_pcb_yolo_mvp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "pcb-defect-yolo" / "pcb-defect.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "pcb_yolo_mvp"


def parse_bool(value: str) -> bool | str:
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PCB-YOLO MVP by injecting SWS and FBC2f modules into an Ultralytics model."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Ultralytics model name or local checkpoint.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO dataset YAML.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="pcb_yolo_mvp_960")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", default="False")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the model and write a manifest without training.")

    parser.add_argument("--no-sws", action="store_true", help="Disable Conv_SWS injection.")
    parser.add_argument("--no-fbc2f", action="store_true", help="Disable FBC2f injection.")
    parser.add_argument("--sws-blocks", type=int, default=2)
    parser.add_argument("--simam-lambda", type=float, default=1e-4)
    parser.add_argument("--pconv-n-div", type=int, default=4)
    parser.add_argument(
        "--target-csp-names",
        default="C2f,C3k2",
        help="Comma-separated Ultralytics CSP-like module class names to replace with FBC2f.",
    )
    return parser.parse_args()


def write_manifest(save_dir: Path, args: argparse.Namespace, stats: ReplacementStats, dry_run: bool) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mvp": "PCB-YOLO-MVP",
        "status": "dry_run" if dry_run else "trained",
        "paper_modules": {
            "implemented": [
                "grayscale sliced dataset via scripts/make_sliced_yolo_dataset.py --grayscale",
                "Conv_SWS-style SimAM with slicing before stride-2 convolution",
                "FBC2f-style FasterNet/PConv cross-stage module",
            ],
            "planned": [
                "NWD-IoU training loss with alpha=0.5",
            ],
        },
        "args": vars(args),
        "replacement_stats": stats.to_dict(),
    }
    (save_dir / "pcb_yolo_mvp_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def print_replacement_stats(stats: ReplacementStats) -> None:
    print("PCB-YOLO-MVP module replacements:")
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


def train_overrides(args: argparse.Namespace) -> dict[str, object]:
    project_dir = Path(args.project).resolve()
    return {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "project": str(project_dir),
        "name": args.name,
        "patience": args.patience,
        "cache": parse_bool(args.cache),
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "exist_ok": args.exist_ok,
        "plots": True,
        "save": True,
        "val": True,
    }


def train_injected_model(args: argparse.Namespace, target_names: set[str]) -> tuple[Path, ReplacementStats]:
    """Let Ultralytics build the dataset-specific model, then inject MVP modules."""
    yolo = YOLO(args.model)
    trainer = yolo._smart_load("trainer")(overrides=train_overrides(args), _callbacks=yolo.callbacks)
    original_setup_model = trainer.setup_model
    stats_holder: dict[str, ReplacementStats] = {}

    def setup_model_with_mvp():
        ckpt = original_setup_model()
        stats = apply_pcb_yolo_mvp(
            trainer.model,
            replace_sws=not args.no_sws,
            replace_fbc2f=not args.no_fbc2f,
            sws_blocks=args.sws_blocks,
            simam_lambda=args.simam_lambda,
            pconv_n_div=args.pconv_n_div,
            target_csp_names=target_names,
        )
        stats_holder["stats"] = stats
        print_replacement_stats(stats)
        return ckpt

    trainer.setup_model = setup_model_with_mvp
    trainer.train()
    return Path(trainer.save_dir), stats_holder.get("stats", ReplacementStats())


def main() -> int:
    args = parse_args()
    target_names = {item.strip() for item in args.target_csp_names.split(",") if item.strip()}
    planned_save_dir = Path(args.project).resolve() / args.name

    if args.dry_run:
        model = YOLO(args.model)
        stats = apply_pcb_yolo_mvp(
            model,
            replace_sws=not args.no_sws,
            replace_fbc2f=not args.no_fbc2f,
            sws_blocks=args.sws_blocks,
            simam_lambda=args.simam_lambda,
            pconv_n_div=args.pconv_n_div,
            target_csp_names=target_names,
        )
        print_replacement_stats(stats)
        write_manifest(planned_save_dir, args, stats, dry_run=True)
        if hasattr(model.model, "info"):
            try:
                model.model.info(verbose=True)
            except TypeError:
                model.model.info()
        print(f"Wrote dry-run manifest to: {planned_save_dir / 'pcb_yolo_mvp_manifest.json'}")
        return 0

    save_dir, stats = train_injected_model(args, target_names)
    write_manifest(save_dir, args, stats, dry_run=False)
    print(f"Wrote PCB-YOLO-MVP manifest to: {save_dir / 'pcb_yolo_mvp_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
