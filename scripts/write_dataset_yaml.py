from __future__ import annotations

import argparse
from pathlib import Path


CLASSES = [
    "missing_pad",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a PCB-Defect Ultralytics dataset YAML.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Absolute dataset root on this machine, e.g. /home/user/pcb-defect-yolo11/datasets/pcb-defect-yolo",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output YAML path. Defaults to <dataset-root>/pcb-defect.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else dataset_root / "pcb-defect.yaml"
    names = ", ".join(f'"{name}"' for name in CLASSES)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

