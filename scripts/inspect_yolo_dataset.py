from __future__ import annotations

import argparse
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Inspect a YOLO-format PCB defect dataset.")
    parser.add_argument(
        "--dataset-root",
        default=r"E:\Research\pcb-defect-yolo11\datasets\pcb-defect-yolo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_count = len(list(image_dir.glob("*.jpg")))
        label_paths = list(label_dir.glob("*.txt"))
        class_counts = Counter()
        empty_labels = 0
        total_boxes = 0
        for label_path in label_paths:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[CLASSES[class_id]] += 1
                total_boxes += 1
        print(f"{split}: images={image_count}, labels={len(label_paths)}, empty_labels={empty_labels}, boxes={total_boxes}")
        print(f"{split} class counts: {dict(class_counts)}")
    print(f"YAML: {root / 'pcb-defect.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

