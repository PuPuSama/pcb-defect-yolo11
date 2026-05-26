from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


CLASSES = [
    "missing_pad",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

COLORS = [
    (46, 204, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (231, 76, 60),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO labels on sample PCB defect images.")
    parser.add_argument(
        "--dataset-root",
        default=r"E:\Research\pcb-defect-yolo11\datasets\pcb-defect-yolo",
    )
    parser.add_argument(
        "--output-dir",
        default=r"E:\Research\pcb-defect-yolo11\runs\label_checks",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def draw_label(image, class_id: int, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[class_id % len(COLORS)]
    name = CLASSES[class_id]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    label = f"{class_id}:{name}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y_text = max(y1, text_h + baseline + 4)
    cv2.rectangle(image, (x1, y_text - text_h - baseline - 4), (x1 + text_w + 6, y_text), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, y_text - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def read_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = [float(value) for value in parts[1:5]]
        x1 = int(round((x_center - box_w / 2.0) * width))
        y1 = int(round((y_center - box_h / 2.0) * height))
        x2 = int(round((x_center + box_w / 2.0) * width))
        y2 = int(round((y_center + box_h / 2.0) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        boxes.append((class_id, (x1, y1, x2, y2)))
    return boxes


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {image_dir}")
    sample_size = min(args.count, len(image_paths))
    sampled = random.Random(args.seed).sample(image_paths, sample_size)

    for image_path in sampled:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV failed to read image: {image_path}")
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_boxes(label_path, width, height)
        for class_id, box in boxes:
            draw_label(image, class_id, box)
        output_path = output_dir / f"{args.split}_{image_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"Wrote {output_path} ({len(boxes)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

