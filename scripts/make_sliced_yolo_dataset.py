from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "pcb-defect-yolo"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "pcb-defect-yolo-sliced-960"
CLASS_NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Window:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sliced YOLO dataset for PCB small-defect detection. "
            "Each original high-resolution image is converted into overlapping local crops."
        )
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--slice-size", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--negative-ratio", type=float, default=0.20)
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert crops to grayscale RGB, matching the PCB-YOLO paper's background-color suppression step.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_labels(label_path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx = float(parts[1]) * width
        cy = float(parts[2]) * height
        bw = float(parts[3]) * width
        bh = float(parts[4]) * height
        boxes.append(Box(cls, cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0))
    return boxes


def grid_windows(width: int, height: int, slice_size: int, overlap: float) -> list[Window]:
    stride = max(1, int(round(slice_size * (1.0 - overlap))))
    xs = list(range(0, max(1, width - slice_size + 1), stride))
    ys = list(range(0, max(1, height - slice_size + 1), stride))
    last_x = max(0, width - slice_size)
    last_y = max(0, height - slice_size)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)

    windows: list[Window] = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            windows.append(Window(x, y, min(width, x + slice_size), min(height, y + slice_size)))
    return windows


def clip_box_to_window(box: Box, window: Window, min_visibility: float, min_box_size: float) -> Box | None:
    ix1 = max(box.x1, window.x1)
    iy1 = max(box.y1, window.y1)
    ix2 = min(box.x2, window.x2)
    iy2 = min(box.y2, window.y2)
    clipped_w = ix2 - ix1
    clipped_h = iy2 - iy1
    if clipped_w < min_box_size or clipped_h < min_box_size:
        return None

    clipped_area = clipped_w * clipped_h
    if box.area <= 0 or clipped_area / box.area < min_visibility:
        return None
    return Box(box.cls, ix1 - window.x1, iy1 - window.y1, ix2 - window.x1, iy2 - window.y1)


def yolo_line(box: Box, window: Window) -> str:
    cx = ((box.x1 + box.x2) / 2.0) / window.width
    cy = ((box.y1 + box.y2) / 2.0) / window.height
    bw = (box.x2 - box.x1) / window.width
    bh = (box.y2 - box.y1) / window.height
    values = [min(1.0, max(0.0, value)) for value in (cx, cy, bw, bh)]
    return f"{box.cls} " + " ".join(f"{value:.6f}" for value in values)


def write_yaml(output_root: Path) -> None:
    names = ", ".join(f'"{name}"' for name in CLASS_NAMES)
    yaml_path = output_root / "pcb-defect-sliced.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def process_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    slice_size: int,
    overlap: float,
    min_visibility: float,
    min_box_size: float,
    negative_ratio: float,
    grayscale: bool,
    rng: random.Random,
) -> dict[str, int]:
    src_image_dir = dataset_root / "images" / split
    src_label_dir = dataset_root / "labels" / split
    dst_image_dir = output_root / "images" / split
    dst_label_dir = output_root / "labels" / split
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_images": 0,
        "positive_slices": 0,
        "negative_slices": 0,
        "boxes": 0,
    }

    for image_path in sorted(src_image_dir.glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        if grayscale:
            image = image.convert("L").convert("RGB")
        width, height = image.size
        boxes = read_labels(src_label_dir / f"{image_path.stem}.txt", width, height)
        stats["source_images"] += 1

        positive_items: list[tuple[Window, list[Box]]] = []
        negative_windows: list[Window] = []
        for window in grid_windows(width, height, slice_size, overlap):
            clipped_boxes = [
                clipped
                for box in boxes
                if (clipped := clip_box_to_window(box, window, min_visibility, min_box_size)) is not None
            ]
            if clipped_boxes:
                positive_items.append((window, clipped_boxes))
            else:
                negative_windows.append(window)

        max_negatives = int(round(len(positive_items) * negative_ratio))
        sampled_negatives = rng.sample(negative_windows, min(max_negatives, len(negative_windows))) if max_negatives > 0 else []

        all_items: list[tuple[Window, list[Box]]] = positive_items + [(window, []) for window in sampled_negatives]
        for index, (window, clipped_boxes) in enumerate(all_items):
            stem = f"{image_path.stem}_x{window.x1}_y{window.y1}_s{index:03d}"
            crop = image.crop((window.x1, window.y1, window.x2, window.y2))
            crop.save(dst_image_dir / f"{stem}.jpg", quality=95)
            label_text = "\n".join(yolo_line(box, window) for box in clipped_boxes)
            (dst_label_dir / f"{stem}.txt").write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
            if clipped_boxes:
                stats["positive_slices"] += 1
                stats["boxes"] += len(clipped_boxes)
            else:
                stats["negative_slices"] += 1

    return stats


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists, pass --force to overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    summary = {
        "source_dataset": str(dataset_root.resolve()),
        "output_dataset": str(output_root.resolve()),
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "min_visibility": args.min_visibility,
        "grayscale": args.grayscale,
        "negative_ratio": args.negative_ratio,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        summary["splits"][split] = process_split(
            dataset_root,
            output_root,
            split,
            args.slice_size,
            args.overlap,
            args.min_visibility,
            args.min_box_size,
            args.negative_ratio,
            args.grayscale,
            rng,
        )
    write_yaml(output_root)
    (output_root / "slicing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote sliced dataset YAML: {output_root / 'pcb-defect-sliced.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
