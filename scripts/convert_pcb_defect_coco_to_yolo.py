from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CLASSES = [
    "missing_pad",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PCB-Defect COCO annotations to YOLO format.")
    parser.add_argument(
        "--raw-root",
        default=r"E:\Research\pcb-defect-yolo11\data\pcb-defect-raw\PCB_Defect",
        help="Raw PCB_Defect directory containing images/ and annotation/_annotations.coco.json",
    )
    parser.add_argument(
        "--output-root",
        default=r"E:\Research\pcb-defect-yolo11\datasets\pcb-defect-yolo",
        help="Output YOLO-format dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the existing output directory before conversion.",
    )
    return parser.parse_args()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive.")


def build_splits(image_ids: list[int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, set[int]]:
    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = int(round(len(shuffled) * train_ratio))
    val_end = train_end + int(round(len(shuffled) * val_ratio))
    return {
        "train": set(shuffled[:train_end]),
        "val": set(shuffled[train_end:val_end]),
        "test": set(shuffled[val_end:]),
    }


def yolo_line(category_index: int, bbox: list[float], width: int, height: int) -> str:
    x, y, box_w, box_h = bbox
    x_center = (x + box_w / 2.0) / width
    y_center = (y + box_h / 2.0) / height
    norm_w = box_w / width
    norm_h = box_h / height
    values = [x_center, y_center, norm_w, norm_h]
    clipped = [min(1.0, max(0.0, value)) for value in values]
    return f"{category_index} " + " ".join(f"{value:.6f}" for value in clipped)


def write_yaml(output_root: Path, classes: list[str]) -> None:
    yaml_path = output_root / "pcb-defect.yaml"
    names = ", ".join(f'"{name}"' for name in classes)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    raw_root = Path(args.raw_root)
    image_root = raw_root / "images"
    annotation_path = raw_root / "annotation" / "_annotations.coco.json"
    output_root = Path(args.output_root)

    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists. Use --force to overwrite: {output_root}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in data.get("categories", [])}
    category_to_yolo = {}
    for category_id, name in categories.items():
        if name in DEFAULT_CLASSES:
            category_to_yolo[category_id] = DEFAULT_CLASSES.index(name)

    missing_classes = [name for name in DEFAULT_CLASSES if name not in set(categories.values())]
    if missing_classes:
        raise ValueError(f"Missing expected classes in COCO categories: {missing_classes}")

    images = {int(item["id"]): item for item in data.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    ignored = Counter()
    for annotation in data.get("annotations", []):
        category_id = int(annotation["category_id"])
        if category_id not in category_to_yolo:
            ignored[category_id] += 1
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    splits = build_splits(
        list(images.keys()),
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    image_to_split = {
        image_id: split_name
        for split_name, ids in splits.items()
        for image_id in ids
    }

    for split_name in ("train", "val", "test"):
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    class_counts = {split_name: Counter() for split_name in ("train", "val", "test")}
    image_counts = Counter()
    box_counts = Counter()

    for image_id, image in images.items():
        split_name = image_to_split[image_id]
        file_name = image["file_name"]
        src_image = image_root / file_name
        if not src_image.is_file():
            raise FileNotFoundError(f"Image referenced by COCO JSON does not exist: {src_image}")

        dst_image = output_root / "images" / split_name / file_name
        shutil.copy2(src_image, dst_image)

        width = int(image["width"])
        height = int(image["height"])
        label_path = output_root / "labels" / split_name / f"{Path(file_name).stem}.txt"
        lines = []
        for annotation in annotations_by_image.get(image_id, []):
            category_index = category_to_yolo[int(annotation["category_id"])]
            lines.append(yolo_line(category_index, annotation["bbox"], width, height))
            class_counts[split_name][DEFAULT_CLASSES[category_index]] += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_counts[split_name] += 1
        box_counts[split_name] += len(lines)

    write_yaml(output_root, DEFAULT_CLASSES)

    print(f"Wrote YOLO dataset to: {output_root}")
    print(f"Classes: {DEFAULT_CLASSES}")
    print(f"Ignored category ids: {dict(ignored)}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: images={image_counts[split_name]} boxes={box_counts[split_name]}")
        print(f"{split_name} class counts: {dict(class_counts[split_name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

