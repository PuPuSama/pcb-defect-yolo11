from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "pcb-defect-yolo"
DEFAULT_OUTPUT = PROJECT_ROOT / "report_assets" / "slicing_scale_study"
CLASS_NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate how local slicing enlarges PCB defect scale.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--slice-size", type=int, default=960)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(26, True)
FONT_AXIS = load_font(16)
FONT_SMALL = load_font(13)


def iter_splits(split: str) -> list[str]:
    return ["train", "val", "test"] if split == "all" else [split]


def collect_records(dataset_root: Path, split: str, slice_size: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split_name in iter_splits(split):
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            width, height = Image.open(image_path).size
            image_area = width * height
            local_area = min(slice_size, width) * min(slice_size, height)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                box_w = float(parts[3]) * width
                box_h = float(parts[4]) * height
                box_area = box_w * box_h
                full_ratio = box_area / image_area
                slice_ratio = box_area / local_area
                gain = slice_ratio / full_ratio if full_ratio > 0 else 0.0
                records.append(
                    {
                        "split": split_name,
                        "image": image_path.name,
                        "class": CLASS_NAMES[cls_id],
                        "image_width": width,
                        "image_height": height,
                        "box_width": box_w,
                        "box_height": box_h,
                        "full_area_ratio": full_ratio,
                        "slice_area_ratio": slice_ratio,
                        "scale_gain": gain,
                    }
                )
    return records


def bin_label(value: float) -> str:
    if value < 0.0005:
        return "<0.05%"
    if value < 0.001:
        return "0.05-0.10%"
    if value < 0.0025:
        return "0.10-0.25%"
    if value < 0.005:
        return "0.25-0.50%"
    if value < 0.01:
        return "0.50-1.00%"
    return ">=1.00%"


def save_gain_chart(records: list[dict[str, object]], path: Path) -> None:
    labels = ["<0.05%", "0.05-0.10%", "0.10-0.25%", "0.25-0.50%", "0.50-1.00%", ">=1.00%"]
    grouped: dict[str, list[float]] = {label: [] for label in labels}
    counts = {label: 0 for label in labels}
    for record in records:
        label = bin_label(float(record["full_area_ratio"]))
        grouped[label].append(float(record["scale_gain"]))
        counts[label] += 1
    medians = [median(grouped[label]) if grouped[label] else 0 for label in labels]

    width, height = 1300, 760
    margin_left, margin_right = 110, 60
    margin_top, margin_bottom = 95, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 34), "Median Scale Gain after 960px Local Slicing", fill="#111827", font=FONT_TITLE, anchor="mm")

    y_max = max(medians + [1]) * 1.18
    for i in range(6):
        y = margin_top + plot_h - int(plot_h * i / 5)
        value = y_max * i / 5
        draw.line((margin_left, y, margin_left + plot_w, y), fill="#e5e7eb", width=1)
        draw.text((margin_left - 12, y), f"{value:.1f}x", fill="#4b5563", font=FONT_SMALL, anchor="rm")
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#374151", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="#374151", width=2)

    group_w = plot_w / len(labels)
    bar_w = int(group_w * 0.58)
    for i, label in enumerate(labels):
        value = medians[i]
        x0 = int(margin_left + group_w * i + (group_w - bar_w) / 2)
        x1 = x0 + bar_w
        y0 = margin_top + plot_h - int(plot_h * value / y_max)
        y1 = margin_top + plot_h
        draw.rectangle((x0, y0, x1, y1), fill="#0ea5e9")
        draw.text((x0 + bar_w / 2, y0 - 9), f"{value:.1f}x", fill="#111827", font=FONT_SMALL, anchor="ms")
        draw.text((x0 + bar_w / 2, y1 + 20), label, fill="#111827", font=FONT_SMALL, anchor="mt")
        draw.text((x0 + bar_w / 2, y1 + 42), f"n={counts[label]}", fill="#64748b", font=FONT_SMALL, anchor="mt")
    draw.text((34, margin_top + plot_h / 2), "median gain", fill="#374151", font=FONT_AXIS, anchor="mm")
    image.save(path)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(dataset_root, args.split, args.slice_size)
    if not records:
        raise ValueError("No label records found.")

    csv_path = output_dir / f"slicing_scale_records_{args.split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    gains = [float(record["scale_gain"]) for record in records]
    small_records = [record for record in records if float(record["full_area_ratio"]) < 0.0025]
    summary = {
        "split": args.split,
        "slice_size": args.slice_size,
        "boxes": len(records),
        "median_gain": median(gains),
        "small_box_count": len(small_records),
        "small_box_ratio": len(small_records) / len(records),
        "small_box_median_gain": median([float(record["scale_gain"]) for record in small_records]) if small_records else 0,
    }
    (output_dir / f"slicing_scale_summary_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_gain_chart(records, output_dir / f"slicing_scale_gain_{args.split}.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote slicing scale study to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
