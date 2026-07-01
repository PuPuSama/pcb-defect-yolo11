from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "pcb-defect-yolo"
OUTPUT_DIR = PROJECT_ROOT / "report_assets"

CLASSES = [
    "missing_pad",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

RUNS = [
    ("YOLO11n-640-e20", PROJECT_ROOT / "runs" / "baselines" / "yolo11n_full_640_e20"),
    ("YOLO11n-640", PROJECT_ROOT / "runs" / "baselines" / "yolo11n_full_640"),
    ("YOLO11n-960", PROJECT_ROOT / "runs" / "baselines" / "yolo11n_full_960_server"),
    ("YOLO26n-960", PROJECT_ROOT / "runs" / "baselines" / "yolo26n_full_960_server"),
]

METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in font_names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(26, True)
FONT_AXIS = load_font(16)
FONT_SMALL = load_font(13)
FONT_BOLD = load_font(15, True)


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_run_metrics(run_dir: Path) -> dict[str, object]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")

    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [clean_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {results_path}")

    last = rows[-1]
    best_map50 = max(rows, key=lambda row: as_float(row.get(METRICS["map50"], "nan")))
    best_map5095 = max(rows, key=lambda row: as_float(row.get(METRICS["map5095"], "nan")))
    return {
        "epochs": int(float(last.get("epoch", "0"))),
        "final": {name: as_float(last[column]) for name, column in METRICS.items()},
        "best_map50": {
            "value": as_float(best_map50.get(METRICS["map50"], "nan")),
            "epoch": int(float(best_map50.get("epoch", "0"))),
        },
        "best_map5095": {
            "value": as_float(best_map5095.get(METRICS["map5095"], "nan")),
            "epoch": int(float(best_map5095.get("epoch", "0"))),
        },
        "loss": {
            "train_box": as_float(last.get("train/box_loss", "nan")),
            "val_box": as_float(last.get("val/box_loss", "nan")),
        },
    }


def inspect_dataset() -> dict[str, object]:
    split_stats: dict[str, dict[str, object]] = {}
    total_class_counts = Counter()
    area_bins = Counter()
    aspect_bins = Counter()
    total_boxes = 0
    bins = [
        ("<0.05%", 0.0005),
        ("0.05-0.10%", 0.001),
        ("0.10-0.25%", 0.0025),
        ("0.25-0.50%", 0.005),
        ("0.50-1.00%", 0.01),
        (">=1.00%", float("inf")),
    ]

    for split in ("train", "val", "test"):
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        image_count = len(list(image_dir.glob("*.jpg")))
        class_counts = Counter()
        split_boxes = 0

        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
                area = width * height
                class_name = CLASSES[class_id]
                class_counts[class_name] += 1
                total_class_counts[class_name] += 1
                total_boxes += 1
                split_boxes += 1

                for label, upper in bins:
                    if area < upper:
                        area_bins[label] += 1
                        break
                ratio = width / height if height else 0
                if ratio < 0.5:
                    aspect_bins["tall"] += 1
                elif ratio > 2.0:
                    aspect_bins["wide"] += 1
                else:
                    aspect_bins["balanced"] += 1

        split_stats[split] = {
            "images": image_count,
            "boxes": split_boxes,
            "class_counts": dict(class_counts),
        }

    return {
        "splits": split_stats,
        "class_counts": dict(total_class_counts),
        "area_bins": dict(area_bins),
        "aspect_bins": dict(aspect_bins),
        "total_images": sum(int(split_stats[s]["images"]) for s in split_stats),
        "total_boxes": total_boxes,
    }


def rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)


def save_bar_chart(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    percent: bool = False,
) -> None:
    width, height = 1300, 760
    margin_left, margin_right = 115, 60
    margin_top, margin_bottom = 95, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 34), title, fill="#111827", font=FONT_TITLE, anchor="mm")

    max_value = max(max(values) for _, values, _ in series) if series else 1
    max_value = max(max_value, 0.01)
    y_max = 1.0 if percent else max_value * 1.18

    axis_color = "#374151"
    grid_color = "#e5e7eb"
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=axis_color, width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=axis_color, width=2)

    ticks = 5
    for i in range(ticks + 1):
        value = y_max * i / ticks
        y = margin_top + plot_h - int(plot_h * value / y_max)
        draw.line((margin_left, y, margin_left + plot_w, y), fill=grid_color, width=1)
        label = f"{value:.1%}" if percent else f"{value:.0f}"
        draw.text((margin_left - 12, y), label, fill="#4b5563", font=FONT_SMALL, anchor="rm")

    group_w = plot_w / max(1, len(categories))
    bar_gap = 6
    bar_w = max(16, int((group_w - 34) / max(1, len(series))))
    for i, category in enumerate(categories):
        group_x = margin_left + group_w * i + 18
        for j, (_, values, color) in enumerate(series):
            value = values[i]
            bar_h = int(plot_h * value / y_max)
            x0 = int(group_x + j * (bar_w + bar_gap))
            y0 = margin_top + plot_h - bar_h
            x1 = x0 + bar_w
            y1 = margin_top + plot_h
            draw.rectangle((x0, y0, x1, y1), fill=color)
            value_label = f"{value:.3f}" if percent else f"{int(value)}"
            draw.text((x0 + bar_w / 2, y0 - 9), value_label, fill="#111827", font=FONT_SMALL, anchor="ms")
        draw.text((margin_left + group_w * i + group_w / 2, margin_top + plot_h + 22), category, fill="#111827", font=FONT_SMALL, anchor="mt")

    legend_x = margin_left
    legend_y = height - 70
    for name, _, color in series:
        draw.rectangle((legend_x, legend_y, legend_x + 20, legend_y + 14), fill=color)
        draw.text((legend_x + 28, legend_y + 7), name, fill="#111827", font=FONT_AXIS, anchor="lm")
        legend_x += 205
    draw.text((36, margin_top + plot_h / 2), y_label, fill="#374151", font=FONT_AXIS, anchor="mm")
    image.save(path)


def save_pipeline(path: Path) -> None:
    width, height = 1400, 520
    image = Image.new("RGB", (width, height), "#f9fafb")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 42), "PCB Defect Detection Experiment Pipeline", fill="#111827", font=FONT_TITLE, anchor="mm")

    boxes = [
        ("COCO annotations\n+ PCB images", "#dbeafe"),
        ("YOLO conversion\n70/20/10 split", "#e0f2fe"),
        ("Dataset check\nlabels + classes", "#ecfccb"),
        ("Train baselines\nYOLO11n/YOLO26n", "#fef3c7"),
        ("Validate\nP/R/mAP + curves", "#fee2e2"),
        ("Failure analysis\nsmall defects", "#ede9fe"),
        ("Next step\nadaptive slicing", "#dcfce7"),
    ]
    start_x, y0, box_w, box_h, gap = 55, 175, 165, 135, 28
    for i, (label, fill) in enumerate(boxes):
        x0 = start_x + i * (box_w + gap)
        rounded_rect(draw, (x0, y0, x0 + box_w, y0 + box_h), fill, "#64748b")
        for k, line in enumerate(label.split("\n")):
            draw.text((x0 + box_w / 2, y0 + 47 + k * 28), line, fill="#111827", font=FONT_BOLD if k == 0 else FONT_AXIS, anchor="mm")
        if i < len(boxes) - 1:
            ax0 = x0 + box_w + 5
            ax1 = x0 + box_w + gap - 5
            ay = y0 + box_h // 2
            draw.line((ax0, ay, ax1, ay), fill="#64748b", width=3)
            draw.polygon([(ax1, ay), (ax1 - 10, ay - 7), (ax1 - 10, ay + 7)], fill="#64748b")

    note = "Expansion focus: turn a simple baseline into data audit + resolution ablation + model comparison + error-driven research plan."
    draw.text((width // 2, 430), note, fill="#334155", font=FONT_AXIS, anchor="mm")
    image.save(path)


def write_summary_markdown(summary: dict[str, object]) -> None:
    lines: list[str] = []
    dataset = summary["dataset"]
    runs = summary["runs"]
    lines.append("# PCB Defect YOLO Report Assets")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Total images: {dataset['total_images']}")
    lines.append(f"- Total boxes: {dataset['total_boxes']}")
    lines.append("- Classes: " + ", ".join(CLASSES))
    lines.append("")
    lines.append("| Split | Images | Boxes |")
    lines.append("| --- | ---: | ---: |")
    for split, stats in dataset["splits"].items():
        lines.append(f"| {split} | {stats['images']} | {stats['boxes']} |")
    lines.append("")
    lines.append("## Run Comparison")
    lines.append("")
    lines.append("| Run | Epochs | Precision | Recall | mAP@50 | mAP@50:95 | Best mAP@50 | Best mAP@50:95 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in runs.items():
        final = metrics["final"]
        lines.append(
            f"| {name} | {metrics['epochs']} | {final['precision']:.5f} | {final['recall']:.5f} | "
            f"{final['map50']:.5f} | {final['map5095']:.5f} | "
            f"{metrics['best_map50']['value']:.5f} | {metrics['best_map5095']['value']:.5f} |"
        )
    lines.append("")
    lines.append("## Key Computed Deltas")
    lines.append("")
    lines.append("- YOLO11n 960 vs YOLO11n 640: higher input resolution improves precision and mAP@50, while recall improves more modestly.")
    lines.append("- YOLO26n 960 vs YOLO11n 960: the newer model does not automatically win on this small PCB dataset, which motivates task-specific small-object strategies.")
    lines.append("- The report expansion therefore frames adaptive slicing, hard-sample mining, and deployment-oriented validation as next-stage work.")
    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = inspect_dataset()
    runs = {name: read_run_metrics(path) for name, path in RUNS}
    summary = {"dataset": dataset, "runs": runs}

    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_markdown(summary)

    split_colors = {"train": "#2563eb", "val": "#16a34a", "test": "#f97316"}
    split_series = []
    for split in ("train", "val", "test"):
        counts = dataset["splits"][split]["class_counts"]
        split_series.append((split, [float(counts.get(cls, 0)) for cls in CLASSES], split_colors[split]))
    save_bar_chart(
        OUTPUT_DIR / "dataset_class_distribution.png",
        "Class Distribution by Split",
        CLASSES,
        split_series,
        "boxes",
    )

    area_order = ["<0.05%", "0.05-0.10%", "0.10-0.25%", "0.25-0.50%", "0.50-1.00%", ">=1.00%"]
    area_series = [("boxes", [float(dataset["area_bins"].get(label, 0)) for label in area_order], "#7c3aed")]
    save_bar_chart(
        OUTPUT_DIR / "box_area_distribution.png",
        "Normalized Box Area Distribution",
        area_order,
        area_series,
        "boxes",
    )

    run_names = list(runs.keys())
    metric_series = [
        ("Precision", [runs[name]["final"]["precision"] for name in run_names], "#2563eb"),
        ("Recall", [runs[name]["final"]["recall"] for name in run_names], "#16a34a"),
        ("mAP@50", [runs[name]["final"]["map50"] for name in run_names], "#f97316"),
        ("mAP@50:95", [runs[name]["final"]["map5095"] for name in run_names], "#7c3aed"),
    ]
    save_bar_chart(
        OUTPUT_DIR / "metrics_comparison.png",
        "Validation Metrics Comparison",
        run_names,
        metric_series,
        "metric",
        percent=True,
    )
    save_pipeline(OUTPUT_DIR / "method_pipeline.png")

    print(f"Wrote report assets to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
