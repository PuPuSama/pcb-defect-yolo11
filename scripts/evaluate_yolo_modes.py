from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "pcb-defect-yolo"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "mode_eval"
CLASS_NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
IOU_THRESHOLDS = [round(0.5 + 0.05 * index, 2) for index in range(10)]


@dataclass
class Box:
    image_id: str
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    source: str = "gt"

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Window:
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO full-image and sliced-inference modes on original PCB images.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--mode", default="sliced", choices=["full", "sliced", "full-sliced"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--slice-size", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images in the split.")
    parser.add_argument("--save-vis", type=int, default=6)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT = load_font(16)
FONT_BOLD = load_font(18, True)


def read_gt(label_path: Path, image_id: str, width: int, height: int) -> list[Box]:
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
        boxes.append(Box(image_id, cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
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
    return [Window(x, y, min(width, x + slice_size), min(height, y + slice_size)) for y in sorted(set(ys)) for x in sorted(set(xs))]


def detections_from_result(result, image_id: str, offset_x: int = 0, offset_y: int = 0, source: str = "pred") -> list[Box]:
    if result.boxes is None:
        return []
    detections: list[Box] = []
    xyxy = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    for box, conf, cls in zip(xyxy, confs, classes):
        detections.append(
            Box(
                image_id=image_id,
                cls=int(cls),
                x1=float(box[0] + offset_x),
                y1=float(box[1] + offset_y),
                x2=float(box[2] + offset_x),
                y2=float(box[3] + offset_y),
                conf=float(conf),
                source=source,
            )
        )
    return detections


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def class_aware_nms(detections: list[Box], iou_threshold: float) -> list[Box]:
    kept: list[Box] = []
    for det in sorted(detections, key=lambda box: box.conf, reverse=True):
        if all(det.cls != old.cls or box_iou(det, old) < iou_threshold for old in kept):
            kept.append(det)
    return kept


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_image(model: YOLO, image_path: Path, args: argparse.Namespace) -> tuple[list[Box], list[Box], list[Window]]:
    image_id = image_path.stem
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    detections: list[Box] = []
    windows: list[Window] = []

    if args.mode in {"full", "full-sliced"}:
        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou, device=args.device, verbose=False)[0]
        detections.extend(detections_from_result(result, image_id, source="full"))

    if args.mode in {"sliced", "full-sliced"}:
        windows = grid_windows(width, height, args.slice_size, args.overlap)
        crops = [image.crop((window.x1, window.y1, window.x2, window.y2)) for window in windows]
        for crop_batch, window_batch in zip(chunks(crops, args.batch), chunks(windows, args.batch)):
            results = model.predict(crop_batch, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou, device=args.device, verbose=False)
            for result, window in zip(results, window_batch):
                detections.extend(detections_from_result(result, image_id, window.x1, window.y1, source="slice"))

    fused = class_aware_nms(detections, args.nms_iou)
    return detections, fused, windows


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def evaluate_class(
    class_id: int,
    threshold: float,
    gt_by_image: dict[str, list[Box]],
    detections: list[Box],
) -> tuple[float, int, int, int]:
    class_gt = {
        image_id: [box for box in boxes if box.cls == class_id]
        for image_id, boxes in gt_by_image.items()
    }
    total_gt = sum(len(boxes) for boxes in class_gt.values())
    if total_gt == 0:
        return 0.0, 0, 0, 0

    matched = {image_id: [False] * len(boxes) for image_id, boxes in class_gt.items()}
    sorted_dets = sorted([box for box in detections if box.cls == class_id], key=lambda box: box.conf, reverse=True)
    tp_values: list[int] = []
    fp_values: list[int] = []
    tp = 0
    fp = 0
    for det in sorted_dets:
        gt_boxes = class_gt.get(det.image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_boxes):
            if matched[det.image_id][index]:
                continue
            iou = box_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_iou >= threshold and best_index >= 0:
            matched[det.image_id][best_index] = True
            tp += 1
            tp_values.append(1)
            fp_values.append(0)
        else:
            fp += 1
            tp_values.append(0)
            fp_values.append(1)

    cum_tp = 0
    cum_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for det_tp, det_fp in zip(tp_values, fp_values):
        cum_tp += det_tp
        cum_fp += det_fp
        recalls.append(cum_tp / total_gt)
        precisions.append(cum_tp / max(1, cum_tp + cum_fp))
    return compute_ap(recalls, precisions), tp, fp, total_gt


def compute_metrics(gt_by_image: dict[str, list[Box]], detections: list[Box]) -> dict[str, object]:
    per_class: dict[str, object] = {}
    map50_values: list[float] = []
    map5095_values: list[float] = []
    total_tp50 = 0
    total_fp50 = 0
    total_gt = 0

    for class_id, class_name in enumerate(CLASS_NAMES):
        aps = []
        tp50 = fp50 = gt_count = 0
        for threshold in IOU_THRESHOLDS:
            ap, tp, fp, gt = evaluate_class(class_id, threshold, gt_by_image, detections)
            aps.append(ap)
            if threshold == 0.5:
                tp50, fp50, gt_count = tp, fp, gt
        if gt_count > 0:
            map50_values.append(aps[0])
            map5095_values.append(sum(aps) / len(aps))
            total_tp50 += tp50
            total_fp50 += fp50
            total_gt += gt_count
        per_class[class_name] = {
            "gt": gt_count,
            "ap50": aps[0],
            "ap50_95": sum(aps) / len(aps),
            "recall50": tp50 / gt_count if gt_count else 0.0,
            "precision50": tp50 / max(1, tp50 + fp50),
        }

    return {
        "precision50": total_tp50 / max(1, total_tp50 + total_fp50),
        "recall50": total_tp50 / max(1, total_gt),
        "mAP50": sum(map50_values) / max(1, len(map50_values)),
        "mAP50_95": sum(map5095_values) / max(1, len(map5095_values)),
        "per_class": per_class,
        "detections": len(detections),
        "gt": total_gt,
    }


def draw_visualization(image_path: Path, gt: list[Box], preds: list[Box], windows: list[Window], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for window in windows:
        draw.rectangle((window.x1, window.y1, window.x2, window.y2), outline="#94a3b8", width=3)
    for box in gt:
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="#22c55e", width=4)
        draw.text((box.x1, max(0, box.y1 - 22)), f"GT {CLASS_NAMES[box.cls]}", fill="#22c55e", font=FONT_BOLD)
    for box in preds:
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="#ef4444", width=3)
        label = f"{CLASS_NAMES[box.cls]} {box.conf:.2f}"
        draw.text((box.x1, min(image.height - 22, box.y2 + 2)), label, fill="#ef4444", font=FONT)
    image.thumbnail((1600, 1600))
    image.save(output_path)


def write_markdown(metrics: dict[str, object], args: argparse.Namespace, output_path: Path) -> None:
    lines = [
        f"# YOLO Mode Evaluation: {args.mode}",
        "",
        f"- Weights: `{args.weights}`",
        f"- Split: `{args.split}`",
        f"- Image size: `{args.imgsz}`",
        f"- Slice size: `{args.slice_size}`",
        f"- Overlap: `{args.overlap}`",
        f"- Confidence threshold: `{args.conf}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision@50 | {metrics['precision50']:.5f} |",
        f"| Recall@50 | {metrics['recall50']:.5f} |",
        f"| mAP@50 | {metrics['mAP50']:.5f} |",
        f"| mAP@50:95 | {metrics['mAP50_95']:.5f} |",
        f"| Predictions after NMS | {metrics['detections']} |",
        f"| Ground-truth boxes | {metrics['gt']} |",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | GT | Precision@50 | Recall@50 | AP@50 | AP@50:95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name, row in metrics["per_class"].items():
        lines.append(
            f"| {class_name} | {row['gt']} | {row['precision50']:.5f} | "
            f"{row['recall50']:.5f} | {row['ap50']:.5f} | {row['ap50_95']:.5f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    image_paths = sorted((dataset_root / "images" / args.split).glob("*.jpg"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    gt_by_image: dict[str, list[Box]] = {}
    all_predictions: list[Box] = []
    image_summaries = []
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        gt = read_gt(dataset_root / "labels" / args.split / f"{image_path.stem}.txt", image_path.stem, image.width, image.height)
        gt_by_image[image_path.stem] = gt
        raw_predictions, fused_predictions, windows = predict_image(model, image_path, args)
        all_predictions.extend(fused_predictions)
        image_summaries.append(
            {
                "image": image_path.name,
                "gt": len(gt),
                "raw_predictions": len(raw_predictions),
                "fused_predictions": len(fused_predictions),
                "windows": len(windows),
            }
        )
        if index < args.save_vis:
            draw_visualization(image_path, gt, fused_predictions, windows, vis_dir / f"{image_path.stem}_{args.mode}.jpg")

    metrics = compute_metrics(gt_by_image, all_predictions)
    payload = {
        "args": vars(args),
        "summary": image_summaries,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(metrics, args, output_dir / "metrics.md")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote evaluation to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
