from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "baselines" / "yolo11n_full_960_server" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "pcb-defect-yolo" / "images" / "val"
DEFAULT_OUTPUT = PROJECT_ROOT / "report_assets" / "adaptive_slicing_demo"
CLASS_NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


@dataclass(frozen=True)
class Window:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    source: str

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage full-image + adaptive sliced inference demo for PCB defect detection."
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image file or directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--full-imgsz", type=int, default=960)
    parser.add_argument("--slice-imgsz", type=int, default=960)
    parser.add_argument("--slice-size", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--small-area", type=float, default=0.003, help="Full-image area ratio threshold.")
    parser.add_argument("--low-conf", type=float, default=0.55)
    parser.add_argument("--context", type=float, default=3.0, help="Crop side >= context * box max side.")
    parser.add_argument("--max-adaptive-windows", type=int, default=8)
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also add overlapping grid windows as a fallback for defects missed by the full-image pass.",
    )
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT = load_font(18)
FONT_BOLD = load_font(20, True)


def image_paths(source: Path, max_images: int) -> list[Path]:
    if source.is_file():
        return [source]
    paths = sorted([*source.glob("*.jpg"), *source.glob("*.png"), *source.glob("*.jpeg")])
    return paths[:max_images]


def result_to_detections(result, source: str, offset_x: int = 0, offset_y: int = 0) -> list[Detection]:
    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    for box, conf, cls_id in zip(xyxy, confs, classes):
        detections.append(
            Detection(
                x1=float(box[0] + offset_x),
                y1=float(box[1] + offset_y),
                x2=float(box[2] + offset_x),
                y2=float(box[3] + offset_y),
                conf=float(conf),
                cls=int(cls_id),
                source=source,
            )
        )
    return detections


def clip_window(cx: float, cy: float, side: int, width: int, height: int, source: str) -> Window:
    side = min(side, max(width, height))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x1 = max(0, min(x1, max(0, width - side)))
    y1 = max(0, min(y1, max(0, height - side)))
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    return Window(x1=x1, y1=y1, x2=x2, y2=y2, source=source)


def grid_windows(width: int, height: int, slice_size: int, overlap: float) -> list[Window]:
    stride = max(1, int(slice_size * (1.0 - overlap)))
    xs = list(range(0, max(1, width - slice_size + 1), stride))
    ys = list(range(0, max(1, height - slice_size + 1), stride))
    if not xs or xs[-1] != max(0, width - slice_size):
        xs.append(max(0, width - slice_size))
    if not ys or ys[-1] != max(0, height - slice_size):
        ys.append(max(0, height - slice_size))
    windows = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            windows.append(Window(x, y, min(width, x + slice_size), min(height, y + slice_size), "grid"))
    return windows


def adaptive_windows(
    detections: list[Detection],
    width: int,
    height: int,
    slice_size: int,
    small_area: float,
    low_conf: float,
    context: float,
    max_windows: int,
) -> list[Window]:
    image_area = width * height
    candidates: list[tuple[float, Window]] = []
    for det in detections:
        area_ratio = det.area / image_area if image_area else 0.0
        if area_ratio <= small_area or det.conf <= low_conf:
            cx = (det.x1 + det.x2) / 2.0
            cy = (det.y1 + det.y2) / 2.0
            box_side = max(det.x2 - det.x1, det.y2 - det.y1)
            side = int(max(slice_size, box_side * context))
            window = clip_window(cx, cy, side, width, height, "adaptive")
            score = det.conf + area_ratio
            candidates.append((score, window))
    candidates.sort(key=lambda item: item[0])
    return unique_windows([window for _, window in candidates])[:max_windows]


def unique_windows(windows: list[Window]) -> list[Window]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[Window] = []
    for window in windows:
        key = (window.x1, window.y1, window.x2, window.y2)
        if key not in seen and window.width > 16 and window.height > 16:
            unique.append(window)
            seen.add(key)
    return unique


def box_iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.conf, reverse=True):
        if all(det.cls != old.cls or box_iou(det, old) < iou_threshold for old in kept):
            kept.append(det)
    return kept


def draw_output(image: Image.Image, full: list[Detection], fused: list[Detection], windows: list[Window], path: Path) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for window in windows:
        color = "#f59e0b" if window.source == "adaptive" else "#64748b"
        draw.rectangle((window.x1, window.y1, window.x2, window.y2), outline=color, width=5)
        draw.text((window.x1 + 8, window.y1 + 8), window.source, fill=color, font=FONT_BOLD)
    for det in full:
        draw.rectangle((det.x1, det.y1, det.x2, det.y2), outline="#60a5fa", width=3)
    for det in fused:
        color = "#22c55e" if det.source != "full" else "#2563eb"
        label = f"{CLASS_NAMES[det.cls]} {det.conf:.2f} {det.source}"
        draw.rectangle((det.x1, det.y1, det.x2, det.y2), outline=color, width=4)
        tx, ty = det.x1, max(0, det.y1 - 24)
        draw.rectangle((tx, ty, tx + max(180, len(label) * 9), ty + 24), fill=color)
        draw.text((tx + 4, ty + 3), label, fill="white", font=FONT)
    canvas.thumbnail((1600, 1600))
    canvas.save(path)


def run_one(model: YOLO, image_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    full_result = model.predict(
        source=str(image_path),
        imgsz=args.full_imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]
    full_detections = result_to_detections(full_result, "full")

    windows = adaptive_windows(
        full_detections,
        width,
        height,
        args.slice_size,
        args.small_area,
        args.low_conf,
        args.context,
        args.max_adaptive_windows,
    )
    if args.grid:
        windows = unique_windows(windows + grid_windows(width, height, args.slice_size, args.overlap))

    slice_detections: list[Detection] = []
    for index, window in enumerate(windows):
        crop = image.crop((window.x1, window.y1, window.x2, window.y2))
        result = model.predict(
            source=crop,
            imgsz=args.slice_imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        slice_detections.extend(result_to_detections(result, window.source, window.x1, window.y1))
        if index < 6:
            crop.save(output_dir / f"{image_path.stem}_crop_{index:02d}_{window.source}.jpg")

    fused = nms(full_detections + slice_detections, args.iou)
    draw_output(image, full_detections, fused, windows, output_dir / f"{image_path.stem}_adaptive_demo.jpg")
    return {
        "image": image_path.name,
        "size": [width, height],
        "full_detections": len(full_detections),
        "slice_windows": len(windows),
        "slice_detections": len(slice_detections),
        "fused_detections": len(fused),
        "windows": [window.__dict__ for window in windows],
    }


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    summary = []
    for image_path in image_paths(source, args.max_images):
        summary.append(run_one(model, image_path, output_dir, args))

    summary_path = output_dir / "adaptive_slicing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote adaptive slicing demo to: {output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
