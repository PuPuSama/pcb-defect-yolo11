# PCB Defect YOLO11

Working project for a two-month paper prototype:

**Adaptive slicing and lightweight YOLO11 for high-resolution PCB small-defect detection.**

## Current MVP

Build a paper-grounded PCB-YOLO MVP on top of the existing YOLO11 workflow.

Implemented pieces:

1. Grayscale + overlapping sliced dataset generation for small PCB defects.
2. Conv_SWS-style SimAM-with-slicing injection before stride-2 convolutions.
3. FBC2f-style PConv/FasterNet replacement for C2f/C3k2-like CSP blocks.
4. Baseline and sliced-mode evaluation scripts for same-split comparison.

Next piece:

1. Wire NWD-IoU into the Ultralytics box loss and run the full ablation.

Start here:

```bash
python scripts/make_sliced_yolo_dataset.py --dataset-root datasets/pcb-defect-yolo --output-root datasets/pcb-defect-yolo-sliced-960-gray --slice-size 960 --overlap 0.25 --min-visibility 0.35 --negative-ratio 0.20 --grayscale --force
python scripts/train_pcb_yolo_mvp.py --data datasets/pcb-defect-yolo-sliced-960-gray/pcb-defect-sliced.yaml --dry-run --exist-ok
```

Detailed runbook: `notes/pcb-yolo-mvp-runbook.md`.

## Project Layout

- `configs/`: dataset YAML and training configs.
- `data/`: raw downloaded archives and temporary extracted files.
- `datasets/`: YOLO-format datasets.
- `notes/`: experiment notes and dataset records.
- `runs/`: training and evaluation outputs.
- `scripts/`: environment checks, conversion scripts, and experiment helpers.
