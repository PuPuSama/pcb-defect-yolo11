# PCB Defect YOLO11

Working project for a two-month paper prototype:

**Adaptive slicing and lightweight YOLO11 for high-resolution PCB small-defect detection.**

## Current First Milestone

Run a clean YOLO11n baseline before modifying the model.

1. Confirm CUDA in the `pcb-yolo` conda environment.
2. Download PCB-Defect from Mendeley Data.
3. Convert COCO annotations to YOLO format.
4. Train YOLO11n on the original full-image resize protocol.
5. Record baseline metrics and qualitative failures.

## Project Layout

- `configs/`: dataset YAML and training configs.
- `data/`: raw downloaded archives and temporary extracted files.
- `datasets/`: YOLO-format datasets.
- `notes/`: experiment notes and dataset records.
- `runs/`: training and evaluation outputs.
- `scripts/`: environment checks, conversion scripts, and experiment helpers.

