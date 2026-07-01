# Sliced-YOLO Server Runbook

This is the real improvement experiment for the course presentation:

**Train YOLO on local 960x960 PCB crops, then evaluate with sliced inference on the original high-resolution images.**

The experiment tests the hypothesis:

> PCB defects are small after full-image resizing. Training and inference on local high-resolution slices should improve small-defect recall and localization compared with full-image YOLO11n-960.

## 1. Prepare Code and Data

```bash
cd ~/pcb-defect-yolo11
git pull
conda activate pcb-yolo
python scripts/check_env.py
python scripts/write_dataset_yaml.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
```

If the sliced scripts are not on the server yet, copy these files from Windows:

```text
scripts/make_sliced_yolo_dataset.py
scripts/evaluate_yolo_modes.py
```

## 2. Build the Sliced Dataset

```bash
python scripts/make_sliced_yolo_dataset.py \
  --dataset-root datasets/pcb-defect-yolo \
  --output-root datasets/pcb-defect-yolo-sliced-960 \
  --slice-size 960 \
  --overlap 0.25 \
  --min-visibility 0.35 \
  --negative-ratio 0.20 \
  --force
```

Expected output:

```text
datasets/pcb-defect-yolo-sliced-960/
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
  pcb-defect-sliced.yaml
  slicing_summary.json
```

## 3. Train the Improved Model

On RTX 5880 24GB, start with batch 16. If memory is enough, try batch 24 or 32.

```bash
python scripts/train_yolo_baseline.py \
  --model yolo11n.pt \
  --data datasets/pcb-defect-yolo-sliced-960/pcb-defect-sliced.yaml \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --workers 8 \
  --name yolo11n_slice960_e100 \
  --project runs/sliced \
  --exist-ok
```

## 4. Evaluate Baseline with the Same Evaluator

This gives a fair reference for the original full-image model.

```bash
python scripts/evaluate_yolo_modes.py \
  --weights runs/baselines/yolo11n_full_960_server/weights/best.pt \
  --dataset-root datasets/pcb-defect-yolo \
  --split val \
  --mode full \
  --imgsz 960 \
  --conf 0.001 \
  --device 0 \
  --output runs/mode_eval/yolo11n_full960_val
```

## 5. Evaluate the Sliced-YOLO Improvement

```bash
python scripts/evaluate_yolo_modes.py \
  --weights runs/sliced/yolo11n_slice960_e100/weights/best.pt \
  --dataset-root datasets/pcb-defect-yolo \
  --split val \
  --mode sliced \
  --imgsz 960 \
  --slice-size 960 \
  --overlap 0.25 \
  --conf 0.001 \
  --device 0 \
  --batch 16 \
  --output runs/mode_eval/yolo11n_slice960_val
```

After validation, run test-set evaluation:

```bash
python scripts/evaluate_yolo_modes.py \
  --weights runs/sliced/yolo11n_slice960_e100/weights/best.pt \
  --dataset-root datasets/pcb-defect-yolo \
  --split test \
  --mode sliced \
  --imgsz 960 \
  --slice-size 960 \
  --overlap 0.25 \
  --conf 0.001 \
  --device 0 \
  --batch 16 \
  --output runs/mode_eval/yolo11n_slice960_test
```

## 6. What to Put in the Presentation

Use this wording if the sliced model improves recall or mAP:

> The proposed sliced-YOLO strategy increases the relative scale of PCB defects during both training and inference. Compared with full-image YOLO11n-960, it improves the target metric on the original validation images, showing that the bottleneck is not only model capacity but also the mismatch between high-resolution PCB images and fixed-size full-image resizing.

If it does not improve:

> The sliced experiment is still meaningful: it verifies that naive slicing may introduce duplicate detections and context loss. The next version should use adaptive candidate-region selection and confidence-aware fusion instead of dense sliding windows.

## 7. Files to Report

```text
datasets/pcb-defect-yolo-sliced-960/slicing_summary.json
runs/sliced/yolo11n_slice960_e100/results.csv
runs/sliced/yolo11n_slice960_e100/results.png
runs/mode_eval/yolo11n_full960_val/metrics.md
runs/mode_eval/yolo11n_slice960_val/metrics.md
runs/mode_eval/yolo11n_slice960_val/visualizations/
```
