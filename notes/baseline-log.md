# Baseline Log

## Environment

- Conda env: `pcb-yolo`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB
- Target model: YOLO11n
- PyTorch: `2.11.0+cu128`
- Ultralytics: `8.4.54`

## Baseline 0: Environment Check

Command:

```powershell
cd /d E:\Research\pcb-defect-yolo11
python scripts\check_env.py
```

Expected:

```text
CUDA available: True
GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU
```

Observed:

```text
CUDA available: True
GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU
```

## Dataset Conversion

- Raw dataset: `E:\Research\pcb-defect-yolo11\data\pcb-defect-raw\PCB_Defect`
- Converted dataset: `E:\Research\pcb-defect-yolo11\datasets\pcb-defect-yolo`
- Dataset YAML: `E:\Research\pcb-defect-yolo11\datasets\pcb-defect-yolo\pcb-defect.yaml`
- Split seed: `42`
- Split ratio: `70/20/10`
- Classes: `missing_pad`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`

Converted split:

| Split | Images | Boxes |
| --- | ---: | ---: |
| train | 161 | 1194 |
| val | 46 | 354 |
| test | 23 | 156 |

Label visualization samples:

```text
E:\Research\pcb-defect-yolo11\runs\label_checks
```

Visual spot check: boxes align with PCB defect regions.

## Smoke Test

Command:

```powershell
& "D:\Programs\miniconda3\envs\pcb-yolo\python.exe" "E:\Research\pcb-defect-yolo11\scripts\train_yolo_baseline.py" --epochs 1 --imgsz 640 --batch 2 --workers 0 --name smoke_yolo11n_640 --exist-ok
```

Result:

- Training completed.
- CUDA was used.
- Dataset YAML loaded correctly.
- Validation completed.
- Output: `E:\Research\pcb-defect-yolo11\runs\baselines\smoke_yolo11n_640`

## Baseline 1: Full-Image YOLO11n

### 20-epoch trial

Command:

```powershell
& "D:\Programs\miniconda3\envs\pcb-yolo\python.exe" scripts\train_yolo_baseline.py --epochs 20 --imgsz 640 --batch 4 --workers 0 --name yolo11n_full_640_e20 --exist-ok
```

Output:

```text
E:\Research\pcb-defect-yolo11\runs\baselines\yolo11n_full_640_e20
```

Final epoch metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.54998 |
| Recall | 0.44554 |
| mAP@50 | 0.46298 |
| mAP@50:95 | 0.22380 |
| train box loss | 1.78107 |
| val box loss | 1.82119 |

Interpretation:

- The model is learning: train/validation losses are decreasing.
- mAP and Recall are still rising at epoch 20.
- This is not converged enough for a paper baseline.
- Continue to the 100-epoch baseline before judging the dataset or method.

### 100-epoch formal baseline

Command:

```powershell
& "D:\Programs\miniconda3\envs\pcb-yolo\python.exe" scripts\train_yolo_baseline.py --epochs 100 --imgsz 640 --batch 4 --workers 0 --name yolo11n_full_640 --exist-ok
```

Output:

```text
E:\Research\pcb-defect-yolo11\runs\baselines\yolo11n_full_640
```

Final epoch metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.79568 |
| Recall | 0.74765 |
| mAP@50 | 0.79002 |
| mAP@50:95 | 0.41293 |
| train box loss | 1.31461 |
| val box loss | 1.59897 |
| train cls loss | 0.93021 |
| val cls loss | 0.97896 |

Interpretation:

- The 100-epoch baseline is usable as the first formal comparison point.
- The dataset is not saturated by YOLO11n at `imgsz=640`; mAP@50 is about 0.79 and mAP@50:95 is about 0.41.
- The normalized confusion matrix suggests weaker recall for `missing_pad` and `mouse_bite`, with many samples falling into background.
- This supports the proposed research angle: high-resolution PCB defects lose detail under full-image resize, and small/local defects need a local perception strategy.

Next experiment:

- Run a larger-input full-image baseline (`imgsz=960`) before implementing slicing.
- This separates "higher resolution alone" from the later "fixed/adaptive slicing" contribution.
