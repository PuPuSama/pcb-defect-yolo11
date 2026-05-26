# Server Quick Commands

Run these from the project root on the GPU server:

```bash
cd ~/pcb-defect-yolo11
```

## Check Environment

```bash
nvidia-smi
python scripts/check_env.py
python -c "import ultralytics; print(ultralytics.__version__)"
```

## Fix Dataset YAML After Pulling Code

```bash
python scripts/write_dataset_yaml.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
python scripts/inspect_yolo_dataset.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
```

## Smoke Test

```bash
python scripts/train_yolo_baseline.py \
  --epochs 1 \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --name server_smoke_yolo11n_640 \
  --exist-ok
```

## Formal 960 Baseline

```bash
python scripts/train_yolo_baseline.py \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --workers 4 \
  --name yolo11n_full_960_server \
  --exist-ok
```

## Result Summary

```bash
python scripts/summarize_yolo_run.py runs/baselines/yolo11n_full_960_server
```

