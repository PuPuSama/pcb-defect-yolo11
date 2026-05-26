# Move This Project to a School Server

This project can be moved to a Linux GPU server. The important rule is:

**Upload the project code and converted YOLO dataset, then rewrite `pcb-defect.yaml` on the server with the server's absolute path.**

## 1. What to Upload

Upload this folder:

```text
E:\Research\pcb-defect-yolo11
```

But you can skip heavy or unnecessary outputs:

```text
runs/
data/pcb-defect-raw/
```

Required for training:

```text
scripts/
datasets/pcb-defect-yolo/images/
datasets/pcb-defect-yolo/labels/
environment-server.yml
README.md
notes/
```

The converted YOLO dataset is already enough for training. The raw COCO data is useful for traceability, but not required for baseline training.

## 2. Recommended Upload Methods

### Option A: Use `scp`

From Windows PowerShell:

```powershell
scp -r E:\Research\pcb-defect-yolo11 username@server_ip:/home/username/projects/
```

### Option B: Use FileZilla / Xftp / MobaXterm

Upload the whole folder to something like:

```text
/home/username/projects/pcb-defect-yolo11
```

## 3. Create the Server Environment

SSH into the server:

```bash
ssh username@server_ip
cd /home/username/projects/pcb-defect-yolo11
```

Create a conda environment:

```bash
conda create -n pcb-yolo python=3.10 -y
conda activate pcb-yolo
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics
```

If the server's NVIDIA driver is older, use the PyTorch install command from:

```text
https://pytorch.org/get-started/locally/
```

Then check CUDA:

```bash
nvidia-smi
python scripts/check_env.py
```

Expected:

```text
CUDA available: True
```

## 4. Rewrite the Dataset YAML

The Windows dataset YAML contains a Windows path. On the server, rewrite it:

```bash
python scripts/write_dataset_yaml.py \
  --dataset-root /home/username/projects/pcb-defect-yolo11/datasets/pcb-defect-yolo
```

Check:

```bash
cat datasets/pcb-defect-yolo/pcb-defect.yaml
```

It should look like:

```yaml
path: /home/username/projects/pcb-defect-yolo11/datasets/pcb-defect-yolo
train: images/train
val: images/val
test: images/test
names: ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
```

## 5. Run the Baseline on the Server

Short test:

```bash
python scripts/train_yolo_baseline.py \
  --epochs 1 \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --name server_smoke_yolo11n_640 \
  --exist-ok
```

Formal baseline:

```bash
python scripts/train_yolo_baseline.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --name yolo11n_full_640_server \
  --exist-ok
```

If GPU memory is insufficient, reduce:

```bash
--batch 4
```

If the server GPU has plenty of memory, you can try later:

```bash
--imgsz 960 --batch 4
```

## 6. Keep Training Running After Logout

Use `tmux`:

```bash
tmux new -s pcb
conda activate pcb-yolo
cd /home/username/projects/pcb-defect-yolo11
python scripts/train_yolo_baseline.py --epochs 100 --imgsz 640 --batch 8 --workers 4 --name yolo11n_full_640_server --exist-ok
```

Detach:

```text
Ctrl+b, then d
```

Reattach:

```bash
tmux attach -t pcb
```

## 7. Copy Results Back

After training:

```powershell
scp -r username@server_ip:/home/username/projects/pcb-defect-yolo11/runs/baselines/yolo11n_full_640_server E:\Research\pcb-defect-yolo11\runs\baselines\
```

Then compare:

```text
results.csv
results.png
confusion_matrix.png
val_batch*_pred.jpg
```
