# Git Sync Workflow

Use Git to sync code between the local machine and the GPU server.

Do not put datasets, training runs, or model weights into Git.

## What Git Tracks

Track:

```text
scripts/
notes/
README.md
environment-server.yml
requirements.txt
```

Ignore:

```text
data/
datasets/
runs/
*.pt
*.pth
*.zip
```

The dataset should stay on the server under:

```text
~/pcb-defect-yolo11/datasets/pcb-defect-yolo
```

If the project is freshly cloned, upload or copy the dataset separately, then run:

```bash
python scripts/write_dataset_yaml.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
```

## First Setup on Local Windows

In PowerShell:

```powershell
cd /d E:\Research\pcb-defect-yolo11
git init
git add .gitignore .gitattributes README.md requirements.txt environment-server.yml scripts notes
git commit -m "Initialize PCB defect YOLO11 project"
```

Create a private empty repository on GitHub or Gitee, then connect it:

```powershell
git remote add origin https://github.com/YOUR_NAME/pcb-defect-yolo11.git
git branch -M main
git push -u origin main
```

## First Setup on Server

If the server already has the uploaded folder, connect it to the same remote:

```bash
cd ~/pcb-defect-yolo11
git init
git remote add origin https://github.com/YOUR_NAME/pcb-defect-yolo11.git
git fetch origin
git checkout -B main origin/main
```

Then restore or keep the dataset folder:

```bash
python scripts/write_dataset_yaml.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
python scripts/inspect_yolo_dataset.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
```

If this command removes local files or complains about conflicts, stop and ask before forcing anything.

## Normal Daily Workflow

Local side:

```powershell
cd /d E:\Research\pcb-defect-yolo11
git status
git add scripts notes README.md requirements.txt environment-server.yml
git commit -m "Describe the change"
git push
```

Server side:

```bash
cd ~/pcb-defect-yolo11
git pull
python scripts/write_dataset_yaml.py --dataset-root $(pwd)/datasets/pcb-defect-yolo
```

Run an experiment:

```bash
python scripts/train_yolo_baseline.py --epochs 100 --imgsz 960 --batch 8 --workers 4 --name yolo11n_full_960_server --exist-ok
```

Summarize the result:

```bash
python scripts/summarize_yolo_run.py runs/baselines/yolo11n_full_960_server
```

Send the summary output back for analysis.

## If GitHub Is Slow

Use Gitee instead. The workflow is the same:

```powershell
git remote add origin https://gitee.com/YOUR_NAME/pcb-defect-yolo11.git
git push -u origin main
```

