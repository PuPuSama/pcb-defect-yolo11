from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "pcb-defect-yolo"


def run(args: list[str]) -> None:
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    run(["scripts/write_dataset_yaml.py", "--dataset-root", str(DATASET_ROOT)])
    run(["scripts/check_env.py"])
    run(
        [
            "scripts/train_yolo_baseline.py",
            "--epochs",
            "100",
            "--imgsz",
            "640",
            "--batch",
            "8",
            "--workers",
            "4",
            "--name",
            "aimax_yolo11n_full_640",
            "--exist-ok",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

