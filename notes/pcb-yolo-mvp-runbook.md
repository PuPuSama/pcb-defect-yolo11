# PCB-YOLO MVP Runbook

## MVP定位

今天先做一个能运行、能对比、能讲清楚论文映射关系的 MVP。它不是完整复现论文的最终版，而是把论文的核心思路落到当前 YOLO11 项目上：

| 论文模块 | 当前 MVP 对应实现 | 状态 |
| --- | --- | --- |
| 灰度化 + 切片预处理 | `make_sliced_yolo_dataset.py --grayscale` | 已实现 |
| Conv_SWS | 在 stride=2 的 Ultralytics `Conv` 前加入 SimAM with slicing | 已实现 |
| FBC2f | 将 `C2f/C3k2` 类 CSP 模块替换为 PConv + FasterNet block 的 `FBC2f` | 已实现 |
| NWD-IoU | 论文公式和 alpha=0.5 已记录，训练损失接入放下一版 | 下一步 |

## 1. 构建灰度切片数据集

```bash
python scripts/make_sliced_yolo_dataset.py \
  --dataset-root datasets/pcb-defect-yolo \
  --output-root datasets/pcb-defect-yolo-sliced-960-gray \
  --slice-size 960 \
  --overlap 0.25 \
  --min-visibility 0.35 \
  --negative-ratio 0.20 \
  --grayscale \
  --force
```

汇报说法：

> 论文指出 PCB 底纹颜色差异会干扰缺陷识别，所以我先在数据层加入灰度化；同时采用重叠切片，避免高分辨率 PCB 直接缩放到固定输入尺寸时，小缺陷被压缩丢失。

## 2. 检查模型注入是否成功

```bash
python scripts/train_pcb_yolo_mvp.py \
  --data datasets/pcb-defect-yolo-sliced-960-gray/pcb-defect-sliced.yaml \
  --dry-run \
  --name pcb_yolo_mvp_dry_run \
  --exist-ok
```

检查输出里的两个字段：

```text
conv_sws: 被替换的 stride=2 下采样卷积
fbc2f: 被替换的 C2f/C3k2 类模块
```

同时会生成：

```text
runs/pcb_yolo_mvp/pcb_yolo_mvp_dry_run/pcb_yolo_mvp_manifest.json
```

## 3. 训练 PCB-YOLO-MVP

服务器或有 CUDA 的环境建议：

```bash
python scripts/train_pcb_yolo_mvp.py \
  --model yolo11n.pt \
  --data datasets/pcb-defect-yolo-sliced-960-gray/pcb-defect-sliced.yaml \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --name pcb_yolo_mvp_sws_fbc2f_slice960_gray \
  --project runs/pcb_yolo_mvp \
  --exist-ok
```

显存不够时先降到：

```bash
--batch 8
```

## 4. 与 baseline 做同口径评估

先跑 full-image baseline：

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

再跑 MVP 切片推理：

```bash
python scripts/evaluate_yolo_modes.py \
  --weights runs/pcb_yolo_mvp/pcb_yolo_mvp_sws_fbc2f_slice960_gray/weights/best.pt \
  --dataset-root datasets/pcb-defect-yolo \
  --split val \
  --mode sliced \
  --imgsz 960 \
  --slice-size 960 \
  --overlap 0.25 \
  --conf 0.001 \
  --device 0 \
  --batch 16 \
  --output runs/mode_eval/pcb_yolo_mvp_slice960_gray_val
```

## 5. 汇报时的三句话

1. Baseline 的问题不是单纯模型不够大，而是 PCB 原图分辨率高、小缺陷面积占比低，整图缩放会损失目标细节。
2. MVP 对应论文做了两层改进：数据层用灰度化和重叠切片减少底纹干扰、提高小目标相对尺度；模型层用 SWS 注意力增强下采样前的小目标响应，用 FBC2f 降低 CSP 模块冗余计算。
3. 当前版本已完成 M1/M2 的可运行实现，下一步把 NWD-IoU 接入 Ultralytics loss，做完整消融：baseline、+FBC2f、+SWS、+FBC2f+SWS、+NWD-IoU。

## 6. 今天优先展示的文件

```text
scripts/make_sliced_yolo_dataset.py
scripts/pcb_yolo_mvp_modules.py
scripts/train_pcb_yolo_mvp.py
runs/pcb_yolo_mvp/*/pcb_yolo_mvp_manifest.json
runs/mode_eval/*/metrics.md
runs/mode_eval/*/visualizations/
```
