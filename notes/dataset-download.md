# Dataset Download Notes

## Primary Dataset

PCB-Defect: An Annotated Dataset for Surface Defect Detection in Printed Circuit Boards

- Data page: https://data.mendeley.com/datasets/vdj74sngvn/1
- Paper page: https://www.sciencedirect.com/science/article/pii/S2352340925010133
- DOI: https://doi.org/10.17632/vdj74sngvn.1
- License: CC BY 4.0
- Images: 230 high-resolution PCB images
- Annotation count: 1704 bounding boxes
- Classes: missing pad, mouse bite, open circuit, short circuit, spur, spurious copper
- Annotation format: COCO JSON
- Image resolution range: 800 x 600 to 6000 x 4000

## Download Placement

Put the downloaded archive or extracted raw files under:

```text
E:\Research\pcb-defect-yolo11\data\pcb-defect-raw
```

After download, keep raw data unchanged. Converted YOLO-format data should go under:

```text
E:\Research\pcb-defect-yolo11\datasets\pcb-defect-yolo
```

## First Check After Download

Confirm these exist:

- image files, such as `.jpg`, `.jpeg`, or `.png`
- one COCO annotation JSON file, usually containing `images`, `annotations`, and `categories`

Do not train before checking whether bounding boxes align with the images.

