# Fruit Counting / Produce Detection

A YOLOv8n object-detection project fine-tuned to detect and count:
- Apple
- Carrot
- Orange

## Project contents
- `models/best.pt`: trained custom YOLOv8n weights
- `src/predict.py`: reusable inference and counting script
- `fruit_dataset.yaml`: dataset configuration
- `results/`: evaluation metrics, counting report, plots and sample predictions

## Requirements
Python 3.10+ recommended
Ultralytics YOLO

Install:
```bash
pip install ultralytics
```

## Run predictions
From the project root:
```bash
python src/predict.py --source path/to/image.jpg
```
For a folder of images:
```bash
python src/predict.py --source path/to/images --conf 0.25
```
The script prints detected counts by class and saves annotated images.

## Dataset and training
The project used the Hugging Face dataset `johnatanvq/fruits-dataset`.
The original dataset and split image files are not included in this ZIP.
The model was fine-tuned for 5 epochs at image size 416 using a CPU runtime.
Classes: apple, carrot, orange.

## Test evaluation
- Test images: 24
- Test instances: 76
- Precision: 93.62%
- Recall: 93.79%
- mAP@0.50: 92.97%
- mAP@0.50:0.95: 86.77%

## Counting evaluation
On 24 test images at confidence threshold 0.25:
- Exact counts for all three classes: 79.17% of images (19/24)
- Mean per-image, per-class absolute count error: 0.0972

These are preliminary results from a small test set and may not represent performance on all real-world images.

## Notes
This is an object-detection and counting model, not a general fruit-quality or ripeness classifier.
