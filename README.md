# Object Detection

## Setup

Install torch and torch vision with cuda, then

```bash
pip install -r requirements.txt
```

## Run

Fast R-CNN

``` bash
python train_fasterrcnn.py --dataset penn --splits pennfudan_splits.json --checkpoints checkpoints/r-cnn/PennFudanPed

python train_fasterrcnn.py --dataset pet --splits pet_splits.json --checkpoints checkpoints/r-cnn/oxford-iiit-pet
```

YOLO

``` bash
python train_yolo_nano.py --dataset penn --splits pennfudan_splits.json --checkpoints checkpoints/yolo/PennFudanPed

python train_yolo_nano.py --dataset pet --splits pet_splits.json --checkpoints checkpoints/yolo/oxford-iiit-pet
```
