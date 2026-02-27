# Object Detection

## Run

Fast R-CNN

``` bash
python train_fasterrcnn.py --splits pennfudan_splits.json --checkpoints checkpoints/r-cnn/PennFudanPed

python train_fasterrcnn.py --splits pet_splits.json --checkpoints checkpoints/r-cnn/oxford-iiit-pet
```

YOLO

``` bash
python train_yolo_nano.py --splits pennfudan_splits.json --checkpoints checkpoints/yolo/PennFudanPed

python train_yolo_nano.py --splits pet_splits.json --checkpoints checkpoints/yolo/oxford-iiit-pet
```
