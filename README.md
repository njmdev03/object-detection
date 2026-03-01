# Object Detection

## Setup

Install torch and torch vision with cuda, then

```bash
pip install -r requirements.txt
```

Download the datasets

```bash
python generate_splits.py --dl-only
```

Or overwrite the included splits

```bash
python generate_splits.py --pet_breeds 10
```

then see **Run**

## Run

Fast R-CNN

``` bash
python train.py --model rcnn --dataset penn --splits pennfudan_splits.json --checkpoints checkpoints/r-cnn/PennFudanPed

python train.py --model rcnn --dataset pet --splits pet_splits.json --checkpoints checkpoints/r-cnn/oxford-iiit-pet
```

YOLO

``` bash
python train.py --model yolo --dataset penn --splits pennfudan_splits.json --checkpoints checkpoints/yolo/PennFudanPed --epochs 300

python train.py --model yolo --dataset pet --splits pet_splits.json --checkpoints checkpoints/yolo/oxford-iiit-pet --epochs 300 --lr 0.0005 --batch-size 8
```

## Vis

```bash
python plot_metrics.py --dataset penn --splits pennfudan_splits.json --model rcnn --checkpoint ./checkpoints/r-cnn/PennFudanPed/epoch_50.pth --output ./vis/r-cnn/PennFudanPed/

python plot_metrics.py --dataset pet --splits pet_splits.json --model rcnn --checkpoint ./checkpoints/r-cnn/oxford-iiit-pet/epoch_50.pth --output ./vis/r-cnn/oxford-iiit-pet/



python plot_metrics.py --dataset penn --splits pennfudan_splits.json --model yolo --checkpoint ./checkpoints/yolo/PennFudanPed/epoch_300.pth --output ./vis/yolo/PennFudanPed/

python plot_metrics.py --dataset pet --splits pet_splits.json --model yolo --checkpoint ./checkpoints/yolo/oxford-iiit-pet/epoch_300.pth --output ./vis/yolo/oxford-iiit-pet/
```
