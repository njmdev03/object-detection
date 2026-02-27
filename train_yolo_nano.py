import torch
import time
import argparse
import json
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(args):
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5n",
        pretrained=True
    )

    model.to(DEVICE)

    start = time.time()

    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=512
    )

    train_time = time.time() - start

    metrics = model.val()

    print("\n--- Evaluation ---")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"Training Time: {train_time:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    train(args)