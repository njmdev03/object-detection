import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
from tqdm import tqdm

from datasets import get_dataset
from transforms import get_detection_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


# ----------------- Helpers ------------------
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


# ---------------- Evaluation ----------------
def evaluate(model, loader):
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval").to(DEVICE)

    start = time.time()

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            outputs = model(imgs)

            # TorchMetrics expects:
            # preds: dict(boxes, scores, labels)
            # targets: dict(boxes, labels)

            preds = []
            gts = []

            for pred, target in zip(outputs, targets):
                preds.append({
                    "boxes": pred["boxes"],
                    "scores": pred["scores"],
                    "labels": pred["labels"]
                })

                gts.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"]
                })

            metric.update(preds, gts)

    results = metric.compute()

    elapsed = time.time() - start
    fps = len(loader.dataset) / elapsed

    return results, fps


# ---------------- Training ----------------
def train(args):
    DatasetClass = get_dataset(args.dataset)
    train_dataset = DatasetClass(root="data", split_json=args.splits, transform=get_detection_transforms(train=True))
    val_dataset   = DatasetClass(root="data", split_json=args.splits, transform=get_detection_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr)

    checkpoint_dir = Path(args.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from epoch {checkpoint['epoch']}")

    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0

        for imgs, targets in tqdm(train_loader):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"Epoch Training Time: {epoch_time:.2f} sec")

        # Validation
        results, fps = evaluate(model, val_loader)

        epoch_time = time.time() - epoch_start_time

        map50 = results["map_50"].item()
        map5095 = results["map"].item()
        recall = results["mar_100"].item()

        print(f"mAP@0.5: {map50:.4f}")
        print(f"mAP@0.5:0.95: {map5095:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Inference Speed: {fps:.2f} img/sec")
        print(f"Epoch Validation Time: {epoch_time:.2f} sec")

        # Save epoch checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_dir / f"epoch_{epoch+1}.pth"
        )

    total_training_time = time.time() - total_start_time
    print("\n===== Training Complete =====")
    print(f"Total Training Time: {total_training_time:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", type=str, default="penn", help="Dataset to train on: penn or pet")
    parser.add_argument("--splits", type=str, default="pennfudan_splits.json")
    parser.add_argument("--checkpoints", default="checkpoints/PennFudanPed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(args)
