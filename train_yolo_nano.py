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
import yaml

import sys
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parent / "external" / "yolov5"

sys.path.insert(0, str(YOLO_ROOT))

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression, check_file

from datasets import get_dataset
from transforms import get_detection_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return tuple(zip(*batch))

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # safest: only save weights
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Save state dict only
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def convert_targets_to_yolo(targets, imgs):
    """
    Convert torchvision-style targets to YOLO format.
    """
    yolo_targets = []

    for batch_idx, (target, img) in enumerate(zip(targets, imgs)):
        boxes = target["boxes"]
        labels = target["labels"]

        h, w = img.shape[1], img.shape[2]

        if boxes.numel() == 0:
            continue

        # Convert xyxy → xywh
        x1, y1, x2, y2 = boxes.T
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        batch_tensor = torch.full_like(x_center, batch_idx)

        yolo_box = torch.stack([
            batch_tensor,
            labels.float(),
            x_center,
            y_center,
            width,
            height
        ], dim=1)

        yolo_targets.append(yolo_box)

    if len(yolo_targets):
        return torch.cat(yolo_targets, dim=0)
    else:
        return torch.zeros((0, 6), device=imgs[0].device)

def load_yolov5(num_classes):
    # import torch
    # from models.yolo import Model
    # from utils.general import check_file

    device = DEVICE
    cfg = YOLO_ROOT / "models" / "yolov5n.yaml"

    # Download official YOLOv5n checkpoint
    weights = check_file("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")

    # Build model with your number of classes
    model = Model(str(cfg), ch=3, nc=num_classes).to(device)

    # Load checkpoint safely
    with torch.serialization.safe_globals([Model]):
        ckpt = torch.load(weights, map_location=device, weights_only=False)

    # 1️⃣ Extract pretrained state dict
    pretrained_dict = ckpt["model"].state_dict()

    # 2️⃣ Remove Detect head keys (last layer) from checkpoint
    filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("model.24")}

    # 3️⃣ Load pretrained weights except Detect head
    model.load_state_dict(filtered_dict, strict=False)

    # 4️⃣ Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model[-1].parameters():  # Detect head
        param.requires_grad = True

    # Add default hyperparameters (needed for ComputeLoss)
    hyp_file = YOLO_ROOT / "data/hyps/hyp.scratch-low.yaml"
    with open(hyp_file) as f:
        model.hyp = yaml.safe_load(f)

    return model

def freeze_backbone(model):
    print("Model Params")
    for name, _ in model.named_parameters():
        print(name)
    print()

    for name, param in model.named_parameters():
        if "model.24" not in name:  # last layer in yolov5n
            param.requires_grad = False

def train_yolo(args):
    DatasetClass = get_dataset(args.dataset)

    train_dataset = DatasetClass(
        root=args.root,
        split_json=args.splits,
        transform=get_detection_transforms(train=True)
    )

    val_dataset = DatasetClass(
        root=args.root,
        split_json=args.splits,
        transform=get_detection_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=collate_fn
    )

    num_classes = len(train_dataset.class_to_label) \
        if hasattr(train_dataset, "class_to_label") \
        else len(train_dataset.breed_to_label)

    model = load_yolov5(num_classes)
    freeze_backbone(model)

    compute_loss = ComputeLoss(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr)

    model.to(DEVICE)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for imgs, targets in tqdm(train_loader):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            imgs_tensor = torch.stack(imgs)

            yolo_targets = convert_targets_to_yolo(targets, imgs)

            preds = model(imgs_tensor)

            loss, loss_items = compute_loss(preds, yolo_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"\nEpoch {epoch+1}")
        print(f"Training Loss: {epoch_loss:.4f}")

        checkpoint_dir = Path(args.checkpoints).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_dir / f"yolo_epoch_{epoch+1}.pth"
        )

def evaluate_yolo(model, loader):
    model.eval()

    metric = MeanAveragePrecision().to(DEVICE)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            imgs_tensor = torch.stack(imgs)

            preds = model(imgs_tensor)
            preds = non_max_suppression(preds)

            # Convert to TorchMetrics format
            formatted_preds = []
            formatted_gts = []

            for pred, target in zip(preds, targets):
                if pred is None:
                    formatted_preds.append({
                        "boxes": torch.zeros((0,4)).to(DEVICE),
                        "scores": torch.zeros(0).to(DEVICE),
                        "labels": torch.zeros(0, dtype=torch.int64).to(DEVICE)
                    })
                else:
                    formatted_preds.append({
                        "boxes": pred[:, :4],
                        "scores": pred[:, 4],
                        "labels": pred[:, 5].long()
                    })

                formatted_gts.append({
                    "boxes": target["boxes"].to(DEVICE),
                    "labels": target["labels"].to(DEVICE)
                })

            metric.update(formatted_preds, formatted_gts)

    return metric.compute()

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

    train_yolo(args)