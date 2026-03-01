import time
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
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

# ------- YOLO Helpers --------
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

    # Extract pretrained state dict
    pretrained_dict = ckpt["model"].state_dict()

    # Remove Detect head keys (last layer) from checkpoint
    filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("model.24")}

    # Load pretrained weights except Detect head
    model.load_state_dict(filtered_dict, strict=False)

    # Freeze backbone
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


# ---------------- Evaluation ----------------
def evaluate(model, loader):
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval").to(DEVICE)

    start = time.time()

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            imgs_tensor = torch.stack(imgs)

            outputs = model(imgs_tensor)

            if args.model == "Yolo":
                outputs = non_max_suppression(outputs)

            preds = []
            gts = []

            for pred, target in zip(outputs, targets):
                if args.model == "yolo":
                    if pred is None or pred.numel() == 0:
                        preds.append({
                            "boxes": torch.zeros((0,4)).to(DEVICE),
                            "scores": torch.zeros(0).to(DEVICE),
                            "labels": torch.zeros(0, dtype=torch.int64).to(DEVICE)
                        })
                    else:
                        preds.append({
                            "boxes": pred[:, :4],
                            "scores": pred[:, 4],
                            "labels": pred[:, 5].long()
                        })

                    gts.append({
                        "boxes": target["boxes"].to(DEVICE),
                        "labels": target["labels"].to(DEVICE)
                    })

                elif args.model == "rcnn":
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


# -------- Actual Running ----------
def train(args):
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

    if args.model == "rcnn":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    elif args.model == "yolo":
        num_classes = len(train_dataset.class_to_label) \
            if hasattr(train_dataset, "class_to_label") \
            else len(train_dataset.breed_to_label)

        model = load_yolov5(num_classes)
        freeze_backbone(model)

        compute_loss = ComputeLoss(model)

    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr)

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

            if args.model.lower() == "yolo":
                # --- YOLO specific preprocessing ---
                # Convert 1-based labels to 0-based
                if args.dataset in ["pet", "some_other_1_based_dataset"]:
                    for t in targets:
                        t["labels"] = t["labels"] - 1

                imgs_tensor = torch.stack(imgs)
                yolo_targets = convert_targets_to_yolo(targets, imgs)
                preds = model(imgs_tensor)
                loss, loss_items = compute_loss(preds, yolo_targets)

            elif args.model.lower() == "rcnn":
                # --- R-CNN uses targets as-is ---
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

        # TODO: Validation
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
        checkpoint_dir = Path(args.checkpoints).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_dir / f"epoch_{epoch+1}.pth"
        )

        print()

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
    parser.add_argument("--model", type= str, default="rcnn", help="Model to train, rcnn or yolo")
    args = parser.parse_args()

    train(args)
