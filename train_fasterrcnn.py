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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Dataset ----------------
class PennDataset(Dataset):
    def __init__(self, root, split_json, split):
        self.root = Path(root) / "PennFudanPed"
        with open(split_json) as f:
            splits = json.load(f)
        self.images = splits[split]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.root / "PNGImages" / img_name
        mask_path = self.root / "PedMasks" / img_name.replace(".png", "_mask.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)[1:]

        boxes = []
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return F.to_tensor(img), target


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
    train_ds = PennDataset(args.root, args.splits, "train")
    val_ds = PennDataset(args.root, args.splits, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

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

        map50 = results["map_50"].item()
        map5095 = results["map"].item()
        recall = results["mar_100"].item()

        print(f"mAP@0.5: {map50:.4f}")
        print(f"mAP@0.5:0.95: {map5095:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Inference Speed: {fps:.2f} img/sec")

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
    parser.add_argument("--splits", default="pennfudan_splits.json")
    parser.add_argument("--checkpoints", default="checkpoints/PennFudanPed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(args)
