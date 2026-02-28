from pathlib import Path
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json


def masks_to_boxes(mask: Image.Image) -> torch.Tensor:
    """
    Converts a PIL mask to bounding boxes.
    Assumes each object is represented by a separate value in the mask.
    Returns tensor of shape [num_boxes, 4] in (x_min, y_min, x_max, y_max)
    """
    mask = torch.tensor(np.array(mask), dtype=torch.uint8)  # H x W

    # In Penn-Fudan, each mask is a binary mask with foreground=255, background=0
    mask = mask > 0  # convert to boolean

    if mask.sum() == 0:
        # No object found; create full image box as fallback
        h, w = mask.shape
        return torch.tensor([[0, 0, w, h]], dtype=torch.float32)

    # Add batch dimension to match masks_to_boxes input
    mask = mask.unsqueeze(0)  # shape [1,H,W]

    boxes = torchvision.ops.masks_to_boxes(mask)  # shape [num_boxes, 4]

    return boxes


class PennFudanDataset(Dataset):
    def __init__(self, root, split_json, transform=None):
        self.root = Path(root) / "PennFudanPed"
        self.transform = transform

        with open(split_json) as f:
            splits = json.load(f)

        self.images = splits["train"]  # or val/test depending on loader
        self.class_to_label = splits["class_to_label"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.root / "PNGImages" / img_name
        img = Image.open(img_path).convert("RGB")

        # Convert image filename to mask filename
        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = self.root / "PedMasks" / mask_name
        mask = Image.open(mask_path)

        boxes = masks_to_boxes(mask)
        labels = torch.ones(len(boxes), dtype=torch.int64) * self.class_to_label["person"]

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target