from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import xml.etree.ElementTree as ET


def normalize_breed_name(name: str) -> str:
    """Lowercase and replace spaces with underscores to match filenames"""
    return name.lower().replace(" ", "_")


def parse_pet_boxes_from_xml(xml_path):
    """Return bounding boxes as a torch tensor of shape [num_boxes, 4]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])

    if len(boxes) == 0:
        # fallback: full image box
        img = Image.open(xml_path.with_suffix(".jpg"))
        w, h = img.size
        boxes.append([0, 0, w, h])

    return torch.tensor(boxes, dtype=torch.float32)


def get_breed_from_filename(img_name):
    """
    Extract breed name from filename and normalize:
    'Bombay_012.jpg' -> 'bombay'
    'Japanese_Chin_023.jpg' -> 'japanese_chin'
    """
    img_name = img_name.split(".")[0]          # remove extension
    breed_name = "_".join(img_name.split("_")[:-1])
    return breed_name.lower()                  # normalize lowercase


class PetDataset(Dataset):
    def __init__(self, root, split_json, split="train", transform=None):
        self.root = Path(root) / "oxford-iiit-pet"
        self.transform = transform

        # Load JSON split
        with open(split_json) as f:
            splits = json.load(f)

        self.images = splits[split]  # "train", "val", or "test"

        # Normalize keys to match filenames
        self.breed_to_label = {normalize_breed_name(k): v for k, v in splits["breed_to_label"].items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.root / "images" / img_name
        img = Image.open(img_path).convert("RGB")

        # Get label from filename
        breed_name = get_breed_from_filename(img_name)
        if breed_name not in self.breed_to_label:
            raise KeyError(f"Breed '{breed_name}' not in breed_to_label dictionary")
        label = self.breed_to_label[breed_name]

        # Get boxes from XML
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = self.root / "annotations" / "xmls" / xml_name
        boxes = parse_pet_boxes_from_xml(xml_path)

        # Labels tensor for each box
        labels = torch.tensor([label] * boxes.shape[0], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target