import torch
import time
import argparse
import json
from pathlib import Path
import tempfile
from shutil import rmtree
from PIL import Image
import xml.etree.ElementTree as ET
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_breed_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def parse_pet_boxes_from_xml(xml_path: Path):
    """Return list of [xmin, ymin, xmax, ymax] for Pet dataset"""
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
    except Exception:
        # fallback to full image box if XML missing or empty
        img_path = xml_path.with_suffix(".jpg")
        with Image.open(img_path) as img:
            w, h = img.size
            boxes.append([0, 0, w, h])
    if not boxes:
        with Image.open(xml_path.with_suffix(".jpg")) as img:
            w, h = img.size
            boxes.append([0, 0, w, h])
    return boxes


def create_yolo_labels(root: Path, splits: dict, labels_dict: dict, dataset_type: str):
    """
    Converts JSON splits + annotations into YOLO TXT format.
    Returns: tmp_dir containing:
        images/train, images/val, labels/train, labels/val
    """
    import tempfile
    from pathlib import Path
    from PIL import Image

    tmp_dir = Path(tempfile.mkdtemp())
    images_dir = tmp_dir / "images"
    labels_dir = tmp_dir / "labels"

    for split in ["train", "val"]:
        img_list = splits[split] if isinstance(splits, dict) else splits  # handle list vs dict
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

        for img_name in img_list:
            # Determine image and annotation paths
            if dataset_type == "pets":
                img_path = root / "images" / img_name
                xml_path = root / "annotations" / img_name.replace(".jpg", ".xml")
            else:
                img_path = root / "PNGImages" / img_name
                xml_path = None

            # Open and save image to tmp folder
            with Image.open(img_path) as img:
                w, h = img.size
                dst_img_path = images_dir / split / img_name
                img.save(dst_img_path)

            # Get bounding boxes
            if dataset_type == "pets":
                boxes = parse_pet_boxes_from_xml(xml_path)
                breed_name = "_".join(img_name.split(".")[0].split("_")[:-1]).lower()
                label = labels_dict.get(breed_name, 0)  # fallback to 0 if breed missing
            else:
                boxes = [[0, 0, w, h]]
                label = 0

            # Write YOLO TXT label file
            ext = ".txt"
            txt_name = img_name.rsplit(".", 1)[0] + ext
            txt_path = labels_dir / split / txt_name

            # Inside the loop where you write TXT
            with open(txt_path, "w") as f:
                for box in boxes:
                    x_center = ((box[0] + box[2]) / 2) / w
                    y_center = ((box[1] + box[3]) / 2) / h
                    bw = (box[2] - box[0]) / w
                    bh = (box[3] - box[1]) / h
                    # Use label directly, do NOT subtract 1
                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    return tmp_dir


def generate_data_yaml(tmp_dir: Path, class_names: list):
    """
    Generates YOLOv5/YOLOv8 data.yaml with separate train/val paths
    """
    yaml_path = tmp_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {tmp_dir}/images/train\n")
        f.write(f"val: {tmp_dir}/images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    return yaml_path


def train(args):
    with open(args.splits) as f:
        splits = json.load(f)

    if args.dataset.lower() == "penn":
        dataset_root = Path(args.data_root) / "PennFudanPed"
        class_names = ["person"]
        labels_dict = {"person": 1}
        dataset_type = "penn"
        images = splits["train"]
    else:
        dataset_root = Path(args.data_root) / "oxford-iiit-pet"
        class_names = [normalize_breed_name(b) for b in splits["breeds"]]
        labels_dict = {normalize_breed_name(k): v for k, v in splits["breed_to_label"].items()}
        dataset_type = "pets"
        images = splits["train"]

    # Generate temporary YOLO dataset
    tmp_dir = create_yolo_labels(dataset_root, images, labels_dict, dataset_type)
    data_yaml = generate_data_yaml(tmp_dir, class_names)

    model = YOLO("yolov5n.pt")
    model.to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        start = time.time()
        model.train(
            data=str(data_yaml),
            epochs=1,
            batch=args.batch,
            imgsz=512,
            project=args.checkpoints,
            name=args.dataset + "_yolov5n",
            exist_ok=True
        )
        epoch_time = time.time() - start

        # Save checkpoint after each epoch
        model_path = Path(args.checkpoints) / (args.dataset + f"_epoch{epoch}.pt")
        model.save(model_path)

        # Evaluate
        metrics = model.val()
        print("\n--- Epoch Evaluation ---")
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} sec")

    # Cleanup
    rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["penn", "pets"], required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--checkpoints", type=str, required=True)
    args = parser.parse_args()

    train(args)