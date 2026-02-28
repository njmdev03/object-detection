import os
import json
import random
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.datasets import OxfordIIITPet
import urllib.request
import zipfile

SEED = 42
random.seed(SEED)


# ---------------- Penn-Fudan Download ----------------
def download_penn_fudan(root):
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "PennFudanPed.zip"

    if not (root / "PennFudanPed").exists():
        print("Downloading Penn-Fudan...")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)

        zip_path.unlink()
        print("Penn-Fudan ready.")
    else:
        print("Penn-Fudan already exists.")


# ---------------- Generic Split ----------------
def create_split_dict(file_list):
    train, temp = train_test_split(file_list, test_size=0.30, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED)

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


# ---------------- Penn-Fudan Splits ----------------
def generate_penn_splits(root):
    img_dir = Path(root) / "PennFudanPed/PNGImages"
    images = sorted([p.name for p in img_dir.glob("*.png")])

    splits = create_split_dict(images)

    # Single class dataset
    splits["classes"] = ["person"]
    splits["class_to_label"] = {"person": 1}

    with open("pennfudan_splits.json", "w") as f:
        json.dump(splits, f, indent=4)

    print("Saved pennfudan_splits.json")


# ---------------- Pet Dataset Splits ----------------
def generate_pet_splits(root, num_breeds=5):
    dataset = OxfordIIITPet(root=root, download=True)

    # Extract breed names
    breed_names = dataset.classes
    breed_names = sorted(breed_names)

    selected_breeds = sorted(random.sample(breed_names, num_breeds))

    images = []
    labels = []

    for img_path, label in zip(dataset._images, dataset._labels):
        breed = dataset.classes[label]
        if breed in selected_breeds:
            images.append(os.path.basename(img_path))
            labels.append(breed)

    splits = create_split_dict(images)

    selected_breeds = [x.lower().replace(" ", "_") for x in selected_breeds]

    # Stable label mapping (start from 1 for detection models)
    breed_to_label = {breed: i + 1 for i, breed in enumerate(selected_breeds)}

    splits["breeds"] = selected_breeds
    splits["breed_to_label"] = breed_to_label

    with open("pet_splits.json", "w") as f:
        json.dump(splits, f, indent=4)

    print("Saved pet_splits.json")


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--pet_breeds", type=int, default=5)
    parser.add_argument("--dl-only", action="store_true", default=False)
    args = parser.parse_args()

    OxfordIIITPet(root=args.root, download=True)
    download_penn_fudan(args.root)

    if not args.dl_only:
        generate_penn_splits(args.root)
        generate_pet_splits(args.root, args.pet_breeds)