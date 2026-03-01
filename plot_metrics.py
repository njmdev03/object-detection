import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math
import matplotlib.patches as mpatches
import random
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.ops import box_iou
import sys

from train import load_yolov5
from datasets import get_dataset
from transforms import get_detection_transforms

YOLO_ROOT = Path(__file__).resolve().parent / "external" / "yolov5"

sys.path.insert(0, str(YOLO_ROOT))

from utils.general import non_max_suppression


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_predictions(
    model,
    dataset,
    model_type,
    out_dir=Path("vis"),
    num_samples=6,
    score_threshold=0.5,
    iou_threshold=0.5,
    class_map=None,
    title=None
):
    """
    Visualize predictions on random samples from the dataset.

    Args:
        model: PyTorch object detection model (RCNN or YOLO).
        dataset: Dataset returning (image, target) where target is dict with 'boxes' and 'labels'.
        model_type: "rcnn" or "yolo".
        out_dir: Directory to save the image.
        num_samples: Number of random samples to show.
        score_threshold: Minimum prediction score to show.
        iou_threshold: IoU threshold to match predictions to ground truth.
        class_map: Optional mapping from class index to class name.
        title: Optional figure title.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    cols = math.ceil(math.sqrt(num_samples))
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img, target = dataset[idx]
            img_device = img.to(DEVICE)

            # -------- PREDICTIONS --------
            if model_type.lower() == "rcnn":
                prediction = model([img_device])[0]
            elif model_type.lower() == "yolo":
                output = model(img_device.unsqueeze(0))
                if isinstance(output, tuple):
                    output = output[0]

                output = non_max_suppression(output)[0]
                if output is None or output.numel() == 0:
                    prediction = {"boxes": torch.zeros((0, 4)),
                                  "scores": torch.zeros((0,)),
                                  "labels": torch.zeros((0,))}
                else:
                    prediction = {"boxes": output[:, :4],
                                  "scores": output[:, 4],
                                  "labels": output[:, 5]}
            else:
                raise ValueError("Unknown model type")

            # Filter predictions by score threshold
            keep = prediction["scores"] >= score_threshold
            pred_boxes = prediction["boxes"][keep].detach().cpu()
            # print(f"pred_boxes {pred_boxes}")
            pred_scores = prediction["scores"][keep].detach().cpu()
            pred_labels = prediction["labels"][keep].detach().cpu()

            # -------- GROUND TRUTH --------
            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()

            if model_type.lower() == "yolo":
                gt_labels = gt_labels - 1

            img_np = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)
            ax.axis("off")

            # -------- MATCHING --------
            matched_pred = set()
            matched_gt = set()
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                for p_idx in range(len(pred_boxes)):
                    best_iou, gt_idx = ious[p_idx].max(0)
                    if best_iou >= iou_threshold and pred_labels[p_idx] == gt_labels[gt_idx]:
                        matched_pred.add(p_idx)
                        matched_gt.add(gt_idx.item())

            # -------- DRAW GROUND TRUTH --------
            for gt_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                # print(f"Matched_gt: {matched_gt}, git_idx: {gt_idx}")
                x1, y1, x2, y2 = box
                color = "green" if gt_idx in matched_gt else "yellow"
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=2, edgecolor=color)
                ax.add_patch(rect)

                if model_type.lower() == "yolo":
                    label = int(label) + 1

                label_name = class_map[int(label)] if class_map else str(int(label))

                ax.text(x2-1, y1+1, label_name,
                        color='white', fontsize=9, weight='bold',
                        ha='right', va='bottom',
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))

            # -------- DRAW PREDICTIONS --------
            for p_idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                # print("PRED DRAW PRINT")
                x1, y1, x2, y2 = box
                color = "blue" if p_idx in matched_pred else "red"
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=2, edgecolor=color)
                ax.add_patch(rect)

                if model_type.lower() == "yolo":
                    label = int(label) + 1

                label_name = class_map.get(int(label), str(int(label))) if class_map else str(int(label))

                ax.text(x1, y1-2, f"{label_name} {score:.2f}",
                        color='white', fontsize=9, weight='bold',
                        ha='left', va='bottom',
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))

            ax.set_title(f"Sample {idx}")

    # Disable unused axes
    for i in range(len(indices), len(axes)):
        axes[i].axis("off")

    # -------- LEGEND --------
    legend_elements = [
        mpatches.Patch(edgecolor="green", facecolor="none", label="Matched Ground Truth"),
        mpatches.Patch(edgecolor="yellow", facecolor="none", label="Missed Ground Truth"),
        mpatches.Patch(edgecolor="blue", facecolor="none", label="True Positive Prediction"),
        mpatches.Patch(edgecolor="red", facecolor="none", label="False Positive Prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=12)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0,0.05,1,0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "sample_predictions.png", dpi=300, bbox_inches="tight")


def load_model(model_type, checkpoint_path, num_classes):
    if model_type == "rcnn":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    elif model_type == "yolo":
        model = load_yolov5(num_classes)
    else:
        raise ValueError("Unknown model type")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", type=str, default="penn", help="Dataset to train on: penn or pet")
    parser.add_argument("--splits", type=str, default="pennfudan_splits.json")
    parser.add_argument("--model", type= str, default="rcnn", help="Model to train, rcnn or yolo")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=str, default="vis")
    args = parser.parse_args()

    metrics_path = Path(args.checkpoint).parent / "metrics.csv"

    df = pd.read_csv(metrics_path)

    out_path = Path(args.output)

    out_path.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.grid()
    plt.savefig(out_path / "loss_curve.png")
    # plt.show()

    plt.figure()
    plt.plot(df["epoch"], df["mAP@0.5"], label="mAP@0.5")
    plt.plot(df["epoch"], df["recall"], label="Recall")
    plt.plot(df["epoch"], df["precision"], label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Detection Metrics")
    plt.legend()
    plt.grid()
    plt.savefig(out_path / "metrics_curve.png")
    # plt.show()

    plt.figure()
    plt.plot(df["epoch"], df["inference_fps"])
    plt.xlabel("Epoch")
    plt.ylabel("Images per Second")
    plt.title("Inference Speed")
    plt.grid()
    plt.savefig(out_path / "speed_curve.png")
    # plt.show()

    DatasetClass = get_dataset(args.dataset)

    # train_dataset = DatasetClass(
    #     root=args.root,
    #     split_json=args.splits,
    #     transform=get_detection_transforms(train=True)
    # )

    val_dataset = DatasetClass(
        root=args.root,
        split_json=args.splits,
        transform=get_detection_transforms(train=False)
    )

    if hasattr(val_dataset, "class_to_label"):
        classes = val_dataset.class_to_label
    elif hasattr(val_dataset, "breed_to_label"):
        classes = val_dataset.breed_to_label
    else:
        raise ValueError("Dataset does not define class mapping attribute")

    model = load_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        num_classes=len(classes)
    )

    if args.model == "yolo":
        class_map = {v - 1: k for k, v in classes.items()}
    else:
        class_map = {v: k for k, v in classes.items()}
    class_map = {v: k for k, v in classes.items()}

    visualize_predictions(
        model,
        val_dataset,
        out_dir=Path(args.output),
        model_type=args.model,
        num_samples=args.samples,
        score_threshold=0.3,
        iou_threshold=0.5,
        class_map=class_map,
        title=f"{args.model.upper()} - {args.dataset}"
    )
