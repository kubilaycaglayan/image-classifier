"""Evaluate the classifier with metrics beyond accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR, create_dataloaders
from src.device import select_device
from src.model import NUM_CLASSES, create_model
from src.train import DEFAULT_CHECKPOINT_PATH


def metrics_from_confusion_matrix(
    confusion_matrix: Tensor,
) -> dict[str, float | list[float]]:
    """Calculate accuracy, per-class metrics, and macro averages.

    Rows in the matrix are actual classes and columns are predicted classes.
    A macro average gives every class equal weight, which makes minority-class
    performance visible instead of allowing a large class to dominate one
    overall accuracy number.
    """

    true_positives = confusion_matrix.diag().float()
    actual_counts = confusion_matrix.sum(dim=1).float()
    predicted_counts = confusion_matrix.sum(dim=0).float()

    precision = torch.where(
        predicted_counts > 0,
        true_positives / predicted_counts,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        actual_counts > 0,
        true_positives / actual_counts,
        torch.zeros_like(true_positives),
    )
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2 * precision * recall / f1_denominator,
        torch.zeros_like(true_positives),
    )

    total_examples = confusion_matrix.sum().item()
    accuracy = (
        true_positives.sum().item() / total_examples if total_examples else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    num_classes: int = NUM_CLASSES,
    device: str | torch.device = "cpu",
) -> dict[str, float | list[float] | Tensor]:
    """Evaluate a model and return metrics plus its confusion matrix."""

    if num_classes < 1:
        raise ValueError(f"num_classes must be positive; received {num_classes}.")

    selected_device = torch.device(device)
    model.to(selected_device)
    model.eval()
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64
    )

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(selected_device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()
            actual_labels = labels.cpu()
            for actual, predicted in zip(actual_labels, predictions):
                confusion_matrix[actual, predicted] += 1

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["confusion_matrix"] = confusion_matrix
    return metrics


def print_metrics(
    metrics: dict[str, float | list[float] | Tensor],
    class_names: list[str] | tuple[str, ...],
) -> None:
    """Print macro and per-class evaluation metrics."""

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['precision']:.4f}")
    print(f"Macro recall: {metrics['recall']:.4f}")
    print(f"Macro F1: {metrics['f1']:.4f}")
    print("Per-class metrics:")
    for index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"precision={metrics['precision_per_class'][index]:.4f}, "
            f"recall={metrics['recall_per_class'][index]:.4f}, "
            f"F1={metrics['f1_per_class'][index]:.4f}"
        )


def plot_confusion_matrix(
    confusion_matrix: Tensor,
    class_names: list[str] | tuple[str, ...],
    show: bool = True,
):
    """Display actual-vs-predicted counts and return the matplotlib figure."""

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion_matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of images")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Confusion matrix")

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(
                column,
                row,
                int(confusion_matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    if show:
        plt.show()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the image classifier.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()

    _, validation_dataset, _, validation_loader = create_dataloaders(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if checkpoint["class_to_idx"] != validation_dataset.class_to_idx:
        raise ValueError("Checkpoint and validation class mappings do not match.")

    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = select_device()
    metrics = evaluate_model(
        model,
        validation_loader,
        num_classes=len(validation_dataset.classes),
        device=device,
    )
    print_metrics(metrics, validation_dataset.classes)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        validation_dataset.classes,
    )


if __name__ == "__main__":
    main()
