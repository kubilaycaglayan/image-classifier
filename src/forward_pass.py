"""Demonstrate one model forward pass without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR, create_dataloaders
from src.model import create_model


def run_forward_pass(
    model: nn.Module,
    train_loader: DataLoader,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run one batch through ``model`` and return images, labels, logits, predictions."""

    images, labels = next(iter(train_loader))

    model.eval()
    with torch.no_grad():
        logits = model(images)

    predicted_class_indices = logits.argmax(dim=1)

    print(f"Input shape: {list(images.shape)}")
    print(f"Output shape (logits): {list(logits.shape)}")
    print(f"Labels: {labels.tolist()}")
    print(f"Predicted class indices: {predicted_class_indices.tolist()}")

    return images, labels, logits, predicted_class_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one image-classifier forward pass.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--without-pretrained-weights", action="store_true")
    args = parser.parse_args()

    _, _, train_loader, _ = create_dataloaders(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = create_model(pretrained=not args.without_pretrained_weights)
    run_forward_pass(model, train_loader)


if __name__ == "__main__":
    main()
