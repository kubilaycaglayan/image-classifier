"""Explicit training and validation loops for the classifier."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR, create_dataloaders
from src.device import select_device
from src.model import configure_fine_tuning, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"


def save_checkpoint(
    model: nn.Module,
    class_to_idx: Mapping[str, int],
    validation_accuracy: float,
    checkpoint_path: str | Path,
) -> None:
    """Save model weights, class mapping, and the score that selected them."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": dict(class_to_idx),
            "validation_accuracy": validation_accuracy,
        },
        path,
    )
    print(f"Saved best model checkpoint to: {path}")


def create_optimizer(
    model: nn.Module,
    classifier_learning_rate: float,
    backbone_learning_rate: float | None = None,
) -> Optimizer:
    """Create Adam, optionally assigning a smaller rate to the backbone."""

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("The model has no trainable parameters.")

    if backbone_learning_rate is None:
        return Adam([parameter for _, parameter in named_parameters], lr=classifier_learning_rate)

    classifier_parameters = [
        parameter for name, parameter in named_parameters if name.startswith("fc.")
    ]
    backbone_parameters = [
        parameter for name, parameter in named_parameters if not name.startswith("fc.")
    ]
    if not classifier_parameters or not backbone_parameters:
        raise ValueError(
            "Differential learning rates require trainable classifier and "
            "backbone parameters."
        )

    return Adam(
        [
            {"params": classifier_parameters, "lr": classifier_learning_rate},
            {"params": backbone_parameters, "lr": backbone_learning_rate},
        ],
        lr=classifier_learning_rate,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""

    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, correct_predictions / total_examples


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch without changing parameters or tracking gradients."""

    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, correct_predictions / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    class_to_idx: Mapping[str, int] | None = None,
    checkpoint_path: str | Path | None = None,
    backbone_learning_rate: float | None = None,
) -> dict[str, list[float]]:
    """Train ``model`` and print train/validation metrics after each epoch.

    The device is explicit here so the loop remains easy to inspect and test.
    The command-line entry point selects CUDA, MPS, or CPU automatically.
    When ``checkpoint_path`` is provided, a checkpoint is written whenever
    validation accuracy improves. The class mapping must be provided with it.
    """

    if epochs < 1:
        raise ValueError(f"epochs must be positive; received {epochs}.")
    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive; received {learning_rate}."
        )
    if backbone_learning_rate is not None and backbone_learning_rate <= 0:
        raise ValueError(
            "backbone_learning_rate must be positive when provided; "
            f"received {backbone_learning_rate}."
        )
    if checkpoint_path is not None and class_to_idx is None:
        raise ValueError("class_to_idx is required when saving a checkpoint.")

    selected_device = torch.device(device)
    model.to(selected_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model,
        classifier_learning_rate=learning_rate,
        backbone_learning_rate=backbone_learning_rate,
    )
    history = {
        "training_loss": [],
        "training_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_validation_accuracy = float("-inf")

    for epoch in range(epochs):
        training_loss, training_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            selected_device,
        )
        validation_loss, validation_accuracy = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            selected_device,
        )

        history["training_loss"].append(training_loss)
        history["training_accuracy"].append(training_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"training loss: {training_loss:.4f} | "
            f"training accuracy: {training_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f}"
        )

        if (
            checkpoint_path is not None
            and class_to_idx is not None
            and validation_accuracy > best_validation_accuracy
        ):
            save_checkpoint(
                model=model,
                class_to_idx=class_to_idx,
                validation_accuracy=validation_accuracy,
                checkpoint_path=checkpoint_path,
            )
            best_validation_accuracy = validation_accuracy

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the image classifier.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument("--without-pretrained-weights", action="store_true")
    args = parser.parse_args()

    train_dataset, _, train_loader, validation_loader = create_dataloaders(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = create_model(pretrained=not args.without_pretrained_weights)
    backbone_learning_rate = None
    if args.fine_tune:
        configure_fine_tuning(model)
        backbone_learning_rate = args.backbone_learning_rate
    device = select_device()
    train_model(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        class_to_idx=train_dataset.class_to_idx,
        checkpoint_path=args.checkpoint_path,
        backbone_learning_rate=backbone_learning_rate,
    )


if __name__ == "__main__":
    main()
