"""Explicit training and validation loops for the classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR, create_dataloaders
from src.device import select_device
from src.model import create_model


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
) -> dict[str, list[float]]:
    """Train ``model`` and print train/validation metrics after each epoch.

    The device is explicit here so the loop remains easy to inspect and test.
    The command-line entry point selects CUDA, MPS, or CPU automatically.
    """

    if epochs < 1:
        raise ValueError(f"epochs must be positive; received {epochs}.")
    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive; received {learning_rate}."
        )

    selected_device = torch.device(device)
    model.to(selected_device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(trainable_parameters, lr=learning_rate)
    history = {
        "training_loss": [],
        "training_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

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
    parser.add_argument("--without-pretrained-weights", action="store_true")
    args = parser.parse_args()

    _, _, train_loader, validation_loader = create_dataloaders(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = create_model(pretrained=not args.without_pretrained_weights)
    device = select_device()
    train_model(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )


if __name__ == "__main__":
    main()
