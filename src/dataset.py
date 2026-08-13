"""Load and inspect the image datasets used by the classifier.

``ImageFolder`` treats each subdirectory below a root directory as one class.
It sorts those directory names alphabetically and uses the resulting position
as the numeric label. For this project that means ``bottle`` is 0, ``pencil``
is 1, and ``wristwatch`` is 2.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.transforms import create_train_transform, create_validation_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "train"
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
EXPECTED_CLASSES = ("bottle", "pencil", "wristwatch")


def load_datasets(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    train_transform: Callable | None = None,
    validation_transform: Callable | None = None,
) -> tuple[ImageFolder, ImageFolder]:
    """Load the training and validation directories with ``ImageFolder``.

    The class mapping is read from the directory names rather than hard-coded
    into the image labels. Both splits must contain the same three classes so
    that a label has the same meaning in training and validation.
    """

    training_dataset = ImageFolder(
        root=str(train_dir),
        transform=train_transform,
    )
    validation_dataset = ImageFolder(
        root=str(validation_dir),
        transform=validation_transform,
    )

    if tuple(training_dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            "Training classes must be exactly "
            f"{EXPECTED_CLASSES}; found {training_dataset.classes}."
        )

    if validation_dataset.classes != training_dataset.classes:
        raise ValueError(
            "Training and validation class directories must match. "
            f"Training: {training_dataset.classes}; "
            f"validation: {validation_dataset.classes}."
        )

    return training_dataset, validation_dataset


def create_dataloaders(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[ImageFolder, ImageFolder, DataLoader, DataLoader]:
    """Create transformed datasets and their training/validation loaders.

    A ``Dataset`` represents individual samples and labels. A ``DataLoader``
    groups those samples into batches, optionally shuffling their order and
    loading samples with worker processes. Training is shuffled to reduce
    order-related learning, while validation is deterministic for comparable
    metrics between epochs. ``num_workers=0`` keeps loading in the main process
    and is a simple default for learning and cross-platform compatibility.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; received {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative; received {num_workers}.")

    training_dataset, validation_dataset = load_datasets(
        train_dir=train_dir,
        validation_dir=validation_dir,
        train_transform=create_train_transform(),
        validation_transform=create_validation_transform(),
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        training_dataset,
        validation_dataset,
        training_loader,
        validation_loader,
    )


def print_batch_summary(training_loader: DataLoader) -> tuple[Tensor, Tensor]:
    """Print and return one training batch for inspecting its tensor shape."""

    images, labels = next(iter(training_loader))
    print(f"Batch image shape: {list(images.shape)}")
    print(f"Batch label shape: {list(labels.shape)}")
    print(f"Labels: {labels.tolist()}")
    return images, labels


def print_dataset_summary(
    training_dataset: ImageFolder, validation_dataset: ImageFolder
) -> None:
    """Print the key facts about both dataset splits."""

    print(f"Number of training images: {len(training_dataset)}")
    print(f"Number of validation images: {len(validation_dataset)}")
    print(f"Class names: {training_dataset.classes}")
    print(f"Class-to-index mapping: {training_dataset.class_to_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the image datasets.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    (
        training_dataset,
        validation_dataset,
        training_loader,
        _,
    ) = create_dataloaders(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print_dataset_summary(training_dataset, validation_dataset)
    print_batch_summary(training_loader)


if __name__ == "__main__":
    main()
