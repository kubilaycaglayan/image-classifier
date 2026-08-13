"""Load and inspect the image datasets used by the classifier.

``ImageFolder`` treats each subdirectory below a root directory as one class.
It sorts those directory names alphabetically and uses the resulting position
as the numeric label. For this project that means ``bottle`` is 0, ``pencil``
is 1, and ``wristwatch`` is 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "train"
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
EXPECTED_CLASSES = ("bottle", "pencil", "wristwatch")


def load_datasets(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
) -> tuple[ImageFolder, ImageFolder]:
    """Load the training and validation directories with ``ImageFolder``.

    The class mapping is read from the directory names rather than hard-coded
    into the image labels. Both splits must contain the same three classes so
    that a label has the same meaning in training and validation.
    """

    training_dataset = ImageFolder(root=str(train_dir))
    validation_dataset = ImageFolder(root=str(validation_dir))

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
    args = parser.parse_args()

    training_dataset, validation_dataset = load_datasets(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
    )
    print_dataset_summary(training_dataset, validation_dataset)


if __name__ == "__main__":
    main()
