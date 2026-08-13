"""Reproducible experiment configurations and CSV result recording."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from src.dataset import DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR, create_dataloaders
from src.device import select_device
from src.model import create_model
from src.train import train_model


AugmentationStrategy = Literal["moderate", "none"]
RESULT_FIELDS = (
    "name",
    "hypothesis",
    "batch_size",
    "learning_rate",
    "augmentation",
    "epochs",
    "final_training_loss",
    "final_training_accuracy",
    "final_validation_loss",
    "final_validation_accuracy",
    "best_validation_accuracy",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One controlled change and the hypothesis it is testing."""

    name: str
    hypothesis: str
    batch_size: int = 32
    learning_rate: float = 1e-3
    augmentation: AugmentationStrategy = "moderate"
    epochs: int = 5


@dataclass(frozen=True)
class ExperimentResult:
    """Metrics recorded after one experiment run."""

    name: str
    hypothesis: str
    batch_size: int
    learning_rate: float
    augmentation: str
    epochs: int
    final_training_loss: float
    final_training_accuracy: float
    final_validation_loss: float
    final_validation_accuracy: float
    best_validation_accuracy: float


DEFAULT_EXPERIMENTS = (
    ExperimentConfig(
        name="baseline",
        hypothesis="A moderate batch and learning rate provide a stable reference.",
    ),
    ExperimentConfig(
        name="batch_size_16",
        batch_size=16,
        hypothesis="Smaller batches may add useful gradient noise and improve generalization.",
    ),
    ExperimentConfig(
        name="batch_size_64",
        batch_size=64,
        hypothesis="Larger batches may make updates smoother but could reduce generalization.",
    ),
    ExperimentConfig(
        name="learning_rate_1e-4",
        learning_rate=1e-4,
        hypothesis="A smaller learning rate may make classifier updates more stable.",
    ),
    ExperimentConfig(
        name="learning_rate_1e-2",
        learning_rate=1e-2,
        hypothesis="A larger learning rate may converge faster but could overshoot useful updates.",
    ),
    ExperimentConfig(
        name="augmentation_none",
        augmentation="none",
        hypothesis="Removing augmentation tests whether the dataset benefits from invariance.",
    ),
    ExperimentConfig(
        name="epochs_10",
        epochs=10,
        hypothesis="More epochs may improve fit, unless validation performance starts to overfit.",
    ),
)


def experiment_configs() -> tuple[ExperimentConfig, ...]:
    """Return the controlled experiment matrix."""

    return DEFAULT_EXPERIMENTS


def run_experiment(
    config: ExperimentConfig,
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    validation_dir: str | Path = DEFAULT_VALIDATION_DIR,
    device: str | torch.device = "cpu",
    model_factory: Callable[[], nn.Module] = create_model,
) -> ExperimentResult:
    """Run one configuration and convert its history into a result record."""

    train_dataset, _, train_loader, validation_loader = create_dataloaders(
        train_dir=train_dir,
        validation_dir=validation_dir,
        batch_size=config.batch_size,
        augmentation=config.augmentation,
    )
    history = train_model(
        model_factory(),
        train_loader,
        validation_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=device,
        class_to_idx=train_dataset.class_to_idx,
        checkpoint_path=None,
    )
    return ExperimentResult(
        name=config.name,
        hypothesis=config.hypothesis,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        augmentation=config.augmentation,
        epochs=config.epochs,
        final_training_loss=history["training_loss"][-1],
        final_training_accuracy=history["training_accuracy"][-1],
        final_validation_loss=history["validation_loss"][-1],
        final_validation_accuracy=history["validation_accuracy"][-1],
        best_validation_accuracy=max(history["validation_accuracy"]),
    )


def record_result(result: ExperimentResult, results_path: str | Path) -> None:
    """Append one result to a CSV file, writing the header when needed."""

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named dataset experiment.")
    parser.add_argument(
        "--experiment",
        choices=[config.name for config in experiment_configs()],
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments and hypotheses.",
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/experiments.csv"),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.list:
        for config in experiment_configs():
            print(f"{config.name}: {config.hypothesis}")
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --list is used.")

    config = next(
        config for config in experiment_configs() if config.name == args.experiment
    )
    result = run_experiment(
        config,
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        device=args.device or select_device(),
    )
    record_result(result, args.results_path)
    print(f"Recorded {result.name} in {args.results_path}")


if __name__ == "__main__":
    main()
