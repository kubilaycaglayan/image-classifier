"""Run inference on one image with a trained classifier."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.transforms import create_validation_transform


@dataclass(frozen=True)
class Prediction:
    """Human-readable result for one image prediction."""

    predicted_class: str
    class_probabilities: dict[str, float]
    confidence: float


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_names: Sequence[str],
    device: str | torch.device = "cpu",
    transform: Callable | None = None,
) -> Prediction:
    """Predict one image's class, probabilities, and confidence.

    ``model.eval()`` disables training-only behavior such as dropout and makes
    batch-normalization use learned statistics. ``torch.no_grad()`` avoids
    constructing a gradient graph because inference does not update weights;
    this reduces memory use and computation.
    """

    if not class_names:
        raise ValueError("class_names must contain at least one class.")

    selected_device = torch.device(device)
    model.to(selected_device)
    model.eval()
    image_transform = transform or create_validation_transform()

    with Image.open(image_path) as image:
        image_tensor = image_transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(selected_device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    if probabilities.numel() != len(class_names):
        raise ValueError(
            f"Model outputs {probabilities.numel()} classes, but "
            f"{len(class_names)} class names were provided."
        )

    predicted_index = int(probabilities.argmax())
    probability_by_class = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(class_names)
    }
    return Prediction(
        predicted_class=class_names[predicted_index],
        class_probabilities=probability_by_class,
        confidence=probability_by_class[class_names[predicted_index]],
    )
