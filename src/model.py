"""Construct the frozen-backbone transfer-learning classifier."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


NUM_CLASSES = 3


def create_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create a ResNet18 with a trainable classifier for ``num_classes``.

    When ``pretrained`` is true, the model starts with weights learned on
    ImageNet. Those feature-extractor parameters are frozen initially, while
    the newly created final linear layer remains trainable.
    """

    if num_classes < 1:
        raise ValueError(f"num_classes must be positive; received {num_classes}.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def trainable_parameter_names(model: nn.Module) -> list[str]:
    """Return names of parameters that will receive gradients."""

    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def configure_fine_tuning(
    model: nn.Module,
    trainable_layers: Sequence[str] = ("layer4",),
) -> list[str]:
    """Freeze the backbone except for selected later layers and ``fc``.

    The default unfreezes ResNet18's final residual block, ``layer4``. Earlier
    blocks retain their general ImageNet features, while ``layer4`` and the
    classifier can adapt to this project's object classes.
    """

    if not trainable_layers:
        raise ValueError("trainable_layers must contain at least one layer.")

    top_level_names = set(dict(model.named_children()))
    unknown_layers = set(trainable_layers) - top_level_names
    if unknown_layers:
        raise ValueError(f"Unknown model layers: {sorted(unknown_layers)}.")

    for name, parameter in model.named_parameters():
        is_classifier = name.startswith("fc.")
        is_selected_layer = any(
            name.startswith(f"{layer_name}.") for layer_name in trainable_layers
        )
        parameter.requires_grad = is_classifier or is_selected_layer

    return trainable_parameter_names(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the transfer-learning model.")
    parser.add_argument("--without-pretrained-weights", action="store_true")
    args = parser.parse_args()

    model = create_model(pretrained=not args.without_pretrained_weights)
    print(model)
    print(f"Number of output classes: {model.fc.out_features}")
    print(f"Trainable parameters: {trainable_parameter_names(model)}")


if __name__ == "__main__":
    main()
