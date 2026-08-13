"""Image transformations for training and validation.

The pretrained torchvision model planned for this project expects a consistent
224 x 224 input and ImageNet-style normalized pixel values. Training receives
small, plausible variations so the model does not memorize the exact training
images. Validation remains deterministic so its metrics measure the model,
not random augmentation.
"""

from torchvision import transforms


IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_train_transform() -> transforms.Compose:
    """Create the training pipeline with moderate image augmentation."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
                ratio=(3 / 4, 4 / 3),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_validation_transform() -> transforms.Compose:
    """Create deterministic preprocessing for validation images."""

    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE, antialias=True),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
