"""
Dataset utilities for the small-sample defect classification project.

Expects a directory laid out in torchvision.datasets.ImageFolder format:

    data_root/
        defect_type_a/
            img001.png
            img002.png
        defect_type_b/
            img001.png
        good/
            img001.png

If your raw images instead live as a flat folder with class-prefixed
filenames (e.g. `scratch_0001.png`, `good_0002.png`), use
`restructure_flat_folder()` below to convert them into the ImageFolder
layout in-place before training.
"""
import os
import shutil

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ImageNet normalization stats, required because we reuse an
# ImageNet-pretrained EfficientNet-B0 backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> transforms.Compose:
    """Grayscale-safe preprocessing/augmentation pipeline.

    Images are converted to 3-channel so they are compatible with an
    ImageNet-pretrained backbone, then lightly augmented (flip + small
    rotation) to help the model generalize from very few labeled examples.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def restructure_flat_folder(base_path: str, delimiter: str = "_") -> None:
    """Convert a flat, class-prefixed folder of images into ImageFolder layout.

    Example: `scratch_0001.png` -> `base_path/scratch/scratch_0001.png`

    This is a one-time, in-place operation. Safe to call multiple times;
    it only moves files that are still sitting directly in `base_path`.
    """
    all_files = [
        f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))
    ]

    class_names = {f.split(delimiter)[0] for f in all_files if delimiter in f}

    for class_name in class_names:
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

    for filename in all_files:
        if delimiter not in filename:
            continue
        class_name = filename.split(delimiter)[0]
        src = os.path.join(base_path, filename)
        dst = os.path.join(base_path, class_name, filename)
        shutil.move(src, dst)


def load_dataloaders(
    data_root: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    image_size: int = 224,
    seed: int = 42,
):
    """Load the defect image dataset and return train/test DataLoaders.

    Returns:
        train_loader, test_loader, class_names
    """
    import torch

    transform = get_transforms(image_size)
    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_dataset.classes
