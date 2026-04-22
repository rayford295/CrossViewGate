from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_transform(
    image_size: int,
    normalize: bool = True,
    augment: bool = False,
    domain: str = "generic",
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> T.Compose:
    if augment and domain == "street":
        ops: list[object] = [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
        ]
    elif augment and domain == "overhead":
        ops = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
        ]
    else:
        ops = [T.Resize((image_size, image_size)), T.ToTensor()]
    if normalize:
        ops.append(T.Normalize(mean or IMAGENET_MEAN, std or IMAGENET_STD))
    return T.Compose(ops)


def normalize_imagenet_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    return (batch - mean) / std


def denormalize_imagenet_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    return (batch * std + mean).clamp(0.0, 1.0)
