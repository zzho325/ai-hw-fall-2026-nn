from pathlib import Path

import torch
from torchvision.datasets import MNIST


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


def load_images(root, train):
    dataset = MNIST(root=root, train=train, download=True)
    images = dataset.data.unsqueeze(1).float() / 255.0
    return images, dataset.targets


def training_split(seed, size):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)
    train_indices, val_indices = indices[5000:], indices[:5000]
    metadata = {
        "seed": seed,
        "training_examples": len(train_indices),
        "validation_examples": len(val_indices),
    }
    return train_indices, val_indices, metadata


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
