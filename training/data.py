#------------------------------------------------------------
# READ CLASSIFICATION DATA WITH A FIXED CATEGORY ORDER
#------------------------------------------------------------
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from production.vit import image_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ProductDataset(Dataset):
    """Read product pictures from one folder per category."""

    def __init__(self, root, classes=None, training=False):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Missing dataset folder: {self.root}")
        found = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.classes = list(classes) if classes is not None else found
        if "unknown" in self.classes:
            raise ValueError("The category name 'unknown' is reserved for rejected predictions.")
        unexpected = set(found) - set(self.classes)
        if unexpected:
            raise ValueError(f"Categories missing from training: {sorted(unexpected)}")
        self.samples = []
        for label, name in enumerate(self.classes):
            files = sorted(p for p in (self.root / name).rglob("*")
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if training and not files:
                raise ValueError(f"Training category has no images: {name}")
            self.samples.extend((p, label) for p in files)
        if not self.samples:
            raise ValueError(f"No images found in {self.root}")
        self.transform = image_transform(training)

    def __len__(self):
        """Return the number of product pictures."""
        return len(self.samples)

    def __getitem__(self, index):
        """Open a picture as RGB and return its category number."""
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def make_loader(dataset, batch_size, workers, training=False, balanced=False, seed=42):
    """Build batches, optionally showing rare categories more often."""
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative.")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balanced:
        counts = Counter(label for _, label in dataset.samples)
        weights = [1.0 / counts[label] for _, label in dataset.samples]
        sampler = WeightedRandomSampler(weights, len(weights), generator=generator)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      shuffle=training and sampler is None, sampler=sampler,
                      generator=generator, pin_memory=torch.cuda.is_available())
