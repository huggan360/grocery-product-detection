#------------------------------------------------------------
# PRETRAINED VISION TRANSFORMER AND IMAGE PREPARATION
#------------------------------------------------------------
import torch
import re
from pathlib import Path
from torch import nn
from torchvision import transforms
from torchvision.models import ViT_L_32_Weights, vit_l_32
from torchvision.transforms import InterpolationMode

from production.preprocessing import PrepareViTImage

ARCHITECTURE = "vit_l_32"
WEIGHTS = ViT_L_32_Weights.IMAGENET1K_V1
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SquarePad:
    """Keep the whole product by adding a grey border before resizing."""

    def __call__(self, image):
        width, height = image.size
        size = max(width, height)
        left, top = (size - width) // 2, (size - height) // 2
        return transforms.functional.pad(
            image, (left, top, size - width - left, size - height - top), fill=127
        )


def image_transform(training=False):
    """Resize crops and use ImageNet normalization for the pretrained ViT."""
    # Use the same gentle preprocessing in training and prediction.
    steps = [PrepareViTImage(), SquarePad()]
    if training:
        # Some views show only part of a product or slightly different lighting.
        steps += [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0),
                                         ratio=(0.85, 1.15),
                                         interpolation=InterpolationMode.BILINEAR),
            transforms.RandomRotation(12, fill=127),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        ]
    else:
        steps += [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=InterpolationMode.BILINEAR)]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if training:
        # Hide a small patch sometimes. This is only a rough occlusion example.
        steps += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))]
    return transforms.Compose(steps)


def build_classifier(number_of_classes, pretrained=True):
    """Keep the pretrained ViT body and replace its final category layer."""
    if number_of_classes < 2:
        raise ValueError("Provide at least two product categories.")
    model = vit_l_32(weights=WEIGHTS if pretrained else None)
    model.heads.head = nn.Linear(model.heads.head.in_features, number_of_classes)
    return model


#------------------------------------------------------------
# SAVE, LOAD, AND REUSE TRAINED CATEGORIES
#------------------------------------------------------------
def read_checkpoint(path):
    """Read our classifier format without unpickling arbitrary Python objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("format_version") != 1:
        raise ValueError("Expected a grocery RGB ViT-L/32 checkpoint, format version 1. Older ViT-B/16 checkpoints cannot be loaded into this model.")
    classes = checkpoint["classes"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Checkpoint has invalid category names.")
    return checkpoint


def load_classifier(path, device):
    """Load a trained model with the exact label order used in training."""
    checkpoint = read_checkpoint(path)
    model = build_classifier(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint["classes"]


#------------------------------------------------------------
# IMAGENET DEMO: KEEP THE ORIGINAL TRAINED CLASSIFICATION HEAD
#------------------------------------------------------------
def load_imagenet_classifier(directory, device):
    """Load all 1,000 pretrained classes without inventing grocery training."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = torch.hub.load_state_dict_from_url(
        WEIGHTS.url, model_dir=str(directory), map_location="cpu", check_hash=True)
    model = vit_l_32(weights=None)
    model.load_state_dict(state)
    # Prefix labels so guesses cannot be mistaken for our grocery categories.
    classes = ["imagenet-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:54]
               for name in WEIGHTS.meta["categories"]]
    return model.to(device).eval(), classes


def initialize_classifier(classes, previous_path=None):
    """Start from ImageNet or reuse an earlier grocery model for fine-tuning."""
    if not previous_path:
        return build_classifier(len(classes), pretrained=True)
    checkpoint = read_checkpoint(previous_path)
    model = build_classifier(len(classes), pretrained=False)
    state = checkpoint["model"].copy()
    old_weight = state.pop("heads.head.weight")
    old_bias = state.pop("heads.head.bias")
    model.load_state_dict(state, strict=False)
    old_indices = {name: i for i, name in enumerate(checkpoint["classes"])}
    with torch.no_grad():
        for index, name in enumerate(classes):
            if name in old_indices:
                model.heads.head.weight[index].copy_(old_weight[old_indices[name]])
                model.heads.head.bias[index].copy_(old_bias[old_indices[name]])
    return model
