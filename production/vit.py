#------------------------------------------------------------
# VIT-SMALL/16: SMALL ENOUGH FOR THE HAILO-8 AI HAT (26 TOPS)
#------------------------------------------------------------
# ViT-L/32 (306M parameters) cannot be compiled for Hailo-8. ViT-Small/16
# (22M) is in the Hailo Model Zoo and runs at roughly 70 FPS on the accelerator,
# and is still usable on the Raspberry Pi CPU while a grocery HEF is compiled.
import re
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from production.preprocessing import PrepareViTImage

ARCHITECTURE = "vit_small_patch16_224"
PRETRAINED = "vit_small_patch16_224.augreg_in21k_ft_in1k"
IMAGE_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class SquarePad:
    """Keep the whole product by adding a grey border before resizing."""

    def __call__(self, image):
        width, height = image.size
        size = max(width, height)
        left, top = (size - width) // 2, (size - height) // 2
        return transforms.functional.pad(
            image, (left, top, size - width - left, size - height - top), fill=127
        )


def square_crop(image):
    """The exact 224x224 RGB image given to both the CPU model and the Hailo HEF."""
    image = SquarePad()(PrepareViTImage()(image))
    return image.resize((IMAGE_SIZE, IMAGE_SIZE))


def image_transform(training=False):
    """Resize crops and normalize them the way ViT-Small was pretrained."""
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


def build_classifier(number_of_classes, pretrained=True, cache_dir=None):
    """Keep the pretrained ViT body and replace its final category layer."""
    import timm
    if number_of_classes < 2:
        raise ValueError("Provide at least two product categories.")
    cache_dir = cache_dir or Path(__file__).resolve().parents[1] / "models/vit/pretrained"
    return timm.create_model(PRETRAINED, pretrained=pretrained, num_classes=number_of_classes,
                             cache_dir=cache_dir)


def head_parameters(model):
    """The new category layer; everything else is the pretrained body."""
    return list(model.get_classifier().parameters())


#------------------------------------------------------------
# SAVE, LOAD, AND REUSE TRAINED CATEGORIES
#------------------------------------------------------------
def read_checkpoint(path):
    """Read our classifier format without unpickling arbitrary Python objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("format_version") != 2:
        raise ValueError("Expected a grocery ViT-Small/16 checkpoint, format version 2. "
                         "Older ViT-L/32 checkpoints cannot run on the Hailo-8; retrain with train.py vit.")
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
def imagenet_classes():
    """Prefix labels so guesses cannot be mistaken for our grocery categories."""
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    return ["imagenet-" + re.sub(r"[^a-z0-9]+", "-", info.index_to_description(i).split(",")[0].lower()).strip("-")[:54]
            for i in range(1000)]


def load_imagenet_classifier(directory, device):
    """Load all 1,000 pretrained classes without inventing grocery training."""
    import timm
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the Hugging Face download with the other model files.
    model = timm.create_model(PRETRAINED, pretrained=True, cache_dir=directory)
    return model.to(device).eval(), imagenet_classes()


def initialize_classifier(classes, previous_path=None):
    """Start from ImageNet or reuse an earlier grocery model for fine-tuning."""
    if not previous_path:
        return build_classifier(len(classes), pretrained=True)
    checkpoint = read_checkpoint(previous_path)
    model = build_classifier(len(classes), pretrained=False)
    state = checkpoint["model"].copy()
    old_weight = state.pop("head.weight")
    old_bias = state.pop("head.bias")
    model.load_state_dict(state, strict=False)
    old_indices = {name: i for i, name in enumerate(checkpoint["classes"])}
    with torch.no_grad():
        for index, name in enumerate(classes):
            if name in old_indices:
                model.head.weight[index].copy_(old_weight[old_indices[name]])
                model.head.bias[index].copy_(old_bias[old_indices[name]])
    return model
