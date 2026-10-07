#------------------------------------------------------------
# YOLO: HAILO-8 ON THE PI (PYTORCH ON A PC). VIT: PYTORCH ON THE CPU/GPU
#------------------------------------------------------------
from pathlib import Path

import numpy as np


def choose_backend(name):
    """auto uses the Hailo-8 when present, otherwise PyTorch on GPU/CPU."""
    if name not in ("auto", "hailo", "torch"):
        raise ValueError("backend must be auto, hailo or torch.")
    if name != "auto":
        return name
    from production.hailo import hailo_available
    return "hailo" if hailo_available() else "torch"


def read_labels(path):
    """One class name per line, in the model's output order."""
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(names) < 1 or len(set(names)) != len(names):
        raise ValueError(f"{path} needs unique class names, one per line.")
    return names


def coco_names():
    """The 80 COCO classes, in the order every pretrained YOLO HEF uses."""
    import yaml
    import ultralytics
    data = Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml"
    names = yaml.safe_load(data.read_text())["names"]
    return [names[i] for i in sorted(names)]


#------------------------------------------------------------
# DETECTOR: [x1, y1, x2, y2, score, class] PER FRAME
#------------------------------------------------------------
class TorchDetector:
    """Ultralytics YOLO .pt weights for development and training checks."""

    def __init__(self, weights, device, image_size, confidence, max_detections):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if self.model.task not in ("detect", "segment"):
            raise ValueError("Use YOLO detection or instance-segmentation weights for tracking.")
        self.names = self.model.names
        self.device, self.image_size = str(device), image_size
        self.confidence, self.max_detections = confidence, max_detections

    def detect(self, rgb):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        result = self.model.predict(bgr, imgsz=self.image_size, conf=self.confidence,
                                    max_det=self.max_detections, device=self.device, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            return np.zeros((0, 6), np.float32)
        return result.boxes.data[:, :6].cpu().numpy().astype(np.float32)


def build_detector(settings, backend, device):
    """Load the configured YOLO for the selected runtime."""
    if backend == "hailo":
        from production.hailo import HailoDetector
        labels = settings.get("hef_labels")
        names = read_labels(labels) if labels else coco_names()
        return HailoDetector(settings["hef"], names, settings["confidence"], settings["max_detections"])
    return TorchDetector(settings["weights"], device, settings["image_size"],
                         settings["confidence"], settings["max_detections"])


#------------------------------------------------------------
# CLASSIFIER: PROBABILITIES FOR A FEW CROPS OF ONE TRACK
#------------------------------------------------------------
class TorchClassifier:
    """ViT-Small/16 in PyTorch: grocery checkpoint, or the ImageNet head for demos."""

    def __init__(self, settings, device):
        from production.vit import load_classifier, load_imagenet_classifier
        self.imagenet = not settings.get("checkpoint")
        if not self.imagenet:
            self.model, self.classes = load_classifier(settings["checkpoint"], device)
        else:
            self.model, self.classes = load_imagenet_classifier(settings["pretrained_directory"], device)
        self.device = device

    def probabilities(self, crops):
        import torch
        from production.vit import image_transform
        transform = image_transform()
        with torch.inference_mode():
            batch = torch.stack([transform(crop) for crop in crops]).to(self.device)
            return self.model(batch).softmax(1).cpu().numpy()


def build_classifier(settings, device):
    """A grocery model when trained; otherwise ImageNet guesses, clearly labelled.

    ViT-Small runs on the Pi's CPU: only a few crops per moving object are classified.
    """
    return TorchClassifier(settings, device)
