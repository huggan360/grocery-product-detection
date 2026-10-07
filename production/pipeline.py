#------------------------------------------------------------
# FULL SHELF -> INSTANCE MASKS -> MASKED CROPS -> OPTIONAL VIT
#------------------------------------------------------------
import threading
import math
from pathlib import Path

from PIL import ImageDraw, ImageOps
import yaml

from production.masks import masked_crop, polygon_bounds, validate_polygon
from production.preprocessing import prepare_yolo_image


class RGBPipeline:
    """Use pretrained segmentation immediately, and a grocery ViT when available."""

    def __init__(self, config):
        """Use one configuration for the real pipeline and the review tool."""
        self.settings = dict(config["prediction"])
        self.settings.setdefault("image_size", 640)
        self.weights = self.settings["segmentation_weights"]
        if not self.weights or not Path(self.weights).is_file():
            raise ValueError(f"Missing segmentation checkpoint: {self.weights}")
        self.classifier_path = config["classifier"].get("checkpoint")
        self.imagenet = not self.classifier_path and config["classifier"].get("baseline") == "imagenet"
        self.imagenet_directory = config["classifier"].get("pretrained_directory", "models/vit/pretrained")
        if self.classifier_path and not Path(self.classifier_path).is_file():
            raise ValueError(f"Missing ViT checkpoint: {self.classifier_path}. Use null for baseline mode.")
        for key in ("detection_confidence", "classification_confidence"):
            if not 0 <= self.settings[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")
        for key in ("batch_size", "max_detections", "image_size"):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        self.padding = self.settings["crop_padding"]
        if not math.isfinite(self.padding) or self.padding < 0:
            raise ValueError("crop_padding must be finite and non-negative.")
        self.threshold = self.settings["classification_confidence"]
        if self.imagenet:
            self.threshold = config["classifier"].get("imagenet_confidence", 0.0)
            if not 0 <= self.threshold <= 1:
                raise ValueError("imagenet_confidence must be between 0 and 1.")
        self.device_name = config.get("device", "auto")
        self.model = self.classifier = None
        self.classes = []
        self.lock = threading.Lock()
        self.available = True
        self.message = ("YOLO26 segmentation + grocery ViT" if self.classifier_path else
                        "YOLO26 masks; basic category hints only. Assign grocery categories manually.")
        if self.imagenet:
            self.message = "YOLO26 + pretrained ViT-L/32 ImageNet guesses (not grocery-trained). Correct labels before saving."

    def predict(self, image):
        """Predict each visible object's outline before any cropping happens."""
        with self.lock:
            return self._predict(image)

    def _predict(self, image):
        """Run one serialized inference call and return editable polygon suggestions."""
        import cv2
        import torch
        from ultralytics import YOLO
        from utils.common import choose_device
        from production.vit import image_transform, load_classifier, load_imagenet_classifier
        if not self.available:
            raise RuntimeError(self.message)
        image = ImageOps.exif_transpose(image).convert("RGB")
        device = choose_device(self.device_name)
        if self.model is None:
            self.model = YOLO(self.weights)
            if self.model.task != "segment":
                self.model = None
                raise ValueError("Use segmentation weights such as yolo26l-seg.pt, not box-only weights.")
        if self.classifier_path and self.classifier is None:
            self.classifier, self.classes = load_classifier(self.classifier_path, device)
        elif self.imagenet and self.classifier is None:
            self.classifier, self.classes = load_imagenet_classifier(self.imagenet_directory, device)
        # Keep the original for editing and for the ViT's masked crops.
        detector_image = prepare_yolo_image(image, self.settings.get("yolo_preprocessing", True))
        result = self.model.predict(detector_image, device=str(device),
            imgsz=self.settings["image_size"],
            conf=self.settings["detection_confidence"],
            max_det=self.settings["max_detections"], retina_masks=True, verbose=False)[0]
        if result.masks is None or result.boxes is None:
            return []
        records = []
        # The shared output supports one exterior outline per item. Retain the largest
        # visible component and record the component count for human review.
        for mask, label, confidence in zip(result.masks.data.cpu().numpy(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            binary = (mask > 0.5).astype("uint8")
            if binary.shape != (image.height, image.width):
                binary = cv2.resize(binary, image.size, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            points = cv2.approxPolyDP(contour, max(0.8, cv2.arcLength(contour, True) * 0.002), True).reshape(-1, 2).tolist()
            try:
                validate_polygon(points, *image.size)
            except ValueError:
                continue
            original_label = result.names[int(label)]
            # A bottle does not tell us whether it contains milk or ketchup.
            hints = {"apple": "apple", "banana": "banana", "orange": "orange", "carrot": "carrots", "broccoli": "broccoli"}
            records.append({"box_xyxy": polygon_bounds(points), "polygon": points,
                "category": hints.get(original_label, "unknown"), "classification_confidence": None,
                "detection_confidence": confidence, "source_label": original_label,
                "label_source": "category_hint", "segmentation_weights": str(self.weights),
                "visible_components": len(contours),
                "preprocessing": "clahe3_sharpen035_v1" if self.settings.get("yolo_preprocessing", True) else "none"})
        if self.classifier is not None:
            transform = image_transform()
            with torch.inference_mode():
                for start in range(0, len(records), self.settings["batch_size"]):
                    chunk = records[start:start + self.settings["batch_size"]]
                    batch = torch.stack([transform(masked_crop(image, record["polygon"], self.padding)) for record in chunk]).to(device)
                    scores, indices = self.classifier(batch).softmax(1).topk(min(3, len(self.classes)), dim=1)
                    for record, row_scores, row_indices in zip(chunk, scores.tolist(), indices.tolist()):
                        score, index = row_scores[0], row_indices[0]
                        record["category"] = self.classes[index] if score >= self.threshold else "unknown"
                        record["classification_confidence"] = score
                        record["label_source"] = "imagenet_vit_l32" if self.imagenet else "grocery_vit_masked_crop"
                        record["top_categories"] = [{"category": self.classes[i], "score": s}
                                                    for s, i in zip(row_scores, row_indices)]
        return records


#------------------------------------------------------------
# LOAD SHARED CONFIGURATION AND DRAW RESULTS
#------------------------------------------------------------
def load_pipeline_config(path, models_dir=None):
    """Resolve model and output paths next to this YAML file."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    model_root = Path(models_dir).resolve() if models_dir else (path.parent / config.get("models_directory", "../models")).resolve()
    config["classifier"]["pretrained_directory"] = str(model_root / "vit/pretrained")
    for section, key in (("prediction", "segmentation_weights"), ("classifier", "checkpoint")):
        value = config[section].get(key)
        if value:
            config[section][key] = str((model_root / value).resolve())
    return config


def draw_predictions(image, records):
    """Draw masks and labels on a copy of the original image."""
    result = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(result)
    for record in records:
        polygon = [tuple(point) for point in record["polygon"]]
        draw.line(polygon + [polygon[0]], fill="cyan", width=2)
        box = record["box_xyxy"]
        draw.rectangle(box, outline="lime", width=2)
        score = record["classification_confidence"]
        label = record["category"]
        if score is not None:
            label += f" {score:.2f}"
        draw.text((max(0, box[0]), max(0, box[1] - 12)), label, fill="red")
    return result
