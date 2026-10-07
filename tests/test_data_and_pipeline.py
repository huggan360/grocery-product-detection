#------------------------------------------------------------
# SYNTHETIC TESTS: NO DATASET DOWNLOADS OR MODEL TRAINING
#------------------------------------------------------------
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from production.vit import image_transform
from utils.common import load_config
from training.data import ProductDataset
from production.geometry import crop_box
from training.metrics import classification_metrics
from production.pipeline import RGBPipeline
from utils.prepare_data import prepare_data, read_annotations


class FakeDetector:
    """Return fixed boxes so crop handling can be checked exactly."""

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image, **kwargs):
        masks = torch.zeros((len(self.boxes), image.height, image.width))
        for index, (x1, y1, x2, y2) in enumerate(self.boxes):
            masks[index, y1:y2, x1:x2] = 1
        return [SimpleNamespace(boxes=SimpleBoxes(self.boxes),
                                masks=SimpleNamespace(data=masks), names={0: "apple"})]


class SimpleBoxes:
    """Provide the parts of the YOLO box API used by the pipeline."""

    def __init__(self, boxes):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.ones(len(boxes)) * 0.9
        self.cls = torch.zeros(len(boxes))

    def __len__(self):
        return len(self.xyxy)


class FakeClassifier(torch.nn.Module):
    """Produce known scores without learning anything."""

    def __init__(self, confident=True):
        super().__init__()
        self.confident = confident

    def forward(self, images):
        return images.new_tensor([0.0, 8.0] if self.confident else [0.0, 0.0]).repeat(len(images), 1)


class DataAndPipelineTests(unittest.TestCase):
    """Check real data conversion and pipeline edge cases using tiny images."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = []
        for split_index, split in enumerate(("train", "val", "test")):
            for class_index, category in enumerate(("apple", "milk")):
                name = f"{split}_{category}.png"
                Image.new("RGB", (40, 30), (30 + split_index * 50, 40 + class_index * 80, 20)).save(self.root / name)
                self.rows.append({"image": name, "split": split, "category": category,
                                  "xmin": 4, "ymin": 3, "xmax": 24, "ymax": 27,
                                  "group": f"{split}_session"})
        self.manifest = self.root / "annotations.csv"
        self.write_manifest()
        self.settings = {"detection_confidence": 0.25, "classification_confidence": 0.65,
                         "batch_size": 1, "max_detections": 100, "crop_padding": 0.05}

    def write_manifest(self):
        """Write the tiny annotation table used by a test."""
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_prepare_and_read_crops(self):
        output = prepare_data([self.manifest], self.root / "prepared")
        # Find any train label; its normalized box must match the original pixels.
        label = next((output / "detection/labels/train").glob("*.txt"))
        self.assertEqual([float(v) for v in label.read_text().split()], [0, 0.35, 0.5, 0.5, 0.8])
        dataset = ProductDataset(output / "classification/train", training=True)
        self.assertEqual(dataset.classes, ["apple", "milk"])
        tensor, target = dataset[0]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(target, 0)
        self.assertTrue(torch.isfinite(tensor).all())
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_data([self.manifest], output)

    def test_same_image_across_splits_is_rejected(self):
        self.rows[2]["image"] = self.rows[0]["image"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "different splits"):
            read_annotations([self.manifest], None)

    def test_renamed_duplicate_pixels_are_rejected(self):
        (self.root / self.rows[2]["image"]).write_bytes((self.root / self.rows[0]["image"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical image"):
            read_annotations([self.manifest], None)

    def test_group_leak_is_rejected(self):
        self.rows[2]["group"] = "train_session"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Group"):
            read_annotations([self.manifest], None)

    def test_bad_box_is_rejected(self):
        self.rows[0]["xmax"] = 500
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Invalid pixel box"):
            read_annotations([self.manifest], None)

    def test_mapping_must_cover_every_label(self):
        with self.assertRaisesRegex(ValueError, "No category mapping"):
            read_annotations([self.manifest], {"apple": "fruit"})
        result = read_annotations([self.manifest], {"apple": "fruit", "milk": "dairy"})
        self.assertEqual(set(next(iter(result.values()))["boxes"].values()), {"fruit"})

    def test_empty_scene_has_empty_label_file(self):
        Image.new("RGB", (40, 30), "black").save(self.root / "empty.png")
        self.rows.append({"image": "empty.png", "split": "train", "category": "",
                          "xmin": "", "ymin": "", "xmax": "", "ymax": "", "group": "train_session"})
        self.write_manifest()
        output = prepare_data([self.manifest], self.root / "prepared")
        labels = [p.read_text() for p in (output / "detection/labels/train").glob("*.txt")]
        self.assertEqual(labels.count(""), 1)

    def test_unknown_validation_class_is_rejected(self):
        folder = self.root / "val/butter"
        folder.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(folder / "image.png")
        with self.assertRaisesRegex(ValueError, "missing from training"):
            ProductDataset(self.root / "val", classes=["apple", "milk"])

    def test_crop_bounds(self):
        self.assertEqual(crop_box((-2, -2, 15, 15), 10, 10, 0.1), (0, 0, 10, 10))
        with self.assertRaises(ValueError):
            crop_box((2, 2, 1, 1), 10, 10)

    def test_pipeline_multiple_crops_and_unknown(self):
        pipeline = self.make_pipeline([[0, 0, 20, 20], [20, 0, 40, 20]])
        result = pipeline.predict(Image.new("RGB", (40, 30)))
        self.assertEqual([item["category"] for item in result], ["milk", "milk"])
        json.dumps(result)
        pipeline.classifier = FakeClassifier(confident=False)
        self.assertTrue(all(item["category"] == "unknown" for item in pipeline.predict(Image.new("RGB", (40, 30)))))

    def test_pipeline_empty_detections(self):
        pipeline = self.make_pipeline([])
        self.assertEqual(pipeline.predict(Image.new("RGB", (40, 30))), [])

    def test_eval_transform_is_repeatable(self):
        image = Image.new("RGB", (15, 40), "red")
        self.assertTrue(torch.equal(image_transform()(image), image_transform()(image)))

    def test_metrics(self):
        result = classification_metrics(torch.tensor([[2, 1], [0, 1]]), ["apple", "milk"])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)

    def test_paths_are_relative_to_config(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/training.yaml")
        self.assertTrue(Path(config["classifier"]["data"]).is_absolute())
        self.assertTrue(Path(config["segmenter"]["weights"]).is_absolute())

    def make_pipeline(self, boxes):
        weights = self.root / "fake.pt"
        weights.touch()
        pipeline = RGBPipeline({"device": "cpu", "classifier": {"checkpoint": None},
                                "prediction": {**self.settings, "segmentation_weights": str(weights)}})
        pipeline.model = FakeDetector(boxes)
        pipeline.classifier = FakeClassifier()
        pipeline.classes = ["apple", "milk"]
        return pipeline


#------------------------------------------------------------
# RUN THE TESTS
#------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
