#------------------------------------------------------------
# REAL ARCHITECTURE TESTS WITHOUT WEIGHT DOWNLOADS OR TRAINING
#------------------------------------------------------------
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from production.vit import build_classifier, initialize_classifier, load_classifier


class TinyClassifier(torch.nn.Module):
    """Stand in for ViT when testing checkpoint bookkeeping."""

    def __init__(self, classes):
        super().__init__()
        self.body = torch.nn.Linear(3, 4)
        self.heads = torch.nn.Module()
        self.heads.head = torch.nn.Linear(4, classes)


class ModelTests(unittest.TestCase):
    """Check architecture compatibility and checkpoint category handling."""

    def test_real_vit_forward_and_gradients_without_weight_updates(self):
        torch.set_num_threads(2)
        model = build_classifier(3, pretrained=False)
        # Backward checks gradient flow only. There is no optimizer or learning.
        model.eval()
        images = torch.zeros(1, 3, 224, 224)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (1, 3))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertIsNotNone(model.heads.head.weight.grad)
        self.assertIsNotNone(model.conv_proj.weight.grad)
        self.assertTrue(torch.isfinite(model.heads.head.weight.grad).all())

    def test_real_yolo26_forward_without_pretrained_download(self):
        from ultralytics import YOLO
        model = YOLO("yolo26l.yaml")
        results = model.predict(Image.new("RGB", (64, 64)), imgsz=64, device="cpu", verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(tuple(results[0].boxes.xyxy.shape[1:]), (4,))

    def test_checkpoint_load_and_category_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            old_model = TinyClassifier(2)
            with torch.no_grad():
                old_model.heads.head.weight[0].fill_(2)
                old_model.heads.head.weight[1].fill_(5)
            torch.save({"format_version": 1, "architecture": "vit_l_32", "classes": ["apple", "milk"],
                        "model": old_model.state_dict()}, path)
            with patch("production.vit.build_classifier", side_effect=lambda n, pretrained: TinyClassifier(n)):
                loaded, classes = load_classifier(path, torch.device("cpu"))
                self.assertEqual(classes, ["apple", "milk"])
                self.assertFalse(loaded.training)
                transferred = initialize_classifier(["milk", "butter", "apple"], path)
            self.assertTrue(torch.equal(transferred.heads.head.weight[0], old_model.heads.head.weight[1]))
            self.assertTrue(torch.equal(transferred.heads.head.weight[2], old_model.heads.head.weight[0]))
            self.assertTrue(torch.equal(transferred.body.weight, old_model.body.weight))


#------------------------------------------------------------
# RUN THE TESTS
#------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
