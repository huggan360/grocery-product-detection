#------------------------------------------------------------
# CHECK SCRIPT CONNECTIONS WITHOUT RUNNING TRAINING
#------------------------------------------------------------
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from utils.common import load_config
from utils.import_grocery_store import import_grocery_store
from training.yolo import train_segmenter


class EntryPointTests(unittest.TestCase):
    """Check output files and training arguments using small stand-in models."""

    def test_detector_training_options_without_calling_real_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset.yaml"
            data.write_text("names: [product]\n")
            config = load_config(Path(__file__).resolve().parents[1] / "configs/training.yaml")
            config["device"] = "cpu"
            fake = MagicMock()
            fake.task = "detect"
            fake.trainer = SimpleNamespace(best=root / "best.pt")
            with patch("ultralytics.YOLO", return_value=fake):
                result = train_segmenter(str(data), "fake.pt", root / "output", device="cpu")
            self.assertEqual(result, root / "best.pt")
            self.assertTrue(fake.train.call_args.kwargs["single_cls"])
            self.assertEqual(fake.train.call_args.kwargs["device"], "cpu")

    def test_swedish_import_preserves_splits_and_coarse_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "classes.csv").write_text(
                "Class Name (str),Class ID (int),Coarse Class Name (str),Coarse Class ID (int)\n"
                "Granny-Smith,0,Apple,0\nRoyal-Gala,1,Apple,0\nMilk,2,Milk,1\n")
            for index, split in enumerate(("train", "val", "test")):
                (source / split).mkdir()
                Image.new("RGB", (10, 10), (index * 40, 0, 0)).save(source / split / "apple.jpg")
                Image.new("RGB", (10, 10), (index * 40, 100, 0)).save(source / split / "milk.jpg")
                (source / f"{split}.txt").write_text(f"{split}/apple.jpg, 1, 0\n{split}/milk.jpg, 2, 1\n")
            result = import_grocery_store(source, root / "prepared")
            self.assertEqual(result["total_images"], 6)
            self.assertEqual(result["classes"], ["apple", "milk"])
            self.assertEqual(result["counts"]["test"], {"apple": 1, "milk": 1})


#------------------------------------------------------------
# RUN THE TESTS
#------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
