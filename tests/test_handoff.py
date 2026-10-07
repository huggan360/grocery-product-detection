#------------------------------------------------------------
# CHECK RETENTION AND REVIEW IMPORT WITHOUT RUNNING MODELS
#------------------------------------------------------------
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from production.run import purge_masks, write_json
from utils.import_acquisition import import_acquisition
from annotation_tool.store import Store


class HandoffTests(unittest.TestCase):
    def test_retention_keeps_latest_three_and_acquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(6):
                folder = root / str(index)
                (folder / "masks").mkdir(parents=True)
                (folder / "masks/one.png").write_bytes(b"sample")
                write_json(folder / ".ml-run.json", {"schema_version": 1,
                    "mode": "acquisition" if index == 0 else "production",
                    "completed_at": f"2026-10-05T12:00:0{index}+00:00", "result": "result.json"})
                write_json(folder / "result.json", {"objects": [{"category": "milk",
                    "polygon": [[0, 0]], "mask_path": "old", "crop_path": "old"}]})
            purge_masks(root, 3)
            self.assertTrue((root / "0/masks").exists())
            self.assertFalse((root / "1/masks").exists())
            self.assertFalse((root / "2/masks").exists())
            self.assertTrue(all((root / str(i) / "masks").exists() for i in (3, 4, 5)))
            self.assertEqual(json.loads((root / "1/result.json").read_text())["objects"],
                             [{"category": "milk"}])

    def test_import_keeps_event_links_masks_and_existing_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (30, 30)).save(image)
            manifest = root / "acquisition.json"
            write_json(manifest, {"schema_version": 1, "records": [{
                "run_id": "run1", "input": {"capture_id": "cap1", "event_id": "door1",
                    "captured_at": "2026-10-05T12:00:00+00:00", "sensor": "shelf1",
                    "image_path": str(image)}, "result": {"mode": "acquisition", "capture_id": "cap1",
                    "objects": [{"category": "milk", "box_xyxy": [1, 1, 20, 20],
                        "polygon": [[1, 1], [20, 1], [20, 20], [1, 20]]}]}}]})
            settings = {"storage": root / "review", "categories": ["milk"], "lease_seconds": 90}
            self.assertEqual(import_acquisition(manifest, settings), 1)
            self.assertEqual(import_acquisition(manifest, settings), 0)
            store = Store(settings["storage"], settings["categories"])
            with store.connect() as connection:
                row = connection.execute("SELECT * FROM images").fetchone()
                self.assertEqual(row["prediction_state"], "ready")
                self.assertEqual(json.loads(row["notes"])["event_id"], "door1")
                self.assertEqual(json.loads(row["boxes"])[0]["category"], "milk")


if __name__ == "__main__":
    unittest.main()
