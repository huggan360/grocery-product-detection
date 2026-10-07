#------------------------------------------------------------
# COLLABORATION AND EXPORT TESTS WITHOUT CAMERAS OR TRAINING
#------------------------------------------------------------
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from annotation_tool.app import create_app
from annotation_tool.settings import read_settings
from annotation_tool.store import StoreError


class ManualPredictor:
    """Keep tests offline and independent of trained checkpoints."""
    available = False
    message = "Manual test mode"


class FakePredictor:
    """Return one known product box to test background inference wiring."""
    available = True
    message = "Synthetic suggestion test"

    def predict(self, image):
        """Supply a fixed result without performing model training."""
        return [{"box_xyxy": [4, 5, 40, 50], "category": "milk",
                 "classification_confidence": 0.9, "detection_confidence": 0.8}]


class WorkspaceTests(unittest.TestCase):
    """Exercise the actual HTTP API, SQLite transactions and file export."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = read_settings(Path(__file__).resolve().parents[1] / "config.yaml")
        self.settings["storage"] = str(self.root / "workspace")
        self.app = create_app(self.settings, ManualPredictor())
        self.store = self.app.state.store
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.alice = self.client.post("/api/editors", json={"name": "Alice"}).json()["token"]
        self.bob = self.client.post("/api/editors", json={"name": "Bob"}).json()["token"]

    def headers(self, token=None):
        """Use Alice by default, or another collaborator when requested."""
        return {"X-Session": token or self.alice}

    def upload(self, collection="session-a", split="train", colour="red"):
        """Upload a real tiny PNG through the same route as the browser."""
        buffer = io.BytesIO()
        Image.new("RGB", (100, 80), colour).save(buffer, format="PNG")
        response = self.client.post("/api/upload", headers=self.headers(),
            data={"shelf": "Top shelf", "collection": collection, "split": split},
            files={"files": ("shelf.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["errors"])
        return response.json()["ids"][0]

    def claim(self, image_id, token=None):
        """Acquire one edit lease through the public API."""
        return self.client.post(f"/api/images/{image_id}/claim", headers=self.headers(token))

    def save(self, image_id, revision=0, token=None, reviewed=False, boxes=None, empty_confirmed=False):
        """Submit an annotation using a known revision."""
        if boxes is None:
            boxes = [{"id": "box-a", "category": "milk", "xyxy": [10, 8, 60, 70]}]
        return self.client.put(f"/api/images/{image_id}/annotations", headers=self.headers(token),
            json={"revision": revision, "boxes": boxes, "notes": "Test observation", "reviewed": reviewed,
                  "empty_confirmed": empty_confirmed})

    def test_home_and_manual_status(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Saved &amp; edited", response.text)
        data = self.client.get("/api/workspace", headers=self.headers()).json()
        self.assertFalse(data["model_available"])
        self.assertEqual(data["cameras"], [])
        self.assertIn("Alice", data["editors"])

    def test_two_people_cannot_claim_the_same_image(self):
        image_id = self.upload()
        self.assertEqual(self.claim(image_id).status_code, 200)
        self.assertEqual(self.claim(image_id, self.bob).status_code, 409)
        workspace = self.client.get("/api/workspace", headers=self.headers(self.bob)).json()
        self.assertEqual(workspace["images"][0]["locked_by"], "Alice")
        self.assertNotIn("lock_token", workspace["images"][0])
        self.assertFalse(workspace["images"][0]["mine"])

    def test_concurrent_claim_has_exactly_one_winner(self):
        image_id = self.upload()
        barrier = threading.Barrier(2)

        def attempt(token):
            """Start both claims together to exercise the database transaction."""
            barrier.wait()
            try:
                self.store.claim(token, image_id)
                return "claimed"
            except StoreError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(attempt, [self.alice, self.bob]))
        self.assertCountEqual(results, ["claimed", "conflict"])

    def test_claim_next_assigns_different_images(self):
        first, second = self.upload(), self.upload(colour="blue")
        alice = self.client.post("/api/claim-next", headers=self.headers()).json()
        bob = self.client.post("/api/claim-next", headers=self.headers(self.bob)).json()
        self.assertEqual({alice["id"], bob["id"]}, {first, second})

    def test_drafts_persist_and_stale_revisions_fail(self):
        image_id = self.upload()
        self.claim(image_id)
        response = self.save(image_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["revision"], 1)
        self.assertEqual(response.json()["state"], "pending")
        self.assertEqual(self.save(image_id, revision=0).status_code, 409)
        self.client.post(f"/api/images/{image_id}/release", headers=self.headers())
        reopened = self.claim(image_id, self.bob).json()
        self.assertEqual(reopened["boxes"][0]["category"], "milk")
        self.assertEqual(reopened["notes"], "Test observation")

    def test_expired_editor_cannot_overwrite_new_editor(self):
        image_id = self.upload()
        self.claim(image_id)
        with self.store.connect(write=True) as connection:
            connection.execute("UPDATE images SET lock_until = ? WHERE id = ?", (time.time() - 1, image_id))
        self.assertEqual(self.claim(image_id, self.bob).status_code, 200)
        self.assertEqual(self.save(image_id, token=self.alice).status_code, 409)
        self.assertEqual(self.save(image_id, token=self.bob).status_code, 200)

    def test_heartbeat_renews_and_release_unlocks(self):
        image_id = self.upload()
        self.claim(image_id)
        response = self.client.post(f"/api/images/{image_id}/heartbeat", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.post(f"/api/images/{image_id}/heartbeat", headers=self.headers(self.bob)).status_code, 409)
        self.client.post(f"/api/images/{image_id}/release", headers=self.headers())
        self.assertEqual(self.claim(image_id, self.bob).status_code, 200)

    def test_review_requires_categories_and_empty_confirmation(self):
        image_id = self.upload()
        self.claim(image_id)
        unknown = [{"id": "a", "category": "unknown", "xyxy": [0, 0, 20, 20]}]
        self.assertEqual(self.save(image_id, reviewed=True, boxes=unknown).status_code, 400)
        self.assertEqual(self.save(image_id, reviewed=True, boxes=[]).status_code, 400)
        self.assertEqual(self.save(image_id, reviewed=True, boxes=[], empty_confirmed=True).status_code, 200)

    def test_invalid_boxes_do_not_change_saved_state(self):
        image_id = self.upload()
        self.claim(image_id)
        for coordinates in ([0, 0, 200, 10], [20, 20, 10, 10], [-1, 0, 20, 20], [True, 0, 20, 20]):
            response = self.save(image_id, boxes=[{"id": "a", "category": "milk", "xyxy": coordinates}])
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.store.get_image(image_id)["revision"], 0)

    def test_session_cannot_cross_splits(self):
        self.upload()
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="PNG")
        response = self.client.post("/api/upload", headers=self.headers(),
            data={"shelf": "Bottom", "collection": "session-a", "split": "val"},
            files={"files": ("other.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 409)

    def test_category_creation_is_shared_and_safe(self):
        response = self.client.post("/api/categories", headers=self.headers(), json={"name": "oat-milk"})
        self.assertEqual(response.status_code, 200)
        data = self.client.get("/api/workspace", headers=self.headers(self.bob)).json()
        self.assertIn("oat-milk", data["categories"])
        for name in ("../escape", "unknown", "Milk"):
            self.assertEqual(self.client.post("/api/categories", headers=self.headers(), json={"name": name}).status_code, 400)

    def test_export_contains_only_reviewed_unlocked_images(self):
        reviewed_id, draft_id = self.upload(), self.upload(colour="blue")
        self.claim(reviewed_id)
        self.assertEqual(self.save(reviewed_id, reviewed=True).status_code, 200)
        self.assertEqual(self.client.get("/api/export", headers=self.headers()).status_code, 400)
        self.client.post(f"/api/images/{reviewed_id}/release", headers=self.headers())
        response = self.client.get("/api/export", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            self.assertIn(f"detection/images/train/{reviewed_id}.png", names)
            self.assertNotIn(f"detection/images/train/{draft_id}.png", names)
            self.assertIn(f"classification/train/milk/{reviewed_id}_0.png", names)
            label = archive.read(f"detection/labels/train/{reviewed_id}.txt").decode().split()
            self.assertEqual([float(value) for value in label], [0, .35, .4875, .5, .775])
            metadata = json.loads(archive.read("annotations.json"))
            self.assertEqual(metadata[0]["edited_by"], "Alice")
            self.assertIn("session-a", archive.read("annotations.csv").decode())

    def test_editing_reviewed_image_returns_it_to_pending(self):
        image_id = self.upload()
        self.claim(image_id)
        self.save(image_id, reviewed=True)
        response = self.save(image_id, revision=1, reviewed=False)
        self.assertEqual(response.json()["state"], "pending")

    def test_export_round_trip_through_existing_data_preparation(self):
        """Verify that exported annotations can feed the project's real converter."""
        from utils.prepare_data import prepare_data
        from training.data import ProductDataset
        for split, colour in (("train", "red"), ("val", "blue")):
            image_id = self.upload(collection=f"session-{split}", split=split, colour=colour)
            self.claim(image_id)
            boxes = [{"id": "milk", "category": "milk", "xyxy": [0, 0, 40, 60]},
                     {"id": "apple", "category": "apple", "xyxy": [50, 10, 90, 70]}]
            self.assertEqual(self.save(image_id, reviewed=True, boxes=boxes).status_code, 200)
            self.client.post(f"/api/images/{image_id}/release", headers=self.headers())
        response = self.client.get("/api/export", headers=self.headers())
        extracted = self.root / "exported"
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(extracted)
        prepared = prepare_data([extracted / "annotations.csv"], self.root / "prepared")
        dataset = ProductDataset(prepared / "classification/train", training=True)
        self.assertEqual(dataset.classes, ["apple", "milk"])
        self.assertEqual(tuple(dataset[0][0].shape), (3, 224, 224))

    def test_store_survives_reopening_without_losing_annotations(self):
        """A server restart must retain reviewed work and editing history."""
        from annotation_tool.store import Store
        image_id = self.upload()
        self.claim(image_id)
        self.save(image_id, reviewed=True)
        reopened = Store(self.settings["storage"], self.settings["categories"])
        image = reopened.get_image(image_id)
        self.assertEqual(image["revision"], 1)
        self.assertEqual(image["state"], "reviewed")
        with reopened.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM history WHERE image_id = ?", (image_id,)).fetchone()[0], 1)

    def test_bad_upload_is_reported_and_no_camera_capture_is_clear(self):
        response = self.client.post("/api/upload", headers=self.headers(),
            data={"shelf": "Top", "collection": "session-a", "split": "train"},
            files={"files": ("bad.png", b"not an image", "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ids"], [])
        self.assertEqual(len(response.json()["errors"]), 1)
        self.assertEqual(self.client.post("/api/capture", headers=self.headers(),
                                         json={"collection": "session-a", "split": "train"}).status_code, 400)

    def test_image_pixels_and_thumbnail_exist(self):
        image_id = self.upload()
        for suffix in ("pixels", "thumbnail"):
            response = self.client.get(f"/api/images/{image_id}/{suffix}")
            self.assertEqual(response.status_code, 200)
            Image.open(io.BytesIO(response.content)).verify()

    def test_background_suggestions_and_late_results_do_not_overwrite_edits(self):
        settings = self.settings | {"storage": str(self.root / "model-workspace")}
        app = create_app(settings, FakePredictor())
        with TestClient(app) as client:
            token = client.post("/api/editors", json={"name": "Hugo"}).json()["token"]
            headers = {"X-Session": token}
            buffer = io.BytesIO()
            Image.new("RGB", (100, 80)).save(buffer, format="PNG")
            image_id = client.post("/api/upload", headers=headers,
                data={"shelf": "Top", "collection": "one", "split": "train"},
                files={"files": ("test.png", buffer.getvalue(), "image/png")}).json()["ids"][0]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                row = app.state.store.get_image(image_id)
                if row["prediction_state"] == "ready":
                    break
                time.sleep(.02)
            self.assertEqual(row["prediction_state"], "ready")
            self.assertEqual(row["boxes"][0]["category"], "milk")
            app.state.store.claim(token, image_id)
            boxes = [{"id": "human", "category": "apple", "xyxy": [1, 1, 20, 30]}]
            app.state.store.save(image_id, token, 0, boxes, "human correction")
            app.state.store.finish_job(image_id, FakePredictor().predict(None))
            self.assertEqual(app.state.store.get_image(image_id)["boxes"], boxes)

    def test_capture_records_each_shelf_and_partial_failure(self):
        settings = self.settings | {"storage": str(self.root / "capture-workspace"), "cameras": [
            {"id": "top", "name": "Top", "enabled": True},
            {"id": "bottom", "name": "Bottom", "enabled": True}]}
        app = create_app(settings, ManualPredictor())
        with TestClient(app) as client:
            token = client.post("/api/editors", json={"name": "Hugo"}).json()["token"]
            with patch("annotation_tool.app.capture_camera", side_effect=[Image.new("RGB", (80, 60)), RuntimeError("Camera unplugged")]):
                response = client.post("/api/capture", headers={"X-Session": token}, json={"collection": "capture-one", "split": "val"})
            self.assertEqual(len(response.json()["ids"]), 1)
            self.assertEqual(response.json()["errors"][0]["message"], "Camera unplugged")
            row = app.state.store.get_image(response.json()["ids"][0])
            self.assertEqual((row["shelf"], row["split"]), ("Top", "val"))


#------------------------------------------------------------
# RUN THE TESTS WITHOUT STARTING A PUBLIC SERVER
#------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
