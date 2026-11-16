#------------------------------------------------------------
# VIDEO UPLOAD, ZONES AND REVIEW THROUGH THE HTTP API (NO MODEL WEIGHTS)
#------------------------------------------------------------
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from annotation_tool.app import create_app
from annotation_tool.settings import read_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from test_video import VideoHandoffTests, temporary_config  # noqa: E402


class FakeVideoPipeline(VideoHandoffTests.FakePipeline):
    message = "Fake video pipeline"

    def run(self, source, directory, camera, zones, event_id, offset=0.0, progress=None):
        from production.video import write_json
        result = super().run(source, directory, camera, zones, event_id, offset, progress)
        from PIL import Image
        Image.new("RGB", (8, 8), "white").save(Path(directory) / "track-1.jpg")
        result["tracks"][0]["crop_file"] = "track-1.jpg"
        result.update(offset_seconds=offset, zones=zones, warnings=[], sample_fps=5,
                      video={"duration": 1.0, "width": 640, "height": 480}, backend="fake")
        write_json(Path(directory) / "result.json", result)
        return result


class VideoReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        settings = read_settings(Path(__file__).resolve().parents[1] / "config.yaml")
        settings.update(storage=str(root / "workspace"), video_config=str(temporary_config(root)))
        app = create_app(settings)
        app.state.videos.pipeline = FakeVideoPipeline()
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        token = self.client.post("/api/editors", json={"name": "Alice"}).json()["token"]
        self.headers = {"X-Session": token}

    def wait(self, video_id):
        for _ in range(50):
            detail = self.client.get(f"/api/videos/{video_id}").json()
            if detail["meta"]["status"] in ("done", "failed"):
                return detail
            time.sleep(0.1)
        self.fail("Video was not processed.")

    def test_upload_zones_reanalyse_and_review(self):
        zones = {"inside": [0, 0, 0.4, 1], "outside": [0.6, 0, 1, 1]}
        self.assertEqual(self.client.put("/api/zones/fridge", json=zones, headers=self.headers).status_code, 200)
        response = self.client.post("/api/videos", headers=self.headers, data={"camera": "fridge"},
                                    files={"file": ("clip.mp4", b"fake video", "video/mp4")})
        self.assertEqual(response.status_code, 200, response.text)
        video_id = response.json()["id"]
        detail = self.wait(video_id)
        self.assertEqual(detail["meta"]["status"], "done", detail["meta"])
        self.assertEqual(detail["result"]["movements"][0]["direction"], "out")
        # Swapping the zones flips the direction without running the models again.
        swapped = {"inside": zones["outside"], "outside": zones["inside"]}
        self.client.put("/api/zones/fridge", json=swapped, headers=self.headers)
        result = self.client.post(f"/api/videos/{video_id}/reanalyse", headers=self.headers).json()
        self.assertEqual(result["movements"][0]["direction"], "in")
        review = {"tracks": {"1": {"category": "milk", "direction": "auto", "ignored": False}}, "reviewed": True}
        self.assertEqual(self.client.put(f"/api/videos/{video_id}/review", json=review,
                                         headers=self.headers).status_code, 200)
        self.assertTrue(self.client.get("/api/videos").json()["items"][0]["reviewed"])
        export = self.client.get("/api/videos/export", headers=self.headers)
        self.assertEqual(export.status_code, 200)
        import io
        import zipfile
        names = zipfile.ZipFile(io.BytesIO(export.content)).namelist()
        self.assertIn(f"classification/milk/{video_id}-1.jpg", names)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_rejects_bad_input(self):
        self.assertEqual(self.client.post("/api/videos", data={"camera": "fridge"},
                         files={"file": ("clip.mp4", b"x", "video/mp4")}).status_code, 422)
        bad = self.client.post("/api/videos", headers=self.headers, data={"camera": "garage"},
                               files={"file": ("clip.mp4", b"x", "video/mp4")})
        self.assertEqual(bad.status_code, 400)
        overlap = {"inside": [0, 0, 0.7, 1], "outside": [0.6, 0, 1, 1]}
        self.assertEqual(self.client.put("/api/zones/fridge", json=overlap, headers=self.headers).status_code, 400)
        self.assertEqual(self.client.get("/api/videos/../../etc/files/video.mp4").status_code, 404)

    def test_bulk_reclassification_backs_up_review_and_replaces_predictions(self):
        response=self.client.post('/api/videos',headers=self.headers,data={'camera':'fridge'},
                                  files={'file':('clip.mp4',b'fake video','video/mp4')})
        video_id=response.json()['id'];self.wait(video_id)
        review={'tracks':{'1':{'category':'milk','direction':'out','ignored':False}},'reviewed':True}
        self.client.put(f'/api/videos/{video_id}/review',json=review,headers=self.headers)
        folder=self.client.app.state.videos.folder(video_id)
        old_result=(folder/'result.json').read_bytes()
        old_review=(folder/'review.json').read_bytes()
        response=self.client.post('/api/videos/reclassify',json={'all_videos':True},headers=self.headers)
        self.assertEqual(response.status_code,200,response.text)
        batch=response.json();self.assertEqual(batch['queued'],1)
        backup=Path(batch['backup_directory'])/video_id
        self.assertEqual((backup/'result.json').read_bytes(),old_result)
        self.assertEqual((backup/'review.json').read_bytes(),old_review)
        self.assertFalse((folder/'review.json').exists())
        self.assertEqual(self.wait(video_id)['meta']['status'],'done')
        self.assertFalse(self.client.get('/api/videos').json()['items'][0]['reviewed'])

    def test_bulk_reclassification_requires_explicit_all_and_skips_active(self):
        self.assertEqual(self.client.post('/api/videos/reclassify',json={},headers=self.headers).status_code,400)
        self.assertEqual(self.client.post('/api/videos/reclassify',json={'all_videos':True}).status_code,422)
        response=self.client.post('/api/videos',headers=self.headers,data={'camera':'fridge'},
                                  files={'file':('clip.mp4',b'fake video','video/mp4')})
        video_id=response.json()['id'];self.wait(video_id)
        self.client.app.state.videos.update(video_id,status='running')
        result=self.client.post('/api/videos/reclassify',json={'all_videos':True},headers=self.headers).json()
        self.assertEqual(result['queued'],0)
        self.assertEqual(result['skipped_ids'],[video_id])


if __name__ == "__main__":
    unittest.main()
