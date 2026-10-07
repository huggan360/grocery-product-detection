#------------------------------------------------------------
# ZONES, HAILO DECODING AND THE VIDEO HANDOFF WITHOUT MODEL WEIGHTS
#------------------------------------------------------------
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from production.hailo import decode_yolo26, parse_nms, suppress_duplicates
from production.run import run_video_request
from production.video import load_config, movements, purge_video_runs, save_zones, write_json
from production.zones import crossing_events, validate_zones

ROOT = Path(__file__).resolve().parents[1]
ZONES = {"inside": [0.0, 0.0, 0.4, 1.0], "outside": [0.6, 0.0, 1.0, 1.0]}


def walk(xs, step=0.2):
    """A track whose box centre moves through the given x positions."""
    return [{"time": i * step, "box": [x - 0.05, 0.4, x + 0.05, 0.6]} for i, x in enumerate(xs)]


def temporary_config(root):
    """A copy of configs/video.yaml that writes only inside the test folder."""
    config = yaml.safe_load((ROOT / "configs/video.yaml").read_text())
    config.update(models_directory=str(ROOT / "models"), acquisition_directory=str(root / "acquisition"),
                  production_directory=str(root / "runs"), zones_file=str(root / "acquisition/zones.json"),
                  backend="torch")
    path = root / "video.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class ZoneTests(unittest.TestCase):
    def test_inside_to_outside_is_out_and_reverse_is_in(self):
        self.assertEqual([e["direction"] for e in crossing_events(walk([0.2, 0.2, 0.5, 0.8, 0.8]), ZONES)], ["out"])
        self.assertEqual([e["direction"] for e in crossing_events(walk([0.8, 0.8, 0.5, 0.2, 0.2]), ZONES)], ["in"])

    def test_one_frame_in_a_zone_or_a_long_gap_is_not_a_movement(self):
        self.assertEqual(crossing_events(walk([0.2, 0.2, 0.8]), ZONES), [])
        gap = walk([0.2, 0.2]) + [{"time": 5.0, "box": [0.75, 0.4, 0.85, 0.6]},
                                  {"time": 5.2, "box": [0.75, 0.4, 0.85, 0.6]}]
        self.assertEqual(crossing_events(gap, ZONES), [])

    def test_zones_are_validated(self):
        with self.assertRaises(ValueError):
            validate_zones({"inside": [0, 0, 0.7, 1], "outside": [0.6, 0, 1, 1]})
        with self.assertRaises(ValueError):
            validate_zones({"inside": [0, 0, 1.2, 1], "outside": None})

    def test_saved_zones_override_the_yaml_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = temporary_config(Path(temporary))
            save_zones(load_config(path), "door", ZONES)
            self.assertEqual(load_config(path)["cameras"]["door"], ZONES)
            self.assertIsNone(load_config(path)["cameras"]["fridge"]["inside"])


class HailoDecodingTests(unittest.TestCase):
    def test_yolo26_raw_head_becomes_pixel_boxes(self):
        outputs = {}
        for size in (80, 40, 20):
            outputs[f"box{size}"] = np.zeros((1, size, size, 4), np.float32)
            outputs[f"cls{size}"] = np.full((1, size, size, 80), -10, np.float32)
        # One confident "bottle" (class 39) at stride 16, cell (x=10, y=5), 2 cells each way.
        outputs["box40"][0, 5, 10] = [2, 2, 2, 2]
        outputs["cls40"][0, 5, 10, 39] = 4
        boxes = decode_yolo26(outputs, 640, 640, 80)
        best = boxes[boxes[:, 4].argmax()]
        np.testing.assert_allclose(best[:4], [136, 56, 200, 120])
        self.assertEqual(int(best[5]), 39)
        self.assertGreater(best[4], 0.98)

    def test_on_chip_nms_output_and_duplicate_suppression(self):
        output = [np.zeros((0, 5)), np.array([[0.1, 0.2, 0.5, 0.6, 0.9], [0.1, 0.2, 0.5, 0.61, 0.8]])]
        boxes = parse_nms(output, 100, 200)
        np.testing.assert_allclose(boxes[0], [20, 20, 60, 100, 0.9, 1], rtol=1e-5)
        self.assertEqual(len(suppress_duplicates(boxes)), 1)


class VideoHandoffTests(unittest.TestCase):
    class FakePipeline:
        """Writes a result like VideoPipeline without loading weights."""

        def run(self, source, directory, camera, zones, event_id, offset=0.0, progress=None):
            from production.zones import analyse_tracks
            tracks = analyse_tracks([{"id": "1", "category": "milk", "classification_confidence": 0.9,
                                      "observations": walk([0.2, 0.2, 0.5, 0.8, 0.8]), "events": [],
                                      "ignored": False}], zones, {})
            for event in tracks[0]["events"]:
                event.update(event_start=offset + event["start"], event_end=offset + event["end"])
            return {"schema_version": 2, "event_id": event_id, "camera": camera, "tracks": tracks,
                    "movements": movements(tracks), "completed_at": "2026-10-07T12:00:00+00:00"}

    def test_schema_2_request_reports_movements_and_keeps_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            save_zones(load_config(config), "fridge", ZONES)
            clip = root / "clip.mp4"
            clip.write_bytes(b"video")
            request = root / "request.json"
            write_json(request, {"schema_version": 2, "capture_id": "c1", "event_id": "door-1",
                                 "captured_at": "2026-10-07T12:00:00+00:00", "camera": "fridge",
                                 "video_path": str(clip), "offset_seconds": 2.0})
            output = run_video_request(request, "c1", root / "run/result.json", config, "production",
                                       managed_retention=True, pipeline=self.FakePipeline())
            result = json.loads(output.read_text())
            self.assertEqual((result["capture_id"], result["event_id"], result["mode"]), ("c1", "door-1", "production"))
            self.assertEqual([(m["direction"], m["category"]) for m in result["movements"]], [("out", "milk")])
            self.assertGreaterEqual(result["movements"][0]["start"], 2.0)
            with self.assertRaises(ValueError):
                run_video_request(request, "other-id", root / "run2/result.json", config, "production",
                                  pipeline=self.FakePipeline())

    def test_test_mode_publishes_clip_and_prediction_to_the_review_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            save_zones(load_config(config), "door", ZONES)
            clip = root / "clip.mp4"
            clip.write_bytes(b"video")
            request = root / "request.json"
            write_json(request, {"schema_version": 2, "capture_id": "c9", "event_id": "door-9",
                                 "captured_at": "2026-10-07T12:00:00+00:00", "camera": "door",
                                 "video_path": str(clip)})
            run = root / "run"
            run.mkdir()
            (run / "track-1.jpg").write_bytes(b"crop")
            run_video_request(request, "c9", run / "result.json", config, "test",
                              managed_retention=True, pipeline=self.FakePipeline())
            published = [p for p in (root / "acquisition/videos").iterdir()]
            self.assertEqual(len(published), 1)
            meta = json.loads((published[0] / "meta.json").read_text())
            self.assertEqual((meta["status"], meta["event_id"], meta["camera"]), ("done", "door-9", "door"))
            self.assertEqual(json.loads((published[0] / "result.json").read_text())["mode"], "test")
            self.assertTrue((published[0] / "track-1.jpg").is_file())
            self.assertEqual((published[0] / "original.mp4").read_bytes(), b"video")

    def test_standalone_retention_keeps_latest_three_video_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(5):
                folder = root / str(index)
                folder.mkdir()
                (folder / "video.mp4").write_bytes(b"v")
                (folder / "track-1.jpg").write_bytes(b"c")
                write_json(folder / ".ml-run.json", {"kind": "video", "mode": "production",
                           "completed_at": f"2026-10-07T12:00:0{index}+00:00", "result": "result.json"})
            purge_video_runs(root, 3)
            self.assertEqual([(root / str(i) / "video.mp4").exists() for i in range(5)],
                             [False, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
