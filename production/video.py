#------------------------------------------------------------
# VIDEO -> TRACKS -> ZONE CROSSINGS -> PRODUCT CLASSIFICATION
#------------------------------------------------------------
import json
import math
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from production.preprocessing import prepare_yolo_image
from production.zones import analyse_tracks, validate_zones


def now():
    """Write timestamps in UTC so clips can be linked to door events."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Publish a complete result in one step."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path, models_dir=None):
    """Resolve model and data folders relative to this YAML file."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key in ("acquisition_directory", "production_directory", "zones_file"):
        config[key] = str((path.parent / config[key]).resolve())
    root = Path(models_dir).resolve() if models_dir else (path.parent / config["models_directory"]).resolve()
    config["models_directory"] = str(root)
    for section, keys in (("detector", ("weights", "hef", "hef_labels")),
                          ("classifier", ("checkpoint",))):
        for key in keys:
            if config[section].get(key):
                config[section][key] = str(root / config[section][key])
    config["classifier"]["pretrained_directory"] = str(root / "vit/pretrained")
    if not 0 < config["tracking"]["sample_fps"] <= 60:
        raise ValueError("sample_fps must be greater than zero and at most 60.")
    # Zones drawn in the annotation tool override the defaults in this file.
    zones_file = Path(config["zones_file"])
    if zones_file.is_file():
        for camera, zones in json.loads(zones_file.read_text()).items():
            if camera in config["cameras"]:
                config["cameras"][camera].update(zones)
    for profile in config["cameras"].values():
        validate_zones(profile)
    return config


def save_zones(config, camera, zones):
    """Store zones for one camera; production and the reviewer both read this file."""
    if camera not in config["cameras"]:
        raise ValueError("Unknown camera.")
    zones = validate_zones(zones)
    path = Path(config["zones_file"])
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved[camera] = zones
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, saved)
    config["cameras"][camera].update(zones)
    return zones


#------------------------------------------------------------
# KEEP ORIGINAL VIDEO; MAKE A BROWSER-PLAYABLE WORKING COPY
#------------------------------------------------------------
def video_info(path):
    """Read video dimensions and timing without guessing a missing frame rate."""
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or frames < 1 or min(width, height) < 2:
            raise ValueError("This video cannot be decoded or has invalid timing.")
        return {"fps": fps, "frames": frames, "width": width, "height": height,
                "duration": frames / fps}
    finally:
        capture.release()


def browser_ready(source, info, settings):
    """H.264 4:2:0 no wider than the proxy plays in every browser as it is."""
    executable = shutil.which("ffprobe")
    if not executable or info["width"] > settings["proxy_width"]:
        return False
    probe = subprocess.run([executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt", "-of", "json", str(source)],
                           capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout or "{}").get("streams") or [{}]
    return streams[0].get("codec_name") == "h264" and streams[0].get("pix_fmt") in ("yuv420p", "yuvj420p")


def prepare_video(source, directory, config):
    """Use FFmpeg to make a seekable MP4, keeping the uploaded original intact."""
    source, directory = Path(source), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["video"]
    if source.stat().st_size > settings["max_upload_mb"] * 1024 * 1024:
        raise ValueError("Video exceeds the configured upload limit.")
    info = video_info(source)
    if info["duration"] > settings["max_duration_seconds"]:
        raise ValueError(f"Upload a clip of at most {settings['max_duration_seconds']} seconds.")
    if info["width"] * info["height"] > settings["max_pixels"]:
        raise ValueError("Video dimensions exceed the configured limit.")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg is required for video playback. Install ffmpeg on the server.")
    proxy = directory / "video.mp4"
    if browser_ready(source, info, settings):
        # Clips cut by the orchestrator are already H.264: copying saves the Pi's CPU.
        encode = ["-c:v", "copy"]
    else:
        scale = f"scale=w='min({int(settings['proxy_width'])},trunc(iw/2)*2)':h=-2"
        encode = ["-vf", scale, "-r", str(info["fps"]), "-c:v", "libx264",
                  "-preset", settings.get("proxy_preset", "veryfast"), "-crf", "20", "-pix_fmt", "yuv420p"]
    with (directory / "ffmpeg.log").open("w") as log:
        subprocess.run([executable, "-nostdin", "-y", "-i", str(source), "-map", "0:v:0", "-an",
                        *encode, "-movflags", "+faststart", str(proxy)],
                       stdout=log, stderr=log, check=True, timeout=600)
    cap = cv2.VideoCapture(str(proxy))
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Could not read the converted video.")
        if not cv2.imwrite(str(directory / "poster.jpg"), frame):
            raise OSError("Could not save the video preview.")
    finally:
        cap.release()
    return video_info(proxy)


def new_tracker(tracking, sample_fps):
    """ByteTrack works on boxes alone, so Hailo and PyTorch share the same tracker."""
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker
    settings = dict(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
                    new_track_thresh=0.25, match_thresh=0.8, fuse_score=True)
    settings.update(tracking.get("bytetrack", {}))
    # Keep a lost object alive for the same time window the zone logic allows.
    settings["track_buffer"] = max(1, math.ceil(tracking["max_gap_seconds"] * sample_fps))
    return BYTETracker(SimpleNamespace(**settings))


class VideoPipeline:
    """Reuse model weights, but start a fresh set of track IDs for each clip."""

    def __init__(self, config):
        from production.backends import choose_backend
        self.config = config
        self.backend = choose_backend(config.get("backend", "auto"))
        self.detector = self.classifier = None
        self.lock = threading.Lock()

    @property
    def message(self):
        runtime = "YOLO on Hailo-8" if self.backend == "hailo" else "YOLO on PyTorch"
        if self.config["classifier"].get("checkpoint"):
            return f"YOLO26m tracking + grocery ViT-Small on CPU ({runtime})."
        return f"YOLO26m tracking + ViT-Small ImageNet guesses on CPU, not grocery-trained ({runtime})."

    def release(self):
        """Give the Hailo-8 back so another process (e.g. the orchestrator) can use it."""
        if self.detector is not None and self.backend == "hailo":
            from production.hailo import release_device
            self.detector = None
            release_device()

    def run(self, source, directory, camera, zones, event_id, offset=0.0, progress=None):
        """Run the same video pipeline from the CLI and the annotation tool."""
        with self.lock:
            return self._run(source, directory, camera, validate_zones(zones), event_id,
                             offset, progress or (lambda *_: None))

    def _run(self, source, directory, camera, zones, event_id, offset, progress):
        from ultralytics.engine.results import Boxes
        from production.backends import build_detector
        from utils.common import choose_device

        config = self.config
        if camera not in config["cameras"] or not math.isfinite(offset) or offset < 0:
            raise ValueError("Choose a configured camera and a nonnegative event offset.")
        started = now()
        directory = Path(directory)
        progress(0, "Preparing video")
        info = prepare_video(source, directory, config)
        proxy = directory / "video.mp4"
        device = choose_device(config["device"])
        settings, tracking = config["detector"], config["tracking"]
        if self.detector is None:
            self.detector = build_detector(settings, self.backend, device)
        excluded = set(settings.get("excluded_labels", []))
        stride = max(1, round(info["fps"] / tracking["sample_fps"]))
        tracker = new_tracker(tracking, info["fps"] / stride)
        capture = cv2.VideoCapture(str(proxy))
        tracks, candidates = {}, {}
        frame_index, sampled = 0, 0
        crop_count = config["classifier"].get("crops_per_track", 3)
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                current = frame_index
                frame_index += 1
                if current % stride:
                    continue
                sampled += 1
                timestamp = current / info["fps"]
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enhanced = np.asarray(prepare_yolo_image(image, settings.get("preprocessing", True)))
                detections = self.detector.detect(enhanced)
                keep = [self.detector.names[int(label)] not in excluded for label in detections[:, 5]]
                detections = detections[np.asarray(keep, bool)] if len(detections) else detections
                for x1, y1, x2, y2, identity, score, label, _ in tracker.update(
                        Boxes(detections, (image.height, image.width)), enhanced):
                    key = str(int(identity))
                    x1, y1, x2, y2 = (float(value) for value in (x1, y1, x2, y2))
                    label = self.detector.names[int(label)]
                    normalized = [max(0., min(1., x1 / image.width)), max(0., min(1., y1 / image.height)),
                                  max(0., min(1., x2 / image.width)), max(0., min(1., y2 / image.height))]
                    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
                        continue
                    if key not in tracks:
                        if len(tracks) >= tracking["max_tracks"]:
                            raise ValueError("Too many tracks. Use a shorter object-movement clip.")
                        tracks[key] = {"id": key, "category": "unknown", "classification_confidence": None,
                            "label_source": "unclassified", "source_label": label,
                            "observations": [], "events": [], "ignored": False}
                        candidates[key] = []
                    tracks[key]["observations"].append({"frame": current, "time": timestamp,
                        "box": normalized, "detection_confidence": float(score)})
                    crop = image.crop((max(0, int(x1)), max(0, int(y1)),
                                       min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))))
                    if min(crop.size) < 2:
                        continue
                    crop.thumbnail((384, 384))
                    # Prefer confident, large and sharp views for classification.
                    sharpness = cv2.Laplacian(np.asarray(crop.convert("L")), cv2.CV_64F).var()
                    quality = score * math.sqrt(crop.width * crop.height) * math.log1p(sharpness)
                    nearby = next((i for i, candidate in enumerate(candidates[key])
                                   if abs(candidate[1] - timestamp) < 0.5), None)
                    candidate = (quality, timestamp, crop.copy())
                    if nearby is None:
                        candidates[key].append(candidate)
                    elif quality > candidates[key][nearby][0]:
                        candidates[key][nearby] = candidate
                    candidates[key].sort(key=lambda entry: entry[0], reverse=True)
                    candidates[key] = candidates[key][:crop_count]
                if sampled % 5 == 0:
                    progress(min(85, int(85 * frame_index / info["frames"])), "Tracking objects")
            if frame_index < max(1, info["frames"] - 2):
                raise ValueError("Video decoding stopped before the end; no partial result was accepted.")
        finally:
            capture.release()
        analysed = analyse_tracks(list(tracks.values()), zones, tracking)
        progress(87, "Classifying moving tracks")
        self._classify(analysed, candidates, directory, device)
        for track in analysed:
            for event in track["events"]:
                event["event_start"] = offset + event["start"]
                event["event_end"] = offset + event["end"]
        result = {"schema_version": 2, "event_id": event_id, "camera": camera,
                  "offset_seconds": offset, "started_at": started, "completed_at": now(),
                  "source": str(Path(source).resolve()), "video": info,
                  "sample_fps": info["fps"] / stride, "zones": zones,
                  "tracking_settings": tracking, "backend": self.backend,
                  "models": {"yolo": settings["hef"] if self.backend == "hailo" else settings["weights"],
                             "vit": config["classifier"].get("checkpoint") or "imagenet-pretrained"},
                  "warnings": (["Set both zones before interpreting in/out movements."]
                               if not all(zones.values()) else []), "tracks": analysed}
        result["movements"] = movements(analysed)
        write_json(directory / "result.json", result)
        progress(100, "Ready")
        return result

    def _classify(self, tracks, candidates, directory, device):
        """Average a few clear views of each moving object, not every video frame."""
        from production.backends import build_classifier

        settings = self.config["classifier"]
        moving = []
        for track in tracks:
            observations = track["observations"]
            xs = [(r["box"][0] + r["box"][2]) / 2 for r in observations]
            ys = [(r["box"][1] + r["box"][3]) / 2 for r in observations]
            displacement = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            track["movement"] = displacement
            crops = candidates[track["id"]]
            if crops:
                path = directory / f"track-{track['id']}.jpg"
                crops[0][2].save(path)
                track["crop_file"] = path.name
            if crops and (track["events"] or displacement >= self.config["tracking"]["minimum_movement"]):
                moving.append(track)
        if not moving:
            return
        if self.classifier is None:
            self.classifier = build_classifier(settings, device)
        classes, imagenet = self.classifier.classes, self.classifier.imagenet
        for track in moving:
            views = candidates[track["id"]]
            mean = self.classifier.probabilities([row[2] for row in views]).mean(0)
            order = np.argsort(-mean)[:3]
            track["top_categories"] = [{"category": classes[i], "confidence": float(mean[i])} for i in order]
            score = float(mean[order[0]])
            track["classification_confidence"] = score
            track["category"] = classes[order[0]] if score >= settings["confidence"] else "unknown"
            track["label_source"] = "imagenet-demo" if imagenet else "grocery-vit"
            track["classification_views"] = [row[1] for row in views]


def movements(tracks):
    """The answer the orchestrator needs: what went in or out, and when."""
    rows = [{"track_id": track["id"], "direction": event["direction"], "category": track["category"],
             "classification_confidence": track["classification_confidence"],
             "start": event["event_start"], "end": event["event_end"]}
            for track in tracks if not track.get("ignored") for event in track["events"]]
    return sorted(rows, key=lambda row: row["start"])


#------------------------------------------------------------
# STANDALONE CLEANUP: KEEP THE ORIGINAL CLIP AND RESULT JSON
#------------------------------------------------------------
def purge_video_runs(root, keep=3):
    """Remove working copies and crops of production runs older than the newest `keep`."""
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be positive.")
    runs = []
    for marker in Path(root).glob("*/.ml-run.json"):
        if marker.is_symlink() or marker.parent.is_symlink():
            continue
        document = json.loads(marker.read_text())
        if document.get("mode") in ("production", "test") and document.get("kind") == "video":
            runs.append((document["completed_at"], marker.parent))
    for _, directory in sorted(runs, reverse=True)[keep:]:
        for pattern in ("video.mp4", "poster.jpg", "track-*.jpg"):
            for path in directory.glob(pattern):
                if path.is_file() and not path.is_symlink():
                    path.unlink()
