#------------------------------------------------------------
# VIDEO REVIEW: UPLOAD/RECORD -> SHARED VIDEO PIPELINE -> CORRECT TRACKS
#------------------------------------------------------------
# Clips are stored in the data-acquisition folder; all model code lives in production.
import json
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from fastapi import File, Form, Header, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from annotation_tool.store import StoreError, category_name
from production.video import VideoPipeline, load_config, movements, save_zones, write_json
from production.zones import analyse_tracks

VIDEO_ID = re.compile(r"[0-9a-f]{24}")
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".h264", ".m4v"}


class ZonesRequest(BaseModel):
    inside: list[float] | None
    outside: list[float] | None


class RecordRequest(BaseModel):
    camera: str
    seconds: float = Field(gt=0, le=120)
    event_id: str = Field(default="", max_length=120)


class ReviewRequest(BaseModel):
    tracks: dict[str, dict]
    notes: str = Field(default="", max_length=4000)
    reviewed: bool = False


class VideoLibrary:
    """One folder per clip; meta.json holds queue state, result.json the model output."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.root = Path(self.config["acquisition_directory"]) / "videos"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pipeline = None
        self.message = "Video models load when the first clip is processed."
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True, name="video-pipeline")

    def folder(self, video_id):
        if not VIDEO_ID.fullmatch(video_id) or not (self.root / video_id / "meta.json").is_file():
            raise StoreError("Video not found.", 404)
        return self.root / video_id

    def meta(self, video_id):
        return json.loads((self.folder(video_id) / "meta.json").read_text())

    def update(self, video_id, **values):
        with self.lock:
            path = self.folder(video_id) / "meta.json"
            meta = json.loads(path.read_text())
            meta.update(values)
            write_json(path, meta)
            return meta

    def add(self, source, filename, camera, event_id, editor, move=False):
        """Copy (or move) a clip in and queue it for the pipeline."""
        if camera not in self.config["cameras"]:
            raise StoreError(f"Camera must be one of: {', '.join(self.config['cameras'])}.")
        suffix = Path(filename).suffix.lower()
        if suffix not in EXTENSIONS:
            raise StoreError("Upload an MP4, MOV, MKV, AVI, WEBM or H.264 video.")
        video_id = secrets.token_hex(12)
        folder = self.root / video_id
        folder.mkdir()
        target = folder / f"original{suffix}"
        (shutil.move if move else shutil.copyfile)(source, target)
        write_json(folder / "meta.json", {
            "id": video_id, "filename": Path(filename).name[:200], "camera": camera,
            "event_id": event_id or f"upload-{video_id[:8]}", "created": time.time(),
            "editor": editor, "status": "queued", "progress": 0, "message": "Waiting",
            "original": target.name})
        return video_id

    def items(self):
        rows = []
        for path in self.root.glob("*/meta.json"):
            if not VIDEO_ID.fullmatch(path.parent.name):
                continue  # Half-published or temporary folders.
            meta = json.loads(path.read_text())
            result = path.parent / "result.json"
            if result.is_file():
                document = json.loads(result.read_text())
                meta["movements"] = len(document.get("movements", []))
                meta["tracks"] = len(document.get("tracks", []))
            meta["reviewed"] = (path.parent / "review.json").is_file() and \
                json.loads((path.parent / "review.json").read_text()).get("reviewed", False)
            rows.append(meta)
        return sorted(rows, key=lambda row: row["created"], reverse=True)

    #------------------------------------------------------------
    # ONE BACKGROUND WORKER: THE PI HAS ONE HAILO-8
    #------------------------------------------------------------
    def start(self):
        for meta in self.items():
            if meta["status"] == "running":
                self.update(meta["id"], status="queued", progress=0, message="Restarted")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def work(self):
        while not self.stop_event.is_set():
            queued = [row for row in self.items() if row["status"] == "queued"]
            if not queued:
                self.stop_event.wait(0.5)
                continue
            meta = queued[-1]  # Oldest first.
            folder = self.root / meta["id"]
            try:
                self.update(meta["id"], status="running", message="Loading models")
                if self.pipeline is None:
                    self.pipeline = VideoPipeline(self.config)
                    self.message = self.pipeline.message
                started = time.monotonic()
                self.pipeline.run(folder / meta["original"], folder, meta["camera"],
                                  self.config["cameras"][meta["camera"]], meta["event_id"],
                                  progress=lambda value, text: self.update(meta["id"], progress=value, message=text))
                self.update(meta["id"], status="done", progress=100,
                            message=f"Processed in {time.monotonic() - started:.1f} s")
            except Exception as error:
                self.update(meta["id"], status="failed", message=str(error)[:1000])
            finally:
                # The live system's ML runs need the Hailo-8 too.
                if self.pipeline is not None and hasattr(self.pipeline, "release"):
                    self.pipeline.release()

    #------------------------------------------------------------
    # REVIEWED TRACK CROPS -> classification/<category>/ FOR train.py vit
    #------------------------------------------------------------
    def export(self):
        """Zip the best crop of every reviewed, labelled, non-ignored track."""
        import tempfile
        import zipfile
        handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        handle.close()
        count = 0
        with zipfile.ZipFile(handle.name, "w") as archive:
            for meta in self.items():
                folder = self.root / meta["id"]
                review = folder / "review.json"
                if not meta["reviewed"] or not (folder / "result.json").is_file():
                    continue
                tracks = json.loads(review.read_text())["tracks"]
                for track in json.loads((folder / "result.json").read_text())["tracks"]:
                    values = tracks.get(track["id"], {})
                    crop = folder / track.get("crop_file", "")
                    if values.get("ignored") or values.get("category", "unknown") == "unknown" or not crop.is_file():
                        continue
                    archive.write(crop, f"classification/{values['category']}/{meta['id']}-{track['id']}.jpg")
                    count += 1
            archive.writestr("README.txt", f"{count} reviewed track crops from fridge videos. "
                             "Merge classification/ into the ViT training data and run: python train.py vit\n")
        if not count:
            Path(handle.name).unlink()
            raise StoreError("No reviewed, labelled tracks to export yet.", 409)
        return Path(handle.name)

    #------------------------------------------------------------
    # ZONES CAN BE REDRAWN WITHOUT RUNNING THE MODELS AGAIN
    #------------------------------------------------------------
    def reanalyse(self, video_id):
        folder = self.folder(video_id)
        path = folder / "result.json"
        if not path.is_file():
            raise StoreError("This clip has no result yet.", 409)
        result = json.loads(path.read_text())
        zones = self.config["cameras"][result["camera"]]
        tracks = analyse_tracks(result["tracks"], zones, self.config["tracking"])
        for track in tracks:
            for event in track["events"]:
                event["event_start"] = result["offset_seconds"] + event["start"]
                event["event_end"] = result["offset_seconds"] + event["end"]
        result.update(zones=zones, tracks=tracks, movements=movements(tracks),
                      warnings=[] if all(zones.values()) else ["Set both zones before interpreting in/out movements."])
        write_json(path, result)
        return result


def add_video_routes(app, settings, store):
    """Register the video API."""
    library = VideoLibrary(settings["video_config"])
    app.state.videos = library
    limit = library.config["video"]["max_upload_mb"] * 1024 * 1024

    def editor(token):
        with store.connect(write=True) as connection:
            return store.editor(token, connection)

    @app.get("/api/videos")
    def list_videos():
        config = library.config
        return {"items": library.items(), "cameras": config["cameras"],
                "recordable": sorted(config.get("capture", {})), "categories": store.categories(),
                "model_message": library.message, "sample_fps": config["tracking"]["sample_fps"]}

    @app.post("/api/videos")
    def upload_video(file: UploadFile = File(), camera: str = Form(), event_id: str = Form(default=""),
                     x_session: str = Header()):
        name = editor(x_session)
        temporary = library.root / f".upload-{secrets.token_hex(6)}"
        try:
            size = 0
            with temporary.open("wb") as handle:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise StoreError("Video exceeds the configured upload limit.")
                    handle.write(chunk)
            return {"id": library.add(temporary, file.filename or "upload.mp4", camera,
                                      event_id.strip(), name, move=True)}
        finally:
            temporary.unlink(missing_ok=True)
            file.file.close()

    @app.get("/api/videos/export")
    def export(x_session: str = Header()):
        from starlette.background import BackgroundTask
        editor(x_session)
        path = library.export()
        return FileResponse(path, filename="fridge-video-crops.zip", media_type="application/zip",
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @app.post("/api/videos/record")
    def record_video(body: RecordRequest, x_session: str = Header()):
        from utils.record_clip import record
        name = editor(x_session)
        capture = library.config.get("capture", {})
        if body.camera not in capture:
            raise StoreError("This camera has no capture settings in configs/video.yaml.")
        try:
            path = record(capture[body.camera], body.seconds,
                          Path(library.config["acquisition_directory"]) / "recordings", body.camera)
        except (RuntimeError, ValueError, OSError) as error:
            raise StoreError(str(error), 503) from error
        return {"id": library.add(path, path.name, body.camera, body.event_id.strip(), name)}

    @app.get("/api/videos/{video_id}")
    def video_detail(video_id: str):
        folder = library.folder(video_id)
        value = {"meta": library.meta(video_id)}
        for name in ("result", "review"):
            path = folder / f"{name}.json"
            value[name] = json.loads(path.read_text()) if path.is_file() else None
        return value

    @app.get("/api/videos/{video_id}/files/{name}")
    def video_file(video_id: str, name: str):
        if not re.fullmatch(r"video\.mp4|poster\.jpg|track-\d+\.jpg", name):
            raise StoreError("File not found.", 404)
        path = library.folder(video_id) / name
        if not path.is_file():
            raise StoreError("File not found.", 404)
        return FileResponse(path)

    @app.put("/api/zones/{camera}")
    def put_zones(camera: str, body: ZonesRequest, x_session: str = Header()):
        editor(x_session)
        try:
            return save_zones(library.config, camera, body.model_dump())
        except ValueError as error:
            raise StoreError(str(error)) from error

    @app.post("/api/videos/{video_id}/reanalyse")
    def reanalyse(video_id: str, x_session: str = Header()):
        editor(x_session)
        return library.reanalyse(video_id)

    @app.post("/api/videos/{video_id}/rerun")
    def rerun(video_id: str, x_session: str = Header()):
        editor(x_session)
        if library.meta(video_id)["status"] in ("queued", "running"):
            raise StoreError("This clip is already queued.", 409)
        return library.update(video_id, status="queued", progress=0, message="Waiting")

    @app.put("/api/videos/{video_id}/review")
    def review(video_id: str, body: ReviewRequest, x_session: str = Header()):
        name = editor(x_session)
        folder = library.folder(video_id)
        tracks = {}
        for track_id, values in body.tracks.items():
            category = values.get("category", "unknown")
            if category != "unknown":
                category_name(category)
            if values.get("direction", "auto") not in ("auto", "in", "out", "none"):
                raise StoreError("Direction must be auto, in, out or none.")
            tracks[str(track_id)] = {"category": category, "ignored": bool(values.get("ignored")),
                                     "direction": values.get("direction", "auto")}
        document = {"tracks": tracks, "notes": body.notes, "reviewed": body.reviewed,
                    "editor": name, "saved_at": time.time()}
        write_json(folder / "review.json", document)
        return document

    return library
