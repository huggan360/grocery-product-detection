#------------------------------------------------------------
# LOCAL WEB SERVER AND SHARED WORKSPACE API
#------------------------------------------------------------
import io
import json
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from PIL import Image, ImageOps, UnidentifiedImageError

from annotation_tool.cameras import capture_camera
from annotation_tool.export_data import build_export
from annotation_tool.predictions import ModelPredictor, PredictionWorker
from annotation_tool.store import Store, StoreError, category_name


class EditorRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class CategoryRequest(BaseModel):
    name: str


class SaveRequest(BaseModel):
    revision: int = Field(ge=0)
    boxes: list[dict]
    notes: str = Field(default="", max_length=4000)
    reviewed: bool = False
    empty_confirmed: bool = False
    masks_reviewed: bool = False


class CaptureRequest(BaseModel):
    collection: str
    split: str


def create_app(settings, predictor=None):
    """Build the server; tests can provide a fake predictor without loading weights."""
    store = Store(settings["storage"], settings["categories"], settings["lease_seconds"])
    predictor = predictor or ModelPredictor(settings)
    worker = PredictionWorker(store, predictor)
    capture_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        """Start model jobs with the server and stop them on shutdown."""
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Fridge Lab", lifespan=lifespan)
    app.state.store = store
    static = Path(__file__).parent / "static"

    @app.exception_handler(StoreError)
    async def store_error(request, error):
        """Display ordinary validation and collaboration errors in plain English."""
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/")
    def home():
        """Serve the annotation interface without a frontend build step."""
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/workspace")
    def workspace(x_session: str | None = Header(default=None)):
        """Send live queue data, category choices and model/camera status."""
        value = store.list_images(x_session)
        value.update(categories=store.categories(), model_available=predictor.available,
                     model_message=predictor.message, lease_seconds=store.lease_seconds,
                     cameras=[{"id": camera["id"], "name": camera["name"]}
                              for camera in settings.get("cameras", []) if camera.get("enabled")])
        return value

    @app.post("/api/editors")
    def register(body: EditorRequest):
        """Register a collaborator's display name."""
        return store.register(body.name)

    @app.post("/api/categories")
    def add_category(body: CategoryRequest, x_session: str = Header()):
        """Make a new category available to every collaborator."""
        return {"name": store.add_category(body.name, x_session)}

    #------------------------------------------------------------
    # UPLOAD AND CAMERA CAPTURE
    #------------------------------------------------------------
    def ingest(image, filename, shelf, collection, split, token):
        """Normalize orientation, keep immutable pixels and enqueue model suggestions."""
        if image.width * image.height > settings["max_image_pixels"]:
            raise StoreError("Image has too many pixels. Resize it before uploading.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image_id = secrets.token_hex(12)
        pixels = store.root / "images" / f"{image_id}.png"
        thumbnail = store.root / "thumbnails" / f"{image_id}.jpg"
        try:
            image.save(pixels)
            preview = image.copy()
            preview.thumbnail((420, 280))
            preview.save(thumbnail, quality=80)
            store.add_image(image_id, filename, shelf, collection, split, image.size,
                            "queued" if predictor.available else "manual", token)
        except BaseException:
            pixels.unlink(missing_ok=True)
            thumbnail.unlink(missing_ok=True)
            raise
        return image_id

    @app.post("/api/upload")
    def upload(files: list[UploadFile] = File(), shelf: str = Form(), collection: str = Form(),
               split: str = Form(), x_session: str = Header()):
        """Accept a batch of saved camera images; report individual file failures."""
        if len(files) > 50:
            raise StoreError("Upload at most 50 images in one batch.")
        with store.connect(write=True) as connection:
            store.editor(x_session, connection)
            store.collection(collection, split, connection)
        ids, errors = [], []
        for upload_file in files:
            try:
                data = upload_file.file.read(settings["max_upload_mb"] * 1024 * 1024 + 1)
                if len(data) > settings["max_upload_mb"] * 1024 * 1024:
                    raise StoreError(f"Image exceeds {settings['max_upload_mb']} MB.")
                with Image.open(io.BytesIO(data)) as image:
                    ids.append(ingest(image, upload_file.filename or "upload", shelf, collection, split, x_session))
            except (StoreError, OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                errors.append({"filename": upload_file.filename, "message": str(error)})
            finally:
                upload_file.file.close()
        return {"ids": ids, "errors": errors}

    @app.post("/api/capture")
    def capture(body: CaptureRequest, x_session: str = Header()):
        """Capture all enabled shelves and add successful frames to the shared queue."""
        with store.connect(write=True) as connection:
            store.editor(x_session, connection)
            store.collection(body.collection, body.split, connection)
        cameras = [camera for camera in settings.get("cameras", []) if camera.get("enabled")]
        if not cameras:
            raise StoreError("No shelf cameras are enabled. Configure cameras or upload images instead.")
        if not capture_lock.acquire(blocking=False):
            raise StoreError("A capture is already in progress. Please wait.", 409)
        ids, errors = [], []
        try:
            for camera in cameras:
                try:
                    with capture_camera(camera) as image:
                        ids.append(ingest(image, f"{camera['id']}.png", camera["name"], body.collection, body.split, x_session))
                except Exception as error:
                    errors.append({"filename": camera["name"], "message": str(error)})
        finally:
            capture_lock.release()
        return {"ids": ids, "errors": errors}

    #------------------------------------------------------------
    # CLAIM, EDIT, AUTOSAVE AND REVIEW
    #------------------------------------------------------------
    @app.post("/api/claim-next")
    def claim_next(x_session: str = Header()):
        """Give this annotator the oldest available pending image."""
        return store.claim(x_session)

    @app.post("/api/images/{image_id}/claim")
    def claim(image_id: str, x_session: str = Header()):
        """Open one chosen image for exclusive editing."""
        return store.claim(x_session, image_id)

    @app.post("/api/images/{image_id}/heartbeat")
    def heartbeat(image_id: str, x_session: str = Header()):
        """Renew the browser's lease before it expires."""
        store.heartbeat(image_id, x_session)
        return {"ok": True}

    @app.post("/api/images/{image_id}/release")
    def release(image_id: str, x_session: str = Header()):
        """Release a frame when the annotator leaves the editor."""
        store.release(image_id, x_session)
        return {"ok": True}

    @app.put("/api/images/{image_id}/annotations")
    def save(image_id: str, body: SaveRequest, x_session: str = Header()):
        """Write a draft or an explicitly reviewed annotation revision."""
        return store.save(image_id, x_session, **body.model_dump())

    @app.post("/api/images/{image_id}/suggest")
    def suggest(image_id: str, x_session: str = Header()):
        """Suggest masks for an existing image; human annotations are not replaced here."""
        if not predictor.available:
            raise StoreError(predictor.message)
        with store.connect(write=True) as connection:
            store.editor(x_session, connection)
            row = store.row(image_id, connection)
            store.require_lock(row, x_session)
            revision = row["revision"]
        with Image.open(store.root / "images" / f"{image_id}.png") as image:
            try:
                records = predictor.predict(image.convert("RGB"))
            except Exception as error:
                raise StoreError(f"Segmentation failed: {error}", 503) from error
        with store.connect(write=True) as connection:
            row = store.row(image_id, connection)
            store.require_lock(row, x_session)
            if row["revision"] != revision:
                raise StoreError("The image changed during inference. Reopen it before applying suggestions.", 409)
            for record in records:
                if record["category"] != "unknown":
                    connection.execute("INSERT OR IGNORE INTO categories VALUES (?)", (category_name(record["category"]),))
            connection.execute("UPDATE images SET suggestions = ? WHERE id = ?", (json.dumps(records), image_id))
        return {"suggestions": records}

    @app.get("/api/images/{image_id}/pixels")
    def pixels(image_id: str):
        """Serve normalized pixels; all box coordinates refer to this exact image."""
        store.get_image(image_id)
        return FileResponse(store.root / "images" / f"{image_id}.png")

    @app.get("/api/images/{image_id}/thumbnail")
    def thumbnail(image_id: str):
        """Serve a small queue preview without loading full-resolution images."""
        store.get_image(image_id)
        return FileResponse(store.root / "thumbnails" / f"{image_id}.jpg")

    @app.get("/api/export")
    def export(x_session: str = Header()):
        """Download reviewed work in formats the project's training scripts accept."""
        with store.connect(write=True) as connection:
            store.editor(x_session, connection)
        path = build_export(store)
        return FileResponse(path, filename="fridge-reviewed.zip", media_type="application/zip",
                            background=BackgroundTask(path.unlink, missing_ok=True))

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
