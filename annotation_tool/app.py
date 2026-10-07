#------------------------------------------------------------
# LOCAL WEB SERVER FOR VIDEO REVIEW
#------------------------------------------------------------
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from annotation_tool.store import Store, StoreError
from annotation_tool.videos import add_video_routes


class EditorRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class CategoryRequest(BaseModel):
    name: str


def create_app(settings):
    """Build the server: clips go through the same video pipeline as production."""
    store = Store(settings["storage"], settings["categories"])
    static = Path(__file__).parent / "static"
    library = None

    @asynccontextmanager
    async def lifespan(app):
        """Start the video worker with the server and stop it on shutdown."""
        library.start()
        yield
        library.stop()

    app = FastAPI(title="Fridge Lab", lifespan=lifespan)
    app.state.store = store

    @app.exception_handler(StoreError)
    async def store_error(request, error):
        """Display ordinary validation errors in plain English."""
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/")
    def home():
        """Serve the review interface without a frontend build step."""
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    @app.post("/api/editors")
    def register(body: EditorRequest):
        """Register a reviewer's display name."""
        return store.register(body.name)

    @app.post("/api/categories")
    def add_category(body: CategoryRequest, x_session: str = Header()):
        """Make a new category available to every reviewer."""
        return {"name": store.add_category(body.name, x_session)}

    library = add_video_routes(app, settings, store)
    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
