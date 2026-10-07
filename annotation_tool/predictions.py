#------------------------------------------------------------
# REVIEW QUEUE ADAPTER: ALL MODEL CODE LIVES IN PRODUCTION
#------------------------------------------------------------
import threading
from PIL import Image
from production.pipeline import RGBPipeline, load_pipeline_config


class ModelPredictor:
    """Connect the review queue to the same pipeline used by main.py."""

    def __init__(self, settings):
        self.pipeline = None
        self.available = False
        self.message = "Manual mode: model suggestions are disabled."
        if not settings.get("suggestions", True):
            return
        try:
            self.pipeline = RGBPipeline(load_pipeline_config(settings["model_config"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.message = f"Model configuration error: {error}"
            return
        self.available = True
        self.message = self.pipeline.message

    def predict(self, image):
        """Pass the original image straight to the shared network."""
        if not self.available:
            raise RuntimeError(self.message)
        return self.pipeline.predict(image)


#------------------------------------------------------------
# ONE SHARED BACKGROUND INFERENCE WORKER
#------------------------------------------------------------
class PredictionWorker:
    """Prepare initial suggestions without blocking other annotators."""

    def __init__(self, store, predictor):
        self.store, self.predictor = store, predictor
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="rgb-suggestions")

    def start(self):
        """Recover interrupted jobs and process the shared queue."""
        self.store.recover_jobs()
        self.thread.start()

    def stop(self):
        """Stop accepting jobs when the server closes."""
        self.stop_event.set()
        self.thread.join(timeout=5)

    def run(self):
        """Store suggestions or record a failure while retaining the source image."""
        while not self.stop_event.is_set():
            row = self.store.next_job()
            if row is None:
                self.stop_event.wait(0.5)
                continue
            try:
                if not self.predictor.available:
                    raise RuntimeError(self.predictor.message)
                with Image.open(self.store.root / "images" / f"{row['id']}.png") as image:
                    records = self.predictor.predict(image.convert("RGB"))
                self.store.finish_job(row["id"], records)
            except Exception as error:
                self.store.finish_job(row["id"], error=str(error)[:1000])
