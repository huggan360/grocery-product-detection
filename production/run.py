#------------------------------------------------------------
# FILE HANDOFF: THE ORCHESTRATOR OWNS SQLITE
#------------------------------------------------------------
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from production.masks import masked_crop
from production.pipeline import RGBPipeline, draw_predictions, load_pipeline_config


def utc_now():
    """Record times with an explicit UTC timezone."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    """Publish complete JSON at once, never a half-written result."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_request(request_path, record_id, output_path, config_path, mode,
                models_dir=None, managed_retention=False, pipeline=None):
    """Process one RGB capture while preserving IDs and sensor metadata."""
    if mode not in ("production", "acquisition"):
        raise ValueError("Unknown run mode.")
    request = json.loads(Path(request_path).read_text())
    if request.get("schema_version") != 1 or str(request.get("capture_id")) != str(record_id):
        raise ValueError("Request schema or capture ID does not match.")
    for key in ("event_id", "captured_at", "sensor", "image_path"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError(f"Request needs {key}.")
    source = Path(request["image_path"])
    if not source.is_absolute() or not source.is_file():
        raise ValueError("image_path must be an existing absolute path.")
    config = load_pipeline_config(config_path, models_dir=models_dir)
    keep = config.get("retention_runs", 3)
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("retention_runs must be a positive integer.")
    output = Path(output_path).resolve()
    if output.exists() or (output.parent / "masks").exists():
        raise ValueError("Use a new run directory; existing results are not overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    pipeline = pipeline or RGBPipeline(config)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    items = pipeline.predict(image)
    masks = output.parent / "masks"
    masks.mkdir()
    for index, item in enumerate(items):
        polygon_path = masks / f"{index:04d}.json"
        crop_path = masks / f"{index:04d}.png"
        write_json(polygon_path, {"polygon": item["polygon"], "box_xyxy": item["box_xyxy"]})
        masked_crop(image, item["polygon"], config["prediction"]["crop_padding"]).save(crop_path)
        item.update(mask_path=str(polygon_path), crop_path=str(crop_path))
    draw_predictions(image, items).save(output.parent / "annotated.jpg")
    result = {"schema_version": 1, "capture_id": str(record_id),
              "event_id": request["event_id"], "mode": mode,
              "captured_at": request["captured_at"], "sensor": request["sensor"],
              "started_at": started, "completed_at": utc_now(),
              "image_path": str(source), "image_size": list(image.size),
              "coordinate_space": "pixels in EXIF-corrected RGB image",
              "sensor_metadata": request.get("sensor_metadata", {}),
              "attachments": request.get("attachments", []),
              "models": {"yolo": config["prediction"]["segmentation_weights"],
                         "vit": config["classifier"].get("checkpoint"),
                         "vit_baseline": config["classifier"].get("baseline")},
              "objects": items, "counts": dict(Counter(item["category"] for item in items)),
              "annotated_path": str(output.parent / "annotated.jpg")}
    write_json(output, result)
    write_json(output.parent / ".ml-run.json", {"schema_version": 1, "mode": mode,
               "completed_at": result["completed_at"], "result": output.name})
    if mode == "production" and not managed_retention:
        purge_masks(output.parent.parent, keep)
    return output


#------------------------------------------------------------
# STANDALONE CLEANUP: NEVER DELETE INPUT IMAGES OR ACQUISITION DATA
#------------------------------------------------------------
def purge_masks(root, keep=3):
    """Keep masks for the latest completed production runs in this output root."""
    if not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be positive.")
    runs = []
    for marker in Path(root).glob("*/.ml-run.json"):
        if marker.parent.is_symlink() or marker.is_symlink():
            continue
        document = json.loads(marker.read_text())
        if document.get("schema_version") == 1 and document.get("mode") == "production":
            runs.append((document["completed_at"], marker.parent))
    for _, directory in sorted(runs, reverse=True)[keep:]:
        masks = directory / "masks"
        if masks.is_dir() and not masks.is_symlink():
            shutil.rmtree(masks)
        marker = json.loads((directory / ".ml-run.json").read_text())
        name = marker["result"]
        if Path(name).name != name:
            continue
        path = directory / name
        if path.is_symlink():
            continue
        result = json.loads(path.read_text())
        for item in result.get("objects", []):
            for key in ("mask_path", "crop_path", "polygon"):
                item.pop(key, None)
        result["masks_purged"] = True
        write_json(path, result)
