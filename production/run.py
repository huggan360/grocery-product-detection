#------------------------------------------------------------
# FILE HANDOFF: THE ORCHESTRATOR OWNS SQLITE
#------------------------------------------------------------
import json
import secrets
import shutil
import time
from pathlib import Path

from production.video import VideoPipeline, load_config, purge_video_runs, write_json

MODES = ("production", "acquisition", "test")


def run_video_request(request_path, record_id, output_path, config_path, mode,
                      models_dir=None, managed_retention=False, pipeline=None):
    """Track objects in one clip and report which ones moved in or out."""
    if mode not in MODES:
        raise ValueError("Unknown run mode.")
    request = json.loads(Path(request_path).read_text())
    if request.get("schema_version") != 2 or str(request.get("capture_id")) != str(record_id):
        raise ValueError("Request schema or capture ID does not match.")
    for key in ("event_id", "captured_at", "camera", "video_path"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError(f"Request needs {key}.")
    source = Path(request["video_path"])
    if not source.is_absolute() or not source.is_file():
        raise ValueError("video_path must be an existing absolute path.")
    config = load_config(config_path, models_dir=models_dir)
    if request["camera"] not in config["cameras"]:
        raise ValueError(f"camera must be one of {sorted(config['cameras'])}.")
    output = Path(output_path).resolve()
    if output.exists() or (output.parent / "video.mp4").exists():
        raise ValueError("Use a new run directory; existing results are not overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline or VideoPipeline(config)
    result = pipeline.run(source, output.parent, request["camera"], config["cameras"][request["camera"]],
                          request["event_id"], float(request.get("offset_seconds", 0.0)))
    result.update(capture_id=str(record_id), mode=mode, captured_at=request["captured_at"],
                  sensor_metadata=request.get("sensor_metadata", {}),
                  attachments=request.get("attachments", []))
    write_json(output, result)
    write_json(output.parent / ".ml-run.json", {"schema_version": 2, "kind": "video", "mode": mode,
               "completed_at": result["completed_at"], "result": output.name})
    if mode == "test":
        publish_for_review(output.parent, source, result, config)
    if mode in ("production", "test") and not managed_retention:
        purge_video_runs(output.parent.parent, config["retention_runs"])
    return output


#------------------------------------------------------------
# --test: SHOW LIVE CLIPS AND PREDICTIONS IN THE REVIEW TOOL
#------------------------------------------------------------
def publish_for_review(run_directory, source, result, config):
    """Copy the clip and its result into the review tool's library as a finished clip."""
    library = Path(config["acquisition_directory"]) / "videos"
    video_id = secrets.token_hex(12)
    staging = library / f".publish-{video_id}"
    staging.mkdir(parents=True)
    try:
        shutil.copyfile(source, staging / f"original{source.suffix.lower()}")
        for path in run_directory.iterdir():
            if path.name in ("video.mp4", "poster.jpg", "result.json") or \
                    (path.name.startswith("track-") and path.suffix == ".jpg"):
                shutil.copyfile(path, staging / path.name)
        name = f"{result['event_id']}-{result['camera']}-{result.get('offset_seconds', 0):.1f}s.mp4"
        write_json(staging / "meta.json", {
            "id": video_id, "filename": name, "camera": result["camera"], "event_id": result["event_id"],
            "capture_id": result["capture_id"], "created": time.time(), "editor": "live system (--test)",
            "status": "done", "progress": 100, "message": f"Live run ({result.get('backend', 'unknown')} YOLO)",
            "original": f"original{source.suffix.lower()}"})
        # The review tool only lists complete folders, so publish with one rename.
        staging.rename(library / video_id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return library / video_id
