#------------------------------------------------------------
# IMPORT SAVED PREDICTIONS INTO THE REVIEW WORKSPACE
#------------------------------------------------------------
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from annotation_tool.settings import read_settings
from annotation_tool.store import Store


def import_acquisition(manifest, settings, split="train"):
    """Copy acquisition images and existing masks; do not rerun the models."""
    document = json.loads(Path(manifest).read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Expected an acquisition export with schema_version 1.")
    store = Store(settings["storage"], settings["categories"], settings["lease_seconds"])
    token = store.register("Acquisition import")["token"]
    count = 0
    for record in document["records"]:
        source, result = record["input"], record["result"]
        if result["mode"] != "acquisition" or source["capture_id"] != result["capture_id"]:
            raise ValueError("Only matching acquisition captures can be imported.")
        image_id = "edge-" + hashlib.sha256(record["run_id"].encode()).hexdigest()[:32]
        # Reimporting never overwrites someone's edits.
        with store.connect() as connection:
            if connection.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone():
                continue
        collection = "event-" + hashlib.sha256(source["event_id"].encode()).hexdigest()[:24]
        with Image.open(source["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.save(store.root / "images" / f"{image_id}.png")
        preview = image.copy()
        preview.thumbnail((420, 280))
        preview.save(store.root / "thumbnails" / f"{image_id}.jpg")
        store.add_image(image_id, source["capture_id"], source["sensor"], collection,
                        split, image.size, "manual", token)
        store.finish_job(image_id, result["objects"])
        # Preserve door-event links in exported source suggestions and draft notes.
        with store.connect(write=True) as connection:
            connection.execute("UPDATE images SET notes = ? WHERE id = ?",
                               (json.dumps({"event_id": source["event_id"],
                                            "capture_id": source["capture_id"],
                                            "run_id": record["run_id"],
                                            "captured_at": source["captured_at"]}), image_id))
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import edge acquisition predictions for editing.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "annotation_tool/config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    print(f"Imported {import_acquisition(args.manifest, read_settings(args.config), args.split)} images.")
