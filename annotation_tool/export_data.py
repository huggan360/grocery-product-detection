#------------------------------------------------------------
# EXPORT REVIEWED IMAGES INTO THE EXISTING TRAINING FORMATS
#------------------------------------------------------------
import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

from PIL import Image
import yaml

from production.geometry import crop_box
from production.masks import masked_crop
from annotation_tool.store import StoreError


def build_export(store):
    """Create a consistent ZIP containing approved labels, crops and provenance."""
    rows = store.export_snapshot()
    if not rows:
        raise StoreError("No reviewed, unlocked images to export. Save and leave the editor first.")
    descriptor = tempfile.NamedTemporaryFile(suffix=".zip", prefix="fridge-export-", delete=False)
    path = Path(descriptor.name)
    descriptor.close()
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            manifest = io.StringIO(newline="")
            writer = csv.writer(manifest)
            writer.writerow(["image", "split", "category", "xmin", "ymin", "xmax", "ymax", "group"])
            metadata = []
            segmentation_ids = []
            for split in ("train", "val", "test"):
                archive.writestr(f"detection/images/{split}/", "")
                archive.writestr(f"detection/labels/{split}/", "")
                archive.writestr(f"segmentation/images/{split}/", "")
                archive.writestr(f"segmentation/labels/{split}/", "")
            for row in rows:
                image_id, split = row["id"], row["split"]
                boxes = json.loads(row["boxes"])
                mask_ready = bool(row["masks_reviewed"]) and all(box.get("polygon") for box in boxes)
                image_name = f"detection/images/{split}/{image_id}.png"
                archive.write(store.root / "images" / f"{image_id}.png", image_name)
                lines = []
                segment_lines = []
                with Image.open(store.root / "images" / f"{image_id}.png") as image:
                    for index, box in enumerate(boxes):
                        x1, y1, x2, y2 = box["xyxy"]
                        width, height = row["width"], row["height"]
                        lines.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                                     f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                        writer.writerow([image_name, split, box["category"], *box["xyxy"], row["collection"]])
                        buffer = io.BytesIO()
                        image.crop(crop_box(box["xyxy"], width, height, 0.05)).save(buffer, format="PNG")
                        archive.writestr(f"classification/{split}/{box['category']}/{image_id}_{index}.png", buffer.getvalue())
                        if mask_ready:
                            polygon = box["polygon"]
                            segment_lines.append("0 " + " ".join(f"{value / (width if axis == 0 else height):.8f}"
                                for point in polygon for axis, value in enumerate(point)))
                            masked = io.BytesIO()
                            masked_crop(image, polygon).save(masked, format="PNG")
                            archive.writestr(f"classification_masked/{split}/{box['category']}/{image_id}_{index}.png", masked.getvalue())
                if not boxes:
                    writer.writerow([image_name, split, "", "", "", "", "", row["collection"]])
                archive.writestr(f"detection/labels/{split}/{image_id}.txt", "\n".join(lines) + ("\n" if lines else ""))
                if mask_ready:
                    segmentation_ids.append(image_id)
                    archive.write(store.root / "images" / f"{image_id}.png", f"segmentation/images/{split}/{image_id}.png")
                    archive.writestr(f"segmentation/labels/{split}/{image_id}.txt", "\n".join(segment_lines) + ("\n" if segment_lines else ""))
                metadata.append({key: row[key] for key in ("id", "filename", "shelf", "collection", "split",
                                                          "created", "edited_by", "reviewed_at", "revision", "notes", "masks_reviewed")}
                                | {"boxes": boxes, "model_suggestions": json.loads(row["suggestions"])})
            archive.writestr("annotations.csv", manifest.getvalue())
            archive.writestr("annotations.json", json.dumps(metadata, indent=2))
            archive.writestr("segmentation/dataset.yaml", yaml.safe_dump({
                "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "product"}}, sort_keys=False))
            archive.writestr("segmentation/export_summary.json", json.dumps({"included_images": segmentation_ids,
                "excluded_without_reviewed_masks": [row["id"] for row in rows if row["id"] not in set(segmentation_ids)]}, indent=2))
            archive.writestr("detection/dataset.yaml", yaml.safe_dump({
                "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "product"}}, sort_keys=False))
            archive.writestr("README.txt", "Reviewed fridge annotation export.\n"
                "Point detector.data at detection/dataset.yaml and classifier.data at classification/.\n"
                "Collect separate train and val sessions (and at least two categories) before training.\n"
                "Session split assignments are preserved. Test data must not be used for tuning.\n"
                "Only reviewed images with no active editing lease are included.\n"
                "Classifier crops use 5% padding. All boxes use EXIF-corrected RGB pixel coordinates.\n")
            archive.writestr("segmentation/README.txt", "Only complete images saved with Masks checked are included.\n"
                "Box-only or unreviewed-mask images are excluded as a whole, not treated as background.\n"
                "Train YOLO26-seg on segmentation/dataset.yaml. Train ViT on classification_masked/.\n"
                "One simple exterior polygon per object is supported; disconnected parts and holes are not represented.\n")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
