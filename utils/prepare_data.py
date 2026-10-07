#------------------------------------------------------------
# CONVERT DATASET ANNOTATIONS INTO YOLO LABELS AND VIT CROPS
#------------------------------------------------------------
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
import yaml

from production.geometry import crop_box

REQUIRED_COLUMNS = {"image", "split", "category", "xmin", "ymin", "xmax", "ymax"}
SPLITS = {"train", "val", "test"}


def read_annotations(manifests, mapping):
    """Combine CSV files and reject split leaks or unclear labels."""
    images = {}
    group_splits = {}
    digest_splits = {}
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                raise ValueError(f"{manifest} needs columns {sorted(REQUIRED_COLUMNS)}")
            for line, row in enumerate(reader, start=2):
                path = (manifest.parent / row["image"]).resolve()
                split = row["split"].strip()
                if split not in SPLITS:
                    raise ValueError(f"{manifest}:{line}: split must be train, val, or test.")
                group = row.get("group", "").strip()
                if group:
                    if group in group_splits and group_splits[group] != split:
                        raise ValueError(f"Group {group!r} occurs in more than one split.")
                    group_splits[group] = split
                if path not in images:
                    # Compare decoded pixels so simple file renaming does not hide leaks.
                    with Image.open(path) as image:
                        if image.getexif().get(274, 1) != 1:
                            raise ValueError(f"Normalize EXIF rotation and box coordinates first: {path}")
                        rgb = image.convert("RGB")
                        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                        size = rgb.size
                    if digest in digest_splits and digest_splits[digest] != split:
                        raise ValueError(f"Identical image pixels occur in different splits: {path}")
                    digest_splits[digest] = split
                    images[path] = {"split": split, "size": size, "boxes": {}, "empty": False}
                item = images[path]
                if item["split"] != split:
                    raise ValueError(f"Image occurs in different splits: {path}")
                raw_category = row["category"].strip()
                coords = [row[key].strip() for key in ("xmin", "ymin", "xmax", "ymax")]
                if not raw_category and not any(coords):
                    if item["boxes"]:
                        raise ValueError(f"Image is marked empty but has boxes: {path}")
                    item["empty"] = True
                    continue
                if item["empty"]:
                    raise ValueError(f"Image is marked empty but has boxes: {path}")
                category = mapping.get(raw_category, raw_category) if mapping is not None else raw_category
                if mapping is not None and raw_category not in mapping:
                    raise ValueError(f"No category mapping for {raw_category!r}")
                if not isinstance(category, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", category):
                    raise ValueError(f"Use simple lower-case category names, such as milk: {category!r}")
                if category == "unknown":
                    raise ValueError("The name 'unknown' is reserved for rejected predictions.")
                box = tuple(float(value) for value in coords)
                width, height = item["size"]
                x1, y1, x2, y2 = box
                if not (all(math.isfinite(v) for v in box) and
                        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Invalid pixel box {box} for {path} with size {item['size']}")
                if box in item["boxes"] and item["boxes"][box] != category:
                    raise ValueError(f"The same box has conflicting categories: {path}")
                item["boxes"][box] = category
    if not images:
        raise ValueError("No images found in the manifests.")
    categories = defaultdict(set)
    for item in images.values():
        categories[item["split"]].update(item["boxes"].values())
    if len(categories["train"]) < 2:
        raise ValueError("Training needs at least two product categories.")
    if categories["val"] != categories["train"]:
        raise ValueError("Train and validation must contain the same category set.")
    if categories["test"] - categories["train"]:
        raise ValueError("Test contains categories that are missing from training.")
    return images


def prepare_data(manifests, output, mapping_path=None, padding=0.05):
    """Make one-class detection labels and category folders without changing originals."""
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("Padding must be finite and non-negative.")
    mapping = json.loads(Path(mapping_path).read_text()) if mapping_path else None
    if mapping is not None and not isinstance(mapping, dict):
        raise ValueError("Category mapping must be a JSON object.")
    images = read_annotations(manifests, mapping)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty so old labels cannot remain: {output}")
    detection = output / "detection"
    counts = defaultdict(lambda: {"images": 0, "crops": 0})
    provenance = []
    for index, (path, item) in enumerate(sorted(images.items())):
        split = item["split"]
        stem = f"{index:08d}"
        image_folder = detection / "images" / split
        label_folder = detection / "labels" / split
        image_folder.mkdir(parents=True, exist_ok=True)
        label_folder.mkdir(parents=True, exist_ok=True)
        labels = []
        with Image.open(path) as original:
            image = original.convert("RGB")
            # PNG preserves pixels and uses the same orientation as the input boxes.
            image.save(image_folder / f"{stem}.png")
            width, height = image.size
            for box_index, (box, category) in enumerate(item["boxes"].items()):
                x1, y1, x2, y2 = box
                labels.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                              f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                folder = output / "classification" / split / category
                folder.mkdir(parents=True, exist_ok=True)
                image.crop(crop_box(box, width, height, padding)).save(folder / f"{stem}_{box_index}.png")
                counts[split]["crops"] += 1
        (label_folder / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        counts[split]["images"] += 1
        provenance.append({"source": str(path), "split": split, "output_stem": stem})
    dataset = {"path": str(detection), "train": "images/train", "val": "images/val",
               "names": {0: "product"}}
    if "test" in counts:
        dataset["test"] = "images/test"
    (detection / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    (output / "preparation.json").write_text(json.dumps({"counts": dict(counts),
        "crop_padding": padding, "mapping": mapping, "images": provenance}, indent=2) + "\n")
    print(json.dumps(dict(counts), indent=2))
    return output


#------------------------------------------------------------
# RUN THIS FILE ON ITS OWN
#------------------------------------------------------------
def add_arguments(parser):
    """Describe the dataset conversion options used by both entry points."""
    parser.add_argument("--manifest", nargs="+", required=True, help="One or more annotation CSV files.")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--mapping", help="JSON mapping from source labels to broad categories.")
    parser.add_argument("--padding", type=float, default=0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make YOLO boxes and ViT crop folders from CSV annotations.")
    add_arguments(parser)
    args = parser.parse_args()
    prepare_data(args.manifest, args.output, args.mapping, args.padding)
