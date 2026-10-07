#------------------------------------------------------------
# PREPARE THE SWEDISH GROCERY STORE CLASSIFICATION DATASET
#------------------------------------------------------------
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


def import_grocery_store(source, output):
    """Keep official splits and group product varieties into broad categories."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output folder: {output}")
    with (source / "classes.csv").open(newline="") as handle:
        class_rows = list(csv.DictReader(handle))
    labels = {int(row["Coarse Class ID (int)"]): row["Coarse Class Name (str)"].lower()
              for row in class_rows}
    rows, hashes, counts = [], {}, {}
    for split in ("train", "val", "test"):
        counts[split] = Counter()
        with (source / f"{split}.txt").open(newline="") as handle:
            for relative, fine, coarse in csv.reader(handle):
                path = (source / relative.strip()).resolve()
                if not path.is_relative_to(source):
                    raise ValueError(f"Image path is outside the dataset: {path}")
                category = labels[int(coarse)]
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                if digest in hashes and hashes[digest] != split:
                    raise ValueError(f"Identical image pixels appear in different official splits: {path}")
                hashes[digest] = split
                rows.append((path, split, category, int(fine)))
                counts[split][category] += 1
    # Validate everything first so a bad input does not leave half a dataset.
    for index, (path, split, category, fine) in enumerate(rows):
        folder = output / split / category
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, folder / f"{index:06d}_{path.name}")
    summary = {
        "source": "https://github.com/marcusklasson/GroceryStoreDataset",
        "citation_key": "klasson2019hierarchical", "task": "classification only; no box labels",
        "split_policy": "Official train/val/test lists kept unchanged; coarse labels used.",
        "counts": {split: dict(sorted(values.items())) for split, values in counts.items()},
        "total_images": len(rows), "classes": sorted(set(labels.values())),
        "validation_missing_classes": sorted(set(counts["train"]) - set(counts["val"])),
    }
    (output / "SOURCE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"total_images": len(rows), "classes": len(labels),
                      "splits": {split: sum(values.values()) for split, values in counts.items()},
                      "validation_missing_classes": summary["validation_missing_classes"]}, indent=2))
    return summary


#------------------------------------------------------------
# RUN THE IMPORTER WITHOUT TRAINING
#------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare coarse labels from the Swedish Grocery Store Dataset.")
    parser.add_argument("--source", default="data/raw/grocery-store-dataset/dataset")
    parser.add_argument("--output", default="data/grocery-store/classification")
    args = parser.parse_args()
    import_grocery_store(args.source, args.output)
