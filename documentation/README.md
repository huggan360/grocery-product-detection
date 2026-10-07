# Grocery models

This repository is the ML step in `smart-fridge-edge`. Two cameras film the fridge
opening and the door. Each clip goes through YOLO26m detection and ByteTrack tracking,
then inside/outside zone crossing, then ViT-Small classification of the moving objects.
The result says what went in or out, and when. On the Raspberry Pi, YOLO runs on the
Hailo-8 AI HAT+ (26 TOPS) and ViT-Small runs on the CPU.

The edge application owns the SQLite database, event IDs, capture queue, result ingestion
and scheduled cleanup. This repository reads a request JSON and writes a result JSON.
It does not write to the orchestrator database. No models are trained during production
or acquisition.

## Layout

```text
main.py                 Run one clip (acquisition or production mode)
train.py                Train YOLO, ViT, or both
production/             Video pipeline, zones, Hailo/PyTorch backends, JSON handoff
training/               Training, datasets, metrics and evaluation
utils/                  Training data preparation, recording, Hailo probe and export
models/yolo/ vit/       PyTorch checkpoints and training outputs
models/hailo/           Compiled .hef models for the Hailo-8 (downloaded on the Pi)
configs/video.yaml      Models, zones, cameras, tracking settings
configs/training.yaml   Data, input models and output folders for training
annotation_tool/        Video upload/record, zone drawing and track review
deploy/                 Copy to and set up the Raspberry Pi
documentation/          Guides, sources and historical notes
data/                   Datasets and data/acquisition (clips, zones, reviews; not in Git)
```

Activate `.venv` and run commands from this repository. Install `requirements.txt`
and `annotation_tool/requirements.txt` if setting up a new environment.
See [HOW-TO.md](../HOW-TO.md) for the whole system on the Raspberry Pi: setup, door pin,
cameras, zones, the Hailo-8 and the live services.

## Review tool

```bash
python annotation_tool/run.py            # http://localhost:9000
```

Upload a clip (or **Record** on the Pi) for the fridge or door camera. It goes through
the same pipeline as `main.py`. Draw the inside and outside zones once per camera.
The zones are saved to `data/acquisition/zones.json`, which production also uses.
Then check each clip's movements, correct the tracks and save the review.
**Export reviewed** downloads labelled track crops for ViT training.

## Production

The orchestrator builds a schema 2 request ([CONTRACT.md](CONTRACT.md)) and calls:

```bash
python main.py --production --request /path/to/request.json \
  --record-id CAPTURE_ID --output /path/to/runs/RUN_ID/result.json
```

Use `--data-aquisition` (also spelled `--data-acquisition`) for persistent collection, and
`--test` to also publish every clip and its prediction to the review tool (live monitoring).
All modes use exactly the same models. `--models-dir` overrides the model root, and
`--managed-retention` leaves cleanup to the orchestrator. Without it, production keeps
the video working copy and crops for the latest `retention_runs` clips. The original
clip and `result.json` are never deleted.

Until grocery models are trained, YOLO26m detects only COCO classes and ViT-Small gives
ImageNet guesses prefixed `imagenet-`. After training, set `classifier.checkpoint` (ViT,
CPU) and `detector.hef` + `hef_labels` (YOLO, Hailo-8) in `configs/video.yaml`.

## Training

```bash
python -m utils.prepare_data --help      # box CSVs -> detection + classification data
python train.py yolo                     # YOLO26m product boxes
python train.py vit                      # ViT-Small/16 grocery classifier
python train.py vit --weights models/vit/previous-run/best.pt
python -m utils.export_hailo --help      # compile YOLO for the Hailo-8 (ViT stays on CPU)
```

Training never starts automatically. YOLO26m is chosen so it compiles for the Hailo-8, and
ViT-Small so it is fast enough on the Pi's CPU (see ../HOW-TO.md). Choose new output folders under `models/yolo` and
`models/vit` for each run. Saved ViT checkpoints carry the category order and architecture;
older ViT-L/32 checkpoints (format 1) cannot be loaded. With no previous grocery checkpoint,
ViT starts from ImageNet; its pretrained downloads are cached under `models/vit/pretrained`.

YOLO training uses unenhanced dataset images, while the pipeline sends YOLO a contrast-
enhanced copy (`detector.preprocessing`). Evaluate this on held-out clips before relying
on it. ViT uses the same gentle preprocessing during training and inference.

Utilities run as modules, for example `python -m utils.import_grocery_store --help`.
Evaluation is available through `python -m training.evaluate --help`.

See [DATASETS.md](DATASETS.md) and [references.bib](references.bib) for datasets.
`history/` preserves notes from the earlier image-based version, which is kept in Git
commit `74f135f` (and the `grocery-product-detection-image-backup-20261007` copy).
Their commands are no longer the current interface.
