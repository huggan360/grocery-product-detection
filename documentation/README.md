# Grocery models

This repository is the ML step in `smart-fridge-edge`. The edge application owns
the SQLite database, event IDs, capture queue, result ingestion and scheduled cleanup.
This repository reads a request JSON and writes a result JSON. It does not write
to the orchestrator database. No models are trained during production or acquisition.

## Layout

```text
main.py                 Run predictions in acquisition or production mode
train.py                Train YOLO, ViT, or both
production/             Shared inference, preprocessing, masks and JSON handoff
training/               Training, datasets, metrics and evaluation
utils/                  Data preparation, public-dataset import and review import
models/yolo/            YOLO input checkpoints and training outputs
models/vit/             ViT input checkpoints and training outputs
configs/inference.yaml  Model selection and prediction settings
configs/training.yaml   Data, input models and output folders for training
annotation_tool/        Shared image/mask editor, using production/pipeline.py
documentation/          Guides, sources and historical notes
data/                   Existing datasets and annotation workspace (not in Git)
```

Activate `.venv` and run commands from this repository. Install `requirements.txt`
and `annotation_tool/requirements.txt` if setting up a new environment.

## Prediction

The current trial enables `classifier.baseline: imagenet`. With no grocery checkpoint,
this loads the original ImageNet-pretrained ViT-L/32 head and displays labels prefixed
`imagenet-`. These are guesses from 1,000 ImageNet classes, not the grocery label set.
The demo threshold is zero so the best guess stays visible even when uncertain.
Correct these labels to grocery categories before approving training annotations.
Remove `baseline: imagenet` to return to YOLO-only hints. A grocery checkpoint takes
precedence over this baseline. Downloaded weights are kept in `models/vit/pretrained`.

The orchestrator normally builds the request and calls this command:

```bash
python main.py --production --request /path/to/request.json \
  --record-id CAPTURE_ID --output /path/to/runs/RUN_ID/result.json
```

Use `--data-aquisition` (also spelled `--data-acquisition`) for persistent collection.
Both modes use exactly the same network. In acquisition mode the orchestrator saves
an additional row in `acquisition_records`, and excludes its files from cleanup.
No server is started by `main.py`.

The shared inference pipeline is YOLO26l-seg on an enhanced copy, then optional
ViT-L/32 on gently enhanced masked crops from the original. Input sizes remain
640 for YOLO and 224 for ViT. Missing detections and merged items remain possible.
One exterior polygon per instance is supported; holes/disconnected parts are lost.

Set `prediction.segmentation_weights` to `yolo/...pt` and `classifier.checkpoint`
to `vit/...pt` in `configs/inference.yaml`. These paths are relative to
`models_directory` (default `../models` relative to the config). `--models-dir`
overrides that root. A null ViT checkpoint explicitly selects category-hint-only
baseline mode; a configured missing checkpoint is an error.

Production retains masks/crops for the latest **three completed image runs**, including
the newest run, by default. Set `retention_runs` in the inference config for standalone
use. Give each run a separate directory under the same output root. Never put unrelated
files in managed run directories. The orchestrator passes `--managed-retention` and uses
its own retention setting, so only it deletes managed files and updates SQLite.

## Training

```bash
python train.py yolo
python train.py vit
python train.py both --config configs/training.yaml
python train.py vit --weights models/vit/previous-run/best.pt
```

Training never starts automatically. Set `segmenter.data` to the exported
`segmentation/dataset.yaml` and `classifier.data` to `classification_masked/` for
real-fridge fine-tuning. The default classifier data points at the existing public
Swedish grocery dataset. Classification-only images cannot train YOLO masks.
Choose new output folders under `models/yolo` and `models/vit` for each run. Saved ViT
checkpoints carry the category order and architecture; old ViT-B/16 weights cannot
be loaded into ViT-L/32. With no previous grocery checkpoint, ViT starts from ImageNet;
its pretrained downloads are cached under `models/vit/pretrained`.

YOLO training currently uses unenhanced dataset images. Evaluate the contrast/sharpening
option on held-out data; turn it off or prepare matching enhanced training copies before
relying on it. ViT uses the same gentle preprocessing during training and inference.

Utilities run as modules, for example `python -m utils.prepare_data --help` and
`python -m utils.import_grocery_store --help`. Evaluation is available through
`python -m training.evaluate --help`.

## Acquisition and review

In the edge repository, export the persistent acquisition rows:

```bash
python run.py --config config/ml.toml --export-acquisition database/acquisition.json
```

Then in this repository:

```bash
python -m utils.import_acquisition ../smart-fridge-edge/database/acquisition.json --split train
python annotation_tool/run.py
```

Open http://localhost:9000. Import copies the original images and existing model masks;
it does not rerun inference. Reimporting the same run skips it to preserve human edits.
Each door event gets one collection, so images from one event cannot be assigned
different splits. The review workspace stores edits separately; raw acquisition rows
in the edge database remain immutable. Export reviewed data from the browser to train.
Use **Suggest masks** only when intentionally replacing a draft with new predictions.

To review with saved models, pass `--model-config configs/my-models.yaml` to
`annotation_tool/run.py`. It calls the same `production.pipeline.RGBPipeline` as main.

See [CONTRACT.md](CONTRACT.md) for the JSON handoff, [DATASETS.md](DATASETS.md) and
[references.bib](references.bib) for datasets. `history/` preserves earlier notes;
their commands are no longer the current interface.
