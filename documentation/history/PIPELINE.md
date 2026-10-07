# One RGB pipeline

Historical notes from before the production/training reorganisation. Commands and paths below may be outdated; use ../README.md for the current setup.

`grocery_rgb/pipeline.py` contains the model loading, YOLO preprocessing,
instance-mask extraction, original-image masked crops, and ViT classification.
Both `predict.py` and the review tool call `RGBPipeline.predict(image)`.
The review tool only handles images, editing, collaboration and its job queue.

## Commands

Run these from the repository with the virtual environment activated:

```bash
# Save masks, category predictions, masked crops and an annotated image.
python main.py predict --config configs/inference.yaml --source path/to/images

# Open the editor with exactly the same model configuration.
python main.py review --config configs/inference.yaml
```

Review is at http://localhost:9000. Run one review server at a time. The existing
`python annotation_tool/run.py` command also uses the shared configuration.
Its `--model-config` option chooses another inference YAML.

The default configuration uses pretrained YOLO26l-seg and no grocery classifier.
`classifier.checkpoint: null` explicitly selects baseline mode. The pipeline
returns basic category hints where possible and `unknown` otherwise, with null
classification confidence. It never pretends these hints came from ViT.

## Use your saved models

Copy `configs/inference.yaml` to `configs/trained.yaml`. Change:

```yaml
classifier:
  checkpoint: ../runs/classifier/best.pt
prediction:
  segmentation_weights: ../runs/segmenter/train/weights/best.pt
```

Keep the other settings in the copied file. Use the actual paths printed by
training. Paths are relative to the YAML file; both checkpoint paths must exist.
The classifier must be our ViT-L/32 checkpoint, and YOLO must be a segmentation
model. An explicitly configured missing classifier is an error, not a fallback.

```bash
python main.py predict --config configs/trained.yaml --source path/to/images
python main.py review --config configs/trained.yaml
```

Choosing a configuration when starting review selects the model for the entire
shared workspace. Restart to switch models. Existing annotations are preserved;
use **Suggest masks** to replace a draft with fresh model predictions. New uploads
are processed automatically. There is no separate review network and no runtime
model-switch API that could change another collaborator's model unexpectedly.

## Call from the fridge application

```python
from PIL import Image
from grocery_rgb.pipeline import RGBPipeline, load_pipeline_config

pipeline = RGBPipeline(load_pipeline_config("configs/trained.yaml"))
with Image.open("shelf.png") as image:
    items = pipeline.predict(image)
```

Create the pipeline once and reuse it. Weights load lazily on the first image;
calls are serialized to avoid concurrent model loading/inference. Each item has
`box_xyxy`, `polygon`, `category`, `detection_confidence`,
`classification_confidence`, source-label and preprocessing metadata. Coordinates
refer to the EXIF-corrected original image. ViT sees the same masked crop in both
prediction and review, with its shared gentle preprocessing.

The file runner saves JSON, masked PNG crops and an annotated JPEG. Set a new
`prediction.output` folder for each run; existing output folders are protected.

## Training and limits

`train-all` now trains the segmenter followed by ViT and writes a configuration
usable by both commands. It requires segmentation annotations; the Swedish
classification dataset alone cannot train the segmenter. `train-detector` remains
an optional box-only baseline experiment, outside the shared mask pipeline.

Only one exterior contour per item is currently retained; holes and detached
visible fragments are not fully represented. See `PREPROCESSING.md` for the
current difference between YOLO training inputs and enhanced inference inputs.
No tests, training or image reruns were performed for this refactor.
