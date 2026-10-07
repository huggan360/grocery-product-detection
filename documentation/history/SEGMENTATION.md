# Segmentation during data collection

Historical notes from before the production/training reorganisation. Commands and paths below may be outdated; use ../README.md for the current setup.

The order is:

```text
Full shelf photo
    -> YOLO26-seg predicts one visible outline per detected object
    -> Crop around each outline and grey out pixels outside it
    -> Optional grocery ViT predicts a category for that masked crop
    -> You correct the outlines, categories and missed objects
    -> Save reviewed annotations for future training
```

There is no separate U-Net. YOLO26-seg predicts masks on the full image before cropping. The original image is kept; masking only changes the classifier input and exported masked crops.

## Baseline model

`data/models/yolo26l-seg.pt` is a COCO-pretrained instance segmentation model. The shared `configs/inference.yaml` points to it. It can suggest masks before you have a grocery-trained model, but it will miss some groceries and produce false detections. The baseline's COCO categories do not cover every grocery category. A `bottle` hint does not identify its contents, so those category fields remain `unknown` until corrected.

The optional grocery ViT checkpoint is taken from `classifier.checkpoint` in the referenced RGB config. Without it, segmentation and manual category labelling still work. With it, ViT receives masked crops. Fine-tune on `classification_masked/` before relying on masked-crop classification; a model trained only on ordinary crops may perform differently.

## Controls

- New uploaded/captured images automatically get segmentation suggestions.
- **Suggest masks** runs the model on the open image, including existing demo images. Replacing a nonempty draft asks first and can be undone.
- **Draw mask / P**: click around the visible product. Click the first point or press **Enter** to finish; **Escape** cancels; **Backspace** removes the last unfinished point.
- Select a product and drag its outline points. **Shift-click an edge** adds a point; **Alt-click a point** removes it. The **Remove point** button does the same for the selected point.
- **Redraw mask** replaces only the selected object's outline, keeping its category.
- Drag inside the selected shape to move it. Its box follows the polygon automatically. Changing numeric box coordinates scales the outline with the box.
- The small grey preview shows the masked crop that will be given to the classifier.
- **Remove mask** keeps the box and category but removes its outline.
- Check **Masks checked** after correcting all object masks, then **Save & next**. Changes clear that check so modified masks are reviewed again.

Annotate only visible pixels, excluding the occluding product. Missing items must be added manually; a model's empty prediction is not proof of an empty shelf. If several products touch, draw one outline per individual product. An empty frame can be included in segmentation training by checking both the empty-frame and masks-checked controls before saving.

This version supports one simple exterior polygon per product. It cannot accurately represent holes or disconnected visible pieces of one instance. The baseline retains the largest visible component and records the component count in its original suggestions. Do not mark such a difficult frame mask-complete if the available outline representation cannot describe it correctly; it can still be used for box training. Self-crossing polygons are rejected on save.

## Export

The original detection and ordinary classification folders remain in the ZIP. Two additional folders contain:

```text
segmentation/
  dataset.yaml
  images/{train,val,test}/
  labels/{train,val,test}/       class 0 followed by normalized polygon points
  export_summary.json           included and excluded image IDs
classification_masked/
  {train,val,test}/{category}/   grey-background product crops
```

Only unlocked, reviewed images explicitly marked **Masks checked**, with a polygon for every product, enter these two folders. An incompletely segmented image is omitted as a whole, so its missing outlines do not teach the model to treat products as background. Ordinary box annotations never become segmentation labels automatically. `annotations.json` retains human polygons, mask-review status, source model hints and annotation provenance.

## Training when enough data has been collected

From the repository root, after extracting the ZIP:

```bash
python train_segmenter.py \
  --data data/fridge-export/segmentation/dataset.yaml \
  --weights data/models/yolo26l-seg.pt \
  --output runs/fridge_segmenter
```

For ViT, copy the RGB config and set `classifier.data` to `../data/fridge-export/classification_masked`. Use the existing `train-classifier` command and set `classifier.initialize_from` if continuing from an earlier grocery checkpoint. Use new output folders.

Then set `prediction.segmentation_weights` and `classifier.checkpoint` in your shared inference YAML to the saved segmenter and ViT checkpoints. Use that same YAML for `main.py predict` and `main.py review`. See [PIPELINE.md](../PIPELINE.md).

Around 3,000 varied shelf images is a collection target, not a guarantee of accuracy. Each can provide several product instances. Use separate recording sessions for training, validation and testing; include different arrangements, lighting, packaging and partial occlusion. Near-identical repeated frames add little variety.

No training or new tests were run for this addition, as requested. Standalone prediction and review now both call the shared segmentation and masked-crop pipeline.

Model and label format references: [YOLO26 segmentation](https://docs.ultralytics.com/tasks/segment/) and [segmentation dataset format](https://docs.ultralytics.com/datasets/segment/).
