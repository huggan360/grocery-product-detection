# Image preprocessing

Historical notes from before the production/training reorganisation. Commands and paths below may be outdated; use ../README.md for the current setup.

The shared code is in `grocery_rgb/preprocessing.py`.

- YOLO prediction uses an RGB copy with local brightness contrast (CLAHE,
  clip limit 3.0, 8 by 8 tiles) and mild sharpening (amount 0.35).
  The image dimensions stay the same; YOLO still uses the configured 640 input size.
- ViT crops come from the original image, with up to 8% extra vibrance and
  5% extra contrast. This is part of the shared classifier transform, so it
  runs during classifier training, evaluation and prediction.
- Original images, editor previews and exported training images stay unchanged.
  Enhanced YOLO images are not passed to ViT.

Set `prediction.yolo_preprocessing: false` in `configs/inference.yaml` to disable
it for both prediction and review, then restart the review server. Both use the
same shared pipeline; see `PIPELINE.md`.

Existing annotations are not regenerated automatically. Open an image and
choose **Suggest masks** to use the new processing. This replaces the current
draft suggestions after the editor's confirmation.

This is an experimental inference preprocessing option, not a measured accuracy
improvement. Stronger contrast can also amplify shadows and printed packaging.
It does not remove the background or guarantee separation of touching products.
No tests or training were run for this change.

The YOLO training and standalone validation scripts currently read the original
dataset images directly. Before adopting preprocessing for a trained detector,
compare it on held-out annotations and either prepare enhanced YOLO dataset
copies with `prepare_yolo_image` for consistent training/validation, or disable
this option. Keep the original dataset for the classifier and the editor.
