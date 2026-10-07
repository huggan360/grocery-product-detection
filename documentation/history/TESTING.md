# Verification performed during setup

Historical notes from before the production/training reorganisation. Commands and paths below may be outdated; use ../README.md for the current setup.

No dataset training, optimizer updates, or pretrained model downloads were performed.

- 20 automated tests passed with `python -m unittest discover -s tests -v`.
- The real YOLO26n architecture completed a CPU forward pass on a synthetic image.
- The real ViT-B/16 architecture completed a CPU forward/backward pass. Gradients reached the body and classification head; no optimizer step was taken.
- Synthetic tests checked crop boundaries, empty detections, unknown rejection, batched crops, saved prediction files, category ordering, checkpoint loading and category transfer, dataset mapping, empty scenes, split leakage, and invalid boxes.
- A mocked YOLO trainer verified script arguments without executing training.
- The Swedish dataset importer decoded the downloaded natural images, preserved the original splits, and checked for identical pixels across splits.
- `python -m pip check` found no broken dependencies.

Tested environment: Python 3.14.7, torch 2.14.0+cpu, torchvision 0.29.0+cpu, ultralytics 8.4.160, Pillow 12.3.0, PyYAML 6.0.3.

These checks establish software/API compatibility in this CPU environment. They do not establish product recognition accuracy, convergence, CUDA behavior, or a successful complete training run. You will verify those when training on your chosen datasets.

## Collaborative annotation tool

The separate `annotation_tool/` has API tests for editing leases, concurrent claims, stale revisions, shared categories, upload validation, draft persistence, reviewed-only exports, mocked camera capture and background model suggestion integration. Its export is also tested through the existing dataset preparation and classifier image-loading code.

Run them with `python -m unittest discover -s annotation_tool/tests -v`. The browser JavaScript passes `node --check annotation_tool/static/app.js`, and the Python modules pass compilation. The local server starts on port 9000.

Per your request, Chromium was not installed and no real-browser automation was run. Pointer interactions, appearance and real camera capture require a manual check in your browser and on the eventual hardware.
