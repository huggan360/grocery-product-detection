# Shared annotation tool

Historical notes from before the production/training reorganisation. Commands and paths below may be outdated; use ../README.md for the current setup.

The editor uses the same `grocery_rgb.pipeline.RGBPipeline` as standalone prediction. See [PIPELINE.md](../PIPELINE.md) for commands and saved-model setup, and [SEGMENTATION.md](SEGMENTATION.md) for mask editing and exports. There is no separate model implementation in the editor.

Run one server on the computer connected to your fridge cameras. Everyone connects to that same server through SSH and opens `http://localhost:9000` in their own browser.

The tool is separate from the training code. It collects and labels RGB images; it never trains a model.

## Start it

From the repository folder:

```bash
source .venv/bin/activate
python -m pip install -r annotation_tool/requirements.txt
python annotation_tool/run.py
```

Dependencies are already installed in this workspace. On this computer, open:

```text
http://localhost:9000
```

The app works immediately with image uploads. Missing trained checkpoints put it in manual annotation mode rather than preventing startup. No browser download, Node build step, external database, or frontend account is needed.

To use a different port:

```bash
python annotation_tool/run.py --port 9001
```

Keep the server terminal open. For an SSH-hosted collection session, run it inside `tmux` so disconnecting the server operator does not stop the app:

```bash
tmux new -s fridge-labels
source .venv/bin/activate
python annotation_tool/run.py
# Detach with Ctrl+B, then D. Return with: tmux attach -t fridge-labels
```

## Connect three or four collaborators

Each collaborator runs this command on **their own computer**, replacing the username and server address:

```bash
ssh -N -L 9000:127.0.0.1:9000 your-user@fridge-computer
```

Then each person opens `http://localhost:9000` and enters their name. Leave the SSH tunnel running. If local port 9000 is already occupied, use:

```bash
ssh -N -L 9001:127.0.0.1:9000 your-user@fridge-computer
```

That person opens `http://localhost:9001` instead. The server still uses port 9000.

Use **one server process**, not one per annotator. Names identify collaborators; they are not passwords. This is a trusted-team tool bound to loopback and intended for access through SSH, not a public internet service. Keep its SQLite database on the server's local disk rather than a network share.

## Your collection workflow

1. One person arranges items on the shelves.
2. Click **Capture shelves** to take one image per enabled camera, or **Upload images** to add saved pictures from one shelf.
3. Enter a collection session name, such as `fridge-day1-session1`, and choose its dataset split. Keep related frames and shelf arrangements together in one session. A session cannot be assigned to two splits.
4. With trained checkpoints configured, the server runs YOLO → crops → ViT in the background. Images become available when suggestions finish. Failed inference still leaves the image available for manual annotation.
5. Each annotator clicks **Next available** or selects a queue image. Other people see who is editing it and take other images.
6. Correct all boxes and categories. Drafts autosave about 650 ms after a change.
7. Click **Save & next**. The image moves to **Saved & edited**, and the next available image opens.
8. Download **Export saved** whenever you want a training snapshot. This does not start training.

The **Unedited** tab includes both untouched images and unfinished drafts. The **Saved & edited** tab contains explicitly reviewed images. Queue and collaborator presence refresh every two seconds. Collaborators edit different images concurrently; this is not simultaneous drawing by several people on the same image.

## Edit boxes

- **Select / V**: select a box, drag its inside to move it, or drag any of its eight handles to resize it.
- **Draw box / D**: drag a rectangle around one visible product.
- Choose a category in the selector. It changes the selected box and becomes the category for your next new box.
- **Add category** shares it with everyone. Use broad names like `milk` or `butter`, with lower-case letters, numbers, hyphens or underscores.
- Edit the four numeric coordinates if you want exact pixel bounds.
- **Delete** removes the selected box. **Ctrl/Cmd+Z** undoes a box edit; **Ctrl/Cmd+Shift+Z** redoes it.
- Zoom buttons and **Fit** help with small products. Coordinates always remain in original image pixels.
- **Ctrl/Cmd+Enter** saves reviewed and opens the next image when focus is outside a text field.
- For an empty shelf, delete false detections and check **No products in this image** before review.
- **Save draft & close** keeps unfinished work in the pending queue and releases the image.

Model confidence is only a suggestion, not proof that a label is correct. Every visible product needs its own box. Look for missed items as well as wrong labels. A saved image can be reopened for correction; changing its annotations returns it to the pending queue until reviewed again.

## Connect your trained models

Set `model_config` in [config.yaml](config.yaml) to the RGB training configuration whose checkpoints you want to use. In that RGB config, set:

```yaml
prediction:
  detector_weights: ../runs/detector/train/weights/best.pt
# ...keep the other prediction settings...
classifier:
  checkpoint: ../runs/swedish_classifier/best.pt
# ...keep the other classifier settings...
```

Use your real trained checkpoint paths. Tool config paths resolve relative to `annotation_tool/config.yaml`; model paths resolve relative to the referenced RGB config. Restart the tool after changing configuration. The model worker loads one YOLO and one ViT instance on the first new image, using the device selected in the RGB config. It never downloads weights automatically.

Your actual trained categories are added to the shared list when their predictions occur. You can add missing categories manually at any time. Low-confidence `unknown` boxes must receive a real category or be deleted before review. Original model suggestions are retained separately from human corrections in the export metadata.

## Set up any number of shelf cameras

Edit the `cameras` list in [config.yaml](config.yaml). There are three disabled examples; add or remove entries as needed. The sources refer to cameras accessible from the **server**, not the annotators' laptops.

USB cameras:

```yaml
cameras:
  - id: shelf-1
    name: Top shelf
    enabled: true
    kind: opencv
    source: 0
    width: 1280
    height: 720
  - id: shelf-2
    name: Bottom shelf
    enabled: true
    kind: opencv
    source: 1
```

Network stream:

```yaml
  - id: shelf-3
    name: Middle shelf
    enabled: true
    kind: opencv
    source: rtsp://192.168.1.50/stream
```

HTTP snapshot camera:

```yaml
  - id: shelf-4
    name: Door shelf
    enabled: true
    kind: snapshot
    source: http://192.168.1.51/snapshot.jpg
```

Restart the server after changing cameras. **Capture shelves** takes frames sequentially, so they are not hardware-synchronized; keep the scene still during capture. A failed camera is reported while successful images are retained. Only one capture batch runs at once. Actual camera indices, stream URLs, permissions and driver behaviour must be tested when the hardware arrives. If using credentials in camera URLs, keep that local config out of Git.

## Where your work is saved

Default storage is:

```text
data/fridge_annotations/
  annotations.sqlite3       shared labels, sessions, editors, revision history
  images/                  original-size, EXIF-corrected RGB PNGs
  thumbnails/              small queue previews
```

Pixel files are immutable; box edits change the database only. Use **Export saved** for dataset snapshots. To back up the complete workspace, stop the server and copy the whole storage folder, including any SQLite auxiliary files. `data/` is already ignored by Git.

## Collaboration and recovery

An image has one editor at a time. Opening it acquires a 90-second lease, renewed every 15 seconds while connected. A closed or disconnected tab eventually releases the image through lease expiry. Every save also checks a revision number, so stale requests fail instead of replacing newer data.

If a lease is lost, editing freezes and the UI offers **Download unsaved draft**. Keep that JSON if needed, leave the image, and reopen the latest revision. Short network failures leave unsaved changes visible in the tab; **Save draft & close** retries saving. Do not close a tab with an unsaved warning unless you intend to discard those local changes. Undo history is local to the current editing session; saved revisions remain in the database history table.

## Export and fine-tune later

The ZIP contains:

```text
detection/
  dataset.yaml
  images/{train,val,test}/
  labels/{train,val,test}/   all boxes use class 0: product
classification/
  {train,val,test}/{category}/   product crops with 5% padding
annotations.csv             compatible with prepare_data.py
annotations.json            human labels, original suggestions and provenance
README.txt
```

Only **reviewed images without an active editing lease** enter an export. Leaving the editor after review releases that lease. Empty reviewed frames receive empty YOLO label files and no classifier crops. The original image IDs, shelf names, collection sessions, splits, editor names and revisions are retained.

Extract the ZIP, copy your RGB config, and set `detector.data` to the extracted `detection/dataset.yaml` and `classifier.data` to the extracted `classification` folder. Set `detector.weights` and `classifier.initialize_from` to your earlier checkpoints, and choose new output folders. Run the existing training scripts when you are ready.

Collect separate training and validation sessions, at least two classifier categories, and representative examples of each category. The app allows small or single-split exports for collection checkpoints; that does not make them sufficient for training. Keep test sessions separate from tuning. The tool enforces consistency within a session name but cannot recognize the same physical arrangement entered under different session names.

## Checks

```bash
python -m unittest discover -s annotation_tool/tests -v
node --check annotation_tool/static/app.js  # optional, if Node is available
```

API tests cover competing editors, simultaneous claims, revision conflicts, lease expiry, drafts, categories, reviewed-only exports, image uploads, model suggestion integration and mocked multi-camera capture. They do not train models. Browser interaction and physical camera tests still need to be performed on your actual setup; Chromium is not required or installed for this tool.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Start the server on port 9000 |
| `config.yaml` | Cameras, model config, categories and storage |
| `app.py` | Web routes, uploads and captures |
| `store.py` | SQLite data, editing leases and revisions |
| `predictions.py` | Background connection to the existing RGB models |
| `cameras.py` | USB, RTSP and snapshot camera adapters |
| `export_data.py` | Reviewed data → training ZIP |
| `static/` | Browser interface, with no build step |
| `tests/` | API and collaboration tests |
