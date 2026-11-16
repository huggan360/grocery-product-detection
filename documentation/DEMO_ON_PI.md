# Use the trained demos on the Pi

The Pi's existing checkout is `/srv/smart-fridge/grocery-product-detection`, on
`main`. Its existing annotation tool is `fridge-review.service`, on port **9000**.
The remote `prod` branch is not updated. Device-specific configuration and videos
remain in the existing checkout, with configuration backups under
`data/deployment_backups/`.

The installed local video configuration uses the trained `yolo_demo1` on the
Hailo-8 (`backend: hailo`, `detector.hef: hailo/yolo_demo1.hef`,
`detector.hef_labels: hailo/yolo_demo1.txt`) and the trained ViT-Small `vit_demo2`
on the CPU. The detector has one class, `item`; ViT supplies the grocery category.

PyTorch inference uses a fixed square letterbox, matching training and the HEF.
Rectangular padding caused very low confidence on the portrait fridge frames.
With matching padding, `yolo_demo1` produces high-confidence boxes covering most
of the frame. Its training boxes covered entire product catalogue images, which
does not teach reliable product localization in fridge scenes. Use boxes from
representative fridge footage for further detector fine-tuning.

The HEF was compiled locally with DFC 3.33.0, obtained from
[DeGirum's public compiler image](https://hub.docker.com/r/degirum/dg_local_compiler).
Calibration used catalogue images and actual fridge frames. Optional fine-tuning
was disabled; quantization and compiler optimization levels were both 0. Build
provenance and hashes are saved beside the HEF in `yolo_demo1.build.json`.

Git excludes model files. Their deployed locations are:

```text
models/yolo/yolo_demo1/train/weights/best.pt
models/hailo/yolo_demo1.hef
models/hailo/yolo_demo1.txt
models/vit/vit_demo2/inference.pt
```

`inference.pt` contains the same ViT weights, architecture and 105 class names as
the best training checkpoint, without optimizer/scheduler/scaler state.

## Reclassify all existing videos

After selecting the models and restarting the reviewer to load them:

```bash
cd /srv/smart-fridge/grocery-product-detection
.venv/bin/python -m utils.reclassify_videos --all
```

Add `--wait` to print progress until the batch completes. The command uses the
reviewer's HTTP API and returns a batch ID, queued video IDs and backup path.
Already queued/running clips are skipped. Original video files are retained.

Previous predictions, track crops and human reviews are backed up under
`data/acquisition/reclassification_backups/<batch>/<video-id>/`. Active reviews are
reset because the new detector may assign different tracking IDs. Reprocessed
clips appear in the same annotation tool as they finish. Failed clips show their
error messages; they are not silently counted as successful.

The per-clip **Run models again** button also backs up the prior results/review.

## Open the annotation tool

From the desktop:

```bash
ssh -N -L 9000:127.0.0.1:9000 hugo@smartfridge
```

Open `http://localhost:9000`. Keep the SSH connection running. Both ends of the
forward must use **9000**, not 900. The dedicated SSH key is configured on this
desktop; Tailscale provides the connection to the Pi.

The optional `annotation_tool/demo.yaml` / `configs/video_demo_cpu.yaml` templates
still support a separate port-9001 library, but the current Pi deployment uses the
original port-9000 tool and video library, as requested.
