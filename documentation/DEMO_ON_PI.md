# Try the trained demos on existing Pi clips

Use the `main` checkout for this demo. The `prod` branch and normal
`configs/video.yaml` are not changed by the demo configuration.

The demo uses YOLO26m `yolo_demo1` and ViT-Small `vit_demo2`. Both run on the CPU
with `configs/video_demo_cpu.yaml`. The trained YOLO `.pt` cannot run directly on
the Hailo-8: compiling a matching `.hef` is a separate step. CPU detection on a Pi
will be slower than the installed accelerator model.

Git excludes checkpoints and data. Transfer these files separately into the main
checkout, keeping their relative paths:

```text
models/yolo/yolo_demo1/train/weights/best.pt
models/vit/vit_demo2/inference.pt
```

`inference.pt` contains the same ViT weights, architecture and 105 class names as
the best training checkpoint, without optimizer/scheduler/scaler state. Do not
use a checkpoint from the earlier partial training run.

From the main checkout, with its Python environment active:

```bash
python annotation_tool/run.py --config annotation_tool/demo.yaml
```

The demo reviewer listens on `127.0.0.1:9001` and uses the separate library
`data/demo/acquisition/videos/`. Normal clips and reviews remain under
`data/acquisition/`. The demo's class list uses the exact trained class names.

To review previous clips without changing their original results or annotations,
copy their complete video-library folders into the demo library first, and copy
the existing zone definitions to `data/demo/acquisition/zones.json`. Then select a
clip in the demo reviewer and choose **Run models again**. Inspect the resulting
tracks and category confidence; catalogue validation is not a fridge-video
accuracy estimate.

Access the localhost-only reviewer over SSH forwarding:

```bash
ssh -N -L 9001:127.0.0.1:9001 hugo@smartfridge
```

Then open `http://localhost:9001`. Keep this SSH connection running while using
the reviewer. With a per-user Tailscale client, the SSH host configuration can use
`ProxyCommand tailscale nc %h %p` to reach the Pi through that client's networking.

The demo does not retrain models, start the door-recording service, or switch the
production detector. It uses the existing ByteTrack and zone-crossing pipeline.
