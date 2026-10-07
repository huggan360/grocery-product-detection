# HOW-TO: the smart fridge on the Raspberry Pi

This file is for whoever installs or runs the system on the Raspberry Pi, including an
AI agent such as Codex. It explains how the system works, how to install it, how the
YOLO model is put on the AI HAT, and how to check that everything works.
Read all of it before changing anything.

## 1. The system in one page

Hardware: Raspberry Pi 5, Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS), a door switch on a
GPIO pin, and two cameras:

- **fridge** camera: looks at the fridge opening (items going in and out of the fridge).
- **door** camera: on top of the fridge, looking outwards at the door shelves.

Two repositories, side by side in the same parent folder:

| Repository | Role |
| --- | --- |
| `smart-fridge-edge` | The **orchestrator**. It owns the door switch, the cameras, the SQLite database and all stored files. |
| `grocery-product-detection` | The **ML step** (`main.py`), the **review tool** (web server, port 9000), the models and their configuration. |

What happens on every door event:

```text
door opens  -> orchestrator starts recording both cameras (rpicam-vid)
door closes -> recording stops
            -> segmentation: ffmpeg finds the moments with motion and cuts them out
            -> each segment is queued in SQLite (table captures)
            -> ML worker runs: python main.py --production|--test ... for each segment
                 YOLO26m on the Hailo-8 finds objects in each sampled frame (5/s)
                 ByteTrack follows each object over time
                 zones: an object seen stably INSIDE then OUTSIDE = "out", the reverse = "in"
                 ViT-Small on the CPU classifies the objects that moved
            -> orchestrator stores the result: predictions (full JSON) + movements (one row per item)
```

| Model | Runs on | File |
| --- | --- | --- |
| YOLO26m detector (640x640) | **Hailo-8 AI HAT** | `models/hailo/yolo26m.hef` (downloaded by setup) |
| ViT-Small/16 classifier (224x224) | **Pi CPU** (PyTorch) | `models/vit/pretrained/` (downloaded by setup), later a trained `models/vit/<run>/best.pt` |

Until grocery models are trained, YOLO knows only the 80 COCO classes (bottle, cup, bowl,
banana, apple, orange, broccoli, carrot, sandwich...). Cartons and jars may be missed.
ViT gives ImageNet guesses, shown as `imagenet-...`. This is expected: the first goal is
correct recording, segmentation, tracking and in/out decisions, plus collecting clips.

### Modes

Both repositories accept the same three modes:

| Mode | Use | Extra behaviour |
| --- | --- | --- |
| `--production` | normal use | full recordings deleted after cutting; newest 3 segments' files kept |
| `--test` | monitoring the live system | every segment, with its prediction, also appears in the review tool; full recordings kept in `smart-fridge-edge/database/recordings/` |
| `--data-aquisition` | collecting training data | nothing is purged; results also go to `acquisition_records` |

### The review tool (port 9000)

`grocery-product-detection/annotation_tool` is a web server on the Pi. From any computer
on the same network, open `http://<pi-hostname>.local:9000`. There you can:

- see every clip, its tracks and the in/out movements (with `--test`, live clips appear automatically),
- upload or **Record** a clip, which runs through exactly the same pipeline,
- **draw the inside/outside zones** for each camera (used by production too),
- correct categories and directions per track, then **Export reviewed** crops for training.

It has no passwords, so only use it on a trusted network.

## 2. Install (on the Pi)

Both repositories must end up next to each other, for example `~/grocery-product-detection`
and `~/smart-fridge-edge`. From the development PC:

```bash
cd grocery-product-detection
bash deploy/sync_to_pi.sh <user>@<pi-hostname>.local
```

This copies both repositories and never touches the Pi's `.venv`, `data/` or `database/`.
`config/pi.toml` is copied only the first time, because it holds the Pi's door pin.
The edge repository can also be cloned from GitHub (`Alolecoc/smart-fridge-edge`,
branch `feat/video-door-capture` until it is merged).

Then on the Pi:

```bash
# ML + Hailo-8 + review tool
cd ~/grocery-product-detection
bash deploy/setup_pi.sh            # if it says the Hailo-8 is not visible: reboot, run again
bash deploy/install_review_service.sh

# Orchestrator
cd ~/smart-fridge-edge
bash scripts/setup_pi.sh
```

`deploy/setup_pi.sh` does the following:

1. Installs `hailo-all` (driver, firmware, HailoRT and its Python bindings), `ffmpeg` and `rpicam-apps`.
2. Creates `.venv` with `--system-site-packages`, because HailoRT's Python module comes from apt.
3. Installs the requirements.
4. **Puts YOLO on the AI HAT.** It downloads the precompiled `yolo26m.hef` matching the installed
   HailoRT (see section 6).
5. Downloads ViT-Small for the CPU.
6. Fails if the Hailo-8 cannot be used.
7. Runs `python -m utils.hailo_probe`.

A good probe looks like this:

```text
Backend: hailo
YOLO HEF: .../models/hailo/yolo26m.hef
  input  yolo26m/input_layer1 (640, 640, 3)
  output ... (80, 80, 80) ... (20, 20, 4)      # six outputs for YOLO26
Detector: 20-60 ms per frame, ...
ViT-Small on CPU: ... ms for 1 crops
```

If it says `Backend: torch`, the Hailo-8 is not being used. Fix that before continuing:
check `hailortcli fw-control identify`, then reboot.

## 3. Door switch (GPIO17)

The door switch is on **BCM GPIO17**, already set in `smart-fridge-edge/config/pi.toml`.
Check it live:

```bash
cd ~/smart-fridge-edge
.venv/bin/python run.py --config config/pi.toml --check-door
```

Open and close the door. The output must change between `level=high` and `level=low`, and
print `door OPEN` while the door is open and `door closed` while it is closed. If the
words are reversed, change `open_level`. If the level never changes, check the wiring, or ask
the user and try another pin with `--check-door --pin <N>`.

```toml
[door]
gpio_pin = 17
open_level = "high"   # the level printed while the door is OPEN
pull_up = true        # switch to GND: true. Switch to 3.3 V: false
```

## 4. Cameras

```bash
rpicam-hello --list-cameras
.venv/bin/python run.py --config config/pi.toml --check-cameras    # in smart-fridge-edge
```

Each camera must report `OK` and about 3 s of video. Configure the cameras in both places,
with the same names (`fridge`, `door`):

- `smart-fridge-edge/config/pi.toml` `[cameras.fridge]`, `[cameras.door]`: used for live recording.
- `grocery-product-detection/configs/video.yaml` `capture:`: used by the review tool's Record button.

Use `kind = "rpicam"` with `index` = CSI port for Pi camera modules, or `kind = "v4l2"` with
`device = "/dev/videoN"` for USB cameras. A camera can be held by only one program at a time,
so do not press Record in the review tool while the door is open.

## 5. Zones: the in/out decision (required)

Without zones, no movement can be decided and results carry a warning
`Set both zones before interpreting in/out movements.`

1. Open `http://<pi-hostname>.local:9000`, enter a name, press **Record → fridge** (about 5 s, door open).
2. When the clip opens, press **Draw inside** and drag a rectangle over the fridge interior.
   Press **Draw outside** and drag one over the area in front of the fridge. The zones
   must not overlap; leave a gap between them, since the gap counts as neither zone.
3. Press **Save zones & re-check**. The zones are saved to
   `grocery-product-detection/data/acquisition/zones.json` and used by every later run.
4. Repeat for **door**: inside = the door shelves, outside = the room.
5. Move an item from outside to inside in front of the camera, record it, and check that the
   clip shows **IN**. Taking it back out must show **OUT**. If a direction is wrong, the
   zones are probably swapped.

## 6. The models on the AI HAT

The Hailo-8 runs only compiled `.hef` files, and a HEF must match the HailoRT version.
`deploy/download_hefs.sh` reads `hailortcli --version` and downloads from the Hailo Model Zoo:

| HailoRT | Model Zoo | Detector |
| --- | --- | --- |
| 4.23 or newer 4.x | v2.18 | `yolo26m.hef` (default) |
| 4.18 to 4.22 | v2.14 to v2.16 | only `yolov8m.hef`: set `detector.hef: hailo/yolov8m.hef` in `configs/video.yaml`, or `sudo apt full-upgrade` to get HailoRT 4.23 |

The code decodes both: YOLO26's raw outputs (six tensors) and YOLOv8's on-chip NMS output.
`configs/video.yaml` → `backend: auto` uses the Hailo-8 when present. ViT always runs on the CPU.

Only one process can hold the Hailo-8. The review tool and the live ML runs take turns:
each releases it after a clip, and a waiting process retries for up to 3 minutes.

**Our own trained YOLO** is compiled on an x86-64 Linux PC (not on the Pi), with the Hailo
Dataflow Compiler 3.x from the Hailo Developer Zone:

```bash
python -m utils.export_hailo models/yolo/<run>/train/weights/best.pt \
    --data data/prepared/detection/dataset.yaml --name fridge-yolo
```

Copy `models/hailo/fridge-yolo.hef` and `fridge-yolo.txt` to the Pi's `models/hailo/`. Set
`detector.hef: hailo/fridge-yolo.hef` and `detector.hef_labels: hailo/fridge-yolo.txt`.
**Our own trained ViT** needs no compiling: copy `models/vit/<run>/best.pt` to the Pi and set
`classifier.checkpoint: vit/<run>/best.pt`. Restart both services afterwards.

## 7. Run the live system

```bash
cd ~/smart-fridge-edge
bash scripts/install_pi_service.sh test         # monitoring: clips + predictions in the review tool
# later: bash scripts/install_pi_service.sh production
journalctl -u smart-fridge -f                   # live log
```

The service runs `python run.py --config config/pi.toml --run --test` and restarts on failure.
Expected log for one door event:

```text
door opened; recording
door closed
recording stopped (door closed)
queued 2 segments
```

Each queued segment is then processed by ML. Every ML run reloads the models, so expect several
seconds per segment; the door can be used again meanwhile.

Check results:

```bash
sqlite3 ~/smart-fridge-edge/database/fridge.sqlite3 \
  "SELECT event_id, camera, direction, category, round(confidence,2), start_seconds FROM movements ORDER BY movement_id DESC LIMIT 20;"
sqlite3 ~/smart-fridge-edge/database/fridge.sqlite3 \
  "SELECT status, error FROM captures ORDER BY created_at DESC LIMIT 10;"
```

In test mode, each segment also appears in the review tool, named `<event>-<camera>-<offset>s.mp4`.

## 8. Tuning the segmentation

`[segmentation]` in `config/pi.toml` decides which parts of a recording go to ML. With `--test`,
full recordings are kept in `database/recordings/<event-id>/`. To see the motion level over time:

```bash
.venv/bin/python run.py --config config/pi.toml --analyse-motion database/recordings/<event>/fridge.mp4
```

Set `motion_threshold` clearly above the level when nothing moves, and below the level when a hand
moves an item. The `recordings` table stores max and mean motion for every recording.
`padding_seconds` adds context before and after; `merge_gap_seconds` joins nearby movements.

## 9. Where things are

| What | Where |
| --- | --- |
| Database (everything) | `smart-fridge-edge/database/fridge.sqlite3`: tables `events`, `recordings`, `captures`, `inference_runs`, `predictions`, `movements`, `acquisition_records` |
| Queued segments | `smart-fridge-edge/database/captures/<capture-id>/clip.mp4` |
| ML run files and logs | `smart-fridge-edge/database/runs/<run-id>/` (`inference.log`, `result.json`) |
| Full recordings (test/acquisition) | `smart-fridge-edge/database/recordings/<event-id>/` |
| Zones | `grocery-product-detection/data/acquisition/zones.json` |
| Review tool clips and reviews | `grocery-product-detection/data/acquisition/videos/` |
| Live system settings | `smart-fridge-edge/config/pi.toml` |
| Model and tracking settings | `grocery-product-detection/configs/video.yaml` |
| Request/result format | `grocery-product-detection/documentation/CONTRACT.md` |
| Orchestrator details | `smart-fridge-edge/docs/ml-integration.md` |

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| Door events never start | `--check-door`; wrong `open_level` means the door looks always open or always closed. |
| `camera failed to start` | `--check-cameras`; is the review tool recording at the same time? Is `index` right? |
| Captures `failed` | `database/runs/<run-id>/inference.log`, or `captures.error` in SQLite. |
| `No module named ...` in inference.log | `ml.python` in `pi.toml` must be `../grocery-product-detection/.venv/bin/python`; rerun `deploy/setup_pi.sh`. |
| `Backend: torch` / slow YOLO | The Hailo-8 is not used: `hailortcli fw-control identify`, reboot, `deploy/setup_pi.sh`. |
| HEF fails to load | The HEF and HailoRT versions do not match: rerun `deploy/download_hefs.sh` (section 6). |
| No movements, warning about zones | Section 5. |
| Movements in the wrong direction | Inside and outside zones are swapped. |
| Everything is `imagenet-...` | Expected until a grocery ViT is trained (section 6). |
| Review tool unreachable | `systemctl status fridge-review`; same network; port 9000. |

Before changing code in `smart-fridge-edge`, read its `AGENTS.md`. Run
`pre-commit run --all-files --hook-stage pre-commit` and `python scripts/run_pre_push.py`
there. In `grocery-product-detection`, run `python -m unittest discover -s tests` and
`python -m unittest discover -s annotation_tool/tests`.
