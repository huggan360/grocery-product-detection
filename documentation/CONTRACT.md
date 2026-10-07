# JSON contract, version 2: video clips

This is the only request `main.py` accepts. Version 1 (one RGB shelf image) was removed
with the image pipeline. The orchestrator still sends version 1 and must be switched by its
team. The command-line flags are unchanged:

```text
<ml-python> <ml-repo>/main.py --production|--data-aquisition
  --record-id <capture-id> --request <absolute-request.json>
  --output <absolute-result.json> [--config <video.yaml>] [--models-dir <model-root>]
  [--managed-retention]
```

The edge row ID is `capture_id`; several clips (for example fridge and door) share
`event_id`. Dates use ISO 8601 with a timezone. Files use absolute paths on the same host.
Failures exit nonzero, and `result.json` is published atomically.

Input (`--request`), one clip from one camera (`fridge` or `door`):

```json
{
  "schema_version": 2,
  "capture_id": "capture-uuid",
  "event_id": "door-event-id",
  "captured_at": "2026-10-07T12:00:00+00:00",
  "camera": "fridge",
  "video_path": "/absolute/database/captures/capture-uuid/fridge.mp4",
  "offset_seconds": 0.0,
  "sensor_metadata": {},
  "attachments": []
}
```

`offset_seconds` is where this clip starts within the door event. The sensors may cut
one event into several clips; times in the result are then event times.

Output (`--output`) keeps the IDs, `mode` and `captured_at`, and adds:

- `movements`: the answer, sorted by time. Each has `track_id`, `direction` (`in`/`out`),
  `category`, `classification_confidence`, `start` and `end` (event seconds).
- `tracks`: every tracked object, with `observations` (frame, time, normalized box,
  detection confidence), `events`, `top_categories`, `label_source`
  (`grocery-vit` or `imagenet-demo`) and `crop_file`.
- `zones`, `tracking_settings`, `backend` (`hailo`/`torch`), `models`, `video`, `warnings`.

A movement is reported only when the same track is seen stably in one zone and then in the
other. An object that disappears is not counted as removed. Results without zones carry a
warning and contain no movements. The run directory also holds `video.mp4` (a working copy),
`poster.jpg` and `track-<id>.jpg` crops. Standalone production keeps these for the latest
`retention_runs` clips; the original clip and `result.json` are never deleted.
