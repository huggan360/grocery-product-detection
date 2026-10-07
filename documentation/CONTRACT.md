# JSON contract, version 1

The edge database row ID is `capture_id`. Several captures (for example one per
shelf camera) share `event_id`. Each attempt also has an edge-owned `run_id`.
Dates use ISO 8601 timestamps with timezone. Files use absolute paths on the same host.

Input (`--request`):

```json
{
  "schema_version": 1,
  "capture_id": "capture-uuid",
  "event_id": "door-event-id",
  "captured_at": "2026-10-05T12:00:00+00:00",
  "sensor": "rgb-shelf-1",
  "image_path": "/absolute/database/captures/capture-uuid/image.png",
  "sensor_metadata": {"radar_mode": "profile-2"},
  "attachments": [{"sensor": "radar", "path": "/absolute/raw.bin"}]
}
```

IR/radar files and settings are preserved by the orchestrator; this pipeline only
infers from RGB. It does not yet perform sensor fusion or added/removed inventory logic.

Output (`--output`) contains the same capture/event IDs, `mode`, `captured_at`,
`started_at`, `completed_at`, `sensor`, `image_path`, `image_size`, sensor metadata,
model paths, `objects`, `counts`, and `annotated_path`. Each object has a category,
box, polygon, detection confidence, optional classification confidence, mask/crop
file paths and prediction-source metadata. Empty detections are a valid empty list.
Results are atomically published after masks and crops are written. Failures cause
a nonzero exit; the edge stores the error and does not accept incomplete results.

The edge passes `--managed-retention`, verifies capture/event/mode IDs and stores the
result in `predictions`. Acquisition also goes into `acquisition_records`. For expired
production runs it removes its managed images/masks/results, preserves category records
in SQLite, removes stale result paths/polygons, and records `purged_at`. Original files
submitted from outside the database directory are never deleted.

The review import takes the edge's acquisition export, not direct SQLite access.
Its draft notes retain capture, event and run IDs for later training provenance.
