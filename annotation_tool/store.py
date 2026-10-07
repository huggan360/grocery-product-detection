#------------------------------------------------------------
# SHARED SQLITE STORAGE AND EXCLUSIVE EDITING LEASES
#------------------------------------------------------------
import json
import math
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from production.masks import polygon_bounds, validate_polygon


class StoreError(Exception):
    """Carry a useful error message and HTTP status back to the browser."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def category_name(value):
    """Keep category names safe for training folder names."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise StoreError("Use a lower-case category name, such as milk or oat-milk.")
    if value in ("unknown", "unlabelled"):
        raise StoreError("Choose a real category; unknown and unlabelled are reserved.")
    return value


class Store:
    """Share images and edits safely between several browser sessions."""

    def __init__(self, root, categories, lease_seconds=90):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        (self.root / "thumbnails").mkdir(exist_ok=True)
        self.database = self.root / "annotations.sqlite3"
        self.lease_seconds = lease_seconds
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS editors (
                    token TEXT PRIMARY KEY, name TEXT NOT NULL, last_seen REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY, split TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, shelf TEXT NOT NULL,
                    collection TEXT NOT NULL, split TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    prediction_state TEXT NOT NULL, prediction_error TEXT,
                    boxes TEXT NOT NULL DEFAULT '[]', suggestions TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0,
                    lock_token TEXT, lock_until REAL NOT NULL DEFAULT 0,
                    edited_by TEXT, reviewed_at REAL);
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY, image_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    editor TEXT NOT NULL, time REAL NOT NULL, state TEXT NOT NULL,
                    boxes TEXT NOT NULL, notes TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS image_state ON images(state, created);
            """)
            # Add mask review state without changing any existing annotations.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
            if "masks_reviewed" not in columns:
                connection.execute("ALTER TABLE images ADD COLUMN masks_reviewed INTEGER NOT NULL DEFAULT 0")
            for name in categories:
                connection.execute("INSERT OR IGNORE INTO categories VALUES (?)", (category_name(name),))

    @contextmanager
    def connect(self, write=False):
        """Open one connection per operation; reserve writes before reading locks."""
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register(self, name):
        """Give this browser tab its own editing identity."""
        name = name.strip()
        if not 1 <= len(name) <= 40:
            raise StoreError("Enter a name between 1 and 40 characters.")
        token = secrets.token_urlsafe(32)
        with self.connect(write=True) as connection:
            connection.execute("INSERT INTO editors VALUES (?, ?, ?)", (token, name, time.time()))
        return {"token": token, "name": name}

    def editor(self, token, connection):
        """Reject editing requests that have no registered browser identity."""
        row = connection.execute("SELECT * FROM editors WHERE token = ?", (token,)).fetchone()
        if row is None:
            raise StoreError("Join the workspace with your name first.", 401)
        connection.execute("UPDATE editors SET last_seen = ? WHERE token = ?", (time.time(), token))
        return row["name"]

    def categories(self):
        """Return the shared category list in alphabetical order."""
        with self.connect() as connection:
            return [row[0] for row in connection.execute("SELECT name FROM categories ORDER BY name")]

    def add_category(self, name, token):
        """Add a category once so all annotators can use it immediately."""
        name = category_name(name)
        with self.connect(write=True) as connection:
            self.editor(token, connection)
            connection.execute("INSERT OR IGNORE INTO categories VALUES (?)", (name,))
        return name

    def collection(self, name, split, connection):
        """Keep every frame from one collection session in the same dataset split."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name):
            raise StoreError("Collection session must use letters, numbers, hyphens or underscores.")
        if split not in ("train", "val", "test"):
            raise StoreError("Split must be train, val or test.")
        existing = connection.execute("SELECT split FROM collections WHERE name = ?", (name,)).fetchone()
        if existing and existing["split"] != split:
            raise StoreError(f"Session {name} already belongs to {existing['split']}. Keep its frames together.", 409)
        connection.execute("INSERT OR IGNORE INTO collections VALUES (?, ?)", (name, split))

    def add_image(self, image_id, filename, shelf, collection, split, size, prediction_state, token):
        """Add a captured or uploaded frame to the pending queue."""
        if not shelf.strip() or len(shelf) > 80:
            raise StoreError("Shelf name must be between 1 and 80 characters.")
        now = time.time()
        with self.connect(write=True) as connection:
            self.editor(token, connection)
            self.collection(collection, split, connection)
            connection.execute("""INSERT INTO images
                (id, filename, shelf, collection, split, width, height, created, updated, prediction_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (image_id, filename[:200], shelf.strip(), collection, split, *size, now, now, prediction_state))

    def row(self, image_id, connection):
        """Find a frame or return a clear not-found response."""
        row = connection.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise StoreError("Image not found.", 404)
        return row

    def public(self, row, connection, token=None, detail=False):
        """Send image details without exposing another person's editing token."""
        value = dict(row)
        owner = value.pop("lock_token")
        active = owner is not None and value["lock_until"] > time.time()
        name = connection.execute("SELECT name FROM editors WHERE token = ?", (owner,)).fetchone() if active else None
        value["locked_by"] = name[0] if name else None
        value["mine"] = active and owner == token
        value["box_count"] = len(json.loads(value["boxes"]))
        value["image_url"] = f"/api/images/{value['id']}/pixels"
        value["thumbnail_url"] = f"/api/images/{value['id']}/thumbnail"
        if detail:
            value["boxes"] = json.loads(value["boxes"])
            value["suggestions"] = json.loads(value["suggestions"])
        else:
            for key in ("boxes", "suggestions", "notes"):
                value.pop(key)
        return value

    def list_images(self, token=None):
        """Show queue updates, saved work, and active collaborators."""
        with self.connect(write=True) as connection:
            if token:
                self.editor(token, connection)
            rows = connection.execute("SELECT * FROM images ORDER BY created DESC").fetchall()
            editors = connection.execute("SELECT name FROM editors WHERE last_seen > ? ORDER BY name",
                                         (time.time() - 45,)).fetchall()
            return {"images": [self.public(row, connection, token) for row in rows],
                    "editors": [row[0] for row in editors]}

    def get_image(self, image_id, token=None):
        """Return current boxes and metadata for one image."""
        with self.connect() as connection:
            return self.public(self.row(image_id, connection), connection, token, detail=True)

    def claim(self, token, image_id=None):
        """Atomically reserve a requested frame or the next available pending one."""
        with self.connect(write=True) as connection:
            self.editor(token, connection)
            now = time.time()
            if image_id:
                row = self.row(image_id, connection)
            else:
                row = connection.execute("""SELECT * FROM images WHERE state = 'pending'
                    AND prediction_state NOT IN ('queued', 'running')
                    AND (lock_token IS NULL OR lock_until <= ? OR lock_token = ?)
                    ORDER BY created LIMIT 1""", (now, token)).fetchone()
                if row is None:
                    raise StoreError("No unclaimed images are ready. Upload or capture more frames.", 404)
            if row["prediction_state"] in ("queued", "running"):
                raise StoreError("Model suggestions are still being prepared.", 409)
            if row["lock_token"] not in (None, token) and row["lock_until"] > now:
                raise StoreError("Someone else is editing this image. Choose another one.", 409)
            # One tab can hold only one frame. Do not strand locks when switching.
            connection.execute("UPDATE images SET lock_token = NULL, lock_until = 0 WHERE lock_token = ?", (token,))
            connection.execute("UPDATE images SET lock_token = ?, lock_until = ? WHERE id = ?",
                               (token, now + self.lease_seconds, row["id"]))
            return self.public(self.row(row["id"], connection), connection, token, detail=True)

    def require_lock(self, row, token):
        """Stop an old tab from overwriting edits after its lease has expired."""
        if row["lock_token"] != token or row["lock_until"] <= time.time():
            raise StoreError("Your edit lease expired or moved to another image. Reopen the image before editing.", 409)

    def heartbeat(self, image_id, token):
        """Keep the current edit lease alive while the tab is connected."""
        with self.connect(write=True) as connection:
            self.editor(token, connection)
            self.require_lock(self.row(image_id, connection), token)
            connection.execute("UPDATE images SET lock_until = ? WHERE id = ?",
                               (time.time() + self.lease_seconds, image_id))

    def release(self, image_id, token):
        """Make an image available to someone else without changing its saved data."""
        with self.connect(write=True) as connection:
            self.editor(token, connection)
            connection.execute("UPDATE images SET lock_token = NULL, lock_until = 0 WHERE id = ? AND lock_token = ?",
                               (image_id, token))

    def validate_boxes(self, boxes, row, connection, reviewed):
        """Require finite in-image boxes and real labels before approving an image."""
        if not isinstance(boxes, list) or len(boxes) > 500:
            raise StoreError("An image can contain up to 500 boxes.")
        known = {entry[0] for entry in connection.execute("SELECT name FROM categories")}
        cleaned, ids = [], set()
        for box in boxes:
            if not isinstance(box, dict) or not isinstance(box.get("id"), str) or not 1 <= len(box["id"]) <= 100:
                raise StoreError("Each box needs a unique ID.")
            if box["id"] in ids:
                raise StoreError("Box IDs must be unique within an image.")
            ids.add(box["id"])
            coordinates = box.get("xyxy")
            if not isinstance(coordinates, list) or len(coordinates) != 4:
                raise StoreError("Each box needs four pixel coordinates.")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coordinates):
                raise StoreError("Box coordinates must be finite numbers.")
            x1, y1, x2, y2 = coordinates
            if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
                raise StoreError("Boxes must have positive size and stay inside the image.")
            category = box.get("category", "")
            if category not in known and (reviewed or category not in ("", "unknown", "unlabelled")):
                raise StoreError("Choose a category for every box before saving as reviewed.")
            item = {"id": box["id"], "xyxy": coordinates, "category": category}
            if box.get("polygon") is not None:
                try:
                    item["polygon"] = validate_polygon(box["polygon"], row["width"], row["height"])
                except ValueError as error:
                    raise StoreError(str(error)) from error
                # Derive boxes from the mask so a moved outline cannot leave a stale crop.
                item["xyxy"] = polygon_bounds(item["polygon"])
            cleaned.append(item)
        return cleaned

    def save(self, image_id, token, revision, boxes, notes, reviewed=False, empty_confirmed=False, masks_reviewed=False):
        """Save a versioned draft or approve the image for training export."""
        if not isinstance(notes, str) or len(notes) > 4000:
            raise StoreError("Notes must be at most 4,000 characters.")
        with self.connect(write=True) as connection:
            name = self.editor(token, connection)
            row = self.row(image_id, connection)
            self.require_lock(row, token)
            if revision != row["revision"]:
                raise StoreError("This image has a newer revision. Reopen it to avoid overwriting changes.", 409)
            boxes = self.validate_boxes(boxes, row, connection, reviewed)
            if masks_reviewed and any(not box.get("polygon") for box in boxes):
                raise StoreError("Every product needs a polygon before marking masks checked.")
            if reviewed and not boxes and not empty_confirmed:
                raise StoreError("Confirm that the shelf image has no visible products before saving zero boxes.")
            now, state = time.time(), "reviewed" if reviewed else "pending"
            connection.execute("""UPDATE images SET boxes = ?, notes = ?, state = ?, revision = revision + 1,
                updated = ?, edited_by = ?, reviewed_at = ?, lock_until = ? WHERE id = ?""",
                (json.dumps(boxes), notes, state, now, name, now if reviewed else None,
                 now + self.lease_seconds, image_id))
            connection.execute("INSERT INTO history (image_id, revision, editor, time, state, boxes, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (image_id, revision + 1, name, now, state, json.dumps(boxes), notes))
            connection.execute("UPDATE images SET masks_reviewed = ? WHERE id = ?",
                               (int(reviewed and masks_reviewed), image_id))
            return self.public(self.row(image_id, connection), connection, token, detail=True)

    #------------------------------------------------------------
    # MODEL JOBS AND CONSISTENT EXPORT SNAPSHOTS
    #------------------------------------------------------------
    def recover_jobs(self):
        """Retry inference interrupted by a server restart without touching edits."""
        with self.connect(write=True) as connection:
            connection.execute("UPDATE images SET prediction_state = 'queued' WHERE prediction_state = 'running' AND revision = 0")

    def next_job(self):
        """Reserve one inference job for the background model worker."""
        with self.connect(write=True) as connection:
            row = connection.execute("SELECT * FROM images WHERE prediction_state = 'queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                connection.execute("UPDATE images SET prediction_state = 'running' WHERE id = ?", (row["id"],))
            return dict(row) if row else None

    def finish_job(self, image_id, records=None, error=None):
        """Store original suggestions and initialize a draft only if no human edited it."""
        records = records or []
        with self.connect(write=True) as connection:
            row = self.row(image_id, connection)
            known = {entry[0] for entry in connection.execute("SELECT name FROM categories")}
            boxes = []
            for index, record in enumerate(records):
                category = record["category"]
                if category not in known and category != "unknown":
                    connection.execute("INSERT OR IGNORE INTO categories VALUES (?)", (category_name(category),))
                    known.add(category)
                x1, y1, x2, y2 = record["box_xyxy"]
                coordinates = [max(0, min(row["width"], x1)), max(0, min(row["height"], y1)),
                               max(0, min(row["width"], x2)), max(0, min(row["height"], y2))]
                if coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]:
                    item = {"id": f"model-{index}", "category": category, "xyxy": coordinates}
                    if record.get("polygon"):
                        item["polygon"] = validate_polygon(record["polygon"], row["width"], row["height"])
                        item["xyxy"] = polygon_bounds(item["polygon"])
                    boxes.append(item)
            connection.execute("""UPDATE images SET prediction_state = ?, prediction_error = ?,
                suggestions = ?, boxes = CASE WHEN revision = 0 THEN ? ELSE boxes END,
                updated = ? WHERE id = ?""",
                ("error" if error else "ready", error, json.dumps(records), json.dumps(boxes), time.time(), image_id))

    def export_snapshot(self):
        """Read reviewed images as a single database snapshot, excluding active edits."""
        with self.connect() as connection:
            connection.execute("BEGIN")
            rows = connection.execute("""SELECT * FROM images WHERE state = 'reviewed'
                AND (lock_token IS NULL OR lock_until <= ?) ORDER BY created""", (time.time(),)).fetchall()
            return [dict(row) for row in rows]
