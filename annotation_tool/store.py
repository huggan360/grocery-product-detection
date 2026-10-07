#------------------------------------------------------------
# REVIEWER NAMES AND THE SHARED CATEGORY LIST
#------------------------------------------------------------
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


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
    """Who is reviewing, and which grocery categories they can choose."""

    def __init__(self, root, categories):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "review.sqlite3"
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS editors (
                    token TEXT PRIMARY KEY, name TEXT NOT NULL, last_seen REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY);
            """)
            for name in categories:
                connection.execute("INSERT OR IGNORE INTO categories VALUES (?)", (category_name(name),))

    @contextmanager
    def connect(self, write=False):
        """Open one connection per operation."""
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
        """Give this browser tab its own reviewer identity."""
        name = name.strip()
        if not 1 <= len(name) <= 40:
            raise StoreError("Enter a name between 1 and 40 characters.")
        token = secrets.token_urlsafe(32)
        with self.connect(write=True) as connection:
            connection.execute("INSERT INTO editors VALUES (?, ?, ?)", (token, name, time.time()))
        return {"token": token, "name": name}

    def editor(self, token, connection):
        """Reject changes that have no registered browser identity."""
        row = connection.execute("SELECT * FROM editors WHERE token = ?", (token,)).fetchone()
        if row is None:
            raise StoreError("Join with your name first.", 401)
        connection.execute("UPDATE editors SET last_seen = ? WHERE token = ?", (time.time(), token))
        return row["name"]

    def categories(self):
        """Return the shared category list in alphabetical order."""
        with self.connect() as connection:
            return [row[0] for row in connection.execute("SELECT name FROM categories ORDER BY name")]

    def add_category(self, name, token):
        """Add a category once so all reviewers can use it immediately."""
        name = category_name(name)
        with self.connect(write=True) as connection:
            self.editor(token, connection)
            connection.execute("INSERT OR IGNORE INTO categories VALUES (?)", (name,))
        return name
