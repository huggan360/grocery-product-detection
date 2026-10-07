#------------------------------------------------------------
# READ THE TOOL SETTINGS
#------------------------------------------------------------
from pathlib import Path

import yaml


def read_settings(path):
    """Resolve paths next to the config file, regardless of the terminal folder."""
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    for key in ("storage", "model_config"):
        if settings.get(key):
            settings[key] = str((path.parent / settings[key]).resolve())
    if settings["lease_seconds"] < 30:
        raise ValueError("lease_seconds must be at least 30.")
    if settings["max_upload_mb"] <= 0 or settings["max_image_pixels"] <= 0:
        raise ValueError("Upload size and image pixel limits must be positive.")
    ids = [camera["id"] for camera in settings.get("cameras", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("Each camera needs a unique id.")
    return settings
