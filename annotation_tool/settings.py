#------------------------------------------------------------
# READ THE TOOL SETTINGS
#------------------------------------------------------------
from pathlib import Path

import yaml


def read_settings(path):
    """Resolve paths next to the config file, regardless of the terminal folder."""
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    for key in ("storage", "video_config"):
        settings[key] = str((path.parent / settings[key]).resolve())
    return settings
