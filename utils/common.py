#------------------------------------------------------------
# CONFIGURATION AND REPEATABLE RUNS
#------------------------------------------------------------
import json
import random
from pathlib import Path

import torch
import yaml


def load_config(path):
    """Read settings and make file paths work from any folder."""
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    for section, keys in {
        "detector": ["data", "output"],
        "segmenter": ["data", "output", "weights", "checkpoint"],
        "classifier": ["data", "output", "checkpoint", "initialize_from"],
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value:
                config[section][key] = str((path.parent / value).resolve())
    # A plain model name is an Ultralytics download name. A path is local.
    weights = config.get("detector", {}).get("weights")
    if weights and ("/" in weights or "\\" in weights):
        config["detector"]["weights"] = str((path.parent / weights).resolve())
    return config


def choose_device(name="auto"):
    """Use a GPU when available, otherwise use the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot use it.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but PyTorch cannot use it.")
    return device


def seed_everything(seed):
    """Make random choices repeatable as far as the hardware allows."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    """Save readable results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def new_run_folder(path):
    """Refuse to mix a new training run with an old one."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output folder is not empty: {path}. Choose a new output path.")
    path.mkdir(parents=True, exist_ok=True)
    return path
