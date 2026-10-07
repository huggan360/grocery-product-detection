#!/usr/bin/env bash
#------------------------------------------------------------
# ONE-TIME SETUP ON THE RASPBERRY PI 5 WITH THE AI HAT+ (HAILO-8, 26 TOPS)
#------------------------------------------------------------
# Run from the repository on the Pi: bash deploy/setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt update
# hailo-all: driver, firmware, HailoRT and its Python bindings (hailo_platform).
sudo apt install -y hailo-all ffmpeg rpicam-apps python3-venv
if ! hailortcli fw-control identify; then
  echo "The Hailo-8 is not visible yet. Reboot once after installing hailo-all, then rerun." >&2
  exit 1
fi
# HailoRT's Python bindings come from apt, so the venv must see system packages.
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
# PyPI's aarch64 torch wheel is CPU-only: used for tracking utilities and ViT fallback.
.venv/bin/pip install -r annotation_tool/requirements.txt
if ! .venv/bin/python -c "import hailo_platform" 2>/dev/null; then
  # The apt bindings may be built against the system NumPy 1.x.
  .venv/bin/pip install "numpy<2"
  .venv/bin/python -c "import hailo_platform"
fi
bash deploy/download_hefs.sh
# Download ViT-Small once now, so the live system never waits on the internet.
.venv/bin/python -c "from production.vit import load_imagenet_classifier as l; l('models/vit/pretrained', 'cpu')"
# YOLO must run on the Hailo-8 here, not silently on the CPU.
.venv/bin/python -c "from production.backends import choose_backend as c; assert c('auto') == 'hailo', 'Hailo-8 not usable'"
.venv/bin/python -m utils.hailo_probe
echo "Setup complete. Next: bash deploy/install_review_service.sh (see HOW-TO.md)"
