#!/usr/bin/env bash
#------------------------------------------------------------
# COPY BOTH REPOSITORIES TO THE PI, SIDE BY SIDE (CODE + SMALL MODELS)
#------------------------------------------------------------
# Usage: bash deploy/sync_to_pi.sh pi@<pi-hostname>.local [remote-parent-directory]
# Result on the Pi:  ~/grocery-product-detection  and  ~/smart-fridge-edge
# (config/pi.toml in the edge repo expects them next to each other).
set -euo pipefail
cd "$(dirname "$0")/.."
target=${1:?Usage: deploy/sync_to_pi.sh user@host [remote-parent-directory]}
parent=${2:-.}
edge=${EDGE_REPOSITORY:-../smart-fridge-edge}
# Never overwrite the Pi's own data: virtualenvs, recordings, database, zones, reviews.
# Excluded paths are also protected from --delete.
rsync -av --delete \
  --exclude .venv/ --exclude __pycache__/ --exclude data/ --exclude runs/ \
  --exclude 'models/vit/pretrained/vit_l_32-*.pth' --exclude 'models/yolo/*-seg.pt' \
  --exclude 'models/yolo/yolo26l.pt' --exclude 'models/hailo/*.hef' \
  ./ "$target:$parent/grocery-product-detection/"
if [ -d "$edge" ]; then
  rsync -av --delete \
    --exclude .venv/ --exclude __pycache__/ --exclude database/ --exclude data/ \
    --exclude .mypy_cache/ --exclude .ruff_cache/ --exclude '*.egg-info/' \
    --exclude config/pi.toml \
    "$edge/" "$target:$parent/smart-fridge-edge/"
  # pi.toml holds the Pi's door pin: copy it only the first time.
  rsync -av --ignore-existing "$edge/config/pi.toml" "$target:$parent/smart-fridge-edge/config/pi.toml"
fi
echo "Next, on the Pi: follow grocery-product-detection/HOW-TO.md"
