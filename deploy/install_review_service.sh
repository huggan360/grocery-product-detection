#!/usr/bin/env bash
#------------------------------------------------------------
# RUN THE REVIEW TOOL AS A SERVICE ON THE PI (PORT 9000, WHOLE NETWORK)
#------------------------------------------------------------
# Usage on the Pi: bash deploy/install_review_service.sh
# Then open http://<pi-hostname>.local:9000 from any computer on the same network.
set -euo pipefail
cd "$(dirname "$0")/.."
repository=$(pwd)
sudo tee /etc/systemd/system/fridge-review.service >/dev/null <<UNIT
[Unit]
Description=Smart fridge video review tool (port 9000)
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$repository
ExecStart=$repository/.venv/bin/python annotation_tool/run.py --host 0.0.0.0 --port 9000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now fridge-review.service
systemctl --no-pager status fridge-review.service | head -5
echo "Open http://$(hostname).local:9000"
