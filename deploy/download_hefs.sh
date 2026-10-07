#!/usr/bin/env bash
#------------------------------------------------------------
# DOWNLOAD THE PRECOMPILED HAILO-8 YOLO (RUN ON THE PI AFTER hailo-all)
#------------------------------------------------------------
# HEFs must match the installed HailoRT. YOLO26 HEFs exist from Model Zoo v2.18
# (HailoRT 4.23). Older HailoRT gets YOLOv8m, which the pipeline also supports.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/hailo
runtime=$(hailortcli --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
case "$runtime" in
  4.2[3-9]) zoo=v2.18.0 ;;  # Newer 4.x runtimes still load these HEFs.
  4.22) zoo=v2.16.0 ;;
  4.21) zoo=v2.15.0 ;;
  4.20|4.19|4.18) zoo=v2.14.0 ;;
  *) echo "HailoRT '$runtime' not recognised. Install it with: sudo apt install hailo-all" >&2; exit 1 ;;
esac
base="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/$zoo/hailo8"
echo "HailoRT $runtime -> Model Zoo $zoo"
fetch() { [ -s "models/hailo/$1.hef" ] || curl -fL --retry 3 -o "models/hailo/$1.hef" "$base/$1.hef"; }
if fetch yolo26m 2>/dev/null; then
  echo "Detector: models/hailo/yolo26m.hef"
else
  rm -f models/hailo/yolo26m.hef
  fetch yolov8m
  echo "YOLO26m needs HailoRT 4.23 (sudo apt full-upgrade). Using YOLOv8m for now:"
  echo "  set detector.hef: hailo/yolov8m.hef in configs/video.yaml"
fi
ls -lh models/hailo
