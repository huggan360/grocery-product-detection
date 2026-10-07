#------------------------------------------------------------
# COMPILE A TRAINED YOLO26M FOR THE HAILO-8 (RUN ON AN x86-64 LINUX PC)
#------------------------------------------------------------
# Needs the Hailo Dataflow Compiler 3.x (Hailo-8 branch) from the Hailo Developer
# Zone, installed in its own Python 3.10 environment together with this repo's
# requirements. The Pi only runs the resulting .hef file. ViT-Small stays a PyTorch
# checkpoint: it runs on the Pi's CPU.
#
#   python -m utils.export_hailo models/yolo/fridge-v1/train/weights/best.pt \
#       --data data/prepared/detection/dataset.yaml --name fridge-yolo
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODELS = Path(__file__).resolve().parents[1] / "models"


def write_labels(path, names):
    path.write_text("\n".join(names) + "\n")
    return path


def export_yolo(weights, data, name):
    """Ultralytics compiles YOLO26 detection to a Hailo-8 HEF with INT8 calibration."""
    from ultralytics import YOLO
    model = YOLO(weights)
    if model.task != "detect":
        raise ValueError("Export YOLO26 detection weights; the Hailo-8 tracker uses boxes.")
    output = Path(model.export(format="hailo", name="hailo8", imgsz=640, data=str(data)))
    hef = next(output.rglob("*.hef"))
    target = MODELS / "hailo" / f"{name}.hef"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hef, target)
    labels = write_labels(target.with_suffix(".txt"), [model.names[i] for i in sorted(model.names)])
    print(f"Set in configs/video.yaml:\n  detector.hef: hailo/{target.name}\n  detector.hef_labels: hailo/{labels.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile trained YOLO26m weights for the Hailo-8.")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML used for calibration.")
    parser.add_argument("--name", required=True, help="Output name in models/hailo/.")
    args = parser.parse_args()
    export_yolo(args.weights, args.data, args.name)
