#------------------------------------------------------------
# FINE-TUNE YOLO26 INSTANCE SEGMENTATION ON REVIEWED OUTLINES
#------------------------------------------------------------
import argparse
from pathlib import Path

from utils.common import choose_device, new_run_folder


def train_segmenter(data, weights, output, epochs=50, batch_size=8, device="auto"):
    """Learn one product mask per visible item from the annotation-tool export."""
    from ultralytics import YOLO
    data = Path(data).resolve()
    if not data.is_file():
        raise ValueError(f"Missing segmentation dataset YAML: {data}")
    model = YOLO(weights)
    if model.task != "segment":
        raise ValueError("Use YOLO segmentation weights ending in -seg.pt or a trained segmentation best.pt.")
    output = new_run_folder(output)
    model.train(data=str(data), epochs=epochs, batch=batch_size, imgsz=640,
                device=str(choose_device(device)), single_cls=True,
                project=str(output), name="train", exist_ok=False, seed=42)
    return Path(model.trainer.best)


#------------------------------------------------------------
# TRAINING STARTS ONLY WHEN YOU RUN THIS SCRIPT
#------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 product instance masks.")
    parser.add_argument("--data", required=True, help="Exported segmentation/dataset.yaml")
    parser.add_argument("--weights", default="models/yolo/yolo26l-seg.pt")
    parser.add_argument("--output", default="models/yolo/fridge-v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(train_segmenter(args.data, args.weights, args.output, args.epochs, args.batch_size, args.device))
