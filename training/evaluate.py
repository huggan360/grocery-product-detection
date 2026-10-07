#------------------------------------------------------------
# EVALUATE THE TWO MODELS SEPARATELY
#------------------------------------------------------------
import argparse
from pathlib import Path

from production.vit import load_classifier
from utils.common import choose_device, load_config, write_json
from training.data import ProductDataset, make_loader
from training.metrics import evaluate_model


def evaluate_classifier(config, split="test"):
    """Report accuracy, macro F1, and a confusion matrix for product crops."""
    settings = config["classifier"]
    device = choose_device(config["device"])
    model, classes = load_classifier(settings["checkpoint"], device)
    dataset = ProductDataset(Path(settings["data"]) / split, classes=classes)
    loader = make_loader(dataset, settings["batch_size"], settings["workers"])
    metrics = evaluate_model(model, loader, device, classes)
    write_json(Path(settings["output"]) / f"evaluation_{split}.json", metrics)
    print(f"Accuracy: {metrics['accuracy']:.3f}; macro F1: {metrics['macro_f1']:.3f}")
    return metrics


def evaluate_detector(config, split="test"):
    """Measure YOLO box quality independently of the ViT classifier."""
    from ultralytics import YOLO
    settings = config["segmenter"]
    weights = Path(settings["checkpoint"])
    if not weights.is_file():
        raise ValueError(f"Missing trained detector: {weights}")
    model = YOLO(str(weights))
    metrics = model.val(data=settings["data"], split=split, single_cls=True,
                        imgsz=640, batch=settings["batch_size"],
                        workers=2, device=str(choose_device(config["device"])),
                        project=settings["output"], name=f"evaluation_{split}")
    result = {key: float(value) for key, value in metrics.results_dict.items()}
    write_json(Path(settings["output"]) / f"evaluation_{split}.json", result)
    return result


#------------------------------------------------------------
# RUN THIS FILE ON ITS OWN
#------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detector or the classifier.")
    parser.add_argument("model", choices=["detector", "classifier"])
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    function = evaluate_detector if args.model == "detector" else evaluate_classifier
    function(load_config(args.config), args.split)
