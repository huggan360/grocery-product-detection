#------------------------------------------------------------
# TRAIN MODELS: THIS SCRIPT NEVER STARTS PRODUCTION
#------------------------------------------------------------
import argparse
from pathlib import Path


def main():
    """Train YOLO, ViT, or both using the model paths in a YAML file."""
    parser = argparse.ArgumentParser(description="Train grocery models.")
    parser.add_argument("model", choices=("yolo", "vit", "both"))
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/training.yaml")
    parser.add_argument("--weights", type=Path, help="Override one model's input checkpoint.")
    args = parser.parse_args()
    if args.weights and args.model == "both":
        parser.error("Set each model's weights in the config when training both.")
    from utils.common import load_config
    config = load_config(args.config)
    if args.model in ("yolo", "both"):
        from training.yolo import train_segmenter
        settings = config["segmenter"]
        weights = str(args.weights.resolve()) if args.weights else settings["weights"]
        print(train_segmenter(settings["data"], weights, settings["output"],
                              settings["epochs"], settings["batch_size"], config["device"]))
    if args.model in ("vit", "both"):
        from training.vit import train_classifier
        if args.weights:
            config["classifier"]["initialize_from"] = str(args.weights.resolve())
        print(train_classifier(config))


if __name__ == "__main__":
    main()
