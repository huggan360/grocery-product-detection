#------------------------------------------------------------
# CHECK THE HAILO-8 YOLO AND THE CPU VIT ON THE PI BEFORE TRYING THE FRIDGE
#------------------------------------------------------------
# python -m utils.hailo_probe [--image photo.jpg]
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def timed(function, repeats=10):
    """Median milliseconds per call after one warm-up call."""
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main():
    from production.backends import build_classifier, build_detector, choose_backend
    from production.video import load_config
    parser = argparse.ArgumentParser(description="Verify HailoRT, the YOLO HEF and the CPU ViT.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/video.yaml"))
    parser.add_argument("--image", help="A photo to detect and classify; defaults to a grey frame.")
    args = parser.parse_args()
    config = load_config(args.config)
    backend = choose_backend(config.get("backend", "auto"))
    print(f"Backend: {backend}")
    if backend == "hailo":
        from hailo_platform import HEF
        path = config["detector"].get("hef")
        if not path or not Path(path).is_file():
            raise SystemExit(f"Missing YOLO HEF {path}. Run deploy/download_hefs.sh.")
        hef = HEF(path)
        print(f"YOLO HEF: {path}")
        for info in hef.get_input_vstream_infos():
            print(f"  input  {info.name} {tuple(info.shape)}")
        for info in hef.get_output_vstream_infos():
            print(f"  output {info.name} {tuple(info.shape)}")
    image = Image.open(args.image).convert("RGB") if args.image else Image.new("RGB", (1280, 720), (128, 128, 128))
    rgb = np.asarray(image)
    detector = build_detector(config["detector"], backend, "cpu")
    detections = detector.detect(rgb)
    print(f"Detector: {timed(lambda: detector.detect(rgb)):.1f} ms per frame, {len(detections)} objects")
    for x1, y1, x2, y2, score, label in detections[:10]:
        print(f"  {detector.names[int(label)]:<14} {score:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    classifier = build_classifier(config["classifier"], "cpu")
    crops = [image.crop(tuple(int(v) for v in row[:4])) for row in detections[:3]] or [image]
    probabilities = classifier.probabilities(crops)
    print(f"ViT-Small on CPU: {timed(lambda: classifier.probabilities(crops), 5):.1f} ms "
          f"for {len(crops)} crops")
    for row in probabilities:
        best = int(row.argmax())
        print(f"  {classifier.classes[best]} {row[best]:.2f}")


if __name__ == "__main__":
    main()
