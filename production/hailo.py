#------------------------------------------------------------
# HAILO-8 (RASPBERRY PI AI HAT+, 26 TOPS): RUN THE COMPILED YOLO .HEF
#------------------------------------------------------------
# HailoRT is installed on the Pi with `sudo apt install hailo-all`; it is not a pip
# package. Use a virtual environment created with --system-site-packages.
import threading
import time

import numpy as np

_device = None
_device_lock = threading.Lock()


def hailo_available():
    """True when the HailoRT Python bindings and an accelerator are present."""
    try:
        from hailo_platform import Device
        return bool(Device.scan())
    except Exception:
        return False


def shared_device(wait_seconds=180):
    """One virtual device per process.

    Only one process can hold the Hailo-8. The review tool and the orchestrator's ML runs
    take turns: each releases it after a clip, and a waiting process retries until free.
    """
    global _device
    with _device_lock:
        if _device is None:
            from hailo_platform import HailoSchedulingAlgorithm, VDevice
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    params = VDevice.create_params()
                    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                    _device = VDevice(params)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1)
        return _device


def release_device():
    """Free the Hailo-8 for other processes."""
    global _device
    with _device_lock:
        if _device is not None:
            _device.release()
            _device = None


class HailoModel:
    """A single-input HEF. Inputs are uint8 RGB; normalization is inside the HEF."""

    def __init__(self, path):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams)
        self.hef = HEF(str(path))
        device = shared_device()
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network = device.configure(self.hef, params)[0]
        self.inputs = InputVStreamParams.make(self.network, format_type=FormatType.UINT8)
        self.outputs = OutputVStreamParams.make(self.network, format_type=FormatType.FLOAT32)
        info = self.hef.get_input_vstream_infos()
        if len(info) != 1:
            raise ValueError(f"{path} has {len(info)} inputs; expected one image input.")
        self.input_name = info[0].name
        self.height, self.width = info[0].shape[:2]
        self.output_names = [item.name for item in self.hef.get_output_vstream_infos()]

    def infer(self, batch):
        """Run uint8 NHWC images; returns {output name: per-image outputs}."""
        from hailo_platform import InferVStreams
        batch = np.ascontiguousarray(batch, dtype=np.uint8)
        if batch.shape[1:] != (self.height, self.width, 3):
            raise ValueError(f"Expected images of {self.width}x{self.height}, got {batch.shape}.")
        # With the scheduler enabled, HailoRT activates the network itself.
        with InferVStreams(self.network, self.inputs, self.outputs) as pipeline:
            return pipeline.infer({self.input_name: batch})


#------------------------------------------------------------
# YOLO HEF: LETTERBOX IN, BOXES OUT
#------------------------------------------------------------
def letterbox(image, width, height):
    """Resize without stretching; returns the padded image, scale and offsets."""
    import cv2
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                         interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, np.uint8)
    top, left = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas, scale, left, top


def parse_nms(output, width, height):
    """Read Hailo's on-chip NMS output (YOLOv8 HEFs): per class [y1, x1, y2, x2, score]."""
    rows = []
    for label, detections in enumerate(output):
        for y1, x1, y2, x2, score in np.asarray(detections, np.float32).reshape(-1, 5):
            rows.append([x1 * width, y1 * height, x2 * width, y2 * height, score, label])
    return np.asarray(rows, np.float32).reshape(-1, 6)


def decode_yolo26(outputs, width, height, classes):
    """Decode YOLO26's NMS-free one-to-one head: per scale 4 box distances + class logits."""
    scales = {}
    for value in outputs.values():
        value = np.asarray(value, np.float32)
        value = value[0] if value.ndim == 4 else value
        kind = "box" if value.shape[-1] == 4 else "cls" if value.shape[-1] == classes else None
        if kind is None:
            raise ValueError(f"Unexpected YOLO26 output shape {value.shape} for {classes} classes.")
        scales.setdefault(value.shape[:2], {})[kind] = value
    boxes, scores = [], []
    for (rows, columns), maps in scales.items():
        if set(maps) != {"box", "cls"}:
            raise ValueError("Each YOLO26 scale needs one box and one class output.")
        stride = height / rows
        y, x = np.mgrid[:rows, :columns].astype(np.float32) + 0.5
        left, top, right, bottom = np.moveaxis(maps["box"], -1, 0)
        boxes.append(np.stack([x - left, y - top, x + right, y + bottom], -1).reshape(-1, 4) * stride)
        scores.append(maps["cls"].reshape(-1, classes))
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    if scores.min() < 0 or scores.max() > 1:  # The Model Zoo HEF leaves the sigmoid to the host.
        scores = 1 / (1 + np.exp(-scores))
    labels = scores.argmax(1)
    best = scores[np.arange(len(scores)), labels]
    return np.concatenate([boxes, best[:, None], labels[:, None]], 1).astype(np.float32)


def suppress_duplicates(boxes, iou=0.7):
    """Per-class NMS, highest score first. YOLO26's one-to-one head rarely needs it."""
    import cv2
    if not len(boxes):
        return boxes
    xywh = np.concatenate([boxes[:, :2], boxes[:, 2:4] - boxes[:, :2]], 1)
    keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), boxes[:, 4].tolist(), boxes[:, 5].astype(int).tolist(), 0.0, iou)
    boxes = boxes[np.asarray(keep, int).reshape(-1)]
    return boxes[np.argsort(-boxes[:, 4])]


class HailoDetector:
    """YOLO HEF on the Hailo-8: YOLO26 raw head (decoded here) or a YOLOv8 HEF with on-chip NMS."""

    def __init__(self, path, names, confidence=0.15, max_detections=100):
        self.model = HailoModel(path)
        self.names = dict(enumerate(names))
        self.confidence, self.max_detections = confidence, max_detections
        self.nms = len(self.model.output_names) == 1

    def detect(self, rgb):
        """Return detections as [x1, y1, x2, y2, score, class] in original pixels."""
        image, scale, left, top = letterbox(rgb, self.model.width, self.model.height)
        outputs = self.model.infer(image[None])
        if self.nms:
            boxes = parse_nms(outputs[self.model.output_names[0]][0], self.model.width, self.model.height)
        else:
            boxes = decode_yolo26(outputs, self.model.width, self.model.height, len(self.names))
        boxes = boxes[boxes[:, 4] >= self.confidence]
        boxes = suppress_duplicates(boxes)[:self.max_detections]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / scale).clip(0, rgb.shape[1])
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / scale).clip(0, rgb.shape[0])
        return boxes
