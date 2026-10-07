#------------------------------------------------------------
# CAPTURE ONE FRAME FROM EACH CONFIGURED SHELF CAMERA
#------------------------------------------------------------
import io

import requests
from PIL import Image


def capture_camera(camera):
    """Read a server-connected USB camera, a stream, or a snapshot URL."""
    if camera["kind"] == "snapshot":
        response = requests.get(camera["source"], timeout=(5, 15), stream=True)
        with response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 20 * 1024 * 1024:
                    raise ValueError("Camera snapshot exceeds 20 MB.")
                chunks.append(chunk)
        with Image.open(io.BytesIO(b"".join(chunks))) as image:
            return image.copy()
    if camera["kind"] != "opencv":
        raise ValueError("Camera kind must be opencv or snapshot.")
    import cv2
    source = camera["source"]
    if isinstance(source, str):
        # These timeouts are supported by the FFmpeg network-stream backend.
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000])
    else:
        capture = cv2.VideoCapture(int(source))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {camera['name']}.")
        if camera.get("width"):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera["width"])
        if camera.get("height"):
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera["height"])
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"No frame received from {camera['name']}.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
