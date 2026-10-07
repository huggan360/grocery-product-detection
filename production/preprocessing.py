#------------------------------------------------------------
# SEPARATE IMAGE COPIES FOR YOLO AND VIT
#------------------------------------------------------------
import numpy as np
from PIL import Image, ImageEnhance


#------------------------------------------------------------
# YOLO: STRONGER LOCAL CONTRAST AND GENTLE EDGE SHARPENING
#------------------------------------------------------------
def prepare_yolo_image(image, enabled=True):
    """Make edges clearer on a copy without moving or resizing any pixels."""
    rgb = image.convert("RGB")
    if not enabled:
        return rgb.copy()
    import cv2

    # Change brightness contrast, keeping colour channels separate.
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    local_contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    light = local_contrast.apply(light)

    # A small unsharp mask strengthens existing edges, not new outlines.
    blurred = cv2.GaussianBlur(light, (0, 0), sigmaX=1.0)
    light = cv2.addWeighted(light, 1.35, blurred, -0.35, 0)
    enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


#------------------------------------------------------------
# VIT: SMALL VIBRANCE AND CONTRAST BOOST ON ORIGINAL CROPS
#------------------------------------------------------------
class PrepareViTImage:
    """Boost dull colours a little while leaving strong colours mostly alone."""

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        highest = rgb.max(axis=2, keepdims=True)
        lowest = rgb.min(axis=2, keepdims=True)
        saturation = (highest - lowest) / np.maximum(highest, 1e-6)
        grey = (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2, keepdims=True)
        # Up to 8% extra colour for muted pixels; already vivid pixels get less.
        vibrant = grey + (rgb - grey) * (1.0 + 0.08 * (1.0 - saturation))
        result = Image.fromarray(np.rint(np.clip(vibrant, 0, 1) * 255).astype(np.uint8))
        return ImageEnhance.Contrast(result).enhance(1.05)
