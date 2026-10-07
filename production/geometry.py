#------------------------------------------------------------
# SAFE IMAGE CROPS
#------------------------------------------------------------
import math


def crop_box(box, width, height, padding=0.0):
    """Add space around a box and keep the crop inside the image."""
    if padding < 0 or not math.isfinite(padding):
        raise ValueError("Crop padding must be finite and non-negative.")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Box coordinates must be finite.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive width and height.")
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    result = (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
              min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Box lies outside the image.")
    return result
