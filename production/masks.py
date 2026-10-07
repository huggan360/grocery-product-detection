#------------------------------------------------------------
# VISIBLE OBJECT POLYGONS AND MASKED CLASSIFIER CROPS
#------------------------------------------------------------
import math

from PIL import Image, ImageDraw

from production.geometry import crop_box


def polygon_bounds(points):
    """Find the box around the visible outline of one object."""
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def validate_polygon(points, width, height):
    """Reject invalid outlines instead of saving unusable mask labels."""
    if not isinstance(points, list) or not 3 <= len(points) <= 500:
        raise ValueError("A polygon needs 3 to 500 points.")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each polygon point needs x and y coordinates.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
            raise ValueError("Polygon coordinates must be finite numbers.")
        if not (0 <= point[0] <= width and 0 <= point[1] <= height):
            raise ValueError("Keep polygon points inside the image.")
    if len({tuple(p) for p in points}) != len(points):
        raise ValueError("Polygon points must be distinct. Do not repeat the closing point.")
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))) / 2
    if area < 1:
        raise ValueError("Polygon must cover at least one square pixel.")

    # Crossing edges make ambiguous masks. Adjacent edges share an endpoint.
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])

    def intersect(a, b, c, d):
        ab_c, ab_d, cd_a, cd_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
        if ab_c * ab_d < 0 and cd_a * cd_b < 0:
            return True
        return ((abs(ab_c) < 1e-8 and on_segment(a, b, c)) or
                (abs(ab_d) < 1e-8 and on_segment(a, b, d)) or
                (abs(cd_a) < 1e-8 and on_segment(c, d, a)) or
                (abs(cd_b) < 1e-8 and on_segment(c, d, b)))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            if intersect(points[i], points[(i + 1) % len(points)], points[j], points[(j + 1) % len(points)]):
                raise ValueError("Polygon edges cross. Move or remove the crossing points.")
    return points


def masked_crop(image, polygon, padding=0.05):
    """Keep the object's visible pixels and replace the surroundings with grey."""
    bounds = crop_box(polygon_bounds(polygon), *image.size, padding)
    crop = image.convert("RGB").crop(bounds)
    mask = Image.new("L", crop.size, 0)
    shifted = [(x - bounds[0], y - bounds[1]) for x, y in polygon]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)
    return Image.composite(crop, Image.new("RGB", crop.size, (127, 127, 127)), mask)
