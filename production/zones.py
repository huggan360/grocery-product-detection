#------------------------------------------------------------
# INSIDE / OUTSIDE ZONES AND OBSERVED CROSSINGS
#------------------------------------------------------------
import math


def validate_zones(zones):
    """Use two separate rectangles, with coordinates between zero and one."""
    result = {}
    for name in ("inside", "outside"):
        value = zones.get(name)
        if value is None:
            result[name] = None
            continue
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError(f"{name} needs [left, top, right, bottom].")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or not 0 <= v <= 1 for v in value):
            raise ValueError("Zone coordinates must be finite numbers between 0 and 1.")
        x1, y1, x2, y2 = value
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            raise ValueError("Draw a zone with a nonzero width and height.")
        result[name] = value
    a, b = result["inside"], result["outside"]
    if a and b and max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
        raise ValueError("Inside and outside zones must not overlap.")
    return result


def locate(box, zones):
    """Use the centre of the object's box to decide which zone it occupies."""
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    matches = [name for name, rect in zones.items() if rect and
               rect[0] < x < rect[2] and rect[1] < y < rect[3]]
    return matches[0] if len(matches) == 1 else "neutral"


def crossing_events(observations, zones, stable_frames=2, dwell_seconds=0.2, max_gap_seconds=1.0):
    """Confirm a movement only after seeing the same track stably on both sides."""
    validate_zones(zones)
    events = []
    stable = candidate = None
    candidate_start = stable_at = previous_time = None
    count = 0
    for sample in sorted(observations, key=lambda row: row["time"]):
        timestamp = sample["time"]
        # A long disappearance breaks the evidence chain, even if an ID is reused.
        if previous_time is not None and timestamp - previous_time > max_gap_seconds:
            stable = candidate = None
            candidate_start = stable_at = None
            count = 0
        previous_time = timestamp
        zone = locate(sample["box"], zones)
        sample["zone"] = zone
        if zone == "neutral":
            candidate, candidate_start, count = None, None, 0
            continue
        if zone != candidate:
            candidate, candidate_start, count = zone, timestamp, 1
        else:
            count += 1
        if count < stable_frames or timestamp - candidate_start + 1e-9 < dwell_seconds:
            continue
        if zone != stable:
            if stable is not None:
                events.append({"direction": "in" if zone == "inside" else "out",
                               "start": stable_at, "end": timestamp,
                               "from_zone": stable, "to_zone": zone,
                               "source": "zone-crossing"})
            stable = zone
        stable_at = timestamp
    return events


def analyse_tracks(tracks, zones, settings):
    """Recompute movements from saved tracks without running the models again."""
    for track in tracks:
        track["events"] = crossing_events(track["observations"], zones,
            settings.get("stable_frames", 2), settings.get("dwell_seconds", 0.2),
            settings.get("max_gap_seconds", 1.0))
    return tracks
