"""Immutable render rules: maximal consecutive runs, no implicit padding/smoothing."""

from bisect import bisect_left
from fractions import Fraction
import hashlib
import json
import math

from services.media.probe import MediaError
from services.model_adapter.adapter import validate_result

RULES = "consecutive-inward-v1"


def seconds_to_frame(seconds, metadata):
    """Nearest boundary, ties up. Validation of the resulting range is separate."""
    value = Fraction(str(seconds)) * metadata["fps_num"] / metadata["fps_den"]
    return math.floor(value + Fraction(1, 2))


def aligned_crop(box, ratio):
    x, y, width = box
    height = width * ratio[1] / ratio[0]
    left, top = math.ceil(x), math.ceil(y)
    # Exact target ratio and even dimensions; entirely inside the floating box.
    divisor = math.gcd(*ratio)
    a, b = (v // divisor for v in ratio)
    k = min(math.floor((x + width - left + 1e-9) / a), math.floor((y + height - top + 1e-9) / b))
    k -= k % 2  # a,b coprime: at least one odd, so k must be even.
    if k <= 0:
        raise MediaError("crop_too_small", "Crop cannot contain an even-sized image at the exact target ratio")
    return [left, top, k * a, k * b]


def make_plan(result, metadata, edits=None):
    try:
        validate_result(result, {**{key: result[key] for key in ("api_version", "video_id", "target_ratio")}, "video_path": "internal"})
    except (ValueError, KeyError, TypeError) as error:
        raise MediaError("invalid_prediction", "Result does not satisfy prediction v1") from error
    ratio = result["target_ratio"]
    predictions = result["predictions"]
    suggestions = []
    boxes = {}
    for item in predictions:
        frame, box = item["frame"], item["bboxes"]
        x, y, width = box
        if frame >= metadata["frame_count"] or x + width > metadata["width"] + 1e-7 or y + width * ratio[1] / ratio[0] > metadata["height"] + 1e-7:
            raise MediaError("invalid_prediction", "Prediction frame or crop exceeds the original video bounds")
        boxes[frame] = box
        if not suggestions or suggestions[-1]["end_frame_exclusive"] != frame:
            suggestions.append({"suggestion_id": str(len(suggestions)), "start_frame": frame, "end_frame_exclusive": frame + 1, "enabled": True})
        else:
            suggestions[-1]["end_frame_exclusive"] = frame + 1
    segments = [dict(s) for s in suggestions] if edits is None else edits
    if not isinstance(segments, list):
        raise MediaError("invalid_segments", "segments must be an array")
    known = {s["suggestion_id"]: s for s in suggestions}
    seen, previous = set(), -1
    planned = []
    frame_crops = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"suggestion_id", "start_frame", "end_frame_exclusive", "enabled"}:
            raise MediaError("invalid_segments", "Each segment requires suggestion_id, start_frame, end_frame_exclusive and enabled")
        sid, start, end, enabled = (segment[k] for k in ("suggestion_id", "start_frame", "end_frame_exclusive", "enabled"))
        if not isinstance(sid, str) or sid not in known or sid in seen or type(start) is not int or type(end) is not int or type(enabled) is not bool or not 0 <= start < end <= metadata["frame_count"] or start < previous:
            raise MediaError("invalid_segments", "Segments must reference unique suggestions in source order with valid non-overlapping ranges")
        seen.add(sid)
        previous = end
        planned.append(dict(segment))
        if enabled:
            original = known[sid]
            for frame in range(start, end):
                reference = min(max(frame, original["start_frame"]), original["end_frame_exclusive"] - 1)
                frame_crops.append({"frame": frame, "crop": aligned_crop(boxes[reference], ratio)})
    output = None
    if frame_crops:
        output = {"width": min(f["crop"][2] for f in frame_crops), "height": min(f["crop"][3] for f in frame_crops)}
    plan = {"api_version": "1", "rules_version": RULES, "video_id": result["video_id"], "target_ratio": ratio,
            "engine": result["engine"], "mock": result["mock"], "metadata": metadata,
            "suggestions": suggestions, "segments": planned, "frame_crops": frame_crops, "output": output,
            "duration_seconds": len(frame_crops) * metadata["fps_den"] / metadata["fps_num"]}
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return plan


def interpolate_box(keyframes, frame):
    """For a future sparse runner only; never used to add unselected frames."""
    points = sorted(keyframes)
    index = bisect_left(points, frame)
    if index == 0:
        return list(keyframes[points[0]])
    if index == len(points):
        return list(keyframes[points[-1]])
    left, right = points[index - 1], points[index]
    t = (frame - left) / (right - left)
    return [a + (b - a) * t for a, b in zip(keyframes[left], keyframes[right])]
