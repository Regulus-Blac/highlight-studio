"""Validate per-video competition prediction records."""
from aic.numeric import finite_number

def prediction_errors(record, expected, meta, require_model_params=False):
    errors = []
    if str(record.get("video_id")) != str(expected["video_id"]):
        errors.append("video_id does not match")
    if record.get("targetRatioWH") != expected["targetRatioWH"]:
        errors.append("targetRatioWH changed or missing")
    for field in ("video_file", "video_filename", "filename", "file_name", "video_path"):
        if field in expected and record.get(field) != expected[field]:
            errors.append(f"Index field changed or missing: {field}")
    params = record.get("model_params_b")
    if require_model_params or "model_params_b" in record:
        if not finite_number(params) or not 0 < params <= 9:
            errors.append("model_params_b must be finite and in (0, 9]")
    predictions = record.get("predictions")
    if not isinstance(predictions, list):
        return errors + ["predictions must be a list"]
    seen = set()
    previous = -1
    tw, th = expected["targetRatioWH"]
    for i, item in enumerate(predictions):
        prefix = f"predictions[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        frame = item.get("frame")
        if type(frame) is not int or not 0 <= frame < meta["frames"]:
            errors.append(f"{prefix}: invalid original frame number")
        else:
            if frame in seen:
                errors.append(f"{prefix}: duplicate frame")
            if frame < previous:
                errors.append(f"{prefix}: frames are not sorted")
            seen.add(frame)
            previous = frame
        box = item.get("bboxes")
        if not isinstance(box, list) or len(box) != 3 or not all(finite_number(x) for x in box):
            errors.append(f"{prefix}: bboxes must contain 3 finite numbers")
            continue
        x, y, w = box
        h = w * th / tw
        if x < 0 or y < 0 or w <= 0 or x + w > meta["width"] + 1e-7 or y + h > meta["height"] + 1e-7:
            errors.append(f"{prefix}: crop is empty or outside source image")
    return errors
