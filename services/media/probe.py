"""Inspect decoded frames; accept irregular timestamps without resampling sources."""

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


class MediaError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def binary(name):
    value = os.environ.get(name.upper() + "_BIN") or shutil.which(name)
    if not value:
        raise MediaError("media_tools_unavailable", f"Install {name} or configure {name.upper()}_BIN")
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path, *options):
    try:
        result = subprocess.run([binary("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov,matroska,webm,avi", *options, "-of", "json", str(Path(path).resolve())], capture_output=True, timeout=120, check=True)
        if result.stderr.strip():
            raise ValueError("Decoder reported errors")
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise MediaError("invalid_media", "Video could not be inspected within 120 seconds") from error


def probe(path):
    data = ffprobe(path, "-show_streams", "-show_format")
    videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if len(videos) != 1:
        raise MediaError("unsupported_media", "Exactly one video stream is required")
    stream = videos[0]
    try:
        width, height = int(stream["width"]), int(stream["height"])
        fps = Fraction(stream["r_frame_rate"])
        duration = float(stream.get("duration", data["format"].get("duration", 0)))
        rotation = float(stream.get("tags", {}).get("rotate", 0))
        for side in stream.get("side_data_list", []):
            rotation = float(side.get("rotation", rotation))
        sar = stream.get("sample_aspect_ratio", "1:1")
        start = float(stream.get("start_time", 0))
        if not all(math.isfinite(v) for v in (duration, rotation, start)) or not (0 < fps <= 120 and 0 < duration <= 600 and 0 < width * height <= 3840 * 2160):
            raise ValueError("limits")
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise MediaError("unsupported_media", "CPU profile requires <=10 minutes, <=4K, <=120 fps and valid stream metadata") from error
    if rotation % 360 or sar not in {"1:1", "N/A"} or abs(start) > 0.0001:
        raise MediaError("unsupported_media_profile", "CPU profile requires zero rotation/start time and square pixels; no implicit normalization")
    if any(abs(float(s.get("start_time", 0))) > 0.0001 for s in data["streams"] if s.get("codec_type") == "audio"):
        raise MediaError("unsupported_media_profile", "CPU profile requires audio starting at source time zero")
    frames = ffprobe(path, "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time").get("frames", [])
    if not frames or len(frames) > 72000:
        raise MediaError("unsupported_media", "Video has no decoded frames or exceeds the CPU frame limit")
    tolerance = max(0.00001, float(Fraction(stream.get("time_base", "1/1000000"))) * 1.1)
    try:
        cfr = all(abs(float(frame["best_effort_timestamp_time"]) - i / float(fps)) <= tolerance for i, frame in enumerate(frames))
    except (KeyError, ValueError):
        cfr = False
    if not cfr:
        # Match the average clock used by the competition decoder. This does not
        # resample frames or implement timestamp-accurate preview/audio mapping.
        try:
            average = Fraction(stream.get("avg_frame_rate", "0/1"))
        except (ValueError, ZeroDivisionError):
            average = Fraction(0)
        fps = average if average > 0 else Fraction(len(frames)) / Fraction(str(duration))
        if fps > 120:
            raise MediaError("unsupported_media", "Average video frame rate exceeds 120 fps")
    return {"width": width, "height": height, "fps_num": fps.numerator, "fps_den": fps.denominator,
            "frame_count": len(frames), "duration_seconds": len(frames) / float(fps),
            "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"]),
            "rotation": 0, "validation_profile": "cpu-cfr-v1" if cfr else "cpu-vfr-v1"}
