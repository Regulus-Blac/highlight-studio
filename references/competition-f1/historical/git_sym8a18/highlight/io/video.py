"""Video metadata probing, with frame-count reconciliation.

Phase 0 statistic C3. Decoders disagree about frame counts -- ``cv2``'s
``CAP_PROP_FRAME_COUNT`` is a container estimate, ``ffprobe``'s ``nb_frames``
may be absent or wrong, and only ``-count_frames`` (slow) actually counts. If
our count is higher than the evaluator's, every prediction past its end is
out of range: those still count in ``N_pred`` but score nothing, so an
overestimate silently bleeds precision from the tail of every video.

The defence is ``n_frames_safe``: when sources disagree, take the minimum and
back off two more frames, and never predict beyond it.

Every probe degrades gracefully -- a missing ``cv2``, ``decord`` or ``ffprobe``
narrows the reconciliation but never raises.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Optional

FRAME_COUNT_BACKOFF = 2


@dataclass
class VideoMeta:
    """Reconciled metadata for one source video."""

    video_id: str
    path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    n_frames: int = 0
    n_frames_safe: int = 0
    sources: Dict[str, int] = field(default_factory=dict)
    vfr: bool = False
    has_audio: bool = False
    risk: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.width > 0 and self.height > 0 and self.n_frames_safe > 0


def _probe_cv2(path: str) -> Dict[str, float]:
    try:
        import cv2
    except ImportError:
        return {}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    try:
        return {
            "n_frames": float(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        cap.release()


def _probe_decord(path: str) -> Dict[str, float]:
    try:
        import decord
    except ImportError:
        return {}
    try:
        vr = decord.VideoReader(path, num_threads=1)
        return {"n_frames": float(len(vr)), "fps": float(vr.get_avg_fps())}
    except Exception:
        return {}


def _probe_ffprobe(path: str, count_frames: bool = False) -> Dict[str, object]:
    if shutil.which("ffprobe") is None:
        return {}
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams"]
    if count_frames:
        cmd.append("-count_frames")
    cmd.append(path)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.SubprocessError, OSError):
        return {}
    if out.returncode != 0:
        return {}
    try:
        streams = json.loads(out.stdout).get("streams", [])
    except json.JSONDecodeError:
        return {}

    info: Dict[str, object] = {"has_audio": any(s.get("codec_type") == "audio" for s in streams)}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return info

    for key, field_name in (("width", "width"), ("height", "height")):
        if video.get(field_name) is not None:
            info[key] = float(video[field_name])
    if video.get("duration") is not None:
        try:
            info["duration"] = float(video["duration"])
        except (TypeError, ValueError):
            pass
    for key, field_name in (("nb_frames", "nb_frames"), ("nb_read_frames", "nb_read_frames")):
        try:
            info[key] = float(video[field_name])
        except (KeyError, TypeError, ValueError):
            pass

    r_rate, avg_rate = video.get("r_frame_rate"), video.get("avg_frame_rate")
    try:
        r_val = float(Fraction(r_rate)) if r_rate and r_rate != "0/0" else None
        avg_val = float(Fraction(avg_rate)) if avg_rate and avg_rate != "0/0" else None
    except (ValueError, ZeroDivisionError):
        r_val = avg_val = None
    if avg_val:
        info["fps"] = avg_val
    # Variable frame rate: the nominal and average rates diverge.
    if r_val and avg_val and abs(r_val - avg_val) > 0.01 * max(r_val, avg_val):
        info["vfr"] = True
    return info


def probe_video(video_id: str, path: str, count_frames: bool = False) -> VideoMeta:
    """Probe one video with every available backend and reconcile the results.

    ``count_frames=True`` adds ``ffprobe -count_frames``, which is authoritative
    but slow -- run it once during the Phase 0 census, not in the hot path.
    """
    meta = VideoMeta(video_id=video_id, path=path)
    if not os.path.exists(path):
        meta.error = "missing file"
        return meta

    cv = _probe_cv2(path)
    dec = _probe_decord(path)
    ff = _probe_ffprobe(path, count_frames=count_frames)

    for src, value in (("cv2", cv.get("n_frames")), ("decord", dec.get("n_frames"))):
        if value and value > 0:
            meta.sources[src] = int(value)
    for key, src in (("nb_frames", "ffprobe"), ("nb_read_frames", "ffprobe_count")):
        value = ff.get(key)
        if isinstance(value, float) and value > 0:
            meta.sources[src] = int(value)

    meta.width = int(cv.get("width") or ff.get("width") or 0)
    meta.height = int(cv.get("height") or ff.get("height") or 0)
    meta.fps = float(ff.get("fps") or dec.get("fps") or cv.get("fps") or 0.0)
    meta.duration = float(ff.get("duration") or 0.0)
    meta.has_audio = bool(ff.get("has_audio", False))
    meta.vfr = bool(ff.get("vfr", False))

    if not meta.sources:
        meta.error = "no decoder could read a frame count"
        return meta

    # An exact count, when we have one, is authoritative.
    if "ffprobe_count" in meta.sources:
        meta.n_frames = meta.sources["ffprobe_count"]
    else:
        meta.n_frames = max(meta.sources.values())

    lo, hi = min(meta.sources.values()), max(meta.sources.values())
    meta.risk = (lo != hi) or meta.vfr
    meta.n_frames_safe = max(0, lo - FRAME_COUNT_BACKOFF) if meta.risk else meta.n_frames
    return meta


def meta_from_dims(video_id: str, width: int, height: int, n_frames: int, fps: float = 30.0) -> VideoMeta:
    """Build a VideoMeta directly -- for tests and for synthetic fixtures."""
    return VideoMeta(
        video_id=video_id,
        width=width,
        height=height,
        fps=fps,
        duration=(n_frames / fps) if fps else 0.0,
        n_frames=n_frames,
        n_frames_safe=n_frames,
        sources={"synthetic": n_frames},
    )
