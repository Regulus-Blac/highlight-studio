"""Submission format validator -- the mandatory gate before any upload.

Deliberately stricter than ``evaluate.py``. The evaluator models what the
organizer's scorer probably does; the validator guards what *we* emit, so it
rejects anything whose handling is uncertain. An invalid prediction still
counts in ``N_pred``, and a malformed file can invalidate the entire
submission (``task.md`` §九), so the asymmetry is deliberate.

Failure modes this exists to catch, all of them live in the current repo:

* ``baseline/predictions.jsonl`` covers 11 of 174 videos and ends with a blank
  line. An evaluator doing ``for line in f: json.loads(line)`` dies on it.
* ``baseline_qwen.py:450-455`` silently emits ``predictions: []`` when segment
  parsing fails; that video then scores 0 with no error anywhere.
* ``isinstance(True, int)`` is True in Python, so a stray boolean passes a
  naive int check and serializes as ``true``.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from highlight.geometry import RoundMode, derive_h
from highlight.io.index import IndexEntry, load_index, read_submission
from highlight.io.video import VideoMeta

COVERAGE_HIGH = 0.95
COVERAGE_LOW = 0.02
MAX_RUNS = 20
TAIL_MARGIN = 2


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _is_int(value: Any) -> bool:
    """True for a genuine int. Rejects bools, which are ints in Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate(
    submission_rows: Sequence[Dict[str, Any]],
    index: Sequence[IndexEntry],
    metas: Optional[Dict[str, VideoMeta]] = None,
    contain_h_mode: RoundMode = "ceil",
    line_problems: Sequence[str] = (),
) -> ValidationReport:
    """Validate a parsed submission against the index and video metadata."""
    rep = ValidationReport()
    metas = metas or {}

    for problem in line_problems:
        rep.error(f"file: {problem}")

    index_ids = [e.video_id for e in index]
    index_by_id = {e.video_id: e for e in index}

    seen: Dict[str, int] = {}
    for i, row in enumerate(submission_rows):
        vid = row.get("video_id")
        if not isinstance(vid, str):
            rep.error(f"row {i}: video_id is {type(vid).__name__}, expected str")
            continue
        if vid in seen:
            rep.error(f"row {i}: duplicate video_id {vid!r} (first seen at row {seen[vid]})")
            continue
        seen[vid] = i

    missing = [v for v in index_ids if v not in seen]
    extra = [v for v in seen if v not in index_by_id]
    if missing:
        shown = ", ".join(missing[:10]) + (f", ... (+{len(missing) - 10} more)" if len(missing) > 10 else "")
        rep.error(f"missing {len(missing)}/{len(index_ids)} video_id(s) from the index: {shown}")
    if extra:
        rep.error(f"{len(extra)} video_id(s) not in the index: {', '.join(sorted(extra)[:10])}")

    n_empty = 0
    total_preds = 0

    for entry in index:
        row = submission_rows[seen[entry.video_id]] if entry.video_id in seen else None
        if row is None:
            continue
        tag = f"video {entry.video_id}"

        tr = row.get("targetRatioWH")
        if tr != [entry.tw, entry.th]:
            rep.error(f"{tag}: targetRatioWH is {tr!r}, index says {[entry.tw, entry.th]} (must not be modified)")

        preds = row.get("predictions")
        if not isinstance(preds, list):
            rep.error(f"{tag}: predictions is {type(preds).__name__}, expected list")
            continue

        total_preds += len(preds)
        if not preds:
            n_empty += 1
            rep.warn(f"{tag}: empty predictions -- scores 0 unless this video genuinely has no highlight")
            continue

        meta = metas.get(entry.video_id)
        frames: List[int] = []
        bad = 0

        for j, p in enumerate(preds):
            where = f"{tag} pred {j}"
            if not isinstance(p, dict):
                rep.error(f"{where}: element is {type(p).__name__}, expected object")
                bad += 1
                continue

            frame = p.get("frame")
            if not _is_int(frame):
                rep.error(f"{where}: frame is {frame!r} ({type(frame).__name__}), expected int")
                bad += 1
                continue
            if frame < 0:
                rep.error(f"{where}: frame {frame} < 0")
                bad += 1
                continue

            box = p.get("bboxes")
            if not isinstance(box, list) or len(box) != 3:
                rep.error(f"{where}: bboxes must be a 3-element list [x, y, w], got {box!r}")
                bad += 1
                continue
            if not all(_is_int(v) for v in box):
                rep.error(f"{where}: bboxes must be ints, got {box!r}")
                bad += 1
                continue
            if not all(math.isfinite(v) for v in box):
                rep.error(f"{where}: bboxes contains a non-finite value: {box!r}")
                bad += 1
                continue

            x, y, w = box
            if x < 0 or y < 0:
                rep.error(f"{where}: x={x}, y={y}; both must be >= 0")
                bad += 1
                continue
            if w <= 0:
                rep.error(f"{where}: w={w}; must be > 0")
                bad += 1
                continue

            frames.append(frame)

            if meta is not None and meta.ok:
                if frame >= meta.n_frames_safe:
                    rep.error(
                        f"{where}: frame {frame} >= n_frames_safe {meta.n_frames_safe} "
                        f"(sources: {meta.sources})"
                    )
                    bad += 1
                    continue
                if meta.risk and frame >= meta.n_frames_safe - TAIL_MARGIN:
                    rep.warn(f"{where}: frame {frame} is within {TAIL_MARGIN} of the safe tail on a risk video")

                h = derive_h(w, entry.tw, entry.th, mode=contain_h_mode)
                if x + w > meta.width or y + h > meta.height:
                    rep.error(
                        f"{where}: box [{x},{y},{w}] with h={h:g} leaves the "
                        f"{meta.width}x{meta.height} frame"
                    )
                    bad += 1

        dupes = len(frames) - len(set(frames))
        if dupes:
            rep.error(f"{tag}: {dupes} duplicate frame index(es); only the first valid one would score")
        if frames != sorted(frames):
            rep.warn(f"{tag}: frames are not ascending (task.md recommends ascending order)")

        if meta is not None and meta.ok and meta.n_frames_safe:
            cov = len(preds) / meta.n_frames_safe
            if cov > COVERAGE_HIGH:
                rep.warn(f"{tag}: coverage {cov:.1%} -- predicting nearly every frame caps F at 2r/(1+r)")
            elif cov < COVERAGE_LOW:
                rep.warn(f"{tag}: coverage {cov:.1%} -- suspiciously few predictions")

        if frames:
            ordered = sorted(set(frames))
            runs = 1 + sum(1 for a, b in zip(ordered, ordered[1:]) if b != a + 1)
            if runs > MAX_RUNS:
                rep.warn(f"{tag}: {runs} non-contiguous runs -- fragmented output risks 人工复核 penalties")

        if bad:
            rep.stats.setdefault("videos_with_bad_preds", []).append(entry.video_id)

    rep.stats.update(
        {
            "n_index": len(index_ids),
            "n_rows": len(submission_rows),
            "n_predictions": total_preds,
            "n_empty_videos": n_empty,
            "n_errors": len(rep.errors),
            "n_warnings": len(rep.warnings),
        }
    )
    return rep


def format_report(rep: ValidationReport, max_items: int = 40) -> str:
    lines = [f"videos {rep.stats.get('n_rows', 0)}/{rep.stats.get('n_index', 0)}  "
             f"predictions {rep.stats.get('n_predictions', 0)}  "
             f"empty {rep.stats.get('n_empty_videos', 0)}"]
    if rep.errors:
        lines.append(f"\nERRORS ({len(rep.errors)}):")
        lines += [f"  {e}" for e in rep.errors[:max_items]]
        if len(rep.errors) > max_items:
            lines.append(f"  ... (+{len(rep.errors) - max_items} more)")
    if rep.warnings:
        lines.append(f"\nWARNINGS ({len(rep.warnings)}):")
        lines += [f"  {w}" for w in rep.warnings[:max_items]]
        if len(rep.warnings) > max_items:
            lines.append(f"  ... (+{len(rep.warnings) - max_items} more)")
    lines.append("\nPASS -- submission is well formed" if rep.ok else "\nFAIL -- do not submit")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a submission JSONL")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", help="video metadata JSON from scripts/highlight/census.py")
    ap.add_argument("--video-dir", help="probe videos directly instead of --meta")
    ap.add_argument("--contain-h-mode", default="ceil", choices=["exact", "ceil", "floor", "round"])
    ap.add_argument("--json-out")
    args = ap.parse_args()

    index = load_index(args.index)
    rows, problems = read_submission(args.pred)

    metas: Dict[str, VideoMeta] = {}
    if args.meta:
        with open(args.meta, "r", encoding="utf-8") as f:
            for vid, m in json.load(f).items():
                metas[str(vid)] = VideoMeta(**m) if isinstance(m, dict) else m
    elif args.video_dir:
        from highlight.io.video import probe_video

        for entry in index:
            metas[entry.video_id] = probe_video(entry.video_id, entry.video_path(args.video_dir))

    rep = validate(rows, index, metas, contain_h_mode=args.contain_h_mode, line_problems=problems)
    print(format_report(rep))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings, "stats": rep.stats}, f,
                      ensure_ascii=False, indent=2)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
