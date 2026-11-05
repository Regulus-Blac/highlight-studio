"""Index and submission file IO.

Two format hazards this module exists to absorb:

1. **The index may be JSON or JSONL.** ``baseline/test_index.json`` is a JSON
   array, but ``contest_info/task.md`` says the real test index ships as JSONL
   ("测试阶段赛方提供 JSONL 索引文件"). ``load_index`` accepts either.

2. **The index may carry a filename.** task.md lists ``video_id``、``视频文件名``
   and ``targetRatioWH``; the sample index omits the filename, and both
   baselines hardcode ``<video_id>.mp4``. ``IndexEntry.filename`` keeps the
   official value when present and falls back to the hardcoded form.

Submission files are written atomically and without a trailing blank line --
``baseline/predictions.jsonl`` currently ends with one, and an evaluator doing
the obvious ``for line in f: json.loads(line)`` would fail the whole file on it.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class IndexEntry:
    """One row of the test index."""

    video_id: str
    target_ratio: Tuple[int, int]
    filename: Optional[str] = None

    @property
    def tw(self) -> int:
        return self.target_ratio[0]

    @property
    def th(self) -> int:
        return self.target_ratio[1]

    def video_path(self, video_dir: str) -> str:
        """Resolve the source video path, honouring an index-supplied filename."""
        return os.path.join(video_dir, self.filename or f"{self.video_id}.mp4")


@dataclass
class SubmissionRow:
    """One line of the submission JSONL."""

    video_id: str
    target_ratio: Tuple[int, int]
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "targetRatioWH": [int(self.target_ratio[0]), int(self.target_ratio[1])],
            "predictions": self.predictions,
        }


def _parse_index_item(item: Dict[str, Any]) -> IndexEntry:
    vid = str(item["video_id"])
    tr = item.get("targetRatioWH", [16, 9])
    filename = None
    for key in ("video_name", "file_name", "filename", "video_file", "video"):
        if isinstance(item.get(key), str) and item[key]:
            filename = item[key]
            break
    return IndexEntry(video_id=vid, target_ratio=(int(tr[0]), int(tr[1])), filename=filename)


def load_index(index_path: str) -> List[IndexEntry]:
    """Load the test index from either a JSON array or a JSONL file.

    Ported from ``baseline/baseline_center.py:34-42``, plus JSONL support and
    the optional filename field.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        text = f.read()

    items: List[Dict[str, Any]]
    try:
        loaded = json.loads(text)
        items = loaded if isinstance(loaded, list) else [loaded]
    except json.JSONDecodeError:
        items = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{index_path}:{lineno}: not valid JSON or JSONL: {exc}") from exc

    entries = [_parse_index_item(it) for it in items]

    seen = set()
    for e in entries:
        if e.video_id in seen:
            raise ValueError(f"{index_path}: duplicate video_id {e.video_id!r}")
        seen.add(e.video_id)
    return entries


def read_submission(path: str, strict: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read a submission JSONL.

    Returns ``(rows, problems)``. In ``strict`` mode a blank or unparseable line
    is recorded as a problem rather than raised, so the validator can report
    every defect in one pass instead of dying on the first.
    """
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                # Not merely cosmetic: a naive line-by-line evaluator dies here.
                problems.append(f"line {lineno}: blank line")
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                problems.append(f"line {lineno}: invalid JSON: {exc}")
                if strict:
                    continue
                raise
            if not isinstance(obj, dict):
                problems.append(f"line {lineno}: top-level value is {type(obj).__name__}, expected object")
                continue
            rows.append(obj)
    return rows, problems


def write_submission(
    path: str,
    rows: Iterable[SubmissionRow],
    index: Optional[Sequence[IndexEntry]] = None,
) -> int:
    """Write a submission JSONL atomically, one object per line, no blank lines.

    When ``index`` is given, rows are emitted in index order and any missing
    video gets an empty-prediction row, so the file always covers every
    ``video_id``. A missing row scores 0 for that video, so silently dropping
    one is the most expensive possible bug.
    """
    by_id = {r.video_id: r for r in rows}

    if index is not None:
        ordered = [by_id.get(e.video_id) or SubmissionRow(e.video_id, e.target_ratio, []) for e in index]
        extra = set(by_id) - {e.video_id for e in index}
        if extra:
            raise ValueError(f"rows not present in index: {sorted(extra)}")
    else:
        ordered = list(by_id.values())

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in ordered:
                f.write(json.dumps(row.to_json_obj(), separators=(",", ":")))
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(ordered)
