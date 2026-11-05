"""Local implementation of the competition metric, plus diagnostics.

The metric (``contest_info/task.md`` §六)::

    P = S_IoU / N_pred     R = S_IoU / N_gt     F_i = 2PR/(P+R) = 2*S_IoU/(N_pred+N_gt)
    Score = 100 * mean_i(F_i)

The part everyone implements wrong
----------------------------------
**Every submitted element increments ``N_pred``, valid or not.** task.md is
explicit: "frame错误、重复预测或格式无效的预测不获得得分，并计入该视频的
N_pred". An invalid prediction is strictly worse than no prediction, which is
why ``shrink_to_fit`` returns None rather than emitting a box that might not
fit.

Why this doubles as a research tool
-----------------------------------
Let ``M`` be the set of submitted frames that hit a GT frame and
``rho = S_IoU/|M|``. Then, exactly and with no approximation::

    F_i = rho_i * F_temporal_i        where  F_temporal_i = 2|M|/(N_pred+N_gt)

So one evaluator run yields both oracle decompositions at once: ``rho`` is the
score with perfect temporal selection, ``F_temporal`` the score with perfect
boxes. Allocate effort by comparing ``-log rho`` against ``-log F_temporal``.

Careful: the factorization is **per video**. ``mean(F)`` is not
``mean(rho) * mean(F_temporal)``, so the aggregates below are diagnostics, not
a decomposition of the final score.
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from highlight.geometry import RoundMode, derive_h, iou_xywh
from highlight.io.index import IndexEntry, load_index, read_submission
from highlight.io.video import VideoMeta

SCORED = "SCORED"
INVALID_SCHEMA = "INVALID_SCHEMA"
INVALID_NAN = "INVALID_NAN"
INVALID_FRAME_RANGE = "INVALID_FRAME_RANGE"
INVALID_COORD = "INVALID_COORD"
INVALID_CONTAIN = "INVALID_CONTAIN"
DUPLICATE = "DUPLICATE"
FRAME_MISS = "FRAME_MISS"

REASONS = (
    SCORED,
    INVALID_SCHEMA,
    INVALID_NAN,
    INVALID_FRAME_RANGE,
    INVALID_COORD,
    INVALID_CONTAIN,
    DUPLICATE,
    FRAME_MISS,
)

# video_id -> frame -> [x, y, w, h]
GroundTruth = Dict[str, Dict[int, Sequence[float]]]


@dataclass
class VideoResult:
    """Per-video score and diagnostics."""

    video_id: str
    n_pred: int = 0
    n_gt: int = 0
    s_iou: float = 0.0
    n_matched: int = 0
    f: float = 0.0
    reasons: Counter = field(default_factory=Counter)
    values: List[float] = field(default_factory=list)
    best_f: float = 0.0
    best_k: int = 0
    n_runs: int = 0

    @property
    def rho(self) -> Optional[float]:
        """Mean spatial IoU on matched frames -- the oracle-temporal score."""
        return self.s_iou / self.n_matched if self.n_matched else None

    @property
    def f_temporal(self) -> float:
        """Score this frame selection would get with perfect boxes."""
        denom = self.n_pred + self.n_gt
        return (2.0 * self.n_matched / denom) if denom else 1.0

    def coverage(self, n_frames: int) -> float:
        return (self.n_pred / n_frames) if n_frames else 0.0


@dataclass
class EvalResult:
    """Aggregate result over the whole index."""

    score: float
    per_video: List[VideoResult]
    reasons: Counter
    mean_rho: Optional[float]
    mean_f_temporal: float
    n_videos: int
    n_zero_f: int
    n_empty_pred: int
    n_no_match: int
    mean_best_f: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "n_videos": self.n_videos,
            "mean_rho": self.mean_rho,
            "mean_f_temporal": self.mean_f_temporal,
            "mean_best_f": self.mean_best_f,
            "n_zero_f": self.n_zero_f,
            "n_empty_pred": self.n_empty_pred,
            "n_no_match": self.n_no_match,
            "reasons": dict(self.reasons),
            "per_video": [
                {
                    "video_id": v.video_id,
                    "n_pred": v.n_pred,
                    "n_gt": v.n_gt,
                    "s_iou": v.s_iou,
                    "n_matched": v.n_matched,
                    "f": v.f,
                    "rho": v.rho,
                    "f_temporal": v.f_temporal,
                    "best_f": v.best_f,
                    "best_k": v.best_k,
                    "n_runs": v.n_runs,
                    "reasons": dict(v.reasons),
                }
                for v in self.per_video
            ],
        }


def _as_frame(value: Any) -> Optional[int]:
    """Coerce a frame index, rejecting bools (``isinstance(True, int)`` is True)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            return None
        return int(value)
    return None


def _as_triplet(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        out.append(float(v))
    return out[0], out[1], out[2]


def coverage_sweep(values: Sequence[float], n_gt: int) -> Tuple[float, int, List[float]]:
    """Best F achievable by keeping only the top-k of these predictions.

    ``values[i]`` is the IoU prediction ``i`` contributed (0 if it scored
    nothing). Returns ``(best_f, best_k, curve)``. The gap between ``best_f``
    and the realized ``F`` is exactly what the decision layer left on the table
    for this ranking -- if it is large, fix the decision layer, not the model.
    """
    if n_gt <= 0:
        return (1.0, 0, []) if not values else (0.0, 0, [])
    ordered = sorted(values, reverse=True)
    curve: List[float] = []
    best_f, best_k, total = 0.0, 0, 0.0
    for k, v in enumerate(ordered, 1):
        total += v
        f = 2.0 * total / (k + n_gt)
        curve.append(f)
        if f > best_f:
            best_f, best_k = f, k
    return best_f, best_k, curve


def _count_runs(frames: Sequence[int]) -> int:
    """Number of maximal contiguous runs -- a fragmentation tell for 人工复核."""
    if not frames:
        return 0
    ordered = sorted(frames)
    runs = 1
    for a, b in zip(ordered, ordered[1:]):
        if b != a + 1:
            runs += 1
    return runs


def score_video(
    video_id: str,
    predictions: Sequence[Any],
    gt_frames: Dict[int, Sequence[float]],
    meta: VideoMeta,
    tw: float,
    th: float,
    iou_h_mode: RoundMode = "exact",
    contain_h_mode: RoundMode = "ceil",
    oracle_spatial: bool = False,
) -> VideoResult:
    """Score one video, walking ``predictions`` in submission order.

    ``oracle_spatial`` forces matched frames to IoU 1.0, turning the returned
    ``f`` into ``f_temporal`` -- the score this frame selection would get with
    perfect boxes.
    """
    res = VideoResult(video_id=video_id, n_gt=len(gt_frames))
    seen_valid: set = set()
    scored_frames: List[int] = []

    for item in predictions:
        res.n_pred += 1
        value = 0.0

        frame = _as_frame(item.get("frame")) if isinstance(item, dict) else None
        triplet = _as_triplet(item.get("bboxes")) if isinstance(item, dict) else None
        if frame is None or triplet is None:
            res.reasons[INVALID_SCHEMA] += 1
            res.values.append(value)
            continue

        x, y, w = triplet
        if not all(math.isfinite(v) for v in (x, y, w)):
            res.reasons[INVALID_NAN] += 1
            res.values.append(value)
            continue

        if frame < 0 or frame >= meta.n_frames:
            res.reasons[INVALID_FRAME_RANGE] += 1
            res.values.append(value)
            continue

        if x < 0 or y < 0 or w <= 0:
            res.reasons[INVALID_COORD] += 1
            res.values.append(value)
            continue

        h_contain = derive_h(w, tw, th, mode=contain_h_mode)
        if x + w > meta.width or y + h_contain > meta.height:
            res.reasons[INVALID_CONTAIN] += 1
            res.values.append(value)
            continue

        # Valid. The rule keeps the first VALID prediction per frame, so an
        # earlier invalid one does not claim the slot.
        if frame in seen_valid:
            res.reasons[DUPLICATE] += 1
            res.values.append(value)
            continue
        seen_valid.add(frame)

        gt_box = gt_frames.get(frame)
        if gt_box is None:
            res.reasons[FRAME_MISS] += 1
            res.values.append(value)
            continue

        if oracle_spatial:
            value = 1.0
        else:
            h = derive_h(w, tw, th, mode=iou_h_mode)
            value = iou_xywh((x, y, w, h), gt_box)

        res.reasons[SCORED] += 1
        res.n_matched += 1
        res.s_iou += value
        res.values.append(value)
        scored_frames.append(frame)

    if res.n_gt == 0 and res.n_pred == 0:
        res.f = 1.0
    elif res.n_gt == 0 or res.n_pred == 0:
        res.f = 0.0
    else:
        res.f = 2.0 * res.s_iou / (res.n_pred + res.n_gt)

    res.best_f, res.best_k, _ = coverage_sweep(res.values, res.n_gt)
    res.n_runs = _count_runs(scored_frames)
    return res


def evaluate(
    submission_rows: Sequence[Dict[str, Any]],
    gt: GroundTruth,
    index: Sequence[IndexEntry],
    metas: Dict[str, VideoMeta],
    iou_h_mode: RoundMode = "exact",
    contain_h_mode: RoundMode = "ceil",
    oracle_spatial: bool = False,
) -> EvalResult:
    """Score a submission against ground truth.

    The denominator is **every video_id in the index**, not the rows present in
    the submission: a missing row is ``N_pred = 0``, which scores 0 whenever the
    video has any GT.
    """
    by_id = {str(r.get("video_id")): r for r in submission_rows}
    per_video: List[VideoResult] = []
    totals: Counter = Counter()

    for entry in index:
        row = by_id.get(entry.video_id)
        preds = row.get("predictions") if isinstance(row, dict) else None
        if not isinstance(preds, list):
            preds = []
        meta = metas.get(entry.video_id) or VideoMeta(video_id=entry.video_id)
        res = score_video(
            video_id=entry.video_id,
            predictions=preds,
            gt_frames=gt.get(entry.video_id, {}),
            meta=meta,
            tw=entry.tw,
            th=entry.th,
            iou_h_mode=iou_h_mode,
            contain_h_mode=contain_h_mode,
            oracle_spatial=oracle_spatial,
        )
        per_video.append(res)
        totals.update(res.reasons)

    n = len(per_video)
    rhos = [v.rho for v in per_video if v.rho is not None]
    return EvalResult(
        score=100.0 * sum(v.f for v in per_video) / n if n else 0.0,
        per_video=per_video,
        reasons=totals,
        mean_rho=(sum(rhos) / len(rhos)) if rhos else None,
        mean_f_temporal=(sum(v.f_temporal for v in per_video) / n) if n else 0.0,
        n_videos=n,
        n_zero_f=sum(1 for v in per_video if v.f == 0.0),
        n_empty_pred=sum(1 for v in per_video if v.n_pred == 0),
        n_no_match=sum(1 for v in per_video if v.n_matched == 0),
        mean_best_f=(sum(v.best_f for v in per_video) / n) if n else 0.0,
    )


def load_gt(path: str) -> GroundTruth:
    """Load ground truth into ``{video_id: {frame: [x, y, w, h]}}``.

    Accepts two shapes:

    * submission-shaped JSONL / JSON list -- rows with ``video_id`` and
      ``predictions: [{"frame": f, "bboxes": [...]}]``, where ``bboxes`` may be
      a triplet (height derived) or a 4-tuple;
    * a mapping ``{video_id: {frame: [x, y, w, h]}}``.

    The real annotation format is not known yet (the training archive has not
    been downloaded). **This is the adapter point** -- when the archive lands,
    normalize whatever it contains into the first shape rather than changing
    the evaluator.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    objs: List[Any]
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict) and "predictions" not in loaded:
            return {
                str(vid): {int(fr): list(box) for fr, box in frames.items()}
                for vid, frames in loaded.items()
            }
        objs = loaded if isinstance(loaded, list) else [loaded]
    except json.JSONDecodeError:
        objs = [json.loads(line) for line in text.splitlines() if line.strip()]

    gt: GroundTruth = {}
    for obj in objs:
        vid = str(obj["video_id"])
        tr = obj.get("targetRatioWH", [16, 9])
        frames: Dict[int, Sequence[float]] = {}
        for p in obj.get("predictions") or obj.get("annotations") or []:
            box = list(p["bboxes"])
            if len(box) == 3:
                box.append(derive_h(box[2], float(tr[0]), float(tr[1]), mode="exact"))
            frames[int(p["frame"])] = box
        gt[vid] = frames
    return gt


def format_report(result: EvalResult, top_n: int = 10) -> str:
    """Human-readable summary; the JSON dump carries the full per-video detail."""
    lines = [
        f"Score           {result.score:.4f}   (mean F over {result.n_videos} videos x 100)",
        f"mean rho        {result.mean_rho:.4f}" if result.mean_rho is not None else "mean rho        n/a",
        f"mean F_temporal {result.mean_f_temporal:.4f}",
        f"mean best_F     {result.mean_best_f:.4f}   (ceiling of the submitted ranking)",
        "",
        f"videos scoring 0      {result.n_zero_f}",
        f"videos with no preds  {result.n_empty_pred}",
        f"videos with no match  {result.n_no_match}",
        "",
        "prediction outcomes:",
    ]
    total = sum(result.reasons.values())
    for reason in REASONS:
        count = result.reasons.get(reason, 0)
        if count:
            lines.append(f"  {reason:<22} {count:>10}  ({100.0 * count / total:.2f}%)")

    worst = sorted(result.per_video, key=lambda v: v.f)[:top_n]
    if worst:
        lines += ["", f"worst {len(worst)} videos:", f"  {'video':<10}{'F':>8}{'rho':>8}{'F_temp':>9}{'best_F':>9}{'n_pred':>8}{'n_gt':>8}"]
        for v in worst:
            rho = f"{v.rho:.3f}" if v.rho is not None else "  n/a"
            lines.append(f"  {v.video_id:<10}{v.f:>8.3f}{rho:>8}{v.f_temporal:>9.3f}{v.best_f:>9.3f}{v.n_pred:>8}{v.n_gt:>8}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a submission against ground truth")
    ap.add_argument("--pred", required=True, help="submission JSONL")
    ap.add_argument("--gt", required=True, help="ground-truth file (see load_gt)")
    ap.add_argument("--index", required=True, help="test index (JSON or JSONL)")
    ap.add_argument("--meta", help="video metadata JSON from scripts/highlight/census.py")
    ap.add_argument("--video-dir", help="probe videos directly instead of --meta")
    ap.add_argument("--iou-h-mode", default="exact", choices=["exact", "ceil", "floor", "round"])
    ap.add_argument("--contain-h-mode", default="ceil", choices=["exact", "ceil", "floor", "round"])
    ap.add_argument("--oracle-spatial", action="store_true", help="force matched IoU to 1.0 -> reports F_temporal")
    ap.add_argument("--json-out", help="write the full per-video report here")
    args = ap.parse_args()

    index = load_index(args.index)
    rows, problems = read_submission(args.pred)
    for p in problems:
        print(f"[submission] {p}")

    metas: Dict[str, VideoMeta] = {}
    if args.meta:
        with open(args.meta, "r", encoding="utf-8") as f:
            for vid, m in json.load(f).items():
                metas[str(vid)] = VideoMeta(**m) if isinstance(m, dict) else m
    elif args.video_dir:
        from highlight.io.video import probe_video

        for entry in index:
            metas[entry.video_id] = probe_video(entry.video_id, entry.video_path(args.video_dir))
    else:
        ap.error("one of --meta or --video-dir is required")

    result = evaluate(
        rows,
        load_gt(args.gt),
        index,
        metas,
        iou_h_mode=args.iou_h_mode,
        contain_h_mode=args.contain_h_mode,
        oracle_spatial=args.oracle_spatial,
    )
    print(format_report(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
