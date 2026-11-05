"""Local evaluator and submission validator for the competition metric."""

from highlight.metric.evaluate import (
    EvalResult,
    VideoResult,
    coverage_sweep,
    evaluate,
    load_gt,
    score_video,
)

__all__ = [
    "EvalResult",
    "VideoResult",
    "coverage_sweep",
    "evaluate",
    "load_gt",
    "score_video",
]
