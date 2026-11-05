"""Crop-box geometry for the highlight re-framing task.

The submission format is a triplet ``[x, y, w]`` per frame; the evaluator
derives ``h = w * target_h / target_w`` itself.

The rounding trap (read this before touching anything here)
-----------------------------------------------------------
``compute_crop_size`` is lifted verbatim from ``baseline/baseline_center.py``
and defines the organizer's "largest target-ratio rectangle". It computes the
width by **rounding up**, but the evaluator derives the height back **from that
width** -- and the two do not round-trip::

    1920x1080 source, 9:16 target
      cw = round(1080 * 9/16) = round(607.5) = 608     (banker's rounding)
      h  = 608 * 16/9         = 1080.889  >  1080      -> box leaves the frame

An out-of-frame box is invalid: it scores nothing yet still counts in
``N_pred``, so it is strictly worse than not predicting that frame at all. The
same defect hits ``1440x1080`` and ``1080x1080`` with a 9:16 target -- i.e. the
most common landscape resolutions, against the 55 vertical-target videos in the
test index.

``safe_crop_size`` is therefore the working definition used by every placement
helper below: the largest ``w`` whose derived height fits under *every* plausible
rounding convention. It costs ~0.2% IoU against the organizer's width and
removes the risk of losing a whole video.

The 1-DOF structure
-------------------
Either way, one axis is pegged to the source dimension, so a max-size crop has
exactly one degree of freedom. For two equal-size boxes offset by ``d`` along a
free axis of extent ``c``::

    IoU(d) = max(0, c - |d|) / (c + |d|)

strictly monotone in ``|d|`` alone. So at fixed scale, minimizing L1 on the
offset is rank-equivalent to maximizing IoU and a differentiable IoU loss buys
nothing. That equivalence breaks as soon as GT boxes vary in scale -- then use
``iou_xywh``.
"""

import math
from typing import List, Literal, Optional, Sequence, Tuple

# [x, y, w, h] in source pixels.
Box = Tuple[float, float, float, float]
# [x, y, w] -- the submission format; h is derived by the evaluator.
Triplet = Tuple[int, int, int]

RoundMode = Literal["exact", "ceil", "floor", "round"]


def compute_crop_size(W: int, H: int, tw: float, th: float) -> Tuple[int, int]:
    """The organizer's max box, lifted verbatim from ``baseline_center.py:56-66``.

    Kept for reference and for measuring GT against the organizer's own
    definition (Phase 0, S1/S2). **Do not place boxes with this** -- its width
    can imply a height that overflows the frame. Use ``safe_crop_size``.
    """
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return W, H
    target = float(tw) / float(th)
    if W / float(H) >= target:
        ch = H
        cw = min(int(round(H * target)), W)
    else:
        cw = W
        ch = min(int(round(W / target)), H)
    return max(1, cw), max(1, ch)


def derive_h(w: float, tw: float, th: float, mode: RoundMode = "exact") -> float:
    """Derive box height from width and target ratio: ``h = w * th / tw``.

    The organizer's rounding convention is unknown (measure it in Phase 0, S2).
    Use ``"exact"`` for IoU -- it is faithful to the stated formula. Use
    ``"ceil"`` for containment: it is the only mode that cannot be invalidated
    by a stricter convention than ours.
    """
    h = float(w) * float(th) / float(tw)
    if mode == "exact":
        return h
    if mode == "ceil":
        return float(math.ceil(h))
    if mode == "floor":
        return float(math.floor(h))
    if mode == "round":
        return float(round(h))
    raise ValueError(f"unknown round mode: {mode!r}")


def _max_width_for_height(H: int, tw: float, th: float) -> int:
    """Largest integer ``w`` with ``w * th / tw <= H`` (exact integer math when possible).

    ``ceil(x) <= H`` is equivalent to ``x <= H`` for integer ``H``, so this
    bound is simultaneously safe under exact, ceil, floor and round.
    """
    if float(tw).is_integer() and float(th).is_integer():
        return (int(H) * int(tw)) // int(th)
    return int(math.floor(H * float(tw) / float(th)))


def safe_crop_size(W: int, H: int, tw: float, th: float) -> Tuple[int, int]:
    """Largest target-ratio box that fits under every rounding convention.

    Returns ``(w, h)`` with ``h = ceil(w * th / tw)``, guaranteeing
    ``w <= W`` and ``h <= H``.
    """
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return max(1, int(W)), max(1, int(H))
    w = max(1, min(int(W), _max_width_for_height(int(H), tw, th)))
    h = int(derive_h(w, tw, th, mode="ceil"))
    # Defensive: shrink if float error left us a pixel over.
    while h > H and w > 1:
        w -= 1
        h = int(derive_h(w, tw, th, mode="ceil"))
    return w, max(1, h)


def free_axis(W: int, H: int, tw: float, th: float) -> Optional[str]:
    """Which axis a safe max-size crop slides along: ``"x"``, ``"y"`` or None.

    Rounding can leave a pixel of slack on the pegged axis too, so this returns
    whichever axis has the larger travel. None means the crop is effectively
    the whole frame: zero degrees of freedom, IoU 1.0 for free against an
    equally-sized GT box.
    """
    w, h = safe_crop_size(W, H, tw, th)
    rx, ry = W - w, H - h
    if rx <= 0 and ry <= 0:
        return None
    return "x" if rx >= ry else "y"


def dof_range(W: int, H: int, tw: float, th: float) -> int:
    """Travel of a safe max-size crop along its free axis, in pixels."""
    w, h = safe_crop_size(W, H, tw, th)
    return max(W - w, H - h, 0)


def free_extent(W: int, H: int, tw: float, th: float) -> int:
    """Crop extent ALONG the free axis -- the ``c`` in ``iou_1d(d, c)``."""
    w, h = safe_crop_size(W, H, tw, th)
    return w if free_axis(W, H, tw, th) == "x" else h


def difficulty(W: int, H: int, tw: float, th: float) -> float:
    """Placement accuracy needed for IoU 0.5, as a fraction of the free travel.

    ``IoU(d) = 0.5`` at ``|d| = c/3``, so this is ``c / (3R)``. Above ~1.5 the
    video is nearly free; below ~0.3 it is hard. ``inf`` means no freedom.
    Phase 0 statistic C2.
    """
    R = dof_range(W, H, tw, th)
    if R <= 0:
        return float("inf")
    return free_extent(W, H, tw, th) / (3.0 * R)


def place_1d(u: float, W: int, H: int, tw: float, th: float) -> Triplet:
    """Place a safe max-size crop at normalized position ``u`` on the free axis.

    ``u`` is the box ORIGIN as a fraction of the free travel: 0 is flush
    left/top, 1 flush right/bottom, 0.5 centered. This is the coordinate Phase 0
    statistic S3 measures on GT boxes and the one the spatial head's K bins
    discretize. The result always satisfies ``contains``.
    """
    w, h = safe_crop_size(W, H, tw, th)
    u = min(1.0, max(0.0, float(u)))
    rx, ry = max(0, W - w), max(0, H - h)
    if free_axis(W, H, tw, th) == "x":
        return int(round(u * rx)), ry // 2, int(w)
    return rx // 2, int(round(u * ry)), int(w)


def center_crop(W: int, H: int, tw: float, th: float) -> Triplet:
    """The centered safe max-size crop -- baseline placement and last-ditch fallback."""
    return place_1d(0.5, W, H, tw, th)


def position_to_u(box: Sequence[float], W: int, H: int, tw: float, th: float) -> Optional[float]:
    """Inverse of ``place_1d``: recover ``u`` from a box ``[x, y, ...]``.

    Returns None when the video has no degrees of freedom. Used to convert GT
    boxes into the spatial head's target coordinate.
    """
    R = dof_range(W, H, tw, th)
    if R <= 0:
        return None
    pos = float(box[0]) if free_axis(W, H, tw, th) == "x" else float(box[1])
    return min(1.0, max(0.0, pos / R))


def iou_1d(d: float, c: float) -> float:
    """IoU of two equal-size boxes offset by ``d`` along one axis of extent ``c``.

    Assumes the other axis overlaps completely, which holds for two max-size
    crops of the same video. If GT boxes turn out to vary in scale (Phase 0,
    S1), this is wrong -- use ``iou_xywh``.
    """
    if c <= 0:
        return 0.0
    d = abs(float(d))
    if d >= c:
        return 0.0
    return (c - d) / (c + d)


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes.

    ``scripts/timelens/timelens_reward.py`` also has a ``compute_iou``, but that
    one is **xyxy**. Passing xywh into it silently returns garbage, which is why
    this exists separately.
    """
    ax, ay, aw, ah = (float(v) for v in a[:4])
    bx, by, bw, bh = (float(v) for v in b[:4])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def contains(
    triplet: Sequence[float],
    W: int,
    H: int,
    tw: float,
    th: float,
    mode: RoundMode = "ceil",
) -> bool:
    """Whether ``[x, y, w]`` stays inside the frame once ``h`` is derived."""
    x, y, w = (float(v) for v in triplet[:3])
    if not all(math.isfinite(v) for v in (x, y, w)):
        return False
    if x < 0 or y < 0 or w <= 0:
        return False
    h = derive_h(w, tw, th, mode=mode)
    if h <= 0:
        return False
    return x + w <= W and y + h <= H


def shrink_to_fit(
    triplet: Sequence[float],
    W: int,
    H: int,
    tw: float,
    th: float,
    mode: RoundMode = "ceil",
    max_shrink: int = 8,
) -> Optional[Triplet]:
    """Nudge a box into the frame, shrinking ``w`` by at most ``max_shrink`` px.

    Returns None if it still does not fit. The caller must then drop the
    prediction rather than submit it: an out-of-frame box scores 0 but still
    counts in ``N_pred``, so it is strictly worse than omitting the frame.
    """
    x, y, w = (int(round(float(v))) for v in triplet[:3])
    x, y = max(0, x), max(0, y)
    for _ in range(max_shrink + 1):
        if w <= 0:
            return None
        h = derive_h(w, tw, th, mode=mode)
        cand = (max(0, min(x, W - w)), max(0, min(y, H - int(h))), w)
        if contains(cand, W, H, tw, th, mode=mode):
            return cand
        w -= 1
    return None


def bin_to_u(k: int, K: int) -> float:
    """Center of spatial-head bin ``k`` of ``K``, in normalized ``u`` coordinates."""
    return 0.5 if K <= 1 else k / (K - 1)


def bin_iou_matrix(K: int, R: float, c: float) -> List[List[float]]:
    """``M[j][k] = iou_1d`` between bins ``j`` and ``k`` -- the spatial decode kernel.

    The spatial head predicts a distribution ``p`` over bins; the metric-optimal
    placement is ``argmax_j sum_k p[k] * M[j][k]``, and that maximum is the
    expected IoU ``q_f`` the decision layer consumes.
    """
    us = [bin_to_u(k, K) for k in range(K)]
    return [[iou_1d((us[j] - us[k]) * R, c) for k in range(K)] for j in range(K)]
