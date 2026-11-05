#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local implementation of the contest metric: video-averaged IoU-weighted F1.
赛事指标的本地实现：视频平均连续 IoU 加权 F1。

Per contest rules (task.md section 6):
  - match predictions to ground truth by (video_id, frame)
  - convert [x, y, w] -> [x, y, w, h] with h = w * target_h / target_w
  - P_i = S_i / N_pred_i,  R_i = S_i / N_gt_i,  F_i = 2*P*R/(P+R)
    (algebraically F_i = 2*S_i / (N_pred_i + N_gt_i))
  - invalid / duplicate / frame-mismatched predictions still count toward
    N_pred but contribute 0 to S
  - N_gt=0 and N_pred=0 -> F=1;  N_gt=0 and N_pred>0 -> F=0;
    N_gt>0 and N_pred=0 -> F=0
  - final score = mean(F_i) over all videos * 100

This module is the scoring core; `evaluate.py` is the CLI around it.
"""

import math

# 评测端由 targetRatioWH 反算高度，参赛者只提交 w。
def derive_h(w, target_ratio):
    """[x,y,w] + targetRatioWH -> box height. 由目标比例反算构图框高度。"""
    tw, th = float(target_ratio[0]), float(target_ratio[1])
    if tw <= 0:
        return float("nan")
    return w * th / tw


def iou_xywh(a, b):
    """Axis-aligned IoU of two [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def check_prediction(pred, target_ratio, meta=None):
    """Validate one prediction dict. Returns (ok, box_xywh, reason).

    `meta` is an optional dict with n_frames / width / height; bound checks
    that need it are skipped when it is None.
    meta 缺省时跳过需要画面尺寸的越界检查。
    """
    if not isinstance(pred, dict):
        return False, None, "prediction is not an object"
    if "frame" not in pred or "bboxes" not in pred:
        return False, None, "missing field frame/bboxes"

    f = pred["frame"]
    if not _finite(f) or float(f) != int(f):
        return False, None, "frame is not a finite integer"
    f = int(f)
    if f < 0:
        return False, None, "frame < 0"

    bb = pred["bboxes"]
    if not isinstance(bb, (list, tuple)) or len(bb) != 3:
        return False, None, "bboxes must be a triplet [x, y, w]"
    if not all(_finite(v) for v in bb):
        return False, None, "bboxes contains NaN/Inf/non-numeric"

    x, y, w = float(bb[0]), float(bb[1]), float(bb[2])
    if w <= 0:
        return False, None, "w <= 0"
    if x < 0 or y < 0:
        return False, None, "x < 0 or y < 0"

    h = derive_h(w, target_ratio)
    if not math.isfinite(h) or h <= 0:
        return False, None, "derived h is invalid"

    if meta:
        n_frames = meta.get("n_frames")
        if n_frames and f >= int(n_frames):
            return False, None, "frame >= n_frames (%d)" % int(n_frames)
        W, H = meta.get("width"), meta.get("height")
        if W and x + w > float(W) + 1e-6:
            return False, None, "x+w exceeds frame width (%s)" % W
        if H and y + h > float(H) + 1e-6:
            return False, None, "y+h exceeds frame height (%s); h=%.2f" % (H, h)

    return True, [x, y, w, h], ""


def score_video(preds, gt_boxes, target_ratio, meta=None):
    """Score one video.

    preds    : list of {"frame": int, "bboxes": [x, y, w]} in submission order
    gt_boxes : {frame -> [x, y, w, h]} ground-truth crop regions
    Returns a dict with F, P, R, S, n_pred, n_gt and per-reason reject counts.
    """
    n_gt = len(gt_boxes)
    n_pred = len(preds)

    if n_gt == 0 and n_pred == 0:
        return {"F": 1.0, "P": None, "R": None, "S": 0.0,
                "n_pred": 0, "n_gt": 0, "n_scored": 0, "rejects": {}}
    if n_gt == 0 or n_pred == 0:
        return {"F": 0.0, "P": None, "R": None, "S": 0.0,
                "n_pred": n_pred, "n_gt": n_gt, "n_scored": 0, "rejects": {}}

    S = 0.0
    n_scored = 0
    seen = set()
    rejects = {}

    def _rej(reason):
        rejects[reason] = rejects.get(reason, 0) + 1

    # 按提交顺序遍历：同一帧只有第一条有效预测得分，其余计为误报。
    for p in preds:
        ok, box, reason = check_prediction(p, target_ratio, meta)
        if not ok:
            _rej(reason)
            continue
        f = int(p["frame"])
        if f in seen:
            _rej("duplicate frame")
            continue
        seen.add(f)
        gt = gt_boxes.get(f)
        if gt is None:
            _rej("frame not in ground truth")
            continue
        S += iou_xywh(box, gt)
        n_scored += 1

    P = S / n_pred
    R = S / n_gt
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return {"F": F, "P": P, "R": R, "S": S,
            "n_pred": n_pred, "n_gt": n_gt, "n_scored": n_scored,
            "rejects": rejects}


def score_dataset(submission, ground_truth, ratios, metas=None):
    """Score every video in `ground_truth`.

    submission   : {video_id -> [pred, ...]}   missing video => N_pred = 0
    ground_truth : {video_id -> {frame -> [x,y,w,h]}}
    ratios       : {video_id -> [tw, th]}
    metas        : {video_id -> {n_frames, width, height}} or None
    Returns (final_score, {video_id -> per-video result}).
    """
    metas = metas or {}
    per_video = {}
    for vid, gt in ground_truth.items():
        preds = submission.get(vid, [])
        ratio = ratios.get(vid, [16, 9])
        per_video[vid] = score_video(preds, gt, ratio, metas.get(vid))
    if not per_video:
        return 0.0, {}
    final = sum(r["F"] for r in per_video.values()) / len(per_video) * 100.0
    return final, per_video
