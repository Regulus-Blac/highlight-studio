#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subject-following crop boxes / 主体跟随构图.

Every submission so far puts the max-size crop at the frame center. That crop
has exactly one degree of freedom (see highlight/geometry.py), and on the val
clips the center crop averages only ~0.62 IoU against the GT crop -- so where
the crop sits costs as much as which frames are kept. This script keeps the
frames of an existing submission and only rewrites their boxes:

  1. sample every `stride`-th selected frame and run an object detector on it;
  2. link detections into tracks (greedy IoU) and pick the main subject by
     sum(area^area_pow * conf * centrality); tracks scoring at least
     `union` x the best one are merged into a group box;
  3. subject point on the free axis -> crop origin u in [0, 1]. Frames with no
     subject fall back to the best single detection, then the motion centroid,
     then interpolation between neighbours;
  4. median + zero-phase EMA smoothing, a dead zone that freezes near-static
     paths, optional shrink toward the center; interpolate u to every frame and
     place the box with geometry.place_1d, which always stays inside the frame.

Detections are cached per video, so `eval_spatial.py` can grid-search every
step after the detector without running it again.
只重写已选帧的框；检测结果按视频缓存，标定时只重跑检测之后的步骤。

    python3 tools/reframe.py --pred outputs/submit_test_scores_ep2.jsonl \
        --video-dir baseline/video --params outputs/reframe_params.json \
        --out outputs/submit_test_reframe.jsonl
"""

import os
import sys
import json
import time
import hashlib
import argparse
import collections

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from highlight.geometry import (center_crop, contains, dof_range, free_axis,  # noqa: E402
                                place_1d, safe_crop_size)

DEFAULT_MODEL = "yolo11s.pt"
# Everything above this is cached; `conf` filters later, so tuning it is free.
DET_CONF_MIN = 0.10
# Motion profile resolution along each axis.
N_BINS = 32
# Travel below this fraction of the crop extent is not worth moving.
MIN_TRAVEL_FRAC = 0.02
BATCH = 16
PERSON = 0  # COCO class id

DEFAULT_PARAMS = {
    "stride": 4,            # detect on every 4th selected frame
    "classes": "all",       # "person" = COCO person only
    "conf": 0.25,
    "link_iou": 0.3,
    "area_pow": 1.0,
    "center_w": 0.5,        # centrality weight 1 - center_w * |c - 0.5| * 2
    "union": 1.0,           # 1.0 = main track only; 0.5 = + tracks >= half its score
    "frame_fallback": True,  # main subject absent -> best detection of that frame
    "anchor_y": 0.5,        # point on the subject box along y (0 top, 1 bottom)
    "anchor_y_animal": None,  # y anchor when every chosen box is an animal (COCO 14-23); None = anchor_y
    "y_skip_classes": None,   # y axis: COCO ids never taken as the subject (e.g. furniture behind a pet)
    "y_cut_person": None,     # y axis: weight factor for a person cut by the top edge (only legs / hands)
    "y_selector": None,       # y axis: JSON of a learned subject selector (selector_features weights) replacing
                              # the area x conf x centrality weight; per frame a softmax over the detections
    "y_pet_owner": None,      # y axis: with a pet in >= 20 % of a shot's frames, a person touching the top edge
                              # (the owner's legs / arms) is never the subject
    "y_head_det": None,       # y axis: weights of a head detector (scripts/plan_c/yhead_det_run.sh); the head found
                              # inside the chosen subject box moves the strip as in the head-aware labels: toward
                              # the head center as the head grows from 10 % to 20 % of the frame height, and just
                              # far enough to contain a head whose center would fall outside the strip
    "motion": True,         # no detection -> motion centroid
    "motion_min": 3.0,      # mean abs gray difference needed to trust motion
    "median": 3,            # median filter width over sampled points (1 = off)
    "ema": 0.5,             # zero-phase EMA factor (1.0 = off)
    "dead": 0.0,            # freeze the path if it moves less than dead * crop extent
    "shrink": 0.0,          # blend u toward 0.5
    "shrink_y": None,       # y-axis (portrait source) override; None = shrink
    "head": False,          # frame on the chosen person's head (pose model): true / "x" / "y"
    "head_pos_x": 0.5,      # where the head sits across the crop (0 left, 1 right)
    "head_pos_y": 0.4,      # where the head sits down the crop (0 top, 1 bottom)
    "shots": False,         # split at hard cuts: fill + smooth each shot on its own
    "shot_subject": True,   # with shots: pick the main subject per shot, not per video
    "cut_k": 4.0,           # cut = sampled-frame motion > max(cut_min, cut_k x video median)
    "cut_min": 20.0,
    "shot_min": 2,          # shots shorter than this (sampled frames) merge into the previous one
}
DEFAULT_POSE_MODEL = "yolo11s-pose.pt"
HEAD_KPTS = (0, 1, 2, 3, 4)  # COCO nose, eyes, ears
KPT_CONF = 0.5
HEAD_MATCH_IOU = 0.5
HEAD_DET_CONF = 0.3          # y_head_det: weakest head detection that moves the strip
_HEAD_DETECTORS = {}


# --------------------------------- detector ---------------------------------
def default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detector_tag(model, imgsz):
    """Cache identity of a detector: weights file + input size."""
    return "%s@%d" % (os.path.basename(model), imgsz)


class Detector:
    """Ultralytics YOLO wrapper -> per frame (n, 6) [cls, conf, x1, y1, x2, y2]."""

    def __init__(self, model=DEFAULT_MODEL, device=None, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.device = device or default_device()
        self.imgsz = imgsz
        self.tag = detector_tag(model, imgsz)

    def __call__(self, frames):
        res = self.model.predict(frames, conf=DET_CONF_MIN, imgsz=self.imgsz,
                                 device=self.device, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), dtype=np.float32))
                continue
            out.append(np.concatenate([b.cls.cpu().numpy()[:, None],
                                       b.conf.cpu().numpy()[:, None],
                                       b.xyxy.cpu().numpy()], axis=1).astype(np.float32))
        return out


class PoseDetector:
    """YOLO pose -> per frame (n, 8) [x1, y1, x2, y2, conf, head_x, head_y, head_conf].

    The head point is the mean of the visible face keypoints (nose, eyes,
    ears); head_conf = 0 when none is visible. 头部点 = 可见面部关键点均值。
    """

    def __init__(self, model=DEFAULT_POSE_MODEL, device=None, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.device = device or default_device()
        self.imgsz = imgsz
        self.tag = "pose:" + detector_tag(model, imgsz)

    def __call__(self, frames):
        res = self.model.predict(frames, conf=DET_CONF_MIN, imgsz=self.imgsz,
                                 device=self.device, verbose=False)
        out = []
        for r in res:
            b, kp = r.boxes, r.keypoints
            if b is None or len(b) == 0 or kp is None:
                out.append(np.zeros((0, 8), dtype=np.float32))
                continue
            K = kp.data.cpu().numpy()[:, list(HEAD_KPTS), :]      # n x 5 x (x, y, conf)
            vis = K[:, :, 2] >= KPT_CONF
            cnt = vis.sum(axis=1)
            hx = np.where(cnt > 0, (K[:, :, 0] * vis).sum(axis=1) / np.maximum(cnt, 1), 0.0)
            hy = np.where(cnt > 0, (K[:, :, 1] * vis).sum(axis=1) / np.maximum(cnt, 1), 0.0)
            hc = np.where(cnt > 0, (K[:, :, 2] * vis).sum(axis=1) / np.maximum(cnt, 1), 0.0)
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                       hx[:, None], hy[:, None], hc[:, None]],
                                      axis=1).astype(np.float32))
        return out


# ------------------------------ frame sampling ------------------------------
def sample_frames(frames, stride):
    """Every `stride`-th of the sorted selected frames, plus the last one."""
    fs = sorted(set(int(f) for f in frames))
    if not fs:
        return []
    out = fs[::max(1, int(stride))]
    if out[-1] != fs[-1]:
        out.append(fs[-1])
    return out


def iter_frames(path, wanted):
    """Yield (frame id, BGR) for `wanted` ids, decoding sequentially.

    No seeking: CAP_PROP_POS_FRAMES lands on the wrong frame for some codecs,
    and a grab() is cheap next to the detector. 顺序解码，不 seek。
    """
    want = set(wanted)
    last = max(want)
    cap = cv2.VideoCapture(path)
    try:
        idx = 0
        while idx <= last:
            if not cap.grab():
                break
            if idx in want:
                ok, fr = cap.retrieve()
                if ok:
                    yield idx, fr
            idx += 1
    finally:
        cap.release()


def _motion_profiles(gray, prev):
    """Column / row profiles of |frame difference| and its mean."""
    if prev is None:
        return [0.0] * N_BINS, [0.0] * N_BINS, 0.0
    d = cv2.absdiff(gray, prev).astype(np.float32)
    mx = d.mean(axis=0).reshape(N_BINS, -1).mean(axis=1)
    my = d.mean(axis=1).reshape(N_BINS, -1).mean(axis=1)
    return ([round(float(v), 2) for v in mx], [round(float(v), 2) for v in my],
            round(float(d.mean()), 3))


def video_size(path):
    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def cache_file(path, sampled, tag, cache_dir):
    """Where collect() keeps the result for (file, sampled frames, detector)."""
    key = hashlib.md5(json.dumps([os.path.abspath(path), os.path.getsize(path),
                                  sampled, tag]).encode("utf-8")).hexdigest()[:20]
    return os.path.join(cache_dir, key + ".json")


def load_cached(path, frames, stride, tag, cache_dir):
    """collect()'s cached result, or None -- never loads a detector."""
    cpath = cache_file(path, sample_frames(frames, stride), tag, cache_dir)
    if not os.path.exists(cpath):
        return None
    with open(cpath, "r", encoding="utf-8") as f:
        return json.load(f)


def collect(path, frames, stride, detector, cache_dir=None):
    """Detections + motion profiles on the sampled frames of one video.

    Cached under `cache_dir` by (file, sampled frames, detector), so the same
    selection is only ever run through the detector once.
    """
    sampled = sample_frames(frames, stride)
    cpath = cache_file(path, sampled, detector.tag, cache_dir) if cache_dir else None
    if cpath and os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)

    W, H = video_size(path)
    got, dets, mx, my, md = [], [], [], [], []
    batch, prev = [], None

    def flush():
        for D in detector([fr for fr in batch]):
            dets.append([[round(float(v), 1) if j > 1 else round(float(v), 3)
                          for j, v in enumerate(row)] for row in D])
        batch.clear()

    for idx, fr in iter_frames(path, sampled):
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (N_BINS * 4, N_BINS * 4),
                       interpolation=cv2.INTER_AREA)
        px, py, m = _motion_profiles(g, prev)
        prev = g
        got.append(idx)
        mx.append(px)
        my.append(py)
        md.append(m)
        batch.append(fr)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()

    out = {"path": path, "W": W, "H": H, "frames": got, "dets": dets,
           "mx": mx, "my": my, "mean_diff": md, "detector": detector.tag}
    if cpath:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def collect_pose(path, frames, stride, pose, cache_dir=None):
    """Pose persons on the same sampled frames as collect(), cached apart from
    the detections so adding the pose model never invalidates them."""
    sampled = sample_frames(frames, stride)
    cpath = cache_file(path, sampled, pose.tag, cache_dir) if cache_dir else None
    if cpath and os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)
    got, persons, batch = [], [], []

    def flush():
        for P in pose(batch):
            persons.append([[round(float(v), 1) if j not in (4, 7) else round(float(v), 3)
                             for j, v in enumerate(row)] for row in P])
        batch.clear()

    for idx, fr in iter_frames(path, sampled):
        got.append(idx)
        batch.append(fr)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    out = {"path": path, "frames": got, "persons": persons, "detector": pose.tag}
    if cpath:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def attach_pose(cache, pose_cache):
    """cache + per-sampled-frame pose persons, when both cover the same frames."""
    if pose_cache is None or pose_cache.get("frames") != cache.get("frames"):
        return cache
    return dict(cache, pose=pose_cache["persons"])


# ------------------------------ subject path --------------------------------
def filter_dets(dets, params):
    out = []
    for D in dets:
        D = np.asarray(D, dtype=np.float64).reshape(-1, 6)
        keep = D[:, 1] >= params["conf"]
        if params["classes"] == "person":
            keep &= D[:, 0] == PERSON
        out.append(D[keep])
    return out


def _iou_matrix(A, B):
    """IoU between xyxy rows of A (n, 4) and B (m, 4)."""
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda Z: np.clip(Z[:, 2] - Z[:, 0], 0, None) * np.clip(Z[:, 3] - Z[:, 1], 0, None)
    union = area(A)[:, None] + area(B)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def link_tracks(dets, iou_thr, max_gap=2):
    """Greedy same-class IoU linking across sampled frames -> track id per row.

    Deliberately simple (no motion model): its only job is to keep one
    subject's identity long enough to score it as a whole.
    """
    last_box, last_k, last_cls, ids = [], [], [], []
    for k, D in enumerate(dets):
        tid = np.full(len(D), -1, dtype=np.int64)
        live = [t for t in range(len(last_box)) if k - last_k[t] <= max_gap + 1]
        if len(D) and live:
            M = _iou_matrix(D[:, 2:6], np.array([last_box[t] for t in live]))
            M = np.where(D[:, 0][:, None] == np.array([last_cls[t] for t in live])[None, :], M, 0.0)
            used = set()
            for flat in np.argsort(-M, axis=None):
                i, j = np.unravel_index(flat, M.shape)
                if M[i, j] < iou_thr:
                    break
                if tid[i] >= 0 or j in used:
                    continue
                tid[i] = live[j]
                used.add(j)
        for i in range(len(D)):
            if tid[i] < 0:
                last_box.append(D[i, 2:6])
                last_k.append(k)
                last_cls.append(D[i, 0])
                tid[i] = len(last_box) - 1
            else:
                last_box[tid[i]] = D[i, 2:6]
                last_k[tid[i]] = k
        ids.append(tid)
    return ids


def shot_segments(cache, params):
    """[(start, end)) sampled-frame ranges split at hard cuts; one range when
    `shots` is off. A cut is a sampled frame whose mean |difference| to the
    previous sample stands far above the video's usual motion.
    按运动突变切镜头；关闭时整段视频一个区间。"""
    n = len(cache["frames"])
    md = np.asarray(cache["mean_diff"], dtype=np.float64)
    if not params.get("shots") or n < 3 or len(md) != n:
        return [(0, n)]
    thr = max(float(params["cut_min"]), float(params["cut_k"]) * float(np.median(md[1:])))
    m = int(params["shot_min"])
    cuts, last = [], 0
    for k in range(1, n):
        if md[k] > thr and k - last >= m:
            cuts.append(k)
            last = k
    if cuts and n - cuts[-1] < m:
        cuts.pop()
    bounds = [0] + cuts + [n]
    return list(zip(bounds[:-1], bounds[1:]))


def fill_gaps(u):
    """NaN -> linear interpolation between known points (edges held)."""
    ok = np.isfinite(u)
    if not ok.any():
        return np.full(len(u), 0.5)
    if ok.all():
        return u
    i = np.arange(len(u))
    return np.interp(i, i[ok], u[ok])


def smooth_path(u, params, R, c):
    """Median -> zero-phase EMA -> dead zone -> shrink toward the center."""
    u = np.asarray(u, dtype=np.float64).copy()
    m = int(params["median"])
    if m > 1 and len(u) > 2:
        half = m // 2
        pad = np.pad(u, half, mode="edge")
        u = np.array([np.median(pad[i:i + 2 * half + 1]) for i in range(len(u))])
    a = float(params["ema"])
    if a < 1.0 and len(u) > 1:
        for rng in (range(1, len(u)), range(len(u) - 2, -1, -1)):
            step = 1 if rng.step > 0 else -1
            for i in rng:
                u[i] = a * u[i] + (1.0 - a) * u[i - step]
    if params["dead"] > 0 and len(u) and (u.max() - u.min()) * R < params["dead"] * c:
        u[:] = np.median(u)
    s = float(params["shrink"])
    if s > 0:
        u = (1.0 - s) * u + s * 0.5
    return np.clip(u, 0.0, 1.0)


def geometry(W, H, ratio):
    """-> (axis, travel R, crop extent c, frame extent L) or None if static."""
    tw, th = float(ratio[0]), float(ratio[1])
    axis = free_axis(W, H, tw, th)
    if axis is None:
        return None
    w, h = safe_crop_size(W, H, tw, th)
    c = w if axis == "x" else h
    R = dof_range(W, H, tw, th)
    if R < MIN_TRAVEL_FRAC * c:
        return None
    return axis, R, c, (W if axis == "x" else H)


def head_point(rows, persons, col):
    """Mean head coordinate (column `col` of a pose row) of the chosen person
    detections, matched to pose persons by box IoU; None if no head is seen."""
    P = np.asarray(persons, dtype=np.float64).reshape(-1, 8)
    A = np.asarray([r for r in rows if int(r[0]) == PERSON], dtype=np.float64).reshape(-1, 6)
    if not len(P) or not len(A):
        return None
    iou = _iou_matrix(A[:, 2:6], P[:, 0:4])
    heads = []
    for i in range(len(A)):
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= HEAD_MATCH_IOU and P[j, 7] > 0:
            heads.append(P[j, col])
    return float(np.mean(heads)) if heads else None


def head_detector(model):
    """The y_head_det detector, loaded once per weights file."""
    if model not in _HEAD_DETECTORS:
        _HEAD_DETECTORS[model] = Detector(model)
    return _HEAD_DETECTORS[model]


def head_in(rows, heads):
    """(y1, y2) of the most confident detected head whose center lies inside the chosen boxes, else None."""
    A = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    x1, y1, x2, y2 = A[:, 2].min(), A[:, 3].min(), A[:, 4].max(), A[:, 5].max()
    best = None
    for h in heads:
        cx, cy = (h[2] + h[4]) / 2.0, (h[3] + h[5]) / 2.0
        if h[1] >= HEAD_DET_CONF and x1 <= cx <= x2 and y1 <= cy <= y2 and (best is None or h[1] > best[1]):
            best = h
    return None if best is None else (float(best[3]), float(best[5]))


def selector_features(D, W, H):
    """Per-detection features of the learned y-axis subject selector; D rows [cls, conf, x1, y1, x2, y2]."""
    D = np.asarray(D, dtype=np.float64).reshape(-1, 6)
    cls = D[:, 0].astype(int)
    person, animal = cls == PERSON, (cls >= 14) & (cls <= 23)
    bw, bh = (D[:, 4] - D[:, 2]) / W, (D[:, 5] - D[:, 3]) / H
    area = bw * bh
    cx, cy = (D[:, 2] + D[:, 4]) / 2.0 / W, (D[:, 3] + D[:, 5]) / 2.0 / H
    top, bottom = D[:, 3] <= 0.02 * H, D[:, 5] >= 0.98 * H
    pet = float((animal & (D[:, 1] >= 0.4)).any())
    rel = area / max(float(area.max()) if len(D) else 1.0, 1e-9)
    return np.stack([person, animal, ~(person | animal), D[:, 1], np.sqrt(area), rel, rel >= 1.0, cx, cy,
                     np.abs(cx - 0.5), bw, bh, top, bottom, person * top * pet, person * pet], 1).astype(np.float64)


def selector_weights(D, W, H, w):
    """Softmax over one frame's detections of the selector scores (w: feature weights)."""
    if not len(D):
        return np.zeros(0)
    z = selector_features(D, W, H) @ np.asarray(w, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def subject_points(cache, axis, params, linked=None, c=None, segments=None):
    """Raw subject point (pixels on the free axis) per sampled frame, NaN = none.

    `linked` = (filtered dets, track ids) to reuse across parameter sweeps;
    `c` = crop extent on the axis, needed only for head framing;
    `segments` = shot ranges from shot_segments(): the main subject is picked
    inside each one (with shot_subject), else once for the whole video.
    """
    W, H = cache["W"], cache["H"]
    L = W if axis == "x" else H
    dets, ids = linked or (None, None)
    if dets is None:
        dets = filter_dets(cache["dets"], params)
        if axis == "y" and params.get("y_skip_classes"):
            skip = np.asarray(params["y_skip_classes"], dtype=np.float64)
            dets = [D[~np.isin(D[:, 0], skip)] for D in dets]
        ids = link_tracks(dets, params["link_iou"])
    a0, a1 = (2, 4) if axis == "x" else (3, 5)
    cw, ap = params["center_w"], params["area_pow"]
    cut = params.get("y_cut_person") if axis == "y" else None

    def weight(row):
        area = max(0.0, row[4] - row[2]) * max(0.0, row[5] - row[3]) / float(W * H)
        cen = (row[a0] + row[a1]) / 2.0 / L
        w = (area ** ap) * row[1] * max(0.0, 1.0 - cw * abs(cen - 0.5) * 2.0)
        if cut is not None and int(row[0]) == PERSON and row[3] <= 0.02 * H and row[5] - row[3] < 0.6 * H:
            w *= float(cut)
        return w

    owner = axis == "y" and params.get("y_pet_owner")
    sw = None
    if axis == "y" and params.get("y_selector"):
        sel = json.load(open(params["y_selector"]))["w"]
        sw = [selector_weights(D, W, H, sel) for D in dets]

    def choose(a, b):
        pet = owner and np.mean([any(14 <= int(r[0]) <= 23 and r[1] >= 0.4 for r in D) for D in dets[a:b]]) >= 0.2
        score = collections.defaultdict(float)
        for k, (D, tid) in enumerate(zip(dets[a:b], ids[a:b]), a):
            for i, (row, t) in enumerate(zip(D, tid)):
                if pet and int(row[0]) == PERSON and row[3] <= 0.02 * H:
                    continue
                score[int(t)] += weight(row) if sw is None else sw[k][i]
        if not score:
            return set()
        best = max(score.values())
        return {t for t, s in score.items() if s >= params["union"] * best - 1e-12}

    if segments and len(segments) > 1 and params.get("shot_subject", True):
        chosen_at = [None] * len(dets)
        for a, b in segments:
            ch = choose(a, b)
            for k in range(a, b):
                chosen_at[k] = ch
    else:
        chosen_at = [choose(0, len(dets))] * len(dets)

    anchor = 0.5 if axis == "x" else float(params["anchor_y"])
    # head: true = both axes, "x" / "y" = that axis only (clean A/B per axis)
    use_head = params.get("head") is True or params.get("head") == axis
    pose = cache.get("pose") if use_head and c else None
    if pose is not None and len(pose) != len(dets):
        pose = None
    head_pos = float(params["head_pos_x"] if axis == "x" else params["head_pos_y"])
    hi_col = 5 if axis == "x" else 6  # head_x / head_y in a pose row
    heads = cache.get("heads") if axis == "y" and c else None
    pts = np.full(len(dets), np.nan)
    for k, (D, tid) in enumerate(zip(dets, ids)):
        rows = [row for row, t in zip(D, tid) if int(t) in chosen_at[k]]
        if not rows and params["frame_fallback"] and len(D):
            rows = [max(D, key=weight)]
        head = head_point(rows, pose[k], hi_col) if pose is not None and rows else None
        if head is not None:
            # crop center such that the head lands at head_pos of the crop
            pts[k] = head + (0.5 - head_pos) * c
        elif rows:
            A = np.asarray(rows)
            lo, hi = A[:, a0].min(), A[:, a1].max()
            # y axis: animal subjects (COCO 14-23) can take their own anchor; people keep anchor_y
            animal = (axis == "y" and params.get("anchor_y_animal") is not None
                      and all(14 <= int(r[0]) <= 23 for r in rows))
            pts[k] = lo + (float(params["anchor_y_animal"]) if animal else anchor) * (hi - lo)
            hb = head_in(rows, heads[k]) if heads is not None else None
            if hb is not None:
                w = min(max(((hb[1] - hb[0]) / H - 0.1) / 0.1, 0.0), 1.0)
                p = w * (hb[0] + hb[1]) / 2.0 + (1.0 - w) * pts[k]
                if abs((hb[0] + hb[1]) / 2.0 - p) > c / 2.0:
                    p = min(max(p, hb[1] - c / 2.0), hb[0] + c / 2.0)
                pts[k] = p
        elif params["motion"] and cache["mean_diff"][k] >= params["motion_min"]:
            prof = np.asarray(cache["mx" if axis == "x" else "my"][k], dtype=np.float64)
            if prof.sum() > 0:
                centers = (np.arange(len(prof)) + 0.5) / len(prof) * L
                pts[k] = float((prof * centers).sum() / prof.sum())
    return pts


def subject_u(cache, ratio, params, linked=None):
    """-> (sampled frame ids, smoothed crop origin u) or None when static."""
    g = geometry(cache["W"], cache["H"], ratio)
    if g is None or not cache["frames"]:
        return None
    axis, R, c, _ = g
    segs = shot_segments(cache, params)
    pts = subject_points(cache, axis, params, linked, c, segs)
    raw = np.clip((pts - c / 2.0) / R, 0.0, 1.0)
    # The labeled clips only calibrate x; y is tuned on the leaderboard, so it
    # may carry its own shrink. 竖屏可单独设 shrink_y。
    if axis == "y" and params.get("shrink_y") is not None:
        params = dict(params, shrink=params["shrink_y"])
    frames = np.asarray(cache["frames"], dtype=np.float64)
    if len(segs) == 1:
        return frames, smooth_path(fill_gaps(raw), params, R, c)
    # Per shot: gaps are filled and the path smoothed inside the shot, so the
    # crop jumps at a cut instead of gliding across it. A shot with no subject
    # at all keeps the whole-video interpolation. 镜头内补位与平滑，切点处直接跳。
    whole = fill_gaps(raw)
    u = np.empty(len(raw))
    for a, b in segs:
        seg = raw[a:b]
        seg = fill_gaps(seg) if np.isfinite(seg).any() else whole[a:b]
        u[a:b] = smooth_path(seg, params, R, c)
    return frames, u


def place_boxes(frames, W, H, ratio, path=None):
    """{frame: [x, y, w]} along `path` = (sampled ids, u), or centered."""
    tw, th = float(ratio[0]), float(ratio[1])
    frames = [int(f) for f in frames]
    if path is None:
        box = list(center_crop(W, H, tw, th))
        return {f: box for f in frames}
    uf = np.interp(np.asarray(frames, dtype=np.float64), path[0], path[1])
    out = {}
    for f, v in zip(frames, uf):
        box = place_1d(float(v), W, H, tw, th)
        if not contains(box, W, H, tw, th):
            box = center_crop(W, H, tw, th)
        out[f] = [int(box[0]), int(box[1]), int(box[2])]
    return out


# ---------------------------------- driver ----------------------------------
def reframe_row(row, video_path, params, get_detector, cache_dir, stats, get_pose=None):
    preds = row.get("predictions") or []
    if not preds:
        stats["empty"] += 1
        return row
    W, H = video_size(video_path)
    if W <= 0 or H <= 0:
        stats["unreadable (kept)"] += 1
        return row
    ratio = row["targetRatioWH"]
    frames = [int(p["frame"]) for p in preds]
    path = None
    if geometry(W, H, ratio) is None:
        stats["no free axis"] += 1
    else:
        cache = collect(video_path, frames, params["stride"], get_detector(), cache_dir)
        if params.get("head") and get_pose is not None:
            cache = attach_pose(cache, collect_pose(video_path, frames, params["stride"],
                                                    get_pose(), cache_dir))
        if params.get("y_head_det") and geometry(W, H, ratio)[0] == "y":
            hd = collect(video_path, frames, params["stride"], head_detector(params["y_head_det"]), cache_dir)
            if hd["frames"] == cache["frames"]:
                cache = dict(cache, heads=hd["dets"])
        path = subject_u(cache, ratio, params)
        stats["reframed"] += 1
        if path is not None:
            stats["_offsets"].append(float(np.mean(np.abs(path[1] - 0.5))))
    boxes = place_boxes(frames, W, H, ratio, path)
    out = dict(row)
    out["predictions"] = [dict(p, bboxes=boxes[int(p["frame"])]) for p in preds]
    return out


def load_params(path=None, **overrides):
    params = dict(DEFAULT_PARAMS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        params.update({k: loaded[k] for k in DEFAULT_PARAMS if k in loaded})
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def main():
    ap = argparse.ArgumentParser(description="Rewrite submission boxes to follow the main subject")
    ap.add_argument("--pred", required=True, help="submission JSONL whose frames are kept")
    ap.add_argument("--video-dir", required=True, help="<video_id>.mp4 lives here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=None, help="JSON from eval_spatial.py --tune")
    ap.add_argument("--stride", type=int, default=None, help="override params stride")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights (auto-downloads)")
    ap.add_argument("--pose-model", default=DEFAULT_POSE_MODEL,
                    help="YOLO pose weights, used only when params head=true")
    ap.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()

    params = load_params(args.params, stride=args.stride)
    print("params: %s" % params)
    det = []

    def get_detector():
        if not det:
            det.append(Detector(args.model, args.device, args.imgsz))
            print("detector: %s on %s" % (det[0].tag, det[0].device))
        return det[0]

    pose = []

    def get_pose():
        if not pose:
            pose.append(PoseDetector(args.pose_model, args.device, args.imgsz))
            print("pose: %s on %s" % (pose[0].tag, pose[0].device))
        return pose[0]

    with open(args.pred, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    stats = collections.Counter()
    stats["_offsets"] = []
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            vid = str(row["video_id"])
            vp = os.path.join(args.video_dir, vid + ".mp4")
            try:
                if not os.path.exists(vp):
                    stats["missing video (kept)"] += 1
                    out = row
                else:
                    out = reframe_row(row, vp, params, get_detector, args.cache_dir, stats,
                                      get_pose)
            except Exception as e:  # one bad video must not stop the batch
                print("[warn] %s: %s -- boxes kept" % (vid, e))
                stats["error (kept)"] += 1
                out = row
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(rows), time.time() - t0))
    offs = stats.pop("_offsets")
    print("reframed -> %s  %s" % (args.out, dict(stats)))
    if offs:
        print("mean |u - 0.5| over moved videos: %.3f (0 = centered)" % float(np.mean(offs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
