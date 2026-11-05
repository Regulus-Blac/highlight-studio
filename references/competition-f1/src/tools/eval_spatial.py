#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate and score the subject-following crop / 标定与评测主体跟随构图.

The rows of data/train.with_video.jsonl that carry `cropRois` are the only
spatial labels we have (9:16 crops on 534x300 landscape clips). This script:

  --analyze  describes the GT crops: where they sit (u = position_to_u, 0 =
             flush left, 0.5 = centered), how often they sit dead-center, and
             how they move -- constant runs, or keyframes + linear interpolation
             (the spacing between slope changes shows which);
  --tune     runs the detector once per clip (cached), then searches
             reframe.py's params on the train split by coordinate descent,
             maximizing the mean per-video IoU on GT frames, and reports val
             against the center crop with a paired bootstrap CI;
  --params   scores a params file on val only;
  --cv K     grouped K-fold over train + val (source-video groups): pooled
             held-out IoU vs center, a much tighter CI than 48 val clips.

Val is reported three ways: mean IoU on GT frames, the contest score with
perfect temporal (62.52 with center boxes), and -- with --probs and
--decode-params -- the contest score with the model's own frame selection
(38.94 with center boxes for ep2). Every step after the detector is numpy, so
the search is cheap once the cache exists.
检测只跑一次并缓存；参数搜索只在 train 上做，val 只用来报告。

    python3 tools/eval_spatial.py --analyze
    python3 tools/eval_spatial.py --tune --params-out outputs/reframe_params.json
    python3 tools/eval_spatial.py --cv 5 --params-out outputs/reframe_params_cv.json
    python3 tools/eval_spatial.py --params outputs/reframe_params.json \
        --probs outputs/scores_val_ep2.probs.jsonl \
        --decode-params outputs/decode_params_ep2.json
"""

import os
import sys
import json
import time
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import reframe as rf  # noqa: E402
from highlight.geometry import free_axis, position_to_u, safe_crop_size  # noqa: E402
from decode_scores import paired_ci, read_jsonl  # noqa: E402

# Coordinate-descent search space. anchor_y is absent on purpose: every
# labeled clip slides on x, so nothing here can calibrate the vertical anchor.
# 训练集只有 x 轴样本，anchor_y 无法在这里标定。
SEARCH = collections.OrderedDict([
    ("classes", ["all", "person"]),
    ("conf", [0.15, 0.25, 0.4]),
    ("area_pow", [0.5, 1.0, 1.5]),
    ("center_w", [0.0, 0.3, 0.6, 0.9]),
    ("union", [1.0, 0.7, 0.4]),
    ("frame_fallback", [True, False]),
    ("motion", [True, False]),
    ("median", [1, 3, 7]),
    ("ema", [1.0, 0.5, 0.25, 0.1]),
    ("dead", [0.0, 0.05, 0.15, 0.3]),
    ("shrink", [0.0, 0.15, 0.3, 0.5]),
])


# ---------------------------------- data -----------------------------------
def load_rows(gt_path, limit=0):
    """Rows with cropRois and a clip on disk -> list of dicts."""
    rows, skipped = [], collections.Counter()
    for r in read_jsonl(gt_path):
        if not r.get("cropRois"):
            skipped["no cropRois"] += 1
            continue
        vp = r.get("video_path") or ""
        if not os.path.exists(vp):
            skipped["no clip on disk"] += 1
            continue
        W, H = rf.video_size(vp)
        if W <= 0 or H <= 0:
            skipped["unreadable clip"] += 1
            continue
        gt = {int(f): [float(c) for c in b] for f, b in r["cropRois"]}
        frames = np.array(sorted(gt), dtype=np.int64)
        seg = set()
        for s in r.get("segments") or []:
            seg.update(range(int(s["start_frame"]), int(s["end_frame"]) + 1))
        seg = seg or set(gt)
        # YouTube id without the QVHighlights window suffix: windows cut from
        # one upload share scenes, so CV folds keep them together.
        src = (r.get("clip") or {}).get("source_vid") or r["video_id"]
        rows.append({
            "vid": r["video_id"], "split": r.get("dataset_split") or "train", "path": vp,
            "group": src.rsplit("_", 2)[0],
            "W": W, "H": H, "ratio": [float(v) for v in r["targetRatioWH"]], "gt": gt,
            "frames": frames, "G": np.array([gt[f] for f in frames], dtype=np.float64),
            "in_seg": np.isin(frames, np.array(sorted(seg), dtype=np.int64)), "n_seg": len(seg),
        })
        if limit and len(rows) >= limit:
            break
    return rows, skipped


def build_caches(rows, stride, args):
    """Detector pass over the GT frames of every row (cached on disk)."""
    tag = rf.detector_tag(args.model, args.imgsz)
    det, caches, t0 = [], {}, time.time()
    for i, r in enumerate(rows):
        frames = r["frames"].tolist()
        cached = rf.load_cached(r["path"], frames, stride, tag, args.cache_dir)
        if cached is not None:
            caches[r["vid"]] = cached
            continue
        if not det:  # a warm cache needs no detector at all
            det.append(rf.Detector(args.model, args.device, args.imgsz))
            print("detector: %s on %s" % (det[0].tag, det[0].device))
        caches[r["vid"]] = rf.collect(r["path"], frames, stride, det[0], args.cache_dir)
        if (i + 1) % 50 == 0:
            print("  detected %d/%d clips  %.0fs" % (i + 1, len(rows), time.time() - t0))
    return caches


def attach_poses(rows, caches, stride, args):
    """Pose persons on the same sampled frames (cached like the detections)."""
    tag = "pose:" + rf.detector_tag(args.pose_model, args.imgsz)
    pose, t0 = [], time.time()
    for i, r in enumerate(rows):
        frames = r["frames"].tolist()
        cpath = rf.cache_file(r["path"], rf.sample_frames(frames, stride), tag, args.cache_dir)
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8") as f:
                pc = json.load(f)
        else:
            if not pose:
                pose.append(rf.PoseDetector(args.pose_model, args.device, args.imgsz))
                print("pose: %s on %s" % (pose[0].tag, pose[0].device))
            pc = rf.collect_pose(r["path"], frames, stride, pose[0], args.cache_dir)
            if (i + 1) % 50 == 0:
                print("  pose %d/%d clips  %.0fs" % (i + 1, len(rows), time.time() - t0))
        caches[r["vid"]] = rf.attach_pose(caches[r["vid"]], pc)
    n = sum("pose" in c for c in caches.values())
    print("pose attached to %d/%d clips" % (n, len(rows)))


# --------------------------------- scoring ---------------------------------
def frame_ious(row, path):
    """IoU on every GT frame for crop path `path` (None = center crop).

    Vectorized form of reframe.place_boxes + the evaluator's IoU (h derived
    from w); `--selfcheck` asserts the two agree.
    """
    W, H = row["W"], row["H"]
    tw, th = row["ratio"]
    w, h = safe_crop_size(W, H, tw, th)
    rx, ry = max(0, W - w), max(0, H - h)
    n = len(row["frames"])
    if path is None:
        u = np.full(n, 0.5)
    else:
        u = np.clip(np.interp(row["frames"].astype(np.float64), path[0], path[1]), 0.0, 1.0)
    if free_axis(W, H, tw, th) == "x":
        x, y = np.round(u * rx), np.full(n, float(ry // 2))
    else:
        x, y = np.full(n, float(rx // 2)), np.round(u * ry)
    ph = w * th / tw
    G = row["G"]
    ix = np.clip(np.minimum(x + w, G[:, 0] + G[:, 2]) - np.maximum(x, G[:, 0]), 0, None)
    iy = np.clip(np.minimum(y + ph, G[:, 1] + G[:, 3]) - np.maximum(y, G[:, 1]), 0, None)
    inter = ix * iy
    union = w * ph + G[:, 2] * G[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


class Scorer:
    """Scores a params dict over rows, memoizing detection filtering + linking."""

    def __init__(self, caches):
        self.caches, self.memo = caches, {}

    def path(self, row, params):
        cache = self.caches[row["vid"]]
        key = (row["vid"], params["classes"], params["conf"], params["link_iou"])
        linked = self.memo.get(key)
        if linked is None:
            dets = rf.filter_dets(cache["dets"], params)
            linked = self.memo[key] = (dets, rf.link_tracks(dets, params["link_iou"]))
        return rf.subject_u(cache, row["ratio"], params, linked)

    def run(self, params, rows):
        """-> {vid: (mean IoU, perfect-temporal F, per-GT-frame IoU array)}."""
        out = {}
        for r in rows:
            ious = frame_ious(r, None if params is None else self.path(r, params))
            out[r["vid"]] = (float(ious.mean()), float(ious[r["in_seg"]].sum()) / r["n_seg"], ious)
        return out


def mean_of(res, i):
    return float(np.mean([v[i] for v in res.values()])) if res else float("nan")


def coordinate_descent(scorer, rows, start, passes=3, verbose=True):
    best = dict(start)
    best_m = mean_of(scorer.run(best, rows), 0)
    if verbose:
        print("  start: mean IoU %.4f" % best_m)
    for p in range(passes):
        improved = False
        for k, vals in SEARCH.items():
            for v in vals:
                if v == best[k]:
                    continue
                cand = dict(best, **{k: v})
                m = mean_of(scorer.run(cand, rows), 0)
                if m > best_m + 1e-6:
                    best, best_m, improved = cand, m, True
                    if verbose:
                        print("  pass %d: %s=%s -> mean IoU %.4f" % (p + 1, k, v, m))
        if not improved:
            break
    return best, best_m


def cross_validate(scorer, rows, start, k, seed=0):
    """Grouped K-fold over every labeled clip (train + val): tune on K-1 folds,
    score the held-out fold, then pool the held-out IoUs for one paired CI
    against the center crop. 48 val clips give a CI too wide to gate on; this
    uses all of them. 全部标注片段分组 K 折：留出分数汇总后与居中框做配对 CI。
    """
    groups = sorted({r["group"] for r in rows})
    np.random.default_rng(seed).shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    held, base = {}, scorer.run(None, rows)
    t0 = time.time()
    for i in range(k):
        fit = [r for r in rows if fold_of[r["group"]] != i]
        out = [r for r in rows if fold_of[r["group"]] == i]
        best, m = coordinate_descent(scorer, fit, start, verbose=False)
        res = scorer.run(best, out)
        held.update(res)
        changed = {p: v for p, v in best.items() if start.get(p) != v}
        print("  fold %d: fit %d -> held-out %d clips  IoU center %.2f -> %.2f   (%.0fs) %s"
              % (i, len(fit), len(out), mean_of({v: base[v] for v in res}, 0) * 100,
                 mean_of(res, 0) * 100, time.time() - t0, changed))
    vids = sorted(held)
    d, lo, hi = paired_ci([held[v][0] for v in vids], [base[v][0] for v in vids])
    print("-" * 76)
    print("  %d-fold CV held-out, %d clips in %d source groups" % (k, len(vids), len(groups)))
    print("  mean IoU: center %.2f -> CV %.2f   delta %+.2f  95%% CI [%+.2f, %+.2f]"
          % (mean_of(base, 0) * 100, mean_of(held, 0) * 100, d, lo, hi))
    won = sum(held[v][0] > base[v][0] + 1e-9 for v in vids)
    lost = sum(held[v][0] < base[v][0] - 1e-9 for v in vids)
    print("  per clip vs center: better %d, worse %d, tied %d" % (won, lost, len(vids) - won - lost))
    for split in ("train", "val"):
        vs = [r["vid"] for r in rows if r["split"] == split]
        if vs:
            print("  %-5s (%3d): center %.2f -> CV %.2f"
                  % (split, len(vs), np.mean([base[v][0] for v in vs]) * 100,
                     np.mean([held[v][0] for v in vs]) * 100))
    return held


def report_val(scorer, val, named, args):
    """Print center vs each params set on val, paired CI against center."""
    base = scorer.run(None, val)
    vids = sorted(base)
    print("  %d val clips" % len(vids))
    print("  %-10s mean IoU %6.2f | contest (perfect temporal) %6.2f"
          % ("center", mean_of(base, 0) * 100, mean_of(base, 1) * 100))
    results = collections.OrderedDict()
    for name, params in named:
        res = results[name] = scorer.run(params, val)
        d0 = paired_ci([res[v][0] for v in vids], [base[v][0] for v in vids])
        d1 = paired_ci([res[v][1] for v in vids], [base[v][1] for v in vids])
        print("  %-10s mean IoU %6.2f (%+.2f [%+.2f, %+.2f]) | contest %6.2f (%+.2f [%+.2f, %+.2f])"
              % (name, mean_of(res, 0) * 100, d0[0], d0[1], d0[2],
                 mean_of(res, 1) * 100, d1[0], d1[1], d1[2]))
        won = sum(res[v][0] > base[v][0] + 1e-9 for v in vids)
        lost = sum(res[v][0] < base[v][0] - 1e-9 for v in vids)
        print("  %-10s per clip vs center: better %d, worse %d" % ("", won, lost))
    if args.probs:
        report_with_selection(val, base, results, args)
    return results


def report_with_selection(val, base, results, args):
    """Contest score when the temporal model picks the frames (not the GT)."""
    import decode_scores as ds
    recs = {r["video_id"]: r for r in read_jsonl(args.probs)}
    dparams = dict(ds.DEFAULT_PARAMS)
    if args.decode_params:
        with open(args.decode_params, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        dparams.update({k: loaded[k] for k in ds.DEFAULT_PARAMS if k in loaded})
    videos = ds.load_val_videos(args.gt, recs)
    rows = {r["vid"]: r for r in val}
    print("-" * 76)
    print("  with the model's frame selection (%s):" % os.path.basename(args.probs))
    for name, res in [("center", base)] + list(results.items()):
        fs = []
        for vid, (_, _, ious) in res.items():
            v = videos.get(vid)
            if v is None:
                continue
            sel = ds.select_for(v, recs.get(vid), dparams)
            arr = np.zeros(v["n"])
            fr = rows[vid]["frames"]
            ok = (fr >= 0) & (fr < v["n"])
            arr[fr[ok]] = ious[ok]
            fs.append(2.0 * arr[sel].sum() / max(sel.sum() + v["n_gt"], 1))
        print("  %-10s contest %6.2f over %d clips" % (name, np.mean(fs) * 100, len(fs)))


# --------------------------------- analysis --------------------------------
def analyze(rows, scorer=None):
    """How the GT crops sit and move. GT 构图框的位置与运动方式。"""
    us, change, knots, n_pos, cover = [], [], collections.Counter(), [], []
    for r in rows:
        tw, th = r["ratio"]
        u = [position_to_u(b, r["W"], r["H"], tw, th) for b in r["G"]]
        us.extend(v for v in u if v is not None)
        axis = 0 if r["W"] / float(r["H"]) > tw / th else 1
        pos = r["G"][:, axis]
        n_pos.append(len(np.unique(np.round(pos))))
        if len(pos) > 1:
            change.append(float(np.mean(np.abs(np.diff(pos)) >= 1.0)))
        if len(pos) > 2:
            kn = np.flatnonzero(np.abs(np.diff(pos, 2)) > 1.0) + 1
            knots.update(np.diff(r["frames"][kn]).tolist())
        cover.append(float(r["in_seg"].sum()) / r["n_seg"])
    us = np.array(us)
    print("=" * 76)
    print("  %d clips, %d GT boxes" % (len(rows), len(us)))
    print("  u (0 = flush left/top, 0.5 = centered): mean %.3f  std %.3f" % (us.mean(), us.std()))
    print("  dead-center |u-0.5|<0.01: %.1f%%   at an edge (u<0.01 or >0.99): %.1f%%"
          % (100 * np.mean(np.abs(us - 0.5) < 0.01), 100 * np.mean((us < 0.01) | (us > 0.99))))
    hist, _ = np.histogram(us, bins=10, range=(0, 1))
    print("  u histogram (10 bins, %%): %s" % np.round(100.0 * hist / max(len(us), 1), 1).tolist())
    print("  distinct positions per clip  min/med/max: %d / %d / %d"
          % (min(n_pos), int(np.median(n_pos)), max(n_pos)))
    if change:
        print("  frames where the crop moves: %.1f%% (median over clips)"
              % (100 * float(np.median(change))))
    if knots:
        print("  spacing between slope changes (frames), most common: %s" % knots.most_common(6))
    print("  GT boxes cover %.1f%% of segment frames (mean over clips)"
          % (100 * float(np.mean(cover))))
    if scorer is not None:
        base = scorer.run(None, rows)
        res = scorer.run(rf.load_params(), rows)
        print("  mean IoU on these clips: center %.2f, reframe defaults %.2f"
              % (mean_of(base, 0) * 100, mean_of(res, 0) * 100))
    print("=" * 76)


def selfcheck(rows, scorer):
    """Vectorized IoU == reframe.place_boxes + highlight_metric IoU."""
    from highlight_metric import derive_h, iou_xywh
    worst = 0.0
    for r in rows[:20]:
        for params in (None, rf.load_params()):
            path = None if params is None else scorer.path(r, params)
            fast = frame_ious(r, path)
            boxes = rf.place_boxes(r["frames"].tolist(), r["W"], r["H"], r["ratio"], path)
            ref = [iou_xywh([b[0], b[1], b[2], derive_h(b[2], r["ratio"])], r["gt"][f])
                   for f, b in ((int(f), boxes[int(f)]) for f in r["frames"])]
            worst = max(worst, float(np.max(np.abs(fast - np.array(ref)))))
    ok = worst < 1e-9
    print("  [%s] vectorized IoU == place_boxes + iou_xywh (max diff %.2e)"
          % ("PASS" if ok else "FAIL", worst))
    return 0 if ok else 1


def save_params(best, tuned_on, args):
    os.makedirs(os.path.dirname(os.path.abspath(args.params_out)), exist_ok=True)
    with open(args.params_out, "w", encoding="utf-8") as f:
        json.dump(dict(best, tuned_on=tuned_on, detector=rf.detector_tag(args.model, args.imgsz)),
                  f, indent=2, ensure_ascii=False)
    print("  params -> %s" % args.params_out)


def main():
    ap = argparse.ArgumentParser(description="Calibrate / score the subject-following crop")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--analyze", action="store_true", help="GT crop statistics")
    ap.add_argument("--tune", action="store_true", help="search params on train, report val")
    ap.add_argument("--params", default=None, help="params JSON to score on val")
    ap.add_argument("--params-out", default=None)
    ap.add_argument("--selfcheck", action="store_true", help="fast IoU vs reference IoU")
    ap.add_argument("--stride", type=int, default=rf.DEFAULT_PARAMS["stride"])
    ap.add_argument("--model", default=rf.DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    ap.add_argument("--probs", default=None, help="val probs -> contest with model selection")
    ap.add_argument("--decode-params", default=None, help="decoding params for --probs")
    ap.add_argument("--limit", type=int, default=0, help="first N labeled clips (smoke test)")
    ap.add_argument("--pose-model", default=None,
                    help="YOLO pose weights: attach head points and search head / head_pos_x")
    ap.add_argument("--cv", type=int, default=0,
                    help="K: grouped K-fold over train+val instead of train->val; "
                         "--params-out then gets params tuned on every clip")
    args = ap.parse_args()

    rows, skipped = load_rows(args.gt, args.limit)
    print("labeled clips: %d  (skipped %s)" % (len(rows), dict(skipped)))
    if not rows:
        return 1
    train = [r for r in rows if r["split"] != "val"]
    val = [r for r in rows if r["split"] == "val"]
    print("train %d / val %d" % (len(train), len(val)))

    if args.analyze and not (args.tune or args.params or args.selfcheck):
        analyze(rows)
        return 0

    caches = build_caches(rows, args.stride, args)
    if args.pose_model:
        attach_poses(rows, caches, args.stride, args)
        SEARCH["head"] = [False, True]
        SEARCH["head_pos_x"] = [0.5, 0.45, 0.55, 0.4, 0.6]
    scorer = Scorer(caches)
    if args.selfcheck:
        return selfcheck(rows, scorer)
    if args.analyze:
        analyze(rows, scorer)

    start = rf.load_params(args.params, stride=args.stride)
    if args.cv:
        print("-" * 76)
        cross_validate(scorer, rows, start, args.cv)
        if args.params_out:
            best, m = coordinate_descent(scorer, rows, start, verbose=False)
            print("  tuned on all %d clips (in-sample %.2f): %s" % (len(rows), m * 100, best))
            save_params(best, "%s (all splits)" % args.gt, args)
        print("=" * 76)
        return 0
    named = [("defaults", rf.load_params(stride=args.stride))]
    if args.params:
        named.append(("params", start))
    if args.tune:
        if not train:
            print("[fatal] no train clips to tune on")
            return 1
        print("-" * 76)
        print("  coordinate descent on %d train clips" % len(train))
        best, m = coordinate_descent(scorer, train, start)
        print("  train mean IoU: center %.2f -> tuned %.2f"
              % (mean_of(scorer.run(None, train), 0) * 100, m * 100))
        print("  tuned params: %s" % best)
        named.append(("tuned", best))
        if args.params_out:
            save_params(best, args.gt, args)
    if val:
        print("-" * 76)
        report_val(scorer, val, named, args)
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
