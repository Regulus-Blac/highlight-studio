#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-second highlight probabilities -> frame selection / 逐秒概率 -> 选帧.

`infer_scores.py` writes one line per video with the model's digit
distribution at every one-second step. This script turns it into the frames
to submit, choosing the set that maximizes the expected contest F:

    F = 2 S / (N_pred + N_gt)
    ->  maximize  sum_{f in sel} p_f / (|sel| + lam * sum_all p_f)

Frames are sorted by p and the best prefix is kept. Adding a frame pays off
while its p exceeds about half the current F, so when the model is unsure
(flat p) the selection grows toward the whole video -- the 35.70 fallback --
and it only narrows where the model is confident.
按期望 F 选帧：模型没把握时自动退回接近全帧，有把握时才收窄。

Evaluation uses the val rows of `data/train.with_video.jsonl` directly:
  * contest score on the rows with `cropRois` (57, center crop, comparable to
    35.70 full video / 62.52 perfect temporal);
  * temporal F on every val row (89, IoU counted as 1), more rows, less noise.

    # self-check against the val GT (no model needed)
    python tools/decode_scores.py --selfcheck

    # untuned + 2-fold CV held-out scores, then params tuned on all val
    python tools/decode_scores.py --probs outputs/scores_val_base.probs.jsonl \
        --tune --params-out outputs/decode_params_base.json

    # submission with frozen params
    python tools/decode_scores.py --probs outputs/scores_test.probs.jsonl \
        --params outputs/decode_params.json --index baseline/test_index.json \
        --video-dir baseline/video --write-submission outputs/submit_test_scores.jsonl
"""

import os
import sys
import json
import math
import random
import argparse
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
from build_sft_temporal import score_steps  # noqa: E402
from highlight_metric import derive_h, iou_xywh, score_video  # noqa: E402
from baseline_qwen import compute_crop_size, load_index, video_meta  # noqa: E402

# Training clips are 5-32 s; longer videos are scored window by window.
# 训练片段 5-32 s，更长的视频分窗推理。
MAX_WINDOW_SEC = 32.0
# Mean highlight ratio of train.jsonl; `alpha` blends p toward it.
PRIOR = 0.43
DIGIT_VALUES = np.arange(10) / 9.0

# interp: p is linearly interpolated between step centers instead of held
# constant over the step. With perfect p that lifts the contest ceiling from
# 57.2 to 59.5 (frame-exact is 62.5), so it is on and not tuned.
# 步中心间线性插值：完美 p 下天花板 57.2 -> 59.5，默认开启、不参与调参。
# empty_tau: output NO frames when the video's peak p stays below it. Only
# searched with --include-empty (val rows without segments scored as N_gt=0),
# since an empty list scores 1 there and 0 everywhere else.
# 整段 p 都低于 empty_tau 时输出空列表；只在 --include-empty 时参与调参。
# shift: subtract from every step's logit before decoding (0 = off). Labeled
# clips are ~45 % highlight, the semifinal's new videos ~15-20 %; a global
# shift trades a little on dense videos for fewer frames on sparse ones.
# Not in GRID: chosen on the dense + simulated-sparse mix (see README 方案 C).
# 整体压低 logit；不参与 --tune 网格，在稠密 + 模拟稀疏混合上选。
DEFAULT_PARAMS = {"temp": 1.0, "smooth": 0, "lam": 1.0, "alpha": 0.0, "min_cov": 0.0,
                  "interp": True, "empty_tau": 0.0, "shift": 0.0}
GRID = {"temp": [1.0, 0.5, 2.0], "smooth": [0, 1, 2], "lam": [1.0, 0.8, 1.25],
        "alpha": [0.0, 0.3], "min_cov": [0.0, 0.3, 0.5]}
EMPTY_TAU_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]


# ----------------------------- steps & windows -----------------------------
def make_windows(duration, max_sec=MAX_WINDOW_SEC):
    """Equal windows of at most `max_sec` -> [(start_sec, end_sec, n_steps)]."""
    n = max(1, int(math.ceil(duration / max_sec - 1e-9)))
    size = duration / n
    out = []
    for i in range(n):
        s = i * size
        e = duration if i == n - 1 else (i + 1) * size
        out.append((s, e, score_steps(e - s)))
    return out


def step_centers(windows):
    """Center time (s) of every global step."""
    out = []
    for s, e, k in windows:
        for j in range(k):
            out.append(s + (j + (e - s)) / 2.0 if j == k - 1 else s + j + 0.5)
    return np.array(out)


def frame_steps(windows, fps, n_frames):
    """Global step index of every frame. 每帧所属的步。"""
    t = np.arange(n_frames) / float(fps)
    starts = np.array([w[0] for w in windows], dtype=np.float64)
    ks = np.array([w[2] for w in windows], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(ks)[:-1]])
    wi = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(windows) - 1)
    local = np.clip(np.floor(t - starts[wi]).astype(np.int64), 0, ks[wi] - 1)
    return offs[wi] + local


# ------------------------------- decoding ----------------------------------
def step_probs(rec, temp=1.0):
    """Digit log-probs -> expected covered fraction per step. 期望覆盖率。

    `temp` > 1 flattens the digit distribution, < 1 sharpens it. Records
    without `digit_logp` (the self-check oracle, train_temporal_head.py,
    ensemble_probs.py) carry `p_sec` directly; `temp` then scales its logit.
    """
    if "digit_logp" not in rec:
        p = np.asarray(rec["p_sec"], dtype=np.float64)
        if temp == 1.0:
            return p
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return 1.0 / (1.0 + np.exp(-(np.log(p) - np.log1p(-p)) / temp))
    lp = np.asarray(rec["digit_logp"], dtype=np.float64) / temp
    lp -= lp.max(axis=1, keepdims=True)
    w = np.exp(lp)
    w /= w.sum(axis=1, keepdims=True)
    return w @ DIGIT_VALUES


def _smooth(p, w):
    """Mean over +-w steps, the window shrinking at the edges.

    Cumulative sums rather than np.convolve(mode="same"), which returns the
    KERNEL's length when it is longer than p -- a clip under ~4 s with
    smooth=2 used to crash the whole submission. 短于核长的序列不再出错。
    """
    if w <= 0 or len(p) < 2:
        return p
    n = len(p)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    i = np.arange(n)
    lo, hi = np.maximum(i - w, 0), np.minimum(i + w + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def frame_probs(p_step, windows, fps, n_frames, params):
    """Step probabilities -> one p per frame (shifted, smoothed, blended, expanded)."""
    p = p_step
    if params.get("shift", 0.0):
        q = np.clip(p, 1e-6, 1.0 - 1e-6)
        p = 1.0 / (1.0 + np.exp(-(np.log(q) - np.log1p(-q) - params["shift"])))
    p = _smooth(p, params["smooth"])
    p = (1.0 - params["alpha"]) * p + params["alpha"] * PRIOR
    if params.get("interp", True):
        return np.interp(np.arange(n_frames) / float(fps), step_centers(windows), p)
    return p[frame_steps(windows, fps, n_frames)]


def select_frames(pf, params):
    """Bool mask over frames maximizing the expected F. 期望 F 最优选帧。

    Never returns an empty selection unless `empty_tau` > 0 and no frame
    reaches it: an empty list scores 1 on a video with no highlight but 0 on
    every other video, so it is only worth it where the model is sure.
    """
    n = len(pf)
    sel = np.zeros(n, dtype=bool)
    total = pf.sum()
    if n == 0:
        return sel
    if params.get("empty_tau", 0.0) > 0 and pf.max() < params["empty_tau"]:
        return sel
    if total <= 1e-9:
        sel[:] = True
        return sel
    order = np.argsort(-pf, kind="stable")
    obj = np.cumsum(pf[order]) / (np.arange(1, n + 1) + params["lam"] * total)
    k = max(int(np.argmax(obj)) + 1, int(math.ceil(params["min_cov"] * n)))
    sel[order[:k]] = True
    return sel


# ------------------------------ ground truth -------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def center_box(W, H, ratio):
    cw, ch = compute_crop_size(W, H, ratio[0], ratio[1])
    return (W - cw) // 2, (H - ch) // 2, cw


def load_val_videos(gt_path, recs=None, split="val", include_empty=False):
    """Val rows -> {vid: dict(n, fps, W, H, ratio, gt mask, n_gt, iou, row)}.

    Video size / length come from the probs record when there is one (what
    inference saw), else from the clip on disk.

    Rows without segments are skipped unless `include_empty`: then they enter
    as N_gt = 0 videos (F = 1 only for an empty selection) on BOTH metrics,
    as the official evaluator averages over every index entry.
    无高光的 val 行默认跳过；include_empty 时按 N_gt=0 计入两个指标。
    """
    videos = {}
    for r in read_jsonl(gt_path):
        if split != "all" and r.get("dataset_split") != split:
            continue
        if not r.get("segments") and not include_empty:
            continue
        vid = r["video_id"]
        rec = (recs or {}).get(vid) or {}
        if "n_frames" in rec:
            n, fps, W, H = rec["n_frames"], rec["fps"], rec["width"], rec["height"]
        else:
            vp = r.get("video_path") or ""
            if not os.path.exists(vp):
                continue
            n, fps, W, H = video_meta(vp)
        if n <= 0:
            continue
        gt_frames = set()
        for s in r.get("segments") or []:
            gt_frames.update(range(int(s["start_frame"]), int(s["end_frame"]) + 1))
        gt = np.zeros(n, dtype=bool)
        gt[[f for f in gt_frames if 0 <= f < n]] = True
        ratio = [float(v) for v in r["targetRatioWH"]]
        v = {"n": n, "fps": fps, "W": W, "H": H, "ratio": ratio, "gt": gt,
             "n_gt": len(gt_frames), "iou": None, "row": r}
        if not gt_frames:
            # No highlight: nothing to overlap, but it still counts on the
            # contest metric (F = 1 iff nothing is selected).
            v["iou"] = np.zeros(n)
        elif r.get("cropRois"):
            # Per-frame IoU of the center box with the GT box, computed the way
            # the evaluator does (h derived from w). 与评测端一致地算居中框 IoU。
            x, y, cw = center_box(W, H, ratio)
            box = [x, y, cw, derive_h(cw, ratio)]
            iou = np.zeros(n)
            for f, b in r["cropRois"]:
                if 0 <= int(f) < n:
                    iou[int(f)] = iou_xywh(box, [float(c) for c in b])
            v["iou"] = iou
        videos[vid] = v
    return videos


def video_scores(v, sel):
    """-> (temporal F, contest F or None) for one selection mask."""
    denom = sel.sum() + v["n_gt"]
    if denom == 0:
        return 1.0, (1.0 if v["iou"] is not None else None)
    ft = 2.0 * (sel & v["gt"]).sum() / denom
    fc = 2.0 * v["iou"][sel].sum() / denom if v["iou"] is not None else None
    return ft, fc


# ------------------------------- evaluation --------------------------------
def select_for(v, rec, params, prob_cache=None):
    """Selection for one val video; a missing / failed record = whole video."""
    if not rec or rec.get("error") or not rec.get("windows"):
        return np.ones(v["n"], dtype=bool)
    key = (id(rec), params["temp"])
    p = prob_cache.get(key) if prob_cache is not None else None
    if p is None:
        p = step_probs(rec, params["temp"])
        if prob_cache is not None:
            prob_cache[key] = p
    if len(p) != sum(w[2] for w in rec["windows"]):
        return np.ones(v["n"], dtype=bool)
    return select_frames(frame_probs(p, rec["windows"], v["fps"], v["n"], params), params)


def evaluate(videos, recs, params, vids, prob_cache=None):
    """-> {vid: (temporal F, contest F or None, selected fraction)}"""
    out = {}
    for vid in vids:
        v = videos[vid]
        sel = select_for(v, recs.get(vid), params, prob_cache)
        ft, fc = video_scores(v, sel)
        out[vid] = (ft, fc, sel.mean())
    return out


def full_video(videos, vids):
    return {vid: video_scores(videos[vid], np.ones(videos[vid]["n"], dtype=bool)) + (1.0,)
            for vid in vids}


def grid_params():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


def tune(videos, recs, vids, prob_cache):
    """Params maximizing mean temporal F over `vids` (first best wins)."""
    best, best_m = None, -1.0
    for params in grid_params():
        res = evaluate(videos, recs, params, vids, prob_cache)
        m = float(np.mean([r[0] for r in res.values()]))
        if m > best_m + 1e-12:
            best, best_m = params, m
    return best, best_m


def paired_ci(a, b, n_boot=2000, seed=0):
    """Mean of a-b and its bootstrap 95% CI, x100. 配对 bootstrap。"""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return d.mean() * 100, np.percentile(means, 2.5) * 100, np.percentile(means, 97.5) * 100


def report(title, res, base):
    """Print model vs full-video on both metrics, paired CI on the difference."""
    vids = sorted(res)
    print("  %s" % title)
    for name, i in (("temporal F", 0), ("contest   ", 1)):
        vs = [v for v in vids if res[v][i] is not None]
        if not vs:
            continue
        a = [res[v][i] for v in vs]
        b = [base[v][i] for v in vs]
        d, lo, hi = paired_ci(a, b)
        print("    %s (%3d): model %6.2f | full video %6.2f | delta %+6.2f  95%% CI [%+.2f, %+.2f]"
              % (name, len(vs), np.mean(a) * 100, np.mean(b) * 100, d, lo, hi))
    print("    selected fraction %.3f" % np.mean([res[v][2] for v in vids]))


def ranking_auc(videos, recs, vids):
    """Mean within-video AUC of the per-frame p against the GT frames.

    Threshold-free: 0.5 = no ranking signal, whatever the decoding does.
    与解码无关的排序信号，0.5 = 无信号。
    """
    aucs = []
    for vid in vids:
        v, rec = videos[vid], recs.get(vid) or {}
        if not rec.get("windows") or rec.get("error"):
            continue
        pos = int(v["gt"].sum())
        neg = v["n"] - pos
        if not pos or not neg:
            continue
        pf = frame_probs(step_probs(rec), rec["windows"], v["fps"], v["n"], DEFAULT_PARAMS)
        order = np.argsort(pf, kind="stable")
        ranks = np.empty(v["n"])
        ranks[order] = np.arange(1, v["n"] + 1)
        # Ties get their average rank. 并列取平均秩。
        _, inv = np.unique(pf, return_inverse=True)
        ranks = (np.bincount(inv, weights=ranks) / np.bincount(inv))[inv]
        aucs.append((ranks[v["gt"]].sum() - pos * (pos + 1) / 2.0) / (pos * neg))
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def run_tune(args, videos, recs):
    vids = sorted(v for v in videos if v in recs)
    missing = len(videos) - len(vids)
    if missing:
        print("[warn] %d val videos have no probs record (scored as whole video)" % missing)
        vids = sorted(videos)
    n_err = sum(1 for v in vids if (recs.get(v) or {}).get("error"))
    gt_ratio = np.mean([videos[v]["gt"].mean() for v in vids])
    auc, n_auc = ranking_auc(videos, recs, vids)
    print("=" * 76)
    print("  %d val videos, %d failed records, GT highlight fraction %.3f"
          % (len(vids), n_err, gt_ratio))
    print("  ranking AUC (within video, %d videos): %.3f   [0.5 = no signal]" % (n_auc, auc))
    base = full_video(videos, vids)
    cache = {}

    print("-" * 76)
    report("default params %s (no tuning)" % DEFAULT_PARAMS,
           evaluate(videos, recs, DEFAULT_PARAMS, vids, cache), base)

    # 2-fold CV: tune on one half, score the other. 两折交叉验证，只报留出分。
    ids = list(vids)
    random.Random(args.seed).shuffle(ids)
    folds = [sorted(ids[i::2]) for i in range(2)]
    held = {}
    print("-" * 76)
    for i in range(2):
        p, m = tune(videos, recs, folds[1 - i], cache)
        print("  fold %d: tuned on %d videos (temporal F %.2f) -> %s"
              % (i, len(folds[1 - i]), m * 100, p))
        held.update(evaluate(videos, recs, p, folds[i], cache))
    report("2-fold CV held-out", held, base)

    p_all, m_all = tune(videos, recs, vids, cache)
    print("-" * 76)
    report("tuned on all val (in-sample, optimistic) %s" % p_all,
           evaluate(videos, recs, p_all, vids, cache), base)
    print("=" * 76)
    if args.params_out:
        d, lo, hi = paired_ci([held[v][0] for v in vids], [base[v][0] for v in vids])
        with open(args.params_out, "w", encoding="utf-8") as f:
            json.dump(dict(p_all, tuned_on=args.probs,
                           cv_temporal_delta=[round(d, 3), round(lo, 3), round(hi, 3)]),
                      f, indent=2, ensure_ascii=False)
        print("  params -> %s" % args.params_out)


# -------------------------------- submission -------------------------------
def decoded_frames(path):
    """Frames a sequential decode actually delivers. 实际可解码帧数。

    CAP_PROP_FRAME_COUNT is a container estimate and can overshoot; frames
    past the real end score 0 but still count in N_pred.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def write_submission(args, recs, params):
    index = load_index(args.index)
    stats = {"videos": 0, "model": 0, "fallback_full": 0, "empty": 0, "frames": 0,
             "truncated": 0}
    with open(args.write_submission, "w", encoding="utf-8") as fout:
        for vid, (tw, th) in index:
            stats["videos"] += 1
            out = {"video_id": vid, "targetRatioWH": [int(tw), int(th)], "predictions": []}
            rec = recs.get(vid) or {}
            if "n_frames" in rec:
                n, fps, W, H = rec["n_frames"], rec["fps"], rec["width"], rec["height"]
            elif args.video_dir:
                n, fps, W, H = video_meta(os.path.join(args.video_dir, vid + ".mp4"))
            else:
                n = fps = W = H = 0
            if n > 0 and W > 0 and H > 0:
                sel = select_for({"n": n, "fps": fps}, rec, params)
                usable = rec.get("windows") and not rec.get("error")
                stats["model" if usable else "fallback_full"] += 1
                if args.safe_frames and args.video_dir:
                    n_dec = decoded_frames(os.path.join(args.video_dir, vid + ".mp4"))
                    if 0 < n_dec < n:
                        stats["truncated"] += 1
                        sel[n_dec:] = False
                x, y, cw = center_box(W, H, [tw, th])
                out["predictions"] = [{"frame": int(f), "bboxes": [x, y, cw]}
                                      for f in np.flatnonzero(sel)]
            if not out["predictions"]:
                stats["empty"] += 1
            stats["frames"] += len(out["predictions"])
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("submission -> %s  %s" % (args.write_submission, stats))


# -------------------------------- self-check -------------------------------
def selfcheck(args):
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))

    # windows / steps
    w = make_windows(285.8)
    check("long video windows <= 32 s", all(e - s <= MAX_WINDOW_SEC + 1e-9 for s, e, _ in w)
          and abs(w[-1][1] - 285.8) < 1e-9, "(%d windows)" % len(w))
    st = frame_steps(w, 30.0, int(285.8 * 30))
    total_steps = sum(k for _, _, k in w)
    check("steps monotone and all used", bool(np.all(np.diff(st) >= 0))
          and st.max() == total_steps - 1 and len(np.unique(st)) == total_steps)
    check("short clip steps", make_windows(10.44)[0][2] == 11 and make_windows(10.2)[0][2] == 10)

    # empty_tau / p_sec temperature -- pure functions, no data needed.
    low = np.full(50, 0.05)
    check("empty_tau: flat low p -> empty, off -> never empty",
          not select_frames(low, dict(DEFAULT_PARAMS, empty_tau=0.1)).any()
          and select_frames(low, DEFAULT_PARAMS).any())
    rec = {"p_sec": [0.2, 0.5, 0.8]}
    hot, cold = step_probs(rec, 2.0), step_probs(rec, 0.5)
    check("p_sec temperature: 2 flattens, 0.5 sharpens, 0.5 fixed",
          hot[0] > 0.2 and hot[2] < 0.8 and cold[0] < 0.2 and cold[2] > 0.8
          and abs(hot[1] - 0.5) < 1e-12 and np.allclose(step_probs(rec, 1.0), [0.2, 0.5, 0.8]))
    w1 = [(0.0, 3.0, 3)]
    ps = np.array([0.2, 0.5, 0.8])
    sh = frame_probs(ps, w1, 1.0, 3, dict(DEFAULT_PARAMS, shift=1.0, interp=False))
    check("shift lowers every p, keeps the order, 0 = off",
          bool(np.all(sh < ps)) and bool(np.all(np.diff(sh) > 0))
          and np.allclose(frame_probs(ps, w1, 1.0, 3, dict(DEFAULT_PARAMS, interp=False)), ps))

    videos = load_val_videos(args.gt)
    vids = sorted(videos)
    spatial = [v for v in vids if videos[v]["iou"] is not None]
    print("  val videos: %d (with spatial GT: %d)" % (len(vids), len(spatial)))
    lost = sum(videos[v]["n_gt"] - int(videos[v]["gt"].sum()) for v in vids)
    check("GT frames inside the clip", lost == 0, "(%d outside)" % lost)

    recs_const, recs_oracle = {}, {}
    for vid in vids:
        v = videos[vid]
        win = make_windows(v["n"] / float(v["fps"]))
        steps = frame_steps(win, v["fps"], v["n"])
        k = sum(x[2] for x in win)
        cov = np.bincount(steps, weights=v["gt"].astype(float), minlength=k) / \
            np.maximum(np.bincount(steps, minlength=k), 1)
        recs_const[vid] = {"windows": win, "p_sec": [0.5] * k}
        recs_oracle[vid] = {"windows": win, "p_sec": cov.tolist()}

    base = full_video(videos, vids)
    const = evaluate(videos, recs_const, DEFAULT_PARAMS, vids)
    oracle = evaluate(videos, recs_oracle, DEFAULT_PARAMS, vids)
    c_const = np.mean([const[v][1] for v in spatial]) * 100
    c_oracle = np.mean([oracle[v][1] for v in spatial]) * 100
    t_oracle = np.mean([oracle[v][0] for v in vids]) * 100
    check("flat p selects the whole video", all(const[v][2] == 1.0 for v in vids))
    # 36.51 on the full 100-row val (62 with cropRois, all 987 sources present);
    # the 4060-era partial val (57 rows) gave 35.70. 全量 val 的全帧分。
    check("flat p = full video 36.51", abs(c_const - 36.51) < 0.05, "(%.2f)" % c_const)
    check("oracle p near perfect temporal 62.52", 59.0 < c_oracle <= 62.6,
          "(contest %.2f, temporal F %.2f)" % (c_oracle, t_oracle))
    check("oracle beats full video on every video",
          all(oracle[v][0] >= base[v][0] - 1e-12 for v in vids))
    auc_o, _ = ranking_auc(videos, recs_oracle, vids)
    auc_c, _ = ranking_auc(videos, recs_const, vids)
    check("ranking AUC: oracle high, flat p = 0.5", auc_o > 0.95 and abs(auc_c - 0.5) < 1e-9,
          "(oracle %.3f, flat %.3f)" % (auc_o, auc_c))

    # Fast scorer vs the reference metric in highlight_metric.score_video.
    # 快速打分与参考实现逐视频对拍。
    worst = 0.0
    for recs in (recs_const, recs_oracle):
        for vid in spatial:
            v = videos[vid]
            sel = select_for(v, recs[vid], DEFAULT_PARAMS)
            x, y, cw = center_box(v["W"], v["H"], v["ratio"])
            preds = [{"frame": int(f), "bboxes": [x, y, cw]} for f in np.flatnonzero(sel)]
            gt = {int(f): [float(c) for c in b] for f, b in v["row"]["cropRois"]}
            ref = score_video(preds, gt, v["ratio"],
                              {"n_frames": v["n"], "width": v["W"], "height": v["H"]})
            worst = max(worst, abs(ref["F"] - video_scores(v, sel)[1]))
    check("fast scorer == highlight_metric.score_video", worst < 1e-9, "(max diff %.2e)" % worst)

    n_fail = results.count(False)
    print("  %d passed, %d failed" % (len(results) - n_fail, n_fail))
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description="Decode per-second highlight probabilities")
    ap.add_argument("--probs", help="*.probs.jsonl from infer_scores.py")
    ap.add_argument("--gt", default="data/train.with_video.jsonl",
                    help="rows with segments / cropRois; the val split is scored")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--tune", action="store_true", help="CV-tune decoding on val")
    ap.add_argument("--params", default=None, help="JSON of decoding params")
    ap.add_argument("--params-out", default=None)
    ap.add_argument("--write-submission", default=None, help="output submission JSONL")
    ap.add_argument("--index", default=None, help="index for --write-submission")
    ap.add_argument("--video-dir", default=None, help="for videos with no probs record")
    ap.add_argument("--include-empty", action="store_true",
                    help="score val rows without segments as N_gt=0 and tune empty_tau")
    ap.add_argument("--safe-frames", action="store_true",
                    help="decode each test video and drop frames past its real end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=("val", "train", "all"),
                    help="GT rows to score; 'all' with out-of-fold train probs + val probs "
                         "gives the CV ~10x the videos")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck(args)
    if not args.probs:
        ap.error("--probs is required unless --selfcheck")
    if args.include_empty:
        GRID["empty_tau"] = EMPTY_TAU_GRID
    recs = {r["video_id"]: r for r in read_jsonl(args.probs)}
    params = dict(DEFAULT_PARAMS)
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        params.update({k: loaded[k] for k in DEFAULT_PARAMS if k in loaded})

    if args.tune:
        run_tune(args, load_val_videos(args.gt, recs, args.split, args.include_empty), recs)
    elif not args.write_submission:
        videos = load_val_videos(args.gt, recs, args.split, args.include_empty)
        vids = sorted(videos)
        print("=" * 76)
        report("params %s" % params, evaluate(videos, recs, params, vids),
               full_video(videos, vids))
    if args.write_submission:
        if not args.index:
            ap.error("--write-submission needs --index")
        print("decoding with %s" % params)
        write_submission(args, recs, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
