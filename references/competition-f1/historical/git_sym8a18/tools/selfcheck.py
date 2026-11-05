#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-check for the local evaluator / 本地评测器自检.

    python3 tools/selfcheck.py [--train train.jsonl]

Part 1 -- hand-built cases covering every scoring rule in task.md section 6,
including the three N_gt/N_pred edge rules and the "invalid predictions still
count toward N_pred" clause.

Part 2 -- replays two reference strategies over the weak-label train set and
checks the resulting scores against the values measured when the metric was
written. This is the regression guard: if a future edit changes the metric,
these two numbers move. 这是回归基准，指标实现被改动时这两个数会变。

Exits non-zero if any check fails.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from highlight_metric import score_video, score_dataset, iou_xywh  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    (PASS if ok else FAIL).append(name)
    print("  [%s] %-58s got=%.6f want=%.6f" % ("ok" if ok else "XX", name, got, want))


def box(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


def p(frame, x, y, w):
    return {"frame": frame, "bboxes": [x, y, w]}


# ---------------------------------------------------------------- part 1
def edge_cases():
    print("-" * 78)
    print("Part 1 / rule-by-rule cases")
    print("-" * 78)
    r169 = [9, 16]          # w=90 -> h=160
    r16x9 = [16, 9]         # w=160 -> h=90

    # IoU primitive
    check("iou: identical boxes", iou_xywh(box(0, 0, 10, 10), box(0, 0, 10, 10)), 1.0)
    check("iou: disjoint boxes", iou_xywh(box(0, 0, 10, 10), box(50, 0, 10, 10)), 0.0)
    # same-size boxes sliding on x: IoU = (w-|dx|)/(w+|dx|)
    check("iou: equal boxes offset by 0.1w",
          iou_xywh(box(0, 0, 100, 50), box(10, 0, 100, 50)), 90.0 / 110.0)

    # the three edge rules
    check("edge: n_gt=0 and n_pred=0 -> F=1",
          score_video([], {}, r16x9)["F"], 1.0)
    check("edge: n_gt=0 and n_pred>0 -> F=0",
          score_video([p(0, 0, 0, 160)], {}, r16x9)["F"], 0.0)
    check("edge: n_gt>0 and n_pred=0 -> F=0",
          score_video([], {0: box(0, 0, 160, 90)}, r16x9)["F"], 0.0)

    # perfect prediction
    gt = {i: box(0, 0, 160, 90) for i in range(10)}
    preds = [p(i, 0, 0, 160) for i in range(10)]
    check("perfect: 10/10 exact -> F=1", score_video(preds, gt, r16x9)["F"], 1.0)

    # height really is derived from the ratio, not submitted
    check("derive h: w=90 with 9:16 gives h=160",
          score_video([p(0, 0, 0, 90)], {0: box(0, 0, 90, 160)}, r169)["F"], 1.0)

    # recall loss: predict half the GT frames, each exact
    #   S=5, n_pred=5, n_gt=10 -> F = 2*5/(5+10)
    check("recall: 5 exact of 10 GT",
          score_video([p(i, 0, 0, 160) for i in range(5)], gt, r16x9)["F"],
          2 * 5.0 / (5 + 10))

    # precision loss: 10 exact + 10 frames that have no GT
    #   S=10, n_pred=20, n_gt=10 -> F = 2*10/(20+10)
    over = preds + [p(i, 0, 0, 160) for i in range(10, 20)]
    check("precision: 10 exact + 10 off-GT frames",
          score_video(over, gt, r16x9)["F"], 2 * 10.0 / (20 + 10))

    # duplicate frame: first valid one scores, the rest are false positives
    dup = [p(0, 0, 0, 160), p(0, 0, 0, 160), p(0, 0, 0, 160)]
    one = {0: box(0, 0, 160, 90)}
    check("duplicate: 3 lines on one frame -> S=1, n_pred=3",
          score_video(dup, one, r16x9)["F"], 2 * 1.0 / (3 + 1))
    check("duplicate: only the first is scored",
          float(score_video(dup, one, r16x9)["n_scored"]), 1.0)

    # first-in-order wins even when a later duplicate would score better
    worse_first = [p(0, 40, 0, 160), p(0, 0, 0, 160)]
    res = score_video(worse_first, one, r16x9)
    check("duplicate: submission order decides which one counts",
          res["S"], iou_xywh(box(40, 0, 160, 90), box(0, 0, 160, 90)))

    # invalid predictions count toward n_pred but score 0
    bad = [p(0, 0, 0, 160),
           p(1, 0, 0, -5),                       # w <= 0
           p(2, -3, 0, 160),                     # x < 0
           {"frame": 3, "bboxes": [0, 0]},       # not a triplet
           {"frame": 4, "bboxes": [0, 0, float("nan")]},   # NaN
           {"frame": 5}]                         # missing bboxes
    check("invalid: 6 submitted, 1 valid -> n_pred=6",
          score_video(bad, one, r16x9)["F"], 2 * 1.0 / (6 + 1))

    # out-of-bound checks only fire when video metadata is supplied
    meta = {"n_frames": 100, "width": 320, "height": 180}
    check("bounds: in-bounds box with meta -> F=1",
          score_video([p(0, 0, 0, 160)], one, r16x9, meta)["F"], 1.0)
    check("bounds: frame >= n_frames is rejected",
          score_video([p(200, 0, 0, 160)], one, r16x9, meta)["F"], 0.0)
    check("bounds: y+h past the frame height is rejected",
          score_video([p(0, 0, 120, 160)], one, r16x9, meta)["F"], 0.0)

    # the round()-vs-floor() trap: 9:16 crop of a 534x300 frame
    #   round(300*9/16) = 169 -> h = 169*16/9 = 300.44 > 300  => rejected
    #   floor(300*9/16) = 168 -> h = 168*16/9 = 298.67 <= 300 => accepted
    m534 = {"n_frames": 10, "width": 534, "height": 300}
    gt534 = {0: box(182, 0, 169, 300)}
    check("trap: w=round(H*9/16)=169 overflows height -> rejected",
          score_video([p(0, 182, 0, 169)], gt534, r169, m534)["F"], 0.0)
    got = score_video([p(0, 183, 0, 168)], gt534, r169, m534)["F"]
    (PASS if got > 0.9 else FAIL).append("trap: w=floor(...)=168 accepted")
    print("  [%s] %-58s got=%.6f want>0.9" % ("ok" if got > 0.9 else "XX",
                                              "trap: w=floor(...)=168 accepted", got))

    # a whole video missing from the submission scores 0
    final, _ = score_dataset({}, {"v1": one}, {"v1": r16x9})
    check("dataset: video missing from submission -> F=0", final, 0.0)

    # final score is a plain mean over videos, long videos get no extra weight
    gt_long = {i: box(0, 0, 160, 90) for i in range(1000)}
    sub = {"short": [p(0, 0, 0, 160)], "long": []}
    final, _ = score_dataset(sub, {"short": one, "long": gt_long},
                             {"short": r16x9, "long": r16x9})
    check("dataset: per-video mean, not per-frame", final, 50.0)


# ---------------------------------------------------------------- part 2
# Reference strategies replayed on the weak-label train set. Measured when the
# metric was written, over the 624 rows that carry spatial annotation.
#
# C is 99.85 rather than 100 on purpose: the GT crop width was built as
# round(H*9/16) = 169, but a [x,y,w] submission makes the evaluator back-derive
# h = 169*16/9 = 300.44 against a GT height of 300, so even a pixel-perfect
# replay of the GT loses ~0.15%. 这是 [x,y,w] 往返换算的固有损失，不是 bug。
REF = {
    "A all-frames + center crop": 35.61,
    "B oracle temporal + center crop": 64.80,
    "C oracle temporal + oracle spatial": 99.85,
}


def ladder(train_path):
    print("-" * 78)
    print("Part 2 / reference strategies on %s" % train_path)
    print("-" * 78)
    if not os.path.exists(train_path):
        print("  [skip] %s not found" % train_path)
        return

    rows = []
    for line in open(train_path, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    # Frame width is not stored in the weak-label schema. Every GT box in this
    # file is the max inscribed 9:16 rectangle with y=0 and h=H, so infer the
    # frame width ONCE from the whole file -- per-row max(x+w) underestimates it
    # whenever the crop never reaches the right edge in that clip.
    # 必须全局推断画面宽度：逐行取 max(x+w) 会在裁剪框从未贴右边时系统性偏小。
    frame_w = max((int(b[0]) + int(b[2]))
                  for d in rows for _, b in (d.get("cropRois") or []))
    print("  inferred frame width (global max of x+w): %d" % frame_w)

    gt, ratios, subs = {}, {}, {k: {} for k in REF}
    fps = 30.0
    skipped = 0
    for d in rows:
        rois = d.get("cropRois") or []
        if not rois:
            # spatial annotation was dropped by the generator on these rows
            skipped += 1
            continue
        vid = d["video_id"]
        gt[vid] = {int(f): [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                   for f, b in rois}
        ratios[vid] = [float(d["targetRatioWH"][0]), float(d["targetRatioWH"][1])]

        w = rois[0][1][2]
        xc = int(round((frame_w - w) / 2.0))
        n_total = max(1, int(round((d["clip"]["end_sec"] - d["clip"]["start_sec"]) * fps)))

        subs["A all-frames + center crop"][vid] = [
            p(f, xc, 0, w) for f in range(n_total)]
        subs["B oracle temporal + center crop"][vid] = [
            p(int(f), xc, 0, w) for f, _ in rois]
        subs["C oracle temporal + oracle spatial"][vid] = [
            p(int(f), int(b[0]), int(b[1]), int(b[2])) for f, b in rois]

    print("  videos scored: %d   (skipped %d rows with no cropRois)"
          % (len(gt), skipped))
    for name, want in REF.items():
        got, _ = score_dataset(subs[name], gt, ratios)
        check(name, got, want, tol=0.01)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Self-check the local evaluator")
    ap.add_argument("--train", default=os.path.join(here, os.pardir, "train.jsonl"))
    args = ap.parse_args()

    edge_cases()
    ladder(os.path.normpath(args.train))

    print("=" * 78)
    print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for name in FAIL:
            print("    FAILED: %s" % name)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
