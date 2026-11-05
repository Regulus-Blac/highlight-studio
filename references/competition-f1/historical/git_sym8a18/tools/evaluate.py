#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local evaluator CLI / 本地评测器命令行入口.

    python3 tools/evaluate.py --gt GT.jsonl --pred PRED.jsonl [--index test_index.json]

Ground-truth file may be either schema:
  * seed weak-label schema (train.jsonl): lines carry `cropRois`
    [[frame, [x, y, w, h]], ...] -- boxes are already [x,y,w,h].
  * submission schema: lines carry `predictions` [{"frame", "bboxes":[x,y,w]}]
    -- height is derived from targetRatioWH.
GT 文件两种 schema 均可：train.jsonl 的 cropRois，或提交格式的 predictions。

Frame-bound / box-bound checks need the source video size. Supply it with
--video-dir (read via OpenCV) or --meta (a JSON map), otherwise those checks
are skipped and the report says so.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from highlight_metric import score_dataset, derive_h  # noqa: E402


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                raise SystemExit("[fatal] %s line %d: bad JSON: %s" % (path, i, e))
    return out


def load_ground_truth(path):
    """-> (gt {vid: {frame: [x,y,w,h]}}, ratios {vid: [tw,th]})"""
    gt, ratios = {}, {}
    for lineno, d in read_jsonl(path):
        vid = str(d.get("video_id", "")).strip()
        if not vid:
            raise SystemExit("[fatal] %s line %d: missing video_id" % (path, lineno))
        ratio = d.get("targetRatioWH") or [16, 9]
        ratios[vid] = [float(ratio[0]), float(ratio[1])]
        boxes = {}
        if "cropRois" in d:
            # weak-label schema: boxes already carry h
            for item in d["cropRois"]:
                f, b = int(item[0]), item[1]
                boxes[f] = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        elif "predictions" in d:
            for p in d["predictions"]:
                f = int(p["frame"])
                x, y, w = (float(v) for v in p["bboxes"])
                boxes[f] = [x, y, w, derive_h(w, ratios[vid])]
        else:
            raise SystemExit("[fatal] %s line %d: no cropRois / predictions"
                             % (path, lineno))
        gt[vid] = boxes
    return gt, ratios


def load_submission(path):
    """-> ({vid: [pred,...]}, {vid: [tw,th]}, [problem strings])"""
    sub, ratios, problems = {}, {}, []
    for lineno, d in read_jsonl(path):
        vid = str(d.get("video_id", "")).strip()
        if not vid:
            problems.append("line %d: missing video_id" % lineno)
            continue
        if vid in sub:
            problems.append("line %d: duplicate video_id %s (first one kept)"
                            % (lineno, vid))
            continue
        preds = d.get("predictions", [])
        if not isinstance(preds, list):
            problems.append("line %d: predictions is not a list" % lineno)
            preds = []
        sub[vid] = preds
        if d.get("targetRatioWH"):
            tr = d["targetRatioWH"]
            ratios[vid] = [float(tr[0]), float(tr[1])]
    return sub, ratios, problems


def load_metas(video_dir=None, meta_path=None, video_ids=()):
    metas = {}
    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as f:
            for vid, m in json.load(f).items():
                metas[str(vid)] = m
    if video_dir:
        try:
            import cv2
        except ImportError:
            print("[warn] --video-dir given but OpenCV is missing; "
                  "bound checks fall back to --meta / are skipped")
            return metas
        for vid in video_ids:
            p = os.path.join(video_dir, str(vid) + ".mp4")
            if not os.path.exists(p):
                continue
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                continue
            metas[str(vid)] = {
                "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            cap.release()
    return metas


def main():
    ap = argparse.ArgumentParser(description="Local contest metric evaluator")
    ap.add_argument("--gt", required=True, help="ground-truth JSONL")
    ap.add_argument("--pred", required=True, help="submission JSONL")
    ap.add_argument("--index", default=None,
                    help="test_index.json; authoritative targetRatioWH source")
    ap.add_argument("--video-dir", default=None, help="source videos for bound checks")
    ap.add_argument("--meta", default=None,
                    help="JSON {video_id: {n_frames, width, height}}")
    ap.add_argument("--per-video", action="store_true", help="print every video")
    ap.add_argument("--json-out", default=None, help="write full report to JSON")
    ap.add_argument("--worst", type=int, default=10, help="show N worst videos")
    args = ap.parse_args()

    gt, gt_ratios = load_ground_truth(args.gt)
    sub, sub_ratios, problems = load_submission(args.pred)

    # targetRatioWH priority: index > ground truth > submission.
    # 目标比例以索引为准——参赛者不得改动它。
    ratios = dict(gt_ratios)
    if args.index:
        with open(args.index, "r", encoding="utf-8") as f:
            for it in json.load(f):
                tr = it.get("targetRatioWH", [16, 9])
                ratios[str(it["video_id"])] = [float(tr[0]), float(tr[1])]
    for vid, r in sub_ratios.items():
        if vid in ratios and [float(v) for v in r] != [float(v) for v in ratios[vid]]:
            problems.append("video %s: submitted targetRatioWH %s != official %s "
                            "(official used)" % (vid, r, ratios[vid]))
        ratios.setdefault(vid, r)

    metas = load_metas(args.video_dir, args.meta, list(gt.keys()))
    if not metas:
        print("[note] no video metadata: frame/box out-of-bound checks skipped")

    missing = [v for v in gt if v not in sub]
    extra = [v for v in sub if v not in gt]
    if missing:
        problems.append("%d video_id in GT missing from submission "
                        "(scored as N_pred=0)" % len(missing))
    if extra:
        problems.append("%d video_id in submission not in GT (ignored)" % len(extra))

    final, per_video = score_dataset(sub, gt, ratios, metas)

    n = len(per_video)
    tot_pred = sum(r["n_pred"] for r in per_video.values())
    tot_gt = sum(r["n_gt"] for r in per_video.values())
    tot_scored = sum(r["n_scored"] for r in per_video.values())
    rejects = {}
    for r in per_video.values():
        for k, v in r["rejects"].items():
            rejects[k] = rejects.get(k, 0) + v

    print("=" * 64)
    print("  Score (F_video x 100) : %.4f" % final)
    print("=" * 64)
    print("  videos                : %d" % n)
    print("  predictions submitted : %d" % tot_pred)
    print("  ground-truth frames   : %d" % tot_gt)
    print("  predictions scored    : %d (%.1f%% of submitted)"
          % (tot_scored, 100.0 * tot_scored / tot_pred if tot_pred else 0.0))
    if rejects:
        print("  rejected predictions  :")
        for k, v in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print("      %-44s %d" % (k, v))
    if problems:
        print("  problems              :")
        for p in problems[:20]:
            print("      - %s" % p)
        if len(problems) > 20:
            print("      ... and %d more" % (len(problems) - 20))

    if args.worst and n:
        worst = sorted(per_video.items(), key=lambda kv: kv[1]["F"])[:args.worst]
        print("  worst %d videos:" % len(worst))
        print("      %-24s %8s %8s %8s" % ("video_id", "F", "n_pred", "n_gt"))
        for vid, r in worst:
            print("      %-24s %8.4f %8d %8d" % (vid, r["F"], r["n_pred"], r["n_gt"]))

    if args.per_video:
        print("-" * 64)
        for vid, r in sorted(per_video.items()):
            print("  %-24s F=%.4f  P=%s  R=%s  n_pred=%d n_gt=%d"
                  % (vid, r["F"],
                     "%.4f" % r["P"] if r["P"] is not None else "   -  ",
                     "%.4f" % r["R"] if r["R"] is not None else "   -  ",
                     r["n_pred"], r["n_gt"]))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"score": final, "per_video": per_video,
                       "problems": problems}, f, ensure_ascii=False, indent=2)
        print("  report -> %s" % args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
