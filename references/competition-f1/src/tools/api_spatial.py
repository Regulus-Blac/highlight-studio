#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-second crop-following subject center from a hosted VLM, TRAINING videos only / 在线大模型逐秒给出主体中心.

The contest crops sit exactly on the doubao subject-center trajectory (IoU
0.997 on the labeled clips), and our detector rules reach IoU 0.69. This asks a
hosted vision model for the same thing -- the point a 9:16 vertical crop should
follow, one per second -- so it can be scored against the contest crops on the
labeled val clips (placed with the same geometry as reframe.py).
NEVER the test set (refused): test data must not go to third-party services.

    python tools/api_spatial.py --gt data/train.with_video.jsonl --split val --labeled-only \\
        --model qwen/qwen3.8-flash --price-in 0.8 --price-out 2.7 --out outputs/teacher/spatial_qwen38f_val.jsonl
    python tools/api_spatial.py --score outputs/teacher/spatial_qwen38f_val.jsonl --gt data/train.with_video.jsonl
"""

import os
import re
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from api_teacher import ask, frames_1fps  # noqa: E402

PROMPT = (
    "你是一名短视频剪辑师。上面是同一段横屏视频按每秒 1 帧抽取的 {n} 张画面，按时间顺序排列，每张前面标了它对应的秒数。\n"
    "现在要把它裁成竖屏（9:16）短视频：裁剪框高度等于画面高度，只能左右移动。请为每一秒给出裁剪框应该对准的主体中心的"
    "水平位置 center_x（0 表示画面最左边，1 表示最右边，保留三位小数）。主体是这一秒最值得观众看到的人或物"
    "（通常是正在说话、动作或被镜头关注的人，看人时以脸和上半身为准）；画面里没有明确主体时给 0.5。\n"
    "只输出 JSON，不要任何解释：{{\"timeline\": [{{\"time_sec\": 0, \"center_x\": 0.5}}, ...]}}，"
    "timeline 必须按顺序覆盖第 0 秒到第 {last} 秒，共 {n} 项。")

# y axis: a portrait source cropped to 16:9 (full width, the box only moves up / down). The contest training clips are
# all landscape (doubao center_y is 0.5 for half the seconds), so portrait videos need their own teacher labels.
PROMPT_Y = (
    "你是一名短视频剪辑师。上面是同一段竖屏视频按每秒 1 帧抽取的 {n} 张画面，按时间顺序排列，每张前面标了它对应的秒数。\n"
    "现在要把它裁成横屏（16:9）视频：裁剪框宽度等于画面宽度，只能上下移动。请为每一秒给出裁剪框应该对准的主体中心的"
    "垂直位置 center_y（0 表示画面最上边，1 表示最下边，保留三位小数）。主体是这一秒最值得观众看到的人或物"
    "（通常是正在说话、动作或被镜头关注的人，看人时以脸和上半身为准）；画面里没有明确主体时给 0.5。\n"
    "只输出 JSON，不要任何解释：{{\"timeline\": [{{\"time_sec\": 0, \"center_y\": 0.5}}, ...]}}，"
    "timeline 必须按顺序覆盖第 0 秒到第 {last} 秒，共 {n} 项。")


def parse(text, n, key="center_x"):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    pts = {}
    for e in d.get("timeline") or []:
        try:
            pts[int(round(float(e["time_sec"])))] = float(e[key])
        except (KeyError, TypeError, ValueError):
            continue
    if len(pts) < max(2, n // 2):
        return None
    xs = np.array(sorted(pts))
    return np.interp(np.arange(n), xs, np.clip([pts[x] for x in xs], 0, 1)).round(4).tolist()


def label(args):
    if "test" in os.path.abspath(args.gt).lower() or "semifinal" in os.path.abspath(args.gt).lower():
        sys.exit("[refuse] this looks like the test set; test data must not go to third-party services")
    key = open(args.key_file, encoding="utf-8").read().strip()
    rows = [json.loads(l) for l in open(args.gt, encoding="utf-8") if l.strip()]
    if args.split:
        rows = [r for r in rows if r.get("dataset_split") == args.split]
    if args.labeled_only:
        rows = [r for r in rows if r.get("cropRois")]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["video_id"] for l in open(args.out) if l.strip() and "error" not in json.loads(l)}
    todo = [r for r in rows if r["video_id"] not in done][:args.limit or None]
    lock, st = threading.Lock(), {"cost": 0.0, "ok": 0, "err": 0}
    print("clips to label: %d, model %s" % (len(todo), args.model), flush=True)
    import api_teacher
    api_teacher.PROMPT = PROMPT_Y if args.axis == "y" else PROMPT  # ask() formats api_teacher.PROMPT
    key_name = "center_" + args.axis

    def work(r):
        with lock:
            if st["cost"] >= args.budget:
                return
        rec = {"video_id": r["video_id"], "model": args.model}
        t0 = time.time()
        try:
            fr = frames_1fps(r["video_path"], args.max_side)
            text, usage = ask(args, key, fr)
            cx = parse(text, len(fr), key_name)
            rec.update({"n_sec": len(fr), "usage": usage, "latency": round(time.time() - t0, 1)})
            if cx is None:
                rec["error"], rec["raw"] = "unparsable reply", text[:400]
            else:
                rec[key_name] = cx
        except Exception as e:  # noqa: BLE001
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        with lock:
            u = rec.get("usage") or {}
            st["cost"] += (int(u.get("prompt_tokens") or 0) * args.price_in + int(u.get("completion_tokens") or 0) * args.price_out) / 1e6
            st["err" if "error" in rec else "ok"] += 1
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print("done: %d ok, %d errors, CNY %.3f" % (st["ok"], st["err"], st["cost"]))


def score(args):
    """Crop IoU on the GT frames of the labeled clips: teacher center vs the contest crops."""
    from highlight.geometry import place_1d, safe_crop_size, dof_range
    from highlight_metric import derive_h, iou_xywh
    import cv2
    rows = {json.loads(l)["video_id"]: json.loads(l) for l in open(args.gt, encoding="utf-8")}
    for path in args.score:
        ious, ious_c = [], []
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)
            if "error" in t or not rows.get(t["video_id"], {}).get("cropRois"):
                continue
            r = rows[t["video_id"]]
            cap = cv2.VideoCapture(r["video_path"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            tw, th = [float(v) for v in r["targetRatioWH"]]
            w, h = safe_crop_size(W, H, tw, th)
            R = dof_range(W, H, tw, th)
            cx = np.asarray(t["center_x"], float)
            for f, (gx, gy, gw, gh) in r["cropRois"]:
                c = float(np.interp(f / fps, np.arange(len(cx)), cx)) * W
                u = min(1.0, max(0.0, (c - w / 2.0) / R)) if R > 0 else 0.5
                x0, y0, w0 = place_1d(u, W, H, tw, th)
                ious.append(iou_xywh([x0, y0, w0, derive_h(w0, [tw, th])], [gx, gy, gw, gh]))
                x1, y1, w1 = place_1d(0.5, W, H, tw, th)
                ious_c.append(iou_xywh([x1, y1, w1, derive_h(w1, [tw, th])], [gx, gy, gw, gh]))
        print("%-48s crop IoU %.3f (center %.3f, %d GT frames)" % (os.path.basename(path), np.mean(ious), np.mean(ious_c), len(ious)))


def main():
    ap = argparse.ArgumentParser(description="Crop-following subject centers from a hosted VLM (training videos only)")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--split", default=None)
    ap.add_argument("--labeled-only", action="store_true", help="only clips with cropRois")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model")
    ap.add_argument("--axis", choices=("x", "y"), default="x", help="y: portrait videos, center_y for a 16:9 crop")
    ap.add_argument("--base-url", default="https://api.ppinfra.com/v3/openai")
    ap.add_argument("--key-file", default="/root/sj-fs/.ppio_key")
    ap.add_argument("--out")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--json-mode", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--score", nargs="*", default=None, help="score these label files and exit")
    args = ap.parse_args()
    if args.score:
        return score(args) or 0
    if not (args.model and args.out):
        ap.error("--model and --out are required for labeling")
    label(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
