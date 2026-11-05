#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-second highlight scores from a hosted VLM, for TRAINING videos only / 用在线大模型给训练视频逐秒打分.

The contest labels came from a closed model (doubao-seed-2-1-pro: a per-second
highlight_score timeline at 1 fps, thresholded into segments). This asks an
OpenAI-compatible vision model (PPIO gateway by default) for the same kind of
timeline, so it can (1) be compared with the contest labels on the labeled
clips and (2) if it agrees well enough, label more public long videos.
NEVER point it at the test set: test data must not go to third-party services.
只用于训练视频（公开数据）；测试集视频绝不能发给第三方服务。

Frames: one per second at t = 0, 1, 2, ... (the timeline's time_sec), longest
side --max-side, JPEG, each preceded by a "第 k 秒" text tag.
Output: one JSON line per video (scores per second, summary, token usage,
latency); finished ids are skipped on rerun. The API key is read from a file
and never printed.

    python tools/api_teacher.py --gt data/train.with_video.jsonl --split val --limit 10 \\
        --model deepseek/deepseek-v4-flash-vision-exp --price-in 3 --price-out 9 \\
        --out outputs/teacher/dsv4vis_val.jsonl --budget 20
"""

import os
import re
import sys
import json
import time
import base64
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import requests

PROMPT = (
    "你是一名短视频剪辑师。上面是同一段视频按每秒 1 帧抽取的 {n} 张画面，按时间顺序排列，每张前面标了它对应的秒数。\n"
    "请为每一秒打一个高光分 highlight_score（0 到 1，保留两位小数），表示这一秒是否值得剪进这段视频的高光短片："
    "内容精彩、动作或情绪的高潮、信息量大、主体清楚、画面好看的给高分；过渡、等待、重复、模糊、黑屏、无聊的给低分。"
    "按这段视频自身的内容比较，不同秒之间要拉开差距。\n"
    "只输出 JSON，不要任何解释：{{\"timeline\": [{{\"time_sec\": 0, \"highlight_score\": 0.35}}, ...], "
    "\"summary\": \"一句话概括视频内容\"}}。timeline 必须按顺序覆盖第 0 秒到第 {last} 秒，共 {n} 项。")


def frames_1fps(path, max_side):
    """[(sec, jpeg bytes)] at t = 0, 1, 2, ... (sequential decode, no seeking)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    want = {int(round(k * fps)): k for k in range(int(np.floor((n - 1) / fps)) + 1)} if n > 0 else {}
    out, idx = [], 0
    try:
        while want and idx <= max(want):
            if not cap.grab():
                break
            if idx in want:
                ok, fr = cap.retrieve()
                if ok:
                    h, w = fr.shape[:2]
                    s = max_side / float(max(h, w))
                    if s < 1.0:
                        fr = cv2.resize(fr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        out.append((want[idx], buf.tobytes()))
            idx += 1
    finally:
        cap.release()
    return out


def parse_timeline(text, n):
    """Model reply -> per-second scores (length n) or None."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    tl = d.get("timeline") or []
    pts = {}
    for e in tl:
        try:
            pts[int(round(float(e["time_sec"])))] = float(e["highlight_score"])
        except (KeyError, TypeError, ValueError):
            continue
    if len(pts) < max(2, n // 2):
        return None, d.get("summary", "")
    xs = np.array(sorted(pts))
    ys = np.clip(np.array([pts[x] for x in xs]), 0.0, 1.0)
    return np.interp(np.arange(n), xs, ys).round(4).tolist(), d.get("summary", "")


def ask(args, key, frames):
    n = len(frames)
    content = []
    for sec, jpg in frames:
        content.append({"type": "text", "text": "第 %d 秒" % sec})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
    content.append({"type": "text", "text": PROMPT.format(n=n, last=n - 1)})
    body = {"model": args.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens, "temperature": 0.2}
    if args.json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(args.base_url.rstrip("/") + "/chat/completions", json=body, timeout=args.timeout,
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    if r.status_code != 200:
        raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:300]))
    d = r.json()
    msg = d["choices"][0]["message"]
    return (msg.get("content") or ""), d.get("usage") or {}


def main():
    ap = argparse.ArgumentParser(description="Per-second highlight timeline from a hosted VLM (training videos only)")
    ap.add_argument("--gt", required=True, help="rows with video_id + video_path (train.with_video.jsonl or ext_*.jsonl)")
    ap.add_argument("--split", default=None, help="keep rows with this dataset_split (e.g. val)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.ppinfra.com/v3/openai")
    ap.add_argument("--key-file", default="/root/sj-fs/.ppio_key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-sec", type=int, default=240, help="skip videos longer than this")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--json-mode", action="store_true", help="send response_format json_object")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--price-in", type=float, default=0.0, help="CNY per million input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="CNY per million output tokens")
    ap.add_argument("--budget", type=float, default=20.0, help="stop starting new videos past this cost (CNY)")
    args = ap.parse_args()

    if "test" in os.path.abspath(args.gt).lower() or "semifinal" in os.path.abspath(args.gt).lower():
        sys.exit("[refuse] this looks like the test set; test data must not go to third-party services")
    with open(args.key_file, "r", encoding="utf-8") as f:
        key = f.read().strip()
    rows = [json.loads(l) for l in open(args.gt, encoding="utf-8") if l.strip()]
    if args.split:
        rows = [r for r in rows if r.get("dataset_split") == args.split]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["video_id"] for l in open(args.out, encoding="utf-8")
                if l.strip() and "error" not in json.loads(l)}
    todo = [r for r in rows if r["video_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    lock, state = threading.Lock(), {"cost": 0.0, "in": 0, "out": 0, "ok": 0, "err": 0}
    print("videos to label: %d (already done %d), model %s" % (len(todo), len(done), args.model), flush=True)

    def work(r):
        with lock:
            if state["cost"] >= args.budget:
                return
        rec = {"video_id": r["video_id"], "model": args.model}
        t0 = time.time()
        try:
            fr = frames_1fps(r["video_path"], args.max_side)
            if not fr:
                raise IOError("no frames")
            if len(fr) > args.max_sec:
                raise ValueError("too long (%d s)" % len(fr))
            text, usage = ask(args, key, fr)
            scores, summary = parse_timeline(text, len(fr))
            rec.update({"n_sec": len(fr), "usage": usage, "latency": round(time.time() - t0, 1)})
            if scores is None:
                rec["error"] = "unparsable reply"
                rec["raw"] = text[:500]
            else:
                rec.update({"scores": scores, "summary": summary})
        except Exception as e:  # noqa: BLE001
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        with lock:
            u = rec.get("usage") or {}
            pi, po = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            state["in"] += pi
            state["out"] += po
            state["cost"] += (pi * args.price_in + po * args.price_out) / 1e6
            state["err" if "error" in rec else "ok"] += 1
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("  %s %s  in %d out %d  %.1fs  | total %d ok %d err, CNY %.3f"
                  % (r["video_id"], "ERR " + rec["error"][:80] if "error" in rec else "ok", pi, po,
                     rec.get("latency", 0), state["ok"], state["err"], state["cost"]), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print("done: %d ok, %d errors, tokens in %d / out %d, CNY %.3f"
          % (state["ok"], state["err"], state["in"], state["out"], state["cost"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
