#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-second crop-following subject center from a LOCAL VLM (same prompt as api_spatial.py) / 本地大模型逐秒给出主体中心.

The hosted Qwen3.7 Plus reached crop IoU 0.752 on the labeled val clips against
0.688 for our detector rules. A local model is allowed at test time, so this
runs the same prompt on TimeLens-8B (already in the pipeline, no extra
parameters) or any transformers image-text-to-text checkpoint, generating the
JSON timeline greedily. Output lines match api_spatial.py (center_x per
second), so `api_spatial.py --score` scores them.
与 api_spatial.py 同一提示词，本地生成；输出格式相同。

    python tools/local_spatial.py --model $TIMELENS --gt data/train.with_video.jsonl --split val --labeled-only \\
        --out outputs/teacher/spatial_local_timelens_val.jsonl
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from api_teacher import frames_1fps  # noqa: E402
from api_spatial import PROMPT, parse  # noqa: E402


GROUND_PROMPT = ("Locate the main subject that a vertical 9:16 short-video crop of this frame should keep in view "
                 "(the person or object viewers should watch; for a person, the face and upper body). "
                 "Output its bbox coordinates in JSON format.")


def ground(model, proc, ik, frames, device, batch=8):
    """Qwen3-VL native grounding, one frame per prompt -> center x (0..1) per frame (0.5 when nothing parses)."""
    import io
    import re
    import torch
    from PIL import Image
    out_cx = []
    for a in range(0, len(frames), batch):
        chunk = frames[a:a + batch]
        texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": GROUND_PROMPT}]}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False) for _ in chunk]
        imgs = [[Image.open(io.BytesIO(jpg)).convert("RGB")] for _, jpg in chunk]
        inputs = proc(text=texts, images=imgs, images_kwargs=ik, text_kwargs={"padding": True}, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        for reply in proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True):
            m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", reply)
            out_cx.append(round(min(1.0, max(0.0, (float(m.group(1)) + float(m.group(3))) / 2000.0)), 4) if m else 0.5)
    return out_cx


def main():
    ap = argparse.ArgumentParser(description="Subject-center timeline from a local VLM")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--index", default=None, help="test index (with --video-dir) instead of --gt")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--labeled-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-sec", type=int, default=64, help="longer videos are split into chunks of this many seconds")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=("timeline", "grounding"), default="timeline",
                    help="timeline: one JSON timeline for the whole clip; grounding: one bbox_2d per frame")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": args.device},
                                                        attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    ik = {"do_resize": True}

    if args.index:
        rows = [json.loads(l) for l in open(args.index, encoding="utf-8") if l.strip()]
        for r in rows:
            r["video_path"] = os.path.join(args.video_dir, str(r["video_id"]) + ".mp4")
    else:
        rows = [json.loads(l) for l in open(args.gt, encoding="utf-8") if l.strip()]
        if args.split:
            rows = [r for r in rows if r.get("dataset_split") == args.split]
        if args.labeled_only:
            rows = [r for r in rows if r.get("cropRois")]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["video_id"] for l in open(args.out) if l.strip() and "error" not in json.loads(l)}
    rows = [r for r in rows if r["video_id"] not in done][:args.limit or None]
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as fout:
        for i, r in enumerate(rows, 1):
            rec = {"video_id": r["video_id"], "model": os.path.basename(os.path.normpath(args.model))}
            try:
                fr = frames_1fps(r["video_path"], args.max_side)
                if not fr:
                    raise IOError("no frames")
                cx = []
                if args.mode == "grounding":
                    cx = ground(model, proc, ik, fr, args.device)
                for a in (range(0, len(fr), args.max_sec) if args.mode == "timeline" else []):
                    chunk = fr[a:a + args.max_sec]
                    content = []
                    for k, (_, jpg) in enumerate(chunk):
                        content.append({"type": "text", "text": "第 %d 秒" % k})
                        content.append({"type": "image"})
                    content.append({"type": "text", "text": PROMPT.format(n=len(chunk), last=len(chunk) - 1)})
                    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                    import io
                    imgs = [Image.open(io.BytesIO(jpg)).convert("RGB") for _, jpg in chunk]
                    inputs = proc(text=[text], images=[imgs], images_kwargs=ik, return_tensors="pt").to(args.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=24 * len(chunk) + 64, do_sample=False)
                    reply = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                    part = parse(reply, len(chunk))
                    if part is None:
                        raise ValueError("unparsable reply: " + reply[:200])
                    cx += part
                rec.update({"n_sec": len(fr), "center_x": cx})
            except Exception as e:  # noqa: BLE001
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print("  [%d/%d] %s %s (%.1fs)" % (i, len(rows), r["video_id"], "ERR " + rec["error"][:80] if "error" in rec else "ok",
                                              time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
