#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal-SFT inference -> submission file / 时序 SFT 推理并生成提交文件.

The model picks the highlight segments; every frame inside them gets the
largest target-ratio box at the frame center (the spatial labels are too weak
to learn, see build_sft_temporal.py). 时序由模型给出，空间用居中最大框。

    # validation (57 clips, GT in outputs/val_gt.jsonl)
    python tools/infer_temporal.py \
        --model E:/AICTrain/models/Qwen2.5-VL-3B-Instruct \
        --adapter outputs/sft_qlora_hl90/epoch3 \
        --index outputs/val_index.json --video-dir data/clips \
        --out outputs/pred_val_sft.jsonl
    python tools/evaluate.py --gt outputs/val_gt.jsonl \
        --pred outputs/pred_val_sft.jsonl --index outputs/val_index.json

Omit --adapter to score the untuned base model with the same prompt.
不加 --adapter 即为未微调基座的对照。
"""

import os
import sys
import re
import json
import time
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import temporal_vl as tv  # noqa: E402
from baseline_qwen import (compute_crop_size, load_index, merge_segments,  # noqa: E402
                           parse_segments_sec, sec_segments_to_frames, video_meta)


_START_END = re.compile(
    r'"start[a-z_]*"\s*:\s*(-?\d+\.?\d*)\s*,\s*"end[a-z_]*"\s*:\s*(-?\d+\.?\d*)')


def parse_segments_any(text):
    """The trained JSON first, then untuned-model spellings such as
    [{"start_sec": a, "end_sec": b}]. 兼容未微调模型的其他写法。"""
    segs = parse_segments_sec(text)
    if not segs:
        segs = [(float(a), float(b)) for a, b in _START_END.findall(text)]
    return segs


def main():
    ap = argparse.ArgumentParser(description="Temporal SFT inference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir")
    ap.add_argument("--index", default="outputs/val_index.json")
    ap.add_argument("--video-dir", default="data/clips")
    ap.add_argument("--out", default="outputs/pred_val_sft.jsonl")
    ap.add_argument("--nframes", type=int, default=tv.DEFAULT_NFRAMES,
                    help="minimum frames per video")
    ap.add_argument("--min-sample-fps", type=float, default=0.0,
                    help="raise the frame count on long videos so sampling stays at "
                         "least this dense (training saw ~1.3 fps); 0 = fixed --nframes")
    ap.add_argument("--max-nframes", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=tv.DEFAULT_MAX_PIXELS)
    ap.add_argument("--quant-vision", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-videos", type=int, default=0)
    args = ap.parse_args()

    index = load_index(args.index)
    if args.num_videos > 0:
        index = index[:args.num_videos]
    processor = tv.load_processor(args.model, max_pixels=args.max_pixels)
    model = tv.load_model(args.model, adapter=args.adapter,
                          quant_vision=args.quant_vision)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stats = {"videos": 0, "missing": 0, "empty": 0, "frames": 0}
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fout, \
            open(args.out + ".raw.jsonl", "w", encoding="utf-8") as raw_f:
        for vi, (vid, (tw, th)) in enumerate(index, 1):
            rec = {"video_id": vid, "targetRatioWH": [int(tw), int(th)], "predictions": []}
            path = os.path.join(args.video_dir, vid + ".mp4")
            stats["videos"] += 1
            if not os.path.exists(path):
                stats["missing"] += 1
                print("  [skip] no video: %s" % path)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            n_frames, fps, W, H = video_meta(path)
            duration = n_frames / float(fps)
            # Long videos get more frames so the gap between frames stays near
            # what training saw. 长视频加帧，保持与训练相近的采样密度。
            nframes = args.nframes
            if args.min_sample_fps > 0:
                nframes = max(nframes, min(args.max_nframes,
                                           int(duration * args.min_sample_fps + 0.999)))
            inputs = tv.build_inputs(processor, path, duration, nframes=nframes)
            inputs = inputs.to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0]
            segs_sec = parse_segments_any(text)
            segs = merge_segments(sec_segments_to_frames(segs_sec, fps, n_frames), n_frames)
            raw_f.write(json.dumps({"video_id": vid, "raw": text, "segments_frame": segs},
                                   ensure_ascii=False) + "\n")

            cw, ch = compute_crop_size(W, H, tw, th)
            x, y = (W - cw) // 2, (H - ch) // 2
            for a, b in segs:
                for f in range(a, b + 1):
                    rec["predictions"].append({"frame": f, "bboxes": [x, y, cw]})
            if not rec["predictions"]:
                stats["empty"] += 1
            stats["frames"] += len(rec["predictions"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print("  [%d/%d] %s %.1fs -> %s" % (vi, len(index), vid, duration, text.strip()),
                  flush=True)

    print("done: %s in %.1f min -> %s" % (stats, (time.time() - t0) / 60, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
