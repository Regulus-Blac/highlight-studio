#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-second highlight scoring inference / 逐秒高光打分推理.

For every video the model emits one digit per one-second step (see
`build_sft_temporal.py --task scores`). Decoding is constrained: the first K
steps may only produce the ten digit tokens, then <|im_end|>, so the output
always parses. At every step the log-probabilities of the ten digits are kept;
`decode_scores.py` turns them into frame selections.
约束解码：前 K 步只允许数字 token，保证可解析；保存每步 10 个数字的 log 概率。

Videos longer than 32 s (training clips are 5-32 s) are split into equal
windows and scored window by window. 超过 32 s 的视频分窗推理。

`--overlap-stride S` scores long videos on overlapping 32 s windows started
every S seconds instead, and averages the digit distributions of every step
that covers a second. The record keeps the usual non-overlapping `windows`
layout, so decode_scores.py and ensemble_probs.py read it unchanged; the
windows actually scored are listed in `infer_windows`. Each second then sees
context on both sides, and no window boundary cuts a highlight blind.
重叠窗：每秒取所有覆盖它的窗口的平均分布，输出布局不变。

    # validation, untuned base model (omit --adapter)
    python tools/infer_scores.py --model E:/AICTrain/models/Qwen3.5-4B \
        --index outputs/val_index_scores.json --video-dir data/clips \
        --out outputs/scores_val_base.probs.jsonl
    python tools/decode_scores.py --probs outputs/scores_val_base.probs.jsonl --tune

Output: one JSON line per video with fps / n_frames / size, `windows`
[[start_sec, end_sec, n_steps], ...], `digit_logp` (one 10-vector per step),
`p_sec` (expected coverage per step at temperature 1) and the greedy `raw`
digits. A failed video gets an `error` field; decoding falls back to the whole
video for it.
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import temporal_vl as tv  # noqa: E402
from baseline_qwen import load_index, video_meta  # noqa: E402
from build_sft_temporal import score_steps, step_bounds  # noqa: E402
from decode_scores import DIGIT_VALUES, MAX_WINDOW_SEC, make_windows, step_centers  # noqa: E402


def overlap_windows(duration, max_sec, stride):
    """Inference windows (start, end) of <= max_sec every `stride` s, the last
    one flush with the end; a single window when the video fits in one."""
    if duration <= max_sec + 1e-9 or stride <= 0:
        return [(0.0, duration)]
    out, s = [], 0.0
    while s + max_sec < duration - 1e-9:
        out.append((s, s + max_sec))
        s += stride
    out.append((max(0.0, duration - max_sec), duration))
    return out


def merge_overlapping(layout, scored):
    """Average digit distributions onto the non-overlapping `layout`.

    scored: [((start, end), logp [k x 10])]. Every layout step takes the mean
    probability vector of all scored steps whose span contains its center.
    -> digit log-probs [n_layout_steps x 10].
    """
    centers = step_centers(layout)
    acc = np.zeros((len(centers), 10))
    cnt = np.zeros(len(centers))
    for (s, e), lp in scored:
        k = len(lp)
        probs = np.exp(np.asarray(lp, dtype=np.float64))
        for j in range(k):
            a, b = step_bounds(j, k, e - s)
            hit = (centers >= s + a) & (centers < s + b)
            acc[hit] += probs[j]
            cnt[hit] += 1
    if (cnt == 0).any():  # cannot happen when the windows cover the video
        raise ValueError("%d steps not covered by any window" % int((cnt == 0).sum()))
    return np.log(np.clip(acc / cnt[:, None], 1e-12, None))


def score_window(model, processor, path, start, end, n_steps, digit_ids, end_id):
    """-> (digit log-probs [n_steps x 10], greedy digit string) for one window."""
    inputs = tv.build_inputs(processor, path, end - start, task="scores",
                             start_sec=start).to("cuda")
    plen = inputs["input_ids"].shape[1]

    def allowed(_batch_id, ids):
        return digit_ids if ids.shape[-1] - plen < n_steps else [end_id]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n_steps + 1, do_sample=False,
                             prefix_allowed_tokens_fn=allowed, output_logits=True,
                             return_dict_in_generate=True)
    # Raw logits (before the allowed-token mask), restricted to the digits.
    # 取原始 logits 中 10 个数字的部分。
    logits = torch.stack([out.logits[k][0, digit_ids] for k in range(n_steps)]).float()
    logp = torch.log_softmax(logits, dim=-1)
    raw = processor.batch_decode(out.sequences[:, plen:plen + n_steps],
                                 skip_special_tokens=True)[0]
    return logp.cpu().tolist(), raw


def main():
    ap = argparse.ArgumentParser(description="Per-second highlight scoring inference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir")
    ap.add_argument("--index", required=True, help="JSON [{video_id, targetRatioWH}]")
    ap.add_argument("--video-dir", default="data/clips")
    ap.add_argument("--out", required=True, help="*.probs.jsonl")
    ap.add_argument("--max-pixels", type=int, default=tv.SCORE_MAX_PIXELS,
                    help="pixels per frame; must match training")
    ap.add_argument("--min-pixels", type=int, default=tv.SCORE_MIN_PIXELS)
    ap.add_argument("--max-window-sec", type=float, default=MAX_WINDOW_SEC)
    ap.add_argument("--overlap-stride", type=float, default=0.0,
                    help="long videos: overlapping windows every S s (0 = off, e.g. 16)")
    ap.add_argument("--quant-vision", action="store_true")
    ap.add_argument("--no-quant", action="store_true",
                    help="bf16 model (must match training's --no-quant)")
    ap.add_argument("--num-videos", type=int, default=0)
    args = ap.parse_args()

    index = load_index(args.index)
    if args.num_videos > 0:
        index = index[:args.num_videos]
    processor = tv.load_processor(args.model, max_pixels=args.max_pixels,
                                  min_pixels=args.min_pixels)
    model = tv.load_model(args.model, adapter=args.adapter,
                          quant_vision=args.quant_vision, quant=not args.no_quant)
    model.eval()
    tok = processor.tokenizer
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    end_id = tok.convert_tokens_to_ids("<|im_end|>")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stats = {"videos": 0, "windows": 0, "failed": 0}
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fout:
        for vi, (vid, (tw, th)) in enumerate(index, 1):
            stats["videos"] += 1
            rec = {"video_id": vid, "targetRatioWH": [int(tw), int(th)]}
            path = os.path.join(args.video_dir, vid + ".mp4")
            try:
                n_frames, fps, W, H = video_meta(path)
                if n_frames <= 1 or W <= 0:
                    raise IOError("cannot read video: %s" % path)
                rec.update(fps=fps, n_frames=n_frames, width=W, height=H)
                duration = n_frames / float(fps)
                windows = make_windows(duration, args.max_window_sec)
                logp, raw = [], ""
                infer = overlap_windows(duration, args.max_window_sec, args.overlap_stride)
                if len(infer) > len(windows):
                    scored = []
                    for s, e in infer:
                        lp, _ = score_window(model, processor, path, s, e, score_steps(e - s),
                                             digit_ids, end_id)
                        scored.append(((s, e), lp))
                        stats["windows"] += 1
                    merged = merge_overlapping(windows, scored)
                    logp = merged.tolist()
                    raw = "".join(str(int(d)) for d in merged.argmax(axis=1))
                    rec["infer_windows"] = [[round(s, 4), round(e, 4)] for s, e in infer]
                else:
                    for s, e, k in windows:
                        lp, r = score_window(model, processor, path, s, e, k, digit_ids, end_id)
                        logp += lp
                        raw += r
                        stats["windows"] += 1
                rec["windows"] = [[round(s, 4), round(e, 4), k] for s, e, k in windows]
                rec["digit_logp"] = [[round(v, 3) for v in row] for row in logp]
                probs = torch.tensor(logp).exp() @ torch.tensor(DIGIT_VALUES, dtype=torch.float32)
                rec["p_sec"] = [round(float(v), 3) for v in probs]
                rec["raw"] = raw
                msg = "%.1fs x%d -> %s" % (duration, len(rec.get("infer_windows") or windows), raw)
            except Exception as e:  # noqa: BLE001  one bad video must not stop the batch
                stats["failed"] += 1
                rec["error"] = "%s: %s" % (type(e).__name__, e)
                msg = "[error] %s" % rec["error"]
                torch.cuda.empty_cache()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print("  [%d/%d] %s %s" % (vi, len(index), vid, msg), flush=True)

    print("done: %s in %.1f min -> %s" % (stats, (time.time() - t0) / 60, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
