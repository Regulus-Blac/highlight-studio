#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-second spatial features from a frozen VLM / 冻结大模型的逐秒空间特征.

The contest crops follow the doubao subject center; zero-shot Qwen3-VL-8B
grounding reaches crop IoU ~0.72-0.75 against 0.69 for our detector rules and
0.84 for the doubao per-clip median. This keeps what a learned head needs to
do better: one frame per second through the VLM with the grounding question,
and at a few decoder layers the image-token grid reduced to
  col  [W', D]  mean over rows    (where along x)
  row  [H', D]  mean over columns (where along y)
plus the zero-shot bbox_2d it generates (0-1000 coordinates, NaN if none).
--no-generate runs the prompt forward only (no box, all NaN): for a backbone
whose grounding output is degraded (TimeLens-8B, 0.64) the grid states may
still place the subject, and it keeps the head on the D3T backbone (<= 9B).
One npz per video: col / row [secs, layers, cells, D] fp16, box [secs, 4].
逐秒一帧：图像 token 网格按列 / 按行平均，外加零样本定位框。

    python tools/spatial_feats.py --model $QWEN --gt data/train.with_video.jsonl --out-dir outputs/spf_qwen
    python tools/spatial_feats.py --model $QWEN --index $T/submission_index.jsonl --video-dir $T/video \\
        --out-dir outputs/spf_test_qwen
"""

import io
import os
import re
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from api_teacher import frames_1fps  # noqa: E402
from local_spatial import GROUND_PROMPT  # noqa: E402

BOX = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def frames_at_indices(path, indices, max_side):
    import cv2
    cap = cv2.VideoCapture(path)
    wanted, out, idx = set(indices), [], 0
    while idx <= max(indices):
        if not cap.grab():
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = max_side / float(max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append((idx, buf.tobytes()))
        idx += 1
    cap.release()
    assert [frame for frame, _ in out] == indices, path
    return out


def prepare_frames(row, args):
    return (frames_at_indices(row["video_path"], row['sampling_frame_indices'], args.max_side)
            if args.annotated_frames else frames_1fps(row["video_path"], args.max_side))


def main():
    ap = argparse.ArgumentParser(description="Per-second spatial VLM features (column / row token profiles)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid-dir", default=None, help="also save the first selected layer's 2D grid for candidate crops")
    ap.add_argument("--annotated-frames", action="store_true", help="use each row's exact sampling_frame_indices")
    ap.add_argument("--layers", default="18,27", help="decoder layer indices to keep")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-generate", action="store_true", help="forward pass only, no zero-shot box")
    ap.add_argument("--hflip", action="store_true", help="mirror every frame left-right (x centers become 1 - x)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": args.device},
                                                        attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    image_token = int(getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    merge = int(getattr(proc.image_processor, "merge_size", 2))
    layers = [int(x) for x in args.layers.split(",")]
    dec = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and "visual" not in name and "vision" not in name:
            if dec is None or len(mod) > len(dec):
                dec = mod
    cap = {}

    def hook(i):
        def f(_m, _inp, out):
            # generate() calls the layer once for the prompt (prefill) and then once per new token; keep the prefill
            if i not in cap:
                cap[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return f
    readout_layers = [18, 27] if args.annotated_frames else []
    for i in sorted(set(layers + readout_layers)):
        dec[i].register_forward_hook(hook(i))

    if args.index:
        rows = [json.loads(l) for l in open(args.index, encoding="utf-8") if l.strip()]
        for r in rows:
            r["video_path"] = os.path.join(args.video_dir, str(r["video_id"]) + ".mp4")
    else:
        rows = [json.loads(l) for l in open(args.gt, encoding="utf-8") if l.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    if args.grid_dir:
        os.makedirs(args.grid_dir, exist_ok=True)
    t0 = time.time()
    jobs = []
    for n, r in enumerate(rows, 1):
        dst = os.path.join(args.out_dir, "%s.npz" % r["video_id"])
        grid_dst = os.path.join(args.grid_dir, "%s.npz" % r["video_id"]) if args.grid_dir else None
        if os.path.exists(dst) and (grid_dst is None or os.path.exists(grid_dst)):
            continue
        jobs.append((n, r, dst, grid_dst))
    # Decode one video ahead while the current video's batches run on the GPU.
    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(prepare_frames, jobs[0][1], args) if jobs else None
    for j, (n, r, dst, grid_dst) in enumerate(jobs):
        current = nxt
        nxt = pool.submit(prepare_frames, jobs[j + 1][1], args) if j + 1 < len(jobs) else None
        try:
            frames = current.result()
        except Exception as e:  # noqa: BLE001
            print("[warn] %s: %s" % (r["video_id"], e))
            continue
        cols, rws, boxes, grids, readouts = [], [], [], [], []
        for a in range(0, len(frames), args.batch):
            chunk = frames[a:a + args.batch]
            text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": GROUND_PROMPT}]}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            imgs = [[Image.open(io.BytesIO(jpg)).convert("RGB")] for _, jpg in chunk]
            if args.hflip:   # left-right mirrored view (10-22, flip augmentation / test-time averaging)
                imgs = [[im[0].transpose(Image.Transpose.FLIP_LEFT_RIGHT)] for im in imgs]
            inputs = proc(text=[text] * len(chunk), images=imgs, images_kwargs={"do_resize": True},
                          text_kwargs={"padding": True}, return_tensors="pt").to(args.device)
            grid = inputs["image_grid_thw"].cpu().numpy()          # [b, 3] (t, h, w) in patches
            cap.clear()
            with torch.no_grad():
                if args.no_generate:
                    model(**inputs, logits_to_keep=1)
                else:
                    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            ids = inputs["input_ids"]
            for b in range(len(chunk)):
                gh, gw = int(grid[b][1]) // merge, int(grid[b][2]) // merge
                pos = (ids[b] == image_token).nonzero().flatten()
                per_layer_c, per_layer_r = [], []
                for i in layers:
                    h = cap[i][b, pos].float().reshape(gh, gw, -1)   # prefill hidden states of the image tokens
                    if args.grid_dir and i == layers[0]:
                        grids.append(h.half().cpu().numpy())
                    per_layer_c.append(h.mean(0).cpu().numpy())     # [gw, D]
                    per_layer_r.append(h.mean(1).cpu().numpy())     # [gh, D]
                cols.append(np.stack(per_layer_c).astype(np.float16))
                rws.append(np.stack(per_layer_r).astype(np.float16))
                if args.annotated_frames:
                    readouts.append(np.stack([cap[i][b, -1].float().cpu().numpy() for i in readout_layers]).astype(np.float16))
            if args.no_generate:
                boxes += [[np.nan] * 4] * len(chunk)
                continue
            for reply in proc.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True):
                m = BOX.search(reply)
                boxes.append([float(m.group(k)) for k in range(1, 5)] if m else [np.nan] * 4)
        if not cols:
            continue
        # all frames of one video share the grid size (same resolution)
        frame_info = {'frame_indices': np.array([idx for idx, _ in frames])} if args.annotated_frames else {}
        readout_info = {'readout': np.stack(readouts), 'readout_layers': np.array(readout_layers)} if args.annotated_frames else {}
        if not os.path.exists(dst):
            np.savez(dst, col=np.stack(cols), row=np.stack(rws), box=np.asarray(boxes, np.float32), **frame_info, **readout_info)
        if grid_dst:
            np.savez(grid_dst, grid=np.stack(grids), layer=np.array(layers[0]), **frame_info)
        if n % 20 == 0:
            print("  [%d/%d] %.0fs" % (n, len(rows), time.time() - t0), flush=True)
    pool.shutdown()
    if args.device.startswith("cuda"):
        print("GPU peak reserved %.0f MiB; allocated %.0f MiB" %
              (torch.cuda.max_memory_reserved() / 2**20, torch.cuda.max_memory_allocated() / 2**20), flush=True)
    print("done in %.1f min -> %s" % ((time.time() - t0) / 60, args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
