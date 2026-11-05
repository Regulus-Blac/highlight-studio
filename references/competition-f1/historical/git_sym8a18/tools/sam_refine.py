#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine cached YOLO detections with SAM2 masks / 用 SAM2 掩码细化检测框.

The contest hint suggests crop boxes built from subject masks (SAM) rather than
detector boxes. This rewrites reframe.py's detection cache: for every sampled
frame of every row with a free axis, each detection box above --conf is used as
a SAM2 box prompt and replaced by the bounding box of its mask (same class /
confidence). The refined cache goes to --out-cache-dir under the same key, so
reframe.py / trim_frames.py run unchanged with `--cache-dir <out>`: the only
change is where the subject boxes come from.
只改"主体框从哪来"：检测框 -> SAM2 掩码外接框，其余流程不变。

    python tools/sam_refine.py --pred outputs/sub/D3_v2m_mrhisum.frames.jsonl --video-dir $T/video \\
        --config tools/configs/final_AM.json --model yolo11s.pt --sam /root/sj-fs/models/sam2.1-hiera-small \\
        --cache-dir outputs/det_cache --out-cache-dir outputs/det_cache_sam
"""

import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reframe as rf  # noqa: E402


class Sam:
    def __init__(self, path, device):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch, self.device = torch, device
        self.model = Sam2Model.from_pretrained(path).to(device).eval()
        self.proc = Sam2Processor.from_pretrained(path)

    def boxes(self, image, boxes):
        """RGB image, [[x1, y1, x2, y2], ...] -> mask bounding boxes (same order; None if the mask is empty)."""
        torch = self.torch
        inputs = self.proc(images=image, input_boxes=[[list(map(float, b)) for b in boxes]],
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        res = []
        for m in masks:
            m = np.asarray(m).reshape(-1, *np.asarray(m).shape[-2:])[0] > 0
            ys, xs = np.nonzero(m)
            res.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else None)
        return res


def main():
    ap = argparse.ArgumentParser(description="SAM2-refined detection cache for reframe.py")
    ap.add_argument("--pred", required=True, help="submission whose frames keyed the detection cache")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=rf.DEFAULT_MODEL, help="detector whose cache is read")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sam", required=True, help="SAM2 checkpoint dir (transformers format)")
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    ap.add_argument("--out-cache-dir", default="outputs/det_cache_sam")
    ap.add_argument("--conf", type=float, default=0.15, help="refine detections at or above this confidence")
    ap.add_argument("--max-boxes", type=int, default=6, help="per frame, highest confidence first")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    params = rf.load_params(**{k: v for k, v in cfg["reframe"].items() if k in rf.DEFAULT_PARAMS})
    tag = rf.detector_tag(args.model, args.imgsz)
    sam = Sam(args.sam, args.device)
    rows = [json.loads(l) for l in open(args.pred, encoding="utf-8") if l.strip()]
    os.makedirs(args.out_cache_dir, exist_ok=True)
    t0, done, refined, moved = time.time(), 0, 0, []
    for i, row in enumerate(rows, 1):
        frames = [int(p["frame"]) for p in row.get("predictions") or []]
        path = os.path.join(args.video_dir, str(row["video_id"]) + ".mp4")
        if not frames:
            continue
        W, H = rf.video_size(path)
        sampled = rf.sample_frames(frames, params["stride"])
        src = rf.cache_file(path, sampled, tag, args.cache_dir)
        dst = rf.cache_file(path, sampled, tag, args.out_cache_dir)
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        if rf.geometry(W, H, row["targetRatioWH"]) is None:
            # whole-frame box: nothing to refine, but trim_frames.py reads this cache too -> copy it unchanged
            with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            continue
        cache = json.load(open(src, encoding="utf-8"))
        want = {int(f): k for k, f in enumerate(cache["frames"])}
        for idx, fr in rf.iter_frames(path, sorted(want)):
            k = want[int(idx)]
            D = cache["dets"][k]
            pick = sorted([j for j, d in enumerate(D) if d[1] >= args.conf], key=lambda j: -D[j][1])[:args.max_boxes]
            if not pick:
                continue
            rgb = fr[:, :, ::-1].copy()
            new = sam.boxes(rgb, [D[j][2:6] for j in pick])
            for j, b in zip(pick, new):
                if b is None:
                    continue
                old = D[j][2:6]
                moved.append(abs((b[0] + b[2]) - (old[0] + old[2])) / 2.0 / W + abs((b[1] + b[3]) - (old[1] + old[3])) / 2.0 / H)
                D[j] = D[j][:2] + [round(v, 1) for v in b]
                refined += 1
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        done += 1
        if done % 20 == 0:
            print("  %d videos refined (row %d/%d, %.0fs)" % (done, i, len(rows), time.time() - t0), flush=True)
    print("videos %d, boxes refined %d, mean center shift %.3f of the frame (%.0fs)"
          % (done, refined, float(np.mean(moved)) if moved else 0.0, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
