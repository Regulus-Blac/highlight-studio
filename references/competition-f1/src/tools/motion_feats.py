#!/usr/bin/env python3
"""Dense VideoMAE features on the same one-second grid as infer_yesno.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from infer_yesno import (clip_times, done_ids, load_jobs, make_windows, read_frames,
                         step_centers, time_to_frame, video_info)


def extract(path, model, processor, args):
    n, fps, width, height = video_info(path)
    if n <= 1:
        raise IOError('cannot read video: ' + path)
    windows = make_windows(n / fps, 32.0)
    centers = step_centers(windows)
    ids = [[time_to_frame(t, fps, n) for t in clip_times(c, args.clip_sec, 16 / args.clip_sec, n / fps)]
           for c in centers]
    frames = read_frames(path, [f for row in ids for f in row], 448)
    have = np.array(sorted(frames))
    def prepare(start):
        clips = [[frames[int(have[np.argmin(abs(have - f))])] for f in row]
                 for row in ids[start:start + args.batch]]
        return clips, processor(clips, return_tensors='pt')

    pooled, change = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for clips, inputs in pool.map(prepare, range(0, len(ids), args.batch)):
            inputs = inputs.to('cuda', dtype=torch.float16)
            with torch.inference_mode():
                hidden = model.videomae(**inputs).last_hidden_state
                # Eight temporal tubelets; retain both appearance and beginning-to-end change.
                temporal = hidden.reshape(len(clips), 8, -1, hidden.shape[-1]).mean(2)
                pooled.append(model.fc_norm(temporal.mean(1)).float().cpu().numpy())
                change.append((temporal[:, -2:].mean(1) - temporal[:, :2].mean(1)).float().cpu().numpy())
    features = {'vis': np.concatenate(pooled)[:, None].astype(np.float16),
                'last': np.concatenate(change)[:, None].astype(np.float16),
                'margin': np.zeros(len(ids), np.float32)}
    record = {'fps': fps, 'n_frames': n, 'width': width, 'height': height,
              'windows': windows, 'p_sec': [0.5] * len(ids), 'feature_only': True}
    return features, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/sj-fs/models/videomae-base-kinetics')
    ap.add_argument('--gt')
    ap.add_argument('--index')
    ap.add_argument('--video-dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--clip-sec', type=float, default=2.0)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--num-videos', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.feats, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model = VideoMAEForVideoClassification.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    params = sum(p.numel() for p in model.parameters())
    print('VideoMAE parameters:', params, flush=True)
    skip = done_ids(args.out)
    jobs = [j for j in load_jobs(args) if j[0] not in skip]
    start = time.time()
    with open(args.out, 'a') as out:
        for i, (vid, ratio, path) in enumerate(jobs, 1):
            row = {'video_id': vid, 'targetRatioWH': ratio,
                   'scorer': {'model': args.model, 'parameters': params, 'clip_sec': args.clip_sec,
                              'frames_per_clip': 16, 'weights': 'frozen', 'feature_only': True}}
            try:
                features, meta = extract(path, model, processor, args)
                row.update(meta)
                np.savez(os.path.join(args.feats, vid + '.npz'), **features)
                message = '%d steps' % len(row['p_sec'])
            except Exception as e:
                row.update(error=str(e), predictions=[])
                message = 'error: ' + str(e)
            out.write(json.dumps(row) + '\n')
            out.flush()
            print('[%d/%d] %s %s %.1fs' % (i, len(jobs), vid, message, time.time() - start), flush=True)


if __name__ == '__main__':
    main()
