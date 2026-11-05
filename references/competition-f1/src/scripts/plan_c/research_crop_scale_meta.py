#!/usr/bin/env python3
"""Read video properties only; select the full16:9 geometry supported by training."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from reframe import geometry


ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, required=True)
ap.add_argument('--video-dir', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
selected, counts, frames, all_metadata = [], Counter(), 0, []
for row in map(json.loads, args.index.open()):
    path = args.video_dir / (str(row['video_id']) + '.mp4')
    cap = cv2.VideoCapture(str(path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    g = geometry(width, height, row['targetRatioWH'])
    kind = 'full' if g is None else g[0]
    counts[kind + '_' + ':'.join(map(str, row['targetRatioWH']))] += 1
    meta = dict(video_id=str(row['video_id']), video_path=str(path), width=width, height=height,
                fps=fps, n_frames=n, duration=n / fps, targetRatioWH=row['targetRatioWH'], geometry=kind)
    all_metadata.append(meta)
    if kind == 'full' and row['targetRatioWH'] == [16, 9]:
        meta = dict(meta, sampling_frame_indices=[round(k * fps) for k in range(math.floor((n - 1) / fps) + 1)])
        selected.append(meta)
        frames += len(meta['sampling_frame_indices'])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(''.join(json.dumps(r) + '\n' for r in selected))
args.out.with_suffix('.all_metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in all_metadata))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = dict(selection='geometry=None and targetRatioWH=[16,9], matching public-human training support',
              index_videos=len(all_metadata), geometry_counts=dict(counts), selected_videos=len(selected),
              sampled_frames=frames, pixels_read_or_viewed=False, API_calls=False, training=False,
              inputs_sha256={str(p): sha(p) for p in [Path(__file__), args.index]},
              rows_sha256=sha(args.out))
args.out.with_suffix('.meta.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
