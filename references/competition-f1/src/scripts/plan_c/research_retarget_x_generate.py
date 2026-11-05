#!/usr/bin/env python3
"""Production inference: only the two RetargetVid x heads; control models are separate."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_spatial_head import SpatialHead, load_clip, medfilt, predict


ap = argparse.ArgumentParser()
ap.add_argument('--rows', required=True)
ap.add_argument('--features', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
torch.set_num_threads(4)
rows = list(map(json.loads, open(args.rows)))
models, inputs = {}, {}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for res in (448, 672):
    p = Path(args.model) / f'head_x_{res}'
    cfg = json.loads((p / 'config.json').read_text())
    assert cfg['axis'] == 'x' and cfg['layer_idx'] == 0
    m = SpatialHead(cfg['D']).eval().requires_grad_(False)
    m.load_state_dict(torch.load(p / 'head.pt', map_location='cpu', weights_only=True))
    models[res] = m
    inputs[str(p / 'head.pt')] = sha(p / 'head.pt')
    inputs[str(p / 'config.json')] = sha(p / 'config.json')
count = sum(p.numel() for m in models.values() for p in m.parameters())
assert count == 4225538
results, errors, feature_sha = [], [], {}
for number, row in enumerate(rows, 1):
    vid = str(row['video_id'])
    try:
        expected = int(np.floor((row['n_frames'] - 1) / row['fps'])) + 1
        values = []
        for res, m in models.items():
            p = Path(args.features) / f'spf_restore_test_{res}' / (vid + '.npz')
            prof, zs = load_clip(p, 'x', 0)
            assert len(prof) == expected
            values.append([round(float(x), 4) for x in medfilt(predict(m, prof, zs, 'cpu'))])
            feature_sha[str(p)] = sha(p)
        fused = [round((x + y) / 2, 4) for x, y in zip(*values)]
        results.append(dict(video_id=vid, times=np.arange(expected, dtype=float).tolist(),
                            center_x=medfilt(fused).tolist()))
    except Exception as e:
        print('error', vid, repr(e), flush=True)
        errors.append(dict(video_id=vid, error=str(e)))
        results.append(dict(video_id=vid, predictions=[]))
    if number % 20 == 0:
        print('RETARGET_X_GENERATE', number, len(rows), flush=True)
out = Path(args.out)
out.write_text(''.join(json.dumps(r) + '\n' for r in results))
inputs[args.rows] = sha(args.rows)
inputs[str(Path(__file__))] = sha(__file__)
inputs['tools/train_spatial_head.py'] = sha('tools/train_spatial_head.py')
report = dict(videos=len(rows), errors=errors, loaded_head_parameters=count, control_models_loaded=False,
              full_pipeline_parameters=8926500280,
              policy='per-resolution median+round4 -> Python round(mean,4) -> second median -> integer-second interpolation -> legal crop',
              inputs_sha256=inputs, features_sha256=feature_sha, centers_sha256=sha(out))
out.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print('RETARGET_X_PRODUCTION_COMPLETE', len(rows), 'errors', len(errors), 'loaded', count, flush=True)
